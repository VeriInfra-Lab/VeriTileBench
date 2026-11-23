import models.NanoGPT.Track1.MLPWeightPack
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact IR and executable PTX checked for the reference packer, the
delayed MLP scale update, and the paired Hopper packer. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
open VeriTile.Triton.FP QKVPack

private def referenceGraph : String := r#"
module {
tt.func public @_quantize_weights_dual_kernel(%w_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row_ptr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col_ptr: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial_amax_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %w_stride_l: i32 {tt.divisibility = 16 : i32}, %w_stride_h: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v = arith.constant dense<0.000000e+00> : tensor<64x64xbf16>
%mask_t = arith.constant dense<2816> : tensor<1x64xi32>
%cst = arith.constant dense<768> : tensor<64x1xi32>
%q = arith.constant dense<-4.480000e+02> : tensor<64x64xf32>
%q_0 = arith.constant dense<4.480000e+02> : tensor<64x64xf32>
%mask = arith.constant dense<768> : tensor<1x64xi32>
%cst_1 = arith.constant dense<2816> : tensor<64x1xi32>
%layer = tt.get_program_id x : i32
%tile = tt.get_program_id y : i32
%tile_h = arith.divsi %tile, 12 : i32
%tile_d = arith.remsi %tile, 12 : i32
%offs_h = arith.muli %tile_h, 64 : i32
%offs_h_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%offs_h_3 = tt.splat %offs_h : i32 -> tensor<64xi32>
%offs_h_4 = arith.addi %offs_h_3, %offs_h_2 : tensor<64xi32>
%offs_d = arith.muli %tile_d, 64 : i32
%offs_d_5 = tt.splat %offs_d : i32 -> tensor<64xi32>
%offs_d_6 = arith.addi %offs_d_5, %offs_h_2 : tensor<64xi32>
%mask_7 = tt.expand_dims %offs_h_4 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%mask_8 = arith.cmpi slt, %mask_7, %cst_1 : tensor<64x1xi32>
%mask_9 = tt.expand_dims %offs_d_6 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%mask_10 = arith.cmpi slt, %mask_9, %mask : tensor<1x64xi32>
%mask_11 = tt.broadcast %mask_8 : tensor<64x1xi1> -> tensor<64x64xi1>
%mask_12 = tt.broadcast %mask_10 : tensor<1x64xi1> -> tensor<64x64xi1>
%mask_13 = arith.andi %mask_11, %mask_12 : tensor<64x64xi1>
%v_14 = arith.muli %layer, %w_stride_l : i32
%v_15 = tt.addptr %w_ptr, %v_14 : !tt.ptr<bf16>, i32
%v_16 = tt.splat %w_stride_h : i32 -> tensor<64x1xi32>
%v_17 = arith.muli %mask_7, %v_16 : tensor<64x1xi32>
%v_18 = tt.splat %v_15 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
%v_19 = tt.addptr %v_18, %v_17 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
%v_20 = tt.broadcast %v_19 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x64x!tt.ptr<bf16>>
%v_21 = tt.broadcast %mask_9 : tensor<1x64xi32> -> tensor<64x64xi32>
%v_22 = tt.addptr %v_20, %v_21 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
%v_23 = tt.load %v_22, %mask_13, %v : tensor<64x64x!tt.ptr<bf16>>
%v_24 = arith.extf %v_23 : tensor<64x64xbf16> to tensor<64x64xf32>
%s = tt.addptr %scale_ptr, %layer : !tt.ptr<f32>, i32
%s_25 = tt.load %s : !tt.ptr<f32>
%q_26 = tt.splat %s_25 : f32 -> tensor<64x64xf32>
%q_27 = arith.divf %v_24, %q_26 : tensor<64x64xf32>
%q_28 = arith.minnumf %q_27, %q_0 : tensor<64x64xf32>
%q_29 = arith.maxnumf %q_28, %q : tensor<64x64xf32>
%q_30 = tt.fp_to_fp %q_29, rounding = rtne : tensor<64x64xf32> -> tensor<64x64xf8E4M3FN>
%0 = arith.muli %layer, 2162688 : i32
%1 = tt.addptr %row_ptr, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = arith.muli %mask_7, %cst : tensor<64x1xi32>
%3 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%4 = tt.addptr %3, %2 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%5 = tt.broadcast %4 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
%6 = tt.addptr %5, %v_21 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
tt.store %6, %q_30, %mask_13 : tensor<64x64x!tt.ptr<f8E4M3FN>>
%mask_t_31 = tt.expand_dims %offs_d_6 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%mask_t_32 = arith.cmpi slt, %mask_t_31, %cst : tensor<64x1xi32>
%mask_t_33 = tt.expand_dims %offs_h_4 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%mask_t_34 = arith.cmpi slt, %mask_t_33, %mask_t : tensor<1x64xi32>
%mask_t_35 = tt.broadcast %mask_t_32 : tensor<64x1xi1> -> tensor<64x64xi1>
%mask_t_36 = tt.broadcast %mask_t_34 : tensor<1x64xi1> -> tensor<64x64xi1>
%mask_t_37 = arith.andi %mask_t_35, %mask_t_36 : tensor<64x64xi1>
%7 = tt.addptr %col_ptr, %0 : !tt.ptr<f8E4M3FN>, i32
%8 = arith.muli %mask_t_31, %cst_1 : tensor<64x1xi32>
%9 = tt.splat %7 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%10 = tt.addptr %9, %8 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%11 = tt.broadcast %10 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x64x!tt.ptr<f8E4M3FN>>
%12 = tt.broadcast %mask_t_33 : tensor<1x64xi32> -> tensor<64x64xi32>
%13 = tt.addptr %11, %12 : tensor<64x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xi32>
%14 = tt.trans %q_30 {order = array<i32: 1, 0>} : tensor<64x64xf8E4M3FN> -> tensor<64x64xf8E4M3FN>
tt.store %13, %14, %mask_t_37 : tensor<64x64x!tt.ptr<f8E4M3FN>>
%15 = arith.muli %layer, 528 : i32
%16 = tt.addptr %partial_amax_ptr, %15 : !tt.ptr<f32>, i32
%17 = tt.addptr %16, %tile : !tt.ptr<f32>, i32
%18 = math.absf %v_24 : tensor<64x64xf32>
%19 = "tt.reduce"(%18) <{axis = 1 : i32}> ({
^bb0(%arg7: f32, %arg8: f32):
%21 = arith.maxnumf %arg7, %arg8 : f32
tt.reduce.return %21 : f32
}) : (tensor<64x64xf32>) -> tensor<64xf32>
%20 = "tt.reduce"(%19) <{axis = 0 : i32}> ({
^bb0(%arg7: f32, %arg8: f32):
%21 = arith.maxnumf %arg7, %arg8 : f32
tt.reduce.return %21 : f32
}) : (tensor<64xf32>) -> f32
tt.store %17, %20 : !tt.ptr<f32>
tt.return
}
}
"#

private def referencePtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _quantize_weights_dual_kernel(
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_0,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_1,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_2,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_3,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_4,
.param .u32 _quantize_weights_dual_kernel_param_5,
.param .u32 _quantize_weights_dual_kernel_param_6,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_7,
.param .u64 .ptr .global .align 1 _quantize_weights_dual_kernel_param_8
)
.reqntid 128
{
.reg .pred %p<8>;
.reg .b16 %rs<102>;
.reg .b32 %r<446>;
.reg .b64 %rd<30>;
$L__func_begin0:
ld.param.b64 %rd11, [_quantize_weights_dual_kernel_param_0];
ld.param.b64 %rd12, [_quantize_weights_dual_kernel_param_1];
$L__tmp0:
mov.u32 %r105, %ctaid.x;
ld.param.b64 %rd13, [_quantize_weights_dual_kernel_param_2];
mov.u32 %r106, %ctaid.y;
cvt.u16.u32 %rs33, %r106;
ld.param.b64 %rd14, [_quantize_weights_dual_kernel_param_3];
mul.hi.u16 %rs34, %rs33, -21845;
shr.u16 %rs35, %rs34, 3;
ld.param.b64 %rd15, [_quantize_weights_dual_kernel_param_4];
ld.param.b32 %r107, [_quantize_weights_dual_kernel_param_5];
mul.lo.s16 %rs36, %rs35, 12;
ld.param.b32 %r108, [_quantize_weights_dual_kernel_param_6];
sub.s16 %rs37, %rs33, %rs36;
mul.wide.u16 %r109, %rs35, 64;
mov.u32 %r110, %tid.x;
and.b32 %r111, %r110, 127;
and.b32 %r112, %r110, 31;
bfe.u32 %r113, %r110, 2, 5;
or.b32 %r114, %r113, 32;
and.b32 %r115, %r110, 3;
shl.b32 %r116, %r115, 4;
or.b32 %r117, %r109, %r113;
or.b32 %r118, %r109, %r114;
or.b32 %r119, %r109, %r116;
mul.wide.u16 %r120, %rs37, 64;
or.b32 %r121, %r116, %r120;
or.b32 %r122, %r113, %r120;
or.b32 %r123, %r114, %r120;
setp.lt.u32 %p1, %r106, 528;
mul.lo.s32 %r124, %r107, %r105;
mad.wide.s32 %rd16, %r124, 2, %rd11;
mul.lo.s32 %r125, %r108, %r117;
mul.lo.s32 %r126, %r108, %r118;
mad.wide.s32 %rd17, %r125, 2, %rd16;
mad.wide.s32 %rd18, %r126, 2, %rd16;
cvt.u64.u32 %rd19, %r121;
mul.wide.u32 %rd20, %r121, 2;
add.s64 %rd1, %rd17, %rd20;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd18, %rd20;
add.s64 %rd4, %rd3, 16;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs38, %rs39}, %r1;
mov.b32 {%rs40, %rs41}, %r2;
mov.b32 {%rs42, %rs43}, %r3;
mov.b32 {%rs44, %rs45}, %r4;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.b32 {%rs46, %rs47}, %r6;
mov.b32 {%rs48, %rs49}, %r7;
mov.b32 {%rs50, %rs51}, %r8;
mov.b32 {%rs52, %rs53}, %r9;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs54, %rs55}, %r10;
mov.b32 {%rs56, %rs57}, %r11;
mov.b32 {%rs58, %rs59}, %r12;
mov.b32 {%rs60, %rs61}, %r13;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs62, %rs63}, %r14;
mov.b32 {%rs64, %rs65}, %r15;
mov.b32 {%rs66, %rs67}, %r16;
mov.b32 {%rs68, %rs69}, %r17;
shl.b32 %r127, %r115, 5;
shl.b32 %r128, %r110, 4;
and.b32 %r129, %r128, 896;
shl.b32 %r130, %r110, 2;
and.b32 %r131, %r130, 16;
and.b32 %r132, %r110, 64;
bfe.s32 %r133, %r110, 6, 1;
and.b32 %r134, %r133, 2080;
or.b32 %r135, %r127, %r129;
xor.b32 %r136, %r135, %r134;
or.b32 %r137, %r136, %r131;
mov.b32 %r138, global_smem;
add.s32 %r139, %r138, %r137;
st.shared.v4.b32 [%r139], {%r1, %r2, %r3, %r4};
st.shared.v4.b32 [%r139+1024], {%r6, %r7, %r8, %r9};
xor.b32 %r140, %r137, 64;
add.s32 %r141, %r138, %r140;
st.shared.v4.b32 [%r141+4096], {%r10, %r11, %r12, %r13};
st.shared.v4.b32 [%r141+5120], {%r14, %r15, %r16, %r17};
bar.sync 0;
shl.b32 %r142, %r110, 10;
and.b32 %r143, %r142, 6144;
and.b32 %r144, %r128, 496;
shl.b32 %r145, %r110, 5;
and.b32 %r146, %r145, 1024;
shr.u32 %r147, %r132, 1;
or.b32 %r148, %r143, %r146;
xor.b32 %r149, %r144, %r147;
or.b32 %r150, %r148, %r149;
add.s32 %r151, %r138, %r150;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r152, %r153, %r154, %r155}, [%r151];
mov.b32 {%rs70, %rs71}, %r152;
mov.b32 {%rs72, %rs73}, %r153;
mov.b32 {%rs74, %rs75}, %r154;
mov.b32 {%rs76, %rs77}, %r155;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r156, %r157, %r158, %r159}, [%r151+512];
mov.b32 {%rs78, %rs79}, %r156;
mov.b32 {%rs80, %rs81}, %r157;
mov.b32 {%rs82, %rs83}, %r158;
mov.b32 {%rs84, %rs85}, %r159;
xor.b32 %r160, %r150, 64;
add.s32 %r161, %r138, %r160;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r162, %r163, %r164, %r165}, [%r161];
mov.b32 {%rs86, %rs87}, %r162;
mov.b32 {%rs88, %rs89}, %r163;
mov.b32 {%rs90, %rs91}, %r164;
mov.b32 {%rs92, %rs93}, %r165;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r166, %r167, %r168, %r169}, [%r161+512];
mov.b32 {%rs94, %rs95}, %r166;
mov.b32 {%rs96, %rs97}, %r167;
mov.b32 {%rs98, %rs99}, %r168;
mov.b32 {%rs100, %rs101}, %r169;
cvt.f32.bf16 %r170, %rs70;
cvt.f32.bf16 %r171, %rs71;
cvt.f32.bf16 %r172, %rs72;
cvt.f32.bf16 %r173, %rs73;
cvt.f32.bf16 %r174, %rs74;
cvt.f32.bf16 %r175, %rs75;
cvt.f32.bf16 %r176, %rs76;
cvt.f32.bf16 %r177, %rs77;
cvt.f32.bf16 %r178, %rs78;
cvt.f32.bf16 %r179, %rs79;
cvt.f32.bf16 %r180, %rs80;
cvt.f32.bf16 %r181, %rs81;
cvt.f32.bf16 %r182, %rs82;
cvt.f32.bf16 %r183, %rs83;
cvt.f32.bf16 %r184, %rs84;
cvt.f32.bf16 %r185, %rs85;
cvt.f32.bf16 %r186, %rs86;
cvt.f32.bf16 %r187, %rs87;
cvt.f32.bf16 %r188, %rs88;
cvt.f32.bf16 %r189, %rs89;
cvt.f32.bf16 %r190, %rs90;
cvt.f32.bf16 %r191, %rs91;
cvt.f32.bf16 %r192, %rs92;
cvt.f32.bf16 %r193, %rs93;
cvt.f32.bf16 %r194, %rs94;
cvt.f32.bf16 %r195, %rs95;
cvt.f32.bf16 %r196, %rs96;
cvt.f32.bf16 %r197, %rs97;
cvt.f32.bf16 %r198, %rs98;
cvt.f32.bf16 %r199, %rs99;
cvt.f32.bf16 %r200, %rs100;
cvt.f32.bf16 %r201, %rs101;
cvt.f32.bf16 %r202, %rs38;
cvt.f32.bf16 %r203, %rs39;
cvt.f32.bf16 %r204, %rs40;
cvt.f32.bf16 %r205, %rs41;
cvt.f32.bf16 %r206, %rs42;
cvt.f32.bf16 %r207, %rs43;
cvt.f32.bf16 %r208, %rs44;
cvt.f32.bf16 %r209, %rs45;
cvt.f32.bf16 %r210, %rs46;
cvt.f32.bf16 %r211, %rs47;
cvt.f32.bf16 %r212, %rs48;
cvt.f32.bf16 %r213, %rs49;
cvt.f32.bf16 %r214, %rs50;
cvt.f32.bf16 %r215, %rs51;
cvt.f32.bf16 %r216, %rs52;
cvt.f32.bf16 %r217, %rs53;
cvt.f32.bf16 %r218, %rs54;
cvt.f32.bf16 %r219, %rs55;
cvt.f32.bf16 %r220, %rs56;
cvt.f32.bf16 %r221, %rs57;
cvt.f32.bf16 %r222, %rs58;
cvt.f32.bf16 %r223, %rs59;
cvt.f32.bf16 %r224, %rs60;
cvt.f32.bf16 %r225, %rs61;
cvt.f32.bf16 %r226, %rs62;
cvt.f32.bf16 %r227, %rs63;
cvt.f32.bf16 %r228, %rs64;
cvt.f32.bf16 %r229, %rs65;
cvt.f32.bf16 %r230, %rs66;
cvt.f32.bf16 %r231, %rs67;
cvt.f32.bf16 %r232, %rs68;
cvt.f32.bf16 %r233, %rs69;
mad.wide.u32 %rd5, %r105, 4, %rd14;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd5 + 0 ];
div.full.f32 %r234, %r170, %r18;
div.full.f32 %r235, %r171, %r18;
div.full.f32 %r236, %r172, %r18;
div.full.f32 %r237, %r173, %r18;
div.full.f32 %r238, %r174, %r18;
div.full.f32 %r239, %r175, %r18;
div.full.f32 %r240, %r176, %r18;
div.full.f32 %r241, %r177, %r18;
div.full.f32 %r242, %r178, %r18;
div.full.f32 %r243, %r179, %r18;
div.full.f32 %r244, %r180, %r18;
div.full.f32 %r245, %r181, %r18;
div.full.f32 %r246, %r182, %r18;
div.full.f32 %r247, %r183, %r18;
div.full.f32 %r248, %r184, %r18;
div.full.f32 %r249, %r185, %r18;
div.full.f32 %r250, %r186, %r18;
div.full.f32 %r251, %r187, %r18;
div.full.f32 %r252, %r188, %r18;
div.full.f32 %r253, %r189, %r18;
div.full.f32 %r254, %r190, %r18;
div.full.f32 %r255, %r191, %r18;
div.full.f32 %r256, %r192, %r18;
div.full.f32 %r257, %r193, %r18;
div.full.f32 %r258, %r194, %r18;
div.full.f32 %r259, %r195, %r18;
div.full.f32 %r260, %r196, %r18;
div.full.f32 %r261, %r197, %r18;
div.full.f32 %r262, %r198, %r18;
div.full.f32 %r263, %r199, %r18;
div.full.f32 %r264, %r200, %r18;
div.full.f32 %r265, %r201, %r18;
div.full.f32 %r266, %r202, %r18;
div.full.f32 %r267, %r203, %r18;
div.full.f32 %r268, %r204, %r18;
div.full.f32 %r269, %r205, %r18;
div.full.f32 %r270, %r206, %r18;
div.full.f32 %r271, %r207, %r18;
div.full.f32 %r272, %r208, %r18;
div.full.f32 %r273, %r209, %r18;
div.full.f32 %r274, %r210, %r18;
div.full.f32 %r275, %r211, %r18;
div.full.f32 %r276, %r212, %r18;
div.full.f32 %r277, %r213, %r18;
div.full.f32 %r278, %r214, %r18;
div.full.f32 %r279, %r215, %r18;
div.full.f32 %r280, %r216, %r18;
div.full.f32 %r281, %r217, %r18;
div.full.f32 %r282, %r218, %r18;
div.full.f32 %r283, %r219, %r18;
div.full.f32 %r284, %r220, %r18;
div.full.f32 %r285, %r221, %r18;
div.full.f32 %r286, %r222, %r18;
div.full.f32 %r287, %r223, %r18;
div.full.f32 %r288, %r224, %r18;
div.full.f32 %r289, %r225, %r18;
div.full.f32 %r290, %r226, %r18;
div.full.f32 %r291, %r227, %r18;
div.full.f32 %r292, %r228, %r18;
div.full.f32 %r293, %r229, %r18;
div.full.f32 %r294, %r230, %r18;
div.full.f32 %r295, %r231, %r18;
div.full.f32 %r296, %r232, %r18;
div.full.f32 %r297, %r233, %r18;
min.f32 %r298, %r234, 0f43E00000;
min.f32 %r299, %r235, 0f43E00000;
min.f32 %r300, %r236, 0f43E00000;
min.f32 %r301, %r237, 0f43E00000;
min.f32 %r302, %r238, 0f43E00000;
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
max.f32 %r19, %r298, 0fC3E00000;
max.f32 %r20, %r299, 0fC3E00000;
max.f32 %r21, %r300, 0fC3E00000;
max.f32 %r22, %r301, 0fC3E00000;
max.f32 %r23, %r302, 0fC3E00000;
max.f32 %r24, %r303, 0fC3E00000;
max.f32 %r25, %r304, 0fC3E00000;
max.f32 %r26, %r305, 0fC3E00000;
max.f32 %r27, %r306, 0fC3E00000;
max.f32 %r28, %r307, 0fC3E00000;
max.f32 %r29, %r308, 0fC3E00000;
max.f32 %r30, %r309, 0fC3E00000;
max.f32 %r31, %r310, 0fC3E00000;
max.f32 %r32, %r311, 0fC3E00000;
max.f32 %r33, %r312, 0fC3E00000;
max.f32 %r34, %r313, 0fC3E00000;
max.f32 %r35, %r314, 0fC3E00000;
max.f32 %r36, %r315, 0fC3E00000;
max.f32 %r37, %r316, 0fC3E00000;
max.f32 %r38, %r317, 0fC3E00000;
max.f32 %r39, %r318, 0fC3E00000;
max.f32 %r40, %r319, 0fC3E00000;
max.f32 %r41, %r320, 0fC3E00000;
max.f32 %r42, %r321, 0fC3E00000;
max.f32 %r43, %r322, 0fC3E00000;
max.f32 %r44, %r323, 0fC3E00000;
max.f32 %r45, %r324, 0fC3E00000;
max.f32 %r46, %r325, 0fC3E00000;
max.f32 %r47, %r326, 0fC3E00000;
max.f32 %r48, %r327, 0fC3E00000;
max.f32 %r49, %r328, 0fC3E00000;
max.f32 %r50, %r329, 0fC3E00000;
max.f32 %r51, %r330, 0fC3E00000;
max.f32 %r52, %r331, 0fC3E00000;
max.f32 %r53, %r332, 0fC3E00000;
max.f32 %r54, %r333, 0fC3E00000;
max.f32 %r55, %r334, 0fC3E00000;
max.f32 %r56, %r335, 0fC3E00000;
max.f32 %r57, %r336, 0fC3E00000;
max.f32 %r58, %r337, 0fC3E00000;
max.f32 %r59, %r338, 0fC3E00000;
max.f32 %r60, %r339, 0fC3E00000;
max.f32 %r61, %r340, 0fC3E00000;
max.f32 %r62, %r341, 0fC3E00000;
max.f32 %r63, %r342, 0fC3E00000;
max.f32 %r64, %r343, 0fC3E00000;
max.f32 %r65, %r344, 0fC3E00000;
max.f32 %r66, %r345, 0fC3E00000;
max.f32 %r67, %r346, 0fC3E00000;
max.f32 %r68, %r347, 0fC3E00000;
max.f32 %r69, %r348, 0fC3E00000;
max.f32 %r70, %r349, 0fC3E00000;
max.f32 %r71, %r350, 0fC3E00000;
max.f32 %r72, %r351, 0fC3E00000;
max.f32 %r73, %r352, 0fC3E00000;
max.f32 %r74, %r353, 0fC3E00000;
max.f32 %r75, %r354, 0fC3E00000;
max.f32 %r76, %r355, 0fC3E00000;
max.f32 %r77, %r356, 0fC3E00000;
max.f32 %r78, %r357, 0fC3E00000;
max.f32 %r79, %r358, 0fC3E00000;
max.f32 %r80, %r359, 0fC3E00000;
max.f32 %r81, %r360, 0fC3E00000;
max.f32 %r82, %r361, 0fC3E00000;
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
cvt.rn.satfinite.e4m3x2.f32 %rs17, %r52, %r51;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r54, %r53;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r56, %r55;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r58, %r57;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r60, %r59;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r62, %r61;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r64, %r63;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r66, %r65;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r68, %r67;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r70, %r69;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r72, %r71;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r74, %r73;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r76, %r75;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r78, %r77;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r80, %r79;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r82, %r81;
mul.lo.s32 %r362, %r105, 2162688;
cvt.s64.s32 %rd21, %r362;
add.s64 %rd22, %rd12, %rd21;
mad.wide.u32 %rd23, %r117, 768, %rd22;
mad.wide.u32 %rd24, %r118, 768, %rd22;
add.s64 %rd6, %rd23, %rd19;
add.s64 %rd7, %rd24, %rd19;
mov.b32 %r83, {%rs17, %rs18};
mov.b32 %r84, {%rs19, %rs20};
mov.b32 %r85, {%rs21, %rs22};
mov.b32 %r86, {%rs23, %rs24};
@%p1 st.global.v4.b32 [ %rd6 + 0 ], { %r83, %r84, %r85, %r86 };
mov.b32 %r87, {%rs25, %rs26};
mov.b32 %r88, {%rs27, %rs28};
mov.b32 %r89, {%rs29, %rs30};
mov.b32 %r90, {%rs31, %rs32};
@%p1 st.global.v4.b32 [ %rd7 + 0 ], { %r87, %r88, %r89, %r90 };
add.s64 %rd25, %rd13, %rd21;
mad.wide.u32 %rd26, %r122, 2816, %rd25;
mad.wide.u32 %rd27, %r123, 2816, %rd25;
cvt.u64.u32 %rd28, %r119;
add.s64 %rd8, %rd26, %rd28;
add.s64 %rd9, %rd27, %rd28;
mov.b32 %r91, {%rs1, %rs2};
mov.b32 %r92, {%rs3, %rs4};
mov.b32 %r93, {%rs5, %rs6};
mov.b32 %r94, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd8 + 0 ], { %r91, %r92, %r93, %r94 };
mov.b32 %r95, {%rs9, %rs10};
mov.b32 %r96, {%rs11, %rs12};
mov.b32 %r97, {%rs13, %rs14};
mov.b32 %r98, {%rs15, %rs16};
@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r95, %r96, %r97, %r98 };
mul.lo.s32 %r363, %r105, 528;
mad.wide.s32 %rd29, %r363, 4, %rd15;
mad.wide.u32 %rd10, %r106, 4, %rd29;
abs.f32 %r364, %r202;
abs.f32 %r365, %r203;
abs.f32 %r366, %r204;
abs.f32 %r367, %r205;
abs.f32 %r368, %r206;
abs.f32 %r369, %r207;
abs.f32 %r370, %r208;
abs.f32 %r371, %r209;
abs.f32 %r372, %r210;
abs.f32 %r373, %r211;
abs.f32 %r374, %r212;
abs.f32 %r375, %r213;
abs.f32 %r376, %r214;
abs.f32 %r377, %r215;
abs.f32 %r378, %r216;
abs.f32 %r379, %r217;
abs.f32 %r380, %r218;
abs.f32 %r381, %r219;
abs.f32 %r382, %r220;
abs.f32 %r383, %r221;
abs.f32 %r384, %r222;
abs.f32 %r385, %r223;
abs.f32 %r386, %r224;
abs.f32 %r387, %r225;
abs.f32 %r388, %r226;
abs.f32 %r389, %r227;
abs.f32 %r390, %r228;
abs.f32 %r391, %r229;
abs.f32 %r392, %r230;
abs.f32 %r393, %r231;
abs.f32 %r394, %r232;
abs.f32 %r395, %r233;
$L__tmp1:
max.f32 %r396, %r364, %r365;
max.f32 %r397, %r396, %r366;
max.f32 %r398, %r397, %r367;
max.f32 %r399, %r398, %r368;
max.f32 %r400, %r399, %r369;
max.f32 %r401, %r400, %r370;
max.f32 %r402, %r401, %r371;
max.f32 %r403, %r402, %r372;
max.f32 %r404, %r403, %r373;
max.f32 %r405, %r404, %r374;
max.f32 %r406, %r405, %r375;
max.f32 %r407, %r406, %r376;
max.f32 %r408, %r407, %r377;
max.f32 %r409, %r408, %r378;
max.f32 %r410, %r409, %r379;
max.f32 %r411, %r380, %r381;
max.f32 %r412, %r411, %r382;
max.f32 %r413, %r412, %r383;
max.f32 %r414, %r413, %r384;
max.f32 %r415, %r414, %r385;
max.f32 %r416, %r415, %r386;
max.f32 %r417, %r416, %r387;
max.f32 %r418, %r417, %r388;
max.f32 %r419, %r418, %r389;
max.f32 %r420, %r419, %r390;
max.f32 %r421, %r420, %r391;
max.f32 %r422, %r421, %r392;
max.f32 %r423, %r422, %r393;
max.f32 %r424, %r423, %r394;
max.f32 %r425, %r424, %r395;
$L__tmp2:
shfl.sync.bfly.b32 %r426, %r410, 2, 31, -1;
$L__tmp3:
max.f32 %r427, %r410, %r426;
$L__tmp4:
shfl.sync.bfly.b32 %r428, %r427, 1, 31, -1;
$L__tmp5:
max.f32 %r429, %r427, %r428;
$L__tmp6:
shfl.sync.bfly.b32 %r430, %r425, 2, 31, -1;
$L__tmp7:
max.f32 %r431, %r425, %r430;
$L__tmp8:
shfl.sync.bfly.b32 %r432, %r431, 1, 31, -1;
$L__tmp9:
max.f32 %r433, %r431, %r432;
$L__tmp10:
bar.sync 0;
$L__tmp11:
max.f32 %r434, %r429, %r433;
$L__tmp12:
shfl.sync.bfly.b32 %r435, %r434, 16, 31, -1;
$L__tmp13:
max.f32 %r436, %r434, %r435;
$L__tmp14:
shfl.sync.bfly.b32 %r437, %r436, 8, 31, -1;
$L__tmp15:
max.f32 %r438, %r436, %r437;
$L__tmp16:
shfl.sync.bfly.b32 %r439, %r438, 4, 31, -1;
$L__tmp17:
max.f32 %r100, %r438, %r439;
$L__tmp18:
setp.eq.b32 %p2, %r112, 0;
shr.u32 %r440, %r110, 3;
and.b32 %r441, %r440, 12;
add.s32 %r99, %r138, %r441;
@%p2 st.shared.b32 [ %r99 + 0 ], %r100;
bar.sync 0;
setp.lt.u32 %p3, %r111, 4;
shl.b32 %r442, %r111, 2;
add.s32 %r102, %r138, %r442;
@%p3 ld.shared.b32 %r101, [ %r102 + 0 ];
shfl.sync.bfly.b32 %r443, %r101, 2, 31, -1;
$L__tmp19:
max.f32 %r444, %r101, %r443;
$L__tmp20:
shfl.sync.bfly.b32 %r445, %r444, 1, 31, -1;
$L__tmp21:
max.f32 %r103, %r444, %r445;
$L__tmp22:
setp.eq.b32 %p6, %r115, 0;
and.pred %p4, %p3, %p6;
@%p4 st.shared.b32 [ %r102 + 0 ], %r103;
bar.sync 0;
ld.shared.b32 %r104, [global_smem];
$L__tmp23:
setp.lt.u32 %p7, %r111, 32;
and.pred %p5, %p2, %p7;
@%p5 st.global.b32 [ %rd10 + 0 ], { %r104 };
ret;
$L__tmp24:
$L__func_end0:
}
"#

private def scaleGraph : String := r#"
module {
tt.func public @_mlp_delayed_scales(%partial0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant 9.99999996E-13 : f32
%cst_0 = arith.constant 2.500000e-03 : f32
%values = arith.constant dense<0xFF800000> : tensor<1024xf32>
%values_1 = arith.constant dense<528> : tensor<1024xi32>
%up = arith.constant 0 : i32
%matrix = tt.get_program_id x : i32
%layer = arith.divsi %matrix, 2 : i32
%up_2 = arith.remsi %matrix, 2 : i32
%up_3 = arith.cmpi eq, %up_2, %up : i32
%partial = arith.select %up_3, %partial0, %partial1 : !tt.ptr<f32>
%scale = arith.select %up_3, %scale0, %scale1 : !tt.ptr<f32>
%lane = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
%values_4 = arith.muli %layer, 528 : i32
%values_5 = tt.addptr %partial, %values_4 : !tt.ptr<f32>, i32
%values_6 = tt.splat %values_5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
%values_7 = tt.addptr %values_6, %lane : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
%values_8 = arith.cmpi slt, %lane, %values_1 : tensor<1024xi32>
%values_9 = tt.load %values_7, %values_8, %values : tensor<1024x!tt.ptr<f32>>
%amax = "tt.reduce"(%values_9) <{axis = 0 : i32}> ({
^bb0(%arg4: f32, %arg5: f32):
%amax_11 = arith.maximumf %arg4, %arg5 : f32
tt.reduce.return %amax_11 : f32
}) : (tensor<1024xf32>) -> f32
%product = arith.mulf %amax, %cst_0 : f32
%clamped = arith.cmpf olt, %product, %cst : f32
%clamped_10 = arith.select %clamped, %cst, %product : f32
%0 = tt.addptr %scale, %layer : !tt.ptr<f32>, i32
tt.store %0, %clamped_10 : !tt.ptr<f32>
tt.return
}
}
"#

private def scalePtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _mlp_delayed_scales(
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_0,
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_1,
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_2,
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_3,
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_4,
.param .u64 .ptr .global .align 1 _mlp_delayed_scales_param_5
)
.reqntid 128
{
.reg .pred %p<11>;
.reg .b32 %r<52>;
.reg .b64 %rd<11>;
$L__func_begin0:
ld.param.b64 %rd4, [_mlp_delayed_scales_param_0];
ld.param.b64 %rd5, [_mlp_delayed_scales_param_1];
$L__tmp0:
mov.u32 %r16, %ctaid.x;
shr.u32 %r17, %r16, 1;
ld.param.b64 %rd6, [_mlp_delayed_scales_param_2];
ld.param.b64 %rd7, [_mlp_delayed_scales_param_3];
and.b32 %r18, %r16, 1;
setp.ne.b32 %p7, %r18, 0;
selp.b64 %rd8, %rd5, %rd4, %p7;
selp.b64 %rd9, %rd7, %rd6, %p7;
mov.u32 %r19, %tid.x;
and.b32 %r20, %r19, 127;
and.b32 %r21, %r19, 31;
shl.b32 %r22, %r19, 2;
and.b32 %r23, %r22, 508;
or.b32 %r24, %r23, 512;
mul.lo.s32 %r25, %r17, 528;
mad.wide.s32 %rd10, %r25, 4, %rd8;
mad.wide.u32 %rd1, %r23, 4, %rd10;
add.s64 %rd2, %rd1, 2048;
setp.lt.u32 %p2, %r24, 528;
mov.b32 %r5, -8388608;
mov.pred %p1, -1;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
$L__tmp1:
max.NaN.f32 %r26, %r1, %r2;
max.NaN.f32 %r27, %r26, %r3;
max.NaN.f32 %r28, %r27, %r4;
max.NaN.f32 %r29, %r28, %r6;
max.NaN.f32 %r30, %r29, %r7;
max.NaN.f32 %r31, %r30, %r8;
max.NaN.f32 %r32, %r31, %r9;
$L__tmp2:
shfl.sync.bfly.b32 %r33, %r32, 16, 31, -1;
$L__tmp3:
max.NaN.f32 %r34, %r32, %r33;
$L__tmp4:
shfl.sync.bfly.b32 %r35, %r34, 8, 31, -1;
$L__tmp5:
max.NaN.f32 %r36, %r34, %r35;
$L__tmp6:
shfl.sync.bfly.b32 %r37, %r36, 4, 31, -1;
$L__tmp7:
max.NaN.f32 %r38, %r36, %r37;
$L__tmp8:
shfl.sync.bfly.b32 %r39, %r38, 2, 31, -1;
$L__tmp9:
max.NaN.f32 %r40, %r38, %r39;
$L__tmp10:
shfl.sync.bfly.b32 %r41, %r40, 1, 31, -1;
$L__tmp11:
max.NaN.f32 %r11, %r40, %r41;
$L__tmp12:
setp.eq.b32 %p3, %r21, 0;
shr.u32 %r42, %r19, 3;
and.b32 %r43, %r42, 12;
mov.b32 %r44, global_smem;
add.s32 %r10, %r44, %r43;
@%p3 st.shared.b32 [ %r10 + 0 ], %r11;
bar.sync 0;
setp.lt.u32 %p4, %r20, 4;
shl.b32 %r45, %r20, 2;
add.s32 %r13, %r44, %r45;
@%p4 ld.shared.b32 %r12, [ %r13 + 0 ];
shfl.sync.bfly.b32 %r46, %r12, 2, 31, -1;
$L__tmp13:
max.NaN.f32 %r47, %r12, %r46;
$L__tmp14:
shfl.sync.bfly.b32 %r48, %r47, 1, 31, -1;
$L__tmp15:
max.NaN.f32 %r14, %r47, %r48;
$L__tmp16:
and.b32 %r49, %r19, 3;
setp.eq.b32 %p8, %r49, 0;
and.pred %p5, %p4, %p8;
@%p5 st.shared.b32 [ %r13 + 0 ], %r14;
bar.sync 0;
ld.shared.b32 %r50, [global_smem];
mul.f32 %r51, %r50, 0f3B23D70A;
setp.lt.f32 %p9, %r51, 0f2B8CBCCC;
selp.f32 %r15, 0f2B8CBCCC, %r51, %p9;
mad.wide.u32 %rd3, %r17, 4, %rd9;
setp.lt.u32 %p10, %r20, 32;
and.pred %p6, %p3, %p10;
@%p6 st.global.b32 [ %rd3 + 0 ], { %r15 };
ret;
$L__tmp17:
$L__func_end0:
}
"#

private def packGraph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 1, 16], threadsPerWarp = [1, 4, 2, 4], warpsPerCTA = [8, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 2, 16], threadsPerWarp = [2, 1, 4, 4], warpsPerCTA = [1, 8, 1, 1], order = [3, 0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2, 1, 16], threadsPerWarp = [2, 1, 4, 1, 4, 1], warpsPerCTA = [1, 8, 1, 1, 1, 1], order = [5, 4, 0, 3, 2, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_mlp_paired_pack(%bank: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %row1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scales0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scales1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32, #blocked>
%cst_1 = arith.constant dense<-4.480000e+02> : tensor<64x128xf32, #blocked>
%cst_2 = arith.constant dense<2816> : tensor<1x128xi32, #blocked1>
%matrix = tt.get_program_id z : i32
%tile_h = tt.get_program_id y : i32
%pair_d = tt.get_program_id x : i32
%start_h = arith.muli %tile_h, 64 : i32
%start_d = arith.muli %pair_d, 128 : i32
%layer = arith.divsi %matrix, 2 : i32
%up = arith.remsi %matrix, 2 : i32
%up_3 = arith.cmpi eq, %up, 0 : i32
%rr = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cc = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%tile_base = arith.muli %tile_h, 49152 : i32
%tile_base_4 = arith.addi %tile_base, %start_d : i32
%offset = tt.expand_dims %rr {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%offset_5 = arith.muli %offset, %cst : tensor<64x1xi32, #blocked>
%offset_6 = tt.expand_dims %cc {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%offset_7 = tt.broadcast %offset_5 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%offset_8 = tt.broadcast %offset_6 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%offset_9 = arith.addi %offset_7, %offset_8 : tensor<64x128xi32, #blocked>
%values = arith.muli %matrix, 2162688 : i32
%values_10 = arith.addi %values, %tile_base_4 : i32
%values_11 = tt.addptr %bank, %values_10 : !tt.ptr<bf16>, i32
%values_12 = tt.splat %values_11 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_13 = tt.addptr %values_12, %offset_9 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%values_14 = tt.load %values_13 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_15 = arith.extf %values_14 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
%scale = arith.select %up_3, %scales0, %scales1 : !tt.ptr<f32>
%scale_16 = tt.addptr %scale, %layer : !tt.ptr<f32>, i32
%scale_17 = tt.load %scale_16 : !tt.ptr<f32>
%q = tt.splat %scale_17 : f32 -> tensor<64x128xf32, #blocked>
%q_18 = arith.divf %values_15, %q : tensor<64x128xf32, #blocked>
%q_19 = arith.minnumf %q_18, %cst_0 : tensor<64x128xf32, #blocked>
%q_20 = arith.maxnumf %q_19, %cst_1 : tensor<64x128xf32, #blocked>
%q_21 = tt.fp_to_fp %q_20, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
%row = arith.select %up_3, %row0, %row1 : !tt.ptr<f8E4M3FN>
%row_22 = arith.muli %layer, 2162688 : i32
%row_23 = tt.addptr %row, %row_22 : !tt.ptr<f8E4M3FN>, i32
%col = arith.select %up_3, %col0, %col1 : !tt.ptr<f8E4M3FN>
%col_24 = tt.addptr %col, %row_22 : !tt.ptr<f8E4M3FN>, i32
%0 = tt.addptr %row_23, %tile_base_4 : !tt.ptr<f8E4M3FN>, i32
%1 = tt.splat %0 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%2 = tt.addptr %1, %offset_9 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %2, %q_21 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%partials = ttg.local_alloc : () -> !ttg.memdesc<2x8xf32, #shared, #smem, mutable>
%t = ttg.convert_layout %q_21 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
%th = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%td = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%start_d_25 = arith.muli %pair_d, 360448 : i32
%3 = arith.addi %start_d_25, %start_h : i32
%4 = tt.addptr %col_24, %3 : !tt.ptr<f8E4M3FN>, i32
%5 = tt.expand_dims %td {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%6 = arith.muli %5, %cst_2 : tensor<1x128xi32, #blocked1>
%7 = tt.expand_dims %th {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%8 = tt.broadcast %6 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%9 = tt.broadcast %7 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%10 = arith.addi %8, %9 : tensor<64x128xi32, #blocked1>
%11 = tt.splat %4 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%12 = tt.addptr %11, %10 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %12, %t : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%tile = arith.muli %tile_h, 12 : i32
%tile_26 = arith.muli %pair_d, 2 : i32
%tile_27 = arith.addi %tile, %tile_26 : i32
%warp_halves = math.absf %values_15 : tensor<64x128xf32, #blocked>
%warp_halves_28 = tt.reshape %warp_halves : tensor<64x128xf32, #blocked> -> tensor<8x8x2x64xf32, #blocked2>
%warp_halves_29 = tt.trans %warp_halves_28 {order = array<i32: 2, 0, 1, 3>} : tensor<8x8x2x64xf32, #blocked2> -> tensor<2x8x8x64xf32, #blocked3>
%row_pairs = tt.reshape %warp_halves_29 : tensor<2x8x8x64xf32, #blocked3> -> tensor<2x8x4x2x4x16xf32, #blocked4>
%local_amax = "tt.reduce"(%row_pairs) <{axis = 5 : i32}> ({
^bb0(%local_amax_32: f32, %local_amax_33: f32):
%local_amax_34 = arith.maxnumf %local_amax_32, %local_amax_33 : f32
tt.reduce.return %local_amax_34 : f32
}) : (tensor<2x8x4x2x4x16xf32, #blocked4>) -> tensor<2x8x4x2x4xf32, #ttg.slice<{dim = 5, parent = #blocked4}>>
%local_amax_30 = "tt.reduce"(%local_amax) <{axis = 3 : i32}> ({
^bb0(%local_amax_32: f32, %local_amax_33: f32):
%local_amax_34 = arith.maxnumf %local_amax_32, %local_amax_33 : f32
tt.reduce.return %local_amax_34 : f32
}) : (tensor<2x8x4x2x4xf32, #ttg.slice<{dim = 5, parent = #blocked4}>>) -> tensor<2x8x4x4xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>>
%warp_amax = "tt.reduce"(%local_amax_30) <{axis = 3 : i32}> ({
^bb0(%warp_amax_32: f32, %warp_amax_33: f32):
%warp_amax_34 = arith.maxnumf %warp_amax_32, %warp_amax_33 : f32
tt.reduce.return %warp_amax_34 : f32
}) : (tensor<2x8x4x4xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>>) -> tensor<2x8x4xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>}>>
%warp_amax_31 = "tt.reduce"(%warp_amax) <{axis = 2 : i32}> ({
^bb0(%warp_amax_32: f32, %warp_amax_33: f32):
%warp_amax_34 = arith.maxnumf %warp_amax_32, %warp_amax_33 : f32
tt.reduce.return %warp_amax_34 : f32
}) : (tensor<2x8x4xf32, #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>}>>) -> tensor<2x8xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>}>}>>
ttg.local_store %warp_amax_31, %partials : tensor<2x8xf32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 3, parent = #ttg.slice<{dim = 5, parent = #blocked4}>}>}>}>> -> !ttg.memdesc<2x8xf32, #shared, #smem, mutable>
%grouped = ttg.local_load %partials : !ttg.memdesc<2x8xf32, #shared, #smem, mutable> -> tensor<2x8xf32, #blocked5>
%amax = "tt.reduce"(%grouped) <{axis = 1 : i32}> ({
^bb0(%amax_32: f32, %amax_33: f32):
%amax_34 = arith.maxnumf %amax_32, %amax_33 : f32
tt.reduce.return %amax_34 : f32
}) : (tensor<2x8xf32, #blocked5>) -> tensor<2xf32, #ttg.slice<{dim = 1, parent = #blocked5}>>
%halves_id = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
%13 = arith.select %up_3, %partial0, %partial1 : !tt.ptr<f32>
%14 = arith.muli %layer, 528 : i32
%15 = tt.addptr %13, %14 : !tt.ptr<f32>, i32
%16 = tt.addptr %15, %tile_27 : !tt.ptr<f32>, i32
%17 = tt.splat %16 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked5}>>
%18 = tt.addptr %17, %halves_id : tensor<2x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked5}>>, tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked5}>>
tt.store %18, %amax : tensor<2x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked5}>>
tt.return
}
}
"#

private def packPtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _mlp_paired_pack(
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_0,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_1,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_2,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_3,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_4,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_5,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_6,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_7,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_8,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_9,
.param .u64 .ptr .global .align 1 _mlp_paired_pack_param_10
)
.reqntid 256
{
.reg .pred %p<3>;
.reg .b16 %rs<49>;
.reg .b32 %r<305>;
.reg .b64 %rd<39>;
$L__func_begin0:
ld.param.b64 %rd13, [_mlp_paired_pack_param_0];
ld.param.b64 %rd14, [_mlp_paired_pack_param_1];
$L__tmp0:
mov.u32 %r67, %ctaid.z;
ld.param.b64 %rd15, [_mlp_paired_pack_param_2];
mov.u32 %r68, %ctaid.y;
ld.param.b64 %rd16, [_mlp_paired_pack_param_3];
mov.u32 %r69, %ctaid.x;
shl.b32 %r70, %r68, 6;
ld.param.b64 %rd17, [_mlp_paired_pack_param_4];
shl.b32 %r71, %r69, 7;
ld.param.b64 %rd18, [_mlp_paired_pack_param_5];
shr.u32 %r72, %r67, 1;
ld.param.b64 %rd19, [_mlp_paired_pack_param_6];
ld.param.b64 %rd20, [_mlp_paired_pack_param_7];
and.b32 %r73, %r67, 1;
setp.ne.b32 %p2, %r73, 0;
ld.param.b64 %rd21, [_mlp_paired_pack_param_8];
mov.u32 %r74, %tid.x;
shr.u32 %r75, %r74, 2;
and.b32 %r76, %r75, 62;
and.b32 %r77, %r74, 7;
shl.b32 %r78, %r77, 4;
mad.lo.s32 %r79, %r68, 49152, %r71;
mul.lo.s32 %r80, %r76, 768;
or.b32 %r81, %r80, %r78;
mad.lo.s32 %r82, %r67, 2162688, %r79;
mad.wide.s32 %rd22, %r82, 2, %rd13;
cvt.u64.u32 %rd23, %r81;
mad.wide.u32 %rd1, %r81, 2, %rd22;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd1, 1536;
or.b32 %r83, %r81, 8;
mad.wide.u32 %rd24, %r83, 2, %rd22;
add.s64 %rd4, %rd24, 1536;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs17, %rs18}, %r1;
mov.b32 {%rs19, %rs20}, %r2;
mov.b32 {%rs21, %rs22}, %r3;
mov.b32 {%rs23, %rs24}, %r4;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mov.b32 {%rs25, %rs26}, %r5;
mov.b32 {%rs27, %rs28}, %r6;
mov.b32 {%rs29, %rs30}, %r7;
mov.b32 {%rs31, %rs32}, %r8;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
mov.b32 {%rs33, %rs34}, %r9;
mov.b32 {%rs35, %rs36}, %r10;
mov.b32 {%rs37, %rs38}, %r11;
mov.b32 {%rs39, %rs40}, %r12;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
mov.b32 {%rs41, %rs42}, %r13;
mov.b32 {%rs43, %rs44}, %r14;
mov.b32 {%rs45, %rs46}, %r15;
mov.b32 {%rs47, %rs48}, %r16;
cvt.f32.bf16 %r84, %rs17;
cvt.f32.bf16 %r85, %rs18;
cvt.f32.bf16 %r86, %rs19;
cvt.f32.bf16 %r87, %rs20;
cvt.f32.bf16 %r88, %rs21;
cvt.f32.bf16 %r89, %rs22;
cvt.f32.bf16 %r90, %rs23;
cvt.f32.bf16 %r91, %rs24;
cvt.f32.bf16 %r92, %rs25;
cvt.f32.bf16 %r93, %rs26;
cvt.f32.bf16 %r94, %rs27;
cvt.f32.bf16 %r95, %rs28;
cvt.f32.bf16 %r96, %rs29;
cvt.f32.bf16 %r97, %rs30;
cvt.f32.bf16 %r98, %rs31;
cvt.f32.bf16 %r99, %rs32;
cvt.f32.bf16 %r100, %rs33;
cvt.f32.bf16 %r101, %rs34;
cvt.f32.bf16 %r102, %rs35;
cvt.f32.bf16 %r103, %rs36;
cvt.f32.bf16 %r104, %rs37;
cvt.f32.bf16 %r105, %rs38;
cvt.f32.bf16 %r106, %rs39;
cvt.f32.bf16 %r107, %rs40;
cvt.f32.bf16 %r108, %rs41;
cvt.f32.bf16 %r109, %rs42;
cvt.f32.bf16 %r110, %rs43;
cvt.f32.bf16 %r111, %rs44;
cvt.f32.bf16 %r112, %rs45;
cvt.f32.bf16 %r113, %rs46;
cvt.f32.bf16 %r114, %rs47;
cvt.f32.bf16 %r115, %rs48;
selp.b64 %rd25, %rd19, %rd18, %p2;
mad.wide.u32 %rd5, %r72, 4, %rd25;
mov.u32 %r17, 0x0;
ld.global.b32 { %r17 }, [ %rd5 + 0 ];
div.full.f32 %r116, %r84, %r17;
div.full.f32 %r117, %r85, %r17;
div.full.f32 %r118, %r86, %r17;
div.full.f32 %r119, %r87, %r17;
div.full.f32 %r120, %r88, %r17;
div.full.f32 %r121, %r89, %r17;
div.full.f32 %r122, %r90, %r17;
div.full.f32 %r123, %r91, %r17;
div.full.f32 %r124, %r92, %r17;
div.full.f32 %r125, %r93, %r17;
div.full.f32 %r126, %r94, %r17;
div.full.f32 %r127, %r95, %r17;
div.full.f32 %r128, %r96, %r17;
div.full.f32 %r129, %r97, %r17;
div.full.f32 %r130, %r98, %r17;
div.full.f32 %r131, %r99, %r17;
div.full.f32 %r132, %r100, %r17;
div.full.f32 %r133, %r101, %r17;
div.full.f32 %r134, %r102, %r17;
div.full.f32 %r135, %r103, %r17;
div.full.f32 %r136, %r104, %r17;
div.full.f32 %r137, %r105, %r17;
div.full.f32 %r138, %r106, %r17;
div.full.f32 %r139, %r107, %r17;
div.full.f32 %r140, %r108, %r17;
div.full.f32 %r141, %r109, %r17;
div.full.f32 %r142, %r110, %r17;
div.full.f32 %r143, %r111, %r17;
div.full.f32 %r144, %r112, %r17;
div.full.f32 %r145, %r113, %r17;
div.full.f32 %r146, %r114, %r17;
div.full.f32 %r147, %r115, %r17;
min.f32 %r148, %r116, 0f43E00000;
min.f32 %r149, %r117, 0f43E00000;
min.f32 %r150, %r118, 0f43E00000;
min.f32 %r151, %r119, 0f43E00000;
min.f32 %r152, %r120, 0f43E00000;
min.f32 %r153, %r121, 0f43E00000;
min.f32 %r154, %r122, 0f43E00000;
min.f32 %r155, %r123, 0f43E00000;
min.f32 %r156, %r124, 0f43E00000;
min.f32 %r157, %r125, 0f43E00000;
min.f32 %r158, %r126, 0f43E00000;
min.f32 %r159, %r127, 0f43E00000;
min.f32 %r160, %r128, 0f43E00000;
min.f32 %r161, %r129, 0f43E00000;
min.f32 %r162, %r130, 0f43E00000;
min.f32 %r163, %r131, 0f43E00000;
min.f32 %r164, %r132, 0f43E00000;
min.f32 %r165, %r133, 0f43E00000;
min.f32 %r166, %r134, 0f43E00000;
min.f32 %r167, %r135, 0f43E00000;
min.f32 %r168, %r136, 0f43E00000;
min.f32 %r169, %r137, 0f43E00000;
min.f32 %r170, %r138, 0f43E00000;
min.f32 %r171, %r139, 0f43E00000;
min.f32 %r172, %r140, 0f43E00000;
min.f32 %r173, %r141, 0f43E00000;
min.f32 %r174, %r142, 0f43E00000;
min.f32 %r175, %r143, 0f43E00000;
min.f32 %r176, %r144, 0f43E00000;
min.f32 %r177, %r145, 0f43E00000;
min.f32 %r178, %r146, 0f43E00000;
min.f32 %r179, %r147, 0f43E00000;
max.f32 %r18, %r148, 0fC3E00000;
max.f32 %r19, %r149, 0fC3E00000;
max.f32 %r20, %r150, 0fC3E00000;
max.f32 %r21, %r151, 0fC3E00000;
max.f32 %r22, %r152, 0fC3E00000;
max.f32 %r23, %r153, 0fC3E00000;
max.f32 %r24, %r154, 0fC3E00000;
max.f32 %r25, %r155, 0fC3E00000;
max.f32 %r26, %r156, 0fC3E00000;
max.f32 %r27, %r157, 0fC3E00000;
max.f32 %r28, %r158, 0fC3E00000;
max.f32 %r29, %r159, 0fC3E00000;
max.f32 %r30, %r160, 0fC3E00000;
max.f32 %r31, %r161, 0fC3E00000;
max.f32 %r32, %r162, 0fC3E00000;
max.f32 %r33, %r163, 0fC3E00000;
max.f32 %r34, %r164, 0fC3E00000;
max.f32 %r35, %r165, 0fC3E00000;
max.f32 %r36, %r166, 0fC3E00000;
max.f32 %r37, %r167, 0fC3E00000;
max.f32 %r38, %r168, 0fC3E00000;
max.f32 %r39, %r169, 0fC3E00000;
max.f32 %r40, %r170, 0fC3E00000;
max.f32 %r41, %r171, 0fC3E00000;
max.f32 %r42, %r172, 0fC3E00000;
max.f32 %r43, %r173, 0fC3E00000;
max.f32 %r44, %r174, 0fC3E00000;
max.f32 %r45, %r175, 0fC3E00000;
max.f32 %r46, %r176, 0fC3E00000;
max.f32 %r47, %r177, 0fC3E00000;
max.f32 %r48, %r178, 0fC3E00000;
max.f32 %r49, %r179, 0fC3E00000;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r19, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r21, %r20;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r23, %r22;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r25, %r24;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r27, %r26;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r29, %r28;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r31, %r30;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r33, %r32;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r35, %r34;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r37, %r36;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r39, %r38;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r41, %r40;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r43, %r42;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r45, %r44;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r47, %r46;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r49, %r48;
selp.b64 %rd26, %rd15, %rd14, %p2;
mul.lo.s32 %r180, %r72, 2162688;
cvt.s64.s32 %rd27, %r180;
add.s64 %rd28, %rd26, %rd27;
selp.b64 %rd29, %rd17, %rd16, %p2;
add.s64 %rd30, %rd29, %rd27;
cvt.s64.s32 %rd31, %r79;
add.s64 %rd32, %rd28, %rd31;
add.s64 %rd6, %rd32, %rd23;
add.s64 %rd7, %rd6, 768;
mov.b32 %r50, {%rs1, %rs2};
mov.b32 %r51, {%rs3, %rs4};
mov.b32 %r52, {%rs5, %rs6};
mov.b32 %r53, {%rs7, %rs8};
st.global.v4.b32 [ %rd6 + 0 ], { %r50, %r51, %r52, %r53 };
mov.b32 %r54, {%rs9, %rs10};
mov.b32 %r55, {%rs11, %rs12};
mov.b32 %r56, {%rs13, %rs14};
mov.b32 %r57, {%rs15, %rs16};
st.global.v4.b32 [ %rd7 + 0 ], { %r54, %r55, %r56, %r57 };
shl.b32 %r181, %r74, 4;
and.b32 %r182, %r181, 4080;
mov.b32 %r183, global_smem;
add.s32 %r184, %r183, %r182;
st.shared.v4.b32 [%r184], {%r50, %r51, %r52, %r53};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r185, %r186, %r187, %r188}, [%r184];
bar.warp.sync -1;
st.shared.v4.b32 [%r184], {%r54, %r55, %r56, %r57};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r184];
and.b32 %r193, %r75, 56;
shl.b32 %r194, %r74, 2;
and.b32 %r195, %r194, 124;
mad.lo.s32 %r196, %r69, 360448, %r70;
cvt.s64.s32 %rd33, %r196;
add.s64 %rd34, %rd30, %rd33;
mul.lo.s32 %r197, %r195, 2816;
or.b32 %r198, %r197, %r193;
cvt.u64.u32 %rd35, %r198;
add.s64 %rd8, %rd34, %rd35;
add.s64 %rd9, %rd8, 2816;
add.s64 %rd10, %rd8, 5632;
add.s64 %rd11, %rd8, 8448;
prmt.b32 %r199, %r185, %r189, 0x40U;
prmt.b32 %r200, %r199, %r186, 0x410U;
prmt.b32 %r58, %r200, %r190, 0x4210U;
prmt.b32 %r201, %r187, %r191, 0x40U;
prmt.b32 %r202, %r201, %r188, 0x410U;
prmt.b32 %r59, %r202, %r192, 0x4210U;
st.global.v2.b32 [ %rd8 + 0 ], { %r58, %r59 };
prmt.b32 %r203, %r185, %r189, 0x51U;
prmt.b32 %r204, %r203, %r186, 0x510U;
prmt.b32 %r60, %r204, %r190, 0x5210U;
prmt.b32 %r205, %r187, %r191, 0x51U;
prmt.b32 %r206, %r205, %r188, 0x510U;
prmt.b32 %r61, %r206, %r192, 0x5210U;
st.global.v2.b32 [ %rd9 + 0 ], { %r60, %r61 };
prmt.b32 %r207, %r185, %r189, 0x62U;
prmt.b32 %r208, %r207, %r186, 0x610U;
prmt.b32 %r62, %r208, %r190, 0x6210U;
prmt.b32 %r209, %r187, %r191, 0x62U;
prmt.b32 %r210, %r209, %r188, 0x610U;
prmt.b32 %r63, %r210, %r192, 0x6210U;
st.global.v2.b32 [ %rd10 + 0 ], { %r62, %r63 };
prmt.b32 %r211, %r185, %r189, 0x73U;
prmt.b32 %r212, %r211, %r186, 0x710U;
prmt.b32 %r64, %r212, %r190, 0x7210U;
prmt.b32 %r213, %r187, %r191, 0x73U;
prmt.b32 %r214, %r213, %r188, 0x710U;
prmt.b32 %r65, %r214, %r192, 0x7210U;
st.global.v2.b32 [ %rd11 + 0 ], { %r64, %r65 };
shl.b32 %r215, %r69, 1;
mad.lo.s32 %r216, %r68, 12, %r215;
abs.f32 %r217, %r84;
abs.f32 %r218, %r85;
abs.f32 %r219, %r86;
abs.f32 %r220, %r87;
abs.f32 %r221, %r88;
abs.f32 %r222, %r89;
abs.f32 %r223, %r90;
abs.f32 %r224, %r91;
abs.f32 %r225, %r92;
abs.f32 %r226, %r93;
abs.f32 %r227, %r94;
abs.f32 %r228, %r95;
abs.f32 %r229, %r96;
abs.f32 %r230, %r97;
abs.f32 %r231, %r98;
abs.f32 %r232, %r99;
abs.f32 %r233, %r100;
abs.f32 %r234, %r101;
abs.f32 %r235, %r102;
abs.f32 %r236, %r103;
abs.f32 %r237, %r104;
abs.f32 %r238, %r105;
abs.f32 %r239, %r106;
abs.f32 %r240, %r107;
abs.f32 %r241, %r108;
abs.f32 %r242, %r109;
abs.f32 %r243, %r110;
abs.f32 %r244, %r111;
abs.f32 %r245, %r112;
abs.f32 %r246, %r113;
abs.f32 %r247, %r114;
abs.f32 %r248, %r115;
$L__tmp1:
max.f32 %r249, %r217, %r218;
max.f32 %r250, %r249, %r219;
max.f32 %r251, %r250, %r220;
max.f32 %r252, %r251, %r221;
max.f32 %r253, %r252, %r222;
max.f32 %r254, %r253, %r223;
max.f32 %r255, %r254, %r224;
max.f32 %r256, %r255, %r225;
max.f32 %r257, %r256, %r226;
max.f32 %r258, %r257, %r227;
max.f32 %r259, %r258, %r228;
max.f32 %r260, %r259, %r229;
max.f32 %r261, %r260, %r230;
max.f32 %r262, %r261, %r231;
max.f32 %r263, %r262, %r232;
max.f32 %r264, %r233, %r234;
max.f32 %r265, %r264, %r235;
max.f32 %r266, %r265, %r236;
max.f32 %r267, %r266, %r237;
max.f32 %r268, %r267, %r238;
max.f32 %r269, %r268, %r239;
max.f32 %r270, %r269, %r240;
max.f32 %r271, %r270, %r241;
max.f32 %r272, %r271, %r242;
max.f32 %r273, %r272, %r243;
max.f32 %r274, %r273, %r244;
max.f32 %r275, %r274, %r245;
max.f32 %r276, %r275, %r246;
max.f32 %r277, %r276, %r247;
max.f32 %r278, %r277, %r248;
$L__tmp2:
max.f32 %r279, %r263, %r278;
$L__tmp3:
shfl.sync.bfly.b32 %r280, %r279, 2, 31, -1;
$L__tmp4:
max.f32 %r281, %r279, %r280;
$L__tmp5:
shfl.sync.bfly.b32 %r282, %r281, 1, 31, -1;
$L__tmp6:
max.f32 %r283, %r281, %r282;
$L__tmp7:
shfl.sync.bfly.b32 %r284, %r283, 16, 31, -1;
$L__tmp8:
max.f32 %r285, %r283, %r284;
$L__tmp9:
shfl.sync.bfly.b32 %r286, %r285, 8, 31, -1;
$L__tmp10:
max.f32 %r287, %r285, %r286;
$L__tmp11:
and.b32 %r288, %r74, 4;
add.s32 %r289, %r183, %r288;
add.s32 %r290, %r289, %r193;
st.shared.b32 [%r290+4096], %r287;
bar.sync 0;
shl.b32 %r291, %r77, 3;
and.b32 %r292, %r74, 8;
shr.u32 %r293, %r292, 1;
add.s32 %r294, %r183, %r291;
add.s32 %r295, %r294, %r293;
ld.shared.b32 %r296, [%r295+4096];
$L__tmp12:
shfl.sync.bfly.b32 %r297, %r296, 4, 31, -1;
$L__tmp13:
max.f32 %r298, %r296, %r297;
$L__tmp14:
shfl.sync.bfly.b32 %r299, %r298, 2, 31, -1;
$L__tmp15:
max.f32 %r300, %r298, %r299;
$L__tmp16:
shfl.sync.bfly.b32 %r301, %r300, 1, 31, -1;
$L__tmp17:
max.f32 %r66, %r300, %r301;
$L__tmp18:
bfe.u32 %r302, %r74, 3, 1;
selp.b64 %rd36, %rd21, %rd20, %p2;
mul.lo.s32 %r303, %r72, 528;
mad.wide.u32 %rd37, %r303, 4, %rd36;
mad.wide.s32 %rd38, %r216, 4, %rd37;
mad.wide.u32 %rd12, %r302, 4, %rd38;
and.b32 %r304, %r74, 247;
setp.eq.b32 %p1, %r304, 0;
@%p1 st.global.b32 [ %rd12 + 0 ], { %r66 };
ret;
$L__tmp19:
$L__func_end0:
}
"#


structure Artifacts where
  referenceIR : String
  referencePTX : String
  scaleIR : String
  scalePTX : String
  packIR : String
  packPTX : String
  deriving BEq

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines referenceGraph) &&
  QKVTTIR.normalizePTX a.referencePTX == lines referencePtx &&
  QKVTTIR.normalizeTTIR a.scaleIR == some (lines scaleGraph) &&
  QKVTTIR.normalizePTX a.scalePTX == lines scalePtx &&
  QKVTTIR.normalizeTTIR a.packIR == some (lines packGraph) &&
  QKVTTIR.normalizePTX a.packPTX == lines packPtx

structure Certificate (a : Artifacts) : Type where
  matched : matchesArtifacts a = true
  storePitch : ∀ (warp : Fin 8) (lane : Fin 32),
    compiledPartialStore warp lane = partialStoreAddress warp (laneHalf lane)
  laneAgreement : ∀ (read : Fin 64 → Fin 64 → Value .bf16) (warp : Fin 8) (a b : Fin 32),
    laneHalf a = laneHalf b → warpMaximum read warp a = warpMaximum read warp b
  published : QKVPack.GluonShared.checkedPhases partialPhases = some [0]
  weights : ∀ (arith : Arithmetic) (delayed : Bool) (input : Inputs),
    readWarp arith delayed input = readReference arith delayed input

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : matchesArtifacts a = true then
    some ⟨h, compiled_partial_store, warp_writers_agree, partials_published, warp_refresh_equal⟩
  else none

structure Program where
  artifacts : Artifacts
  checked : Certificate artifacts

def Program.check (a : Artifacts) : Option Program := do
  return ⟨a, ← certify a⟩

def Program.run (program : Program) (hopper : Bool) (arith : Arithmetic) (delayed : Bool)
    (input : Inputs) : WeightRefresh :=
  if hopper && matchesArtifacts program.artifacts then readWarp arith delayed input
  else readReference arith delayed input

theorem Program.run_eq (program : Program) :
    program.run = fun _ arith delayed input => readReference arith delayed input := by
  funext hopper arith delayed input
  simp only [Program.run, program.checked.matched, Bool.and_true]
  cases hopper
  · rfl
  · exact program.checked.weights arith delayed input

#axiomsClean certify
#axiomsClean Program.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
