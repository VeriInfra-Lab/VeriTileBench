import models.NanoGPT.Track1.RopeShift
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact wide-head artifacts at the four training lengths and the
validation length. The templates contain the emitted reduction and
BF16 conversion order, as well as all three store predicates. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RopeShift
open VeriTile.Triton FP

private def referenceIRTemplate : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_qk_norm_rope_forward_kernel(%qk: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<6> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<1> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_2 = arith.constant dense<64> : tensor<1x128xi32, #blocked>
%cst_3 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_4 = arith.constant dense<6> : tensor<4x1xi32, #blocked>
%cst_5 = arith.constant dense<128> : tensor<4x1xi32, #blocked>
%cst_6 = arith.constant dense<2304> : tensor<4x1xi32, #blocked>
%cst_7 = arith.constant dense<12> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_8 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
%cst_9 = arith.constant dense<@ROWS@> : tensor<4x1xi32, #blocked>
%cst_10 = arith.constant dense<0.000000e+00> : tensor<4x128xbf16, #blocked>
%cst_11 = arith.constant dense<1.1920929E-7> : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_12 = arith.constant dense<1.280000e+02> : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m = tt.get_program_id x : i32
%offs_m_13 = arith.muli %offs_m, 4 : i32
%offs_m_14 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_15 = tt.splat %offs_m_13 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_16 = arith.addi %offs_m_15, %offs_m_14 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%mask = tt.expand_dims %offs_m_16 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%mask_17 = arith.cmpi slt, %mask, %cst_9 : tensor<4x1xi32, #blocked>
%mask_18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%mask_19 = tt.expand_dims %mask_18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%mask_20 = arith.cmpi slt, %mask_19, %cst_8 : tensor<1x128xi32, #blocked>
%mask_21 = tt.broadcast %mask_17 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%mask_22 = tt.broadcast %mask_20 : tensor<1x128xi1, #blocked> -> tensor<4x128xi1, #blocked>
%mask_23 = arith.andi %mask_21, %mask_22 : tensor<4x128xi1, #blocked>
%token = arith.divsi %offs_m_16, %cst_7 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%input_head = arith.remsi %offs_m_16, %cst_7 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%x = tt.expand_dims %token {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%x_24 = arith.muli %x, %cst_6 : tensor<4x1xi32, #blocked>
%x_25 = tt.splat %qk : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%x_26 = tt.addptr %x_25, %x_24 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_27 = tt.expand_dims %input_head {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%x_28 = arith.muli %x_27, %cst_5 : tensor<4x1xi32, #blocked>
%x_29 = tt.addptr %x_26, %x_28 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_30 = tt.broadcast %x_29 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_31 = tt.broadcast %mask_19 : tensor<1x128xi32, #blocked> -> tensor<4x128xi32, #blocked>
%x_32 = tt.addptr %x_30, %x_31 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%x_33 = tt.load %x_32, %mask_23, %cst_10 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_flip = tt.reshape %x_33 : tensor<4x128xbf16, #blocked> -> tensor<4x64x2xbf16, #blocked1>
%y = tt.bitcast %x_flip : tensor<4x64x2xbf16, #blocked1> -> tensor<4x64x2xi16, #blocked1>
%y_34 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_89: i16, %y_90: i16):
%y_91 = arith.xori %y_89, %y_90 : i16
tt.reduce.return %y_91 : i16
}) : (tensor<4x64x2xi16, #blocked1>) -> tensor<4x64xi16, #ttg.slice<{dim = 2, parent = #blocked1}>>
%y_35 = tt.expand_dims %y_34 {axis = 2 : i32} : tensor<4x64xi16, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<4x64x1xi16, #blocked1>
%y_36 = tt.broadcast %y_35 : tensor<4x64x1xi16, #blocked1> -> tensor<4x64x2xi16, #blocked1>
%y_37 = arith.xori %y, %y_36 : tensor<4x64x2xi16, #blocked1>
%x_38 = tt.bitcast %y_37 : tensor<4x64x2xi16, #blocked1> -> tensor<4x64x2xbf16, #blocked1>
%x_flip_39 = tt.reshape %x_38 : tensor<4x64x2xbf16, #blocked1> -> tensor<4x128xbf16, #blocked>
%x_40 = arith.extf %x_33 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%x_flip_41 = arith.extf %x_flip_39 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%rstd = arith.mulf %x_40, %x_40 : tensor<4x128xf32, #blocked>
%rstd_42 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_89: f32, %rstd_90: f32):
%rstd_91 = arith.addf %rstd_89, %rstd_90 : f32
tt.reduce.return %rstd_91 : f32
}) : (tensor<4x128xf32, #blocked>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_43 = arith.divf %rstd_42, %cst_12 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_44 = arith.addf %rstd_43, %cst_11 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_45 = math.rsqrt %rstd_44 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%logical_head = arith.remsi %input_head, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%factor_ptrs = arith.muli %x, %cst_5 : tensor<4x1xi32, #blocked>
%factor_ptrs_46 = tt.broadcast %factor_ptrs : tensor<4x1xi32, #blocked> -> tensor<4x128xi32, #blocked>
%factor_ptrs_47 = arith.addi %factor_ptrs_46, %x_31 : tensor<4x128xi32, #blocked>
%f1 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%f1_48 = tt.addptr %f1, %factor_ptrs_47 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%f1_49 = tt.load %f1_48, %mask_23, %cst_10 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%f1_50 = arith.extf %f1_49 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%f2 = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%f2_51 = tt.addptr %f2, %factor_ptrs_47 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%f2_52 = tt.load %f2_51, %mask_23, %cst_10 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%f2_53 = arith.extf %f2_52 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%y_54 = tt.expand_dims %rstd_45 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
%y_55 = tt.broadcast %y_54 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
%y_56 = arith.mulf %x_40, %y_55 : tensor<4x128xf32, #blocked>
%y_57 = arith.mulf %f1_50, %y_56 : tensor<4x128xf32, #blocked>
%y_58 = arith.mulf %x_flip_41, %y_55 : tensor<4x128xf32, #blocked>
%y_59 = arith.mulf %f2_53, %y_58 : tensor<4x128xf32, #blocked>
%y_60 = arith.addf %y_57, %y_59 : tensor<4x128xf32, #blocked>
%shift_row = arith.cmpi sge, %input_head, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%shift_row_61 = arith.cmpi sgt, %token, %cst_0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%shift_row_62 = arith.andi %shift_row, %shift_row_61 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%previous_token = arith.subi %token, %cst_1 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%previous_token_63 = arith.maxsi %previous_token, %cst_0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%x_previous = tt.expand_dims %shift_row_62 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%x_previous_64 = tt.broadcast %x_previous : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%x_previous_65 = arith.andi %mask_23, %x_previous_64 : tensor<4x128xi1, #blocked>
%x_previous_66 = tt.expand_dims %previous_token_63 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%x_previous_67 = arith.muli %x_previous_66, %cst_6 : tensor<4x1xi32, #blocked>
%x_previous_68 = tt.addptr %x_25, %x_previous_67 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_previous_69 = tt.addptr %x_previous_68, %x_28 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_previous_70 = tt.broadcast %x_previous_69 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_previous_71 = tt.addptr %x_previous_70, %x_31 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%x_previous_72 = tt.load %x_previous_71, %x_previous_65, %cst_10 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_previous_73 = arith.extf %x_previous_72 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%previous_rstd = arith.mulf %x_previous_73, %x_previous_73 : tensor<4x128xf32, #blocked>
%previous_rstd_74 = "tt.reduce"(%previous_rstd) <{axis = 1 : i32}> ({
^bb0(%previous_rstd_89: f32, %previous_rstd_90: f32):
%previous_rstd_91 = arith.addf %previous_rstd_89, %previous_rstd_90 : f32
tt.reduce.return %previous_rstd_91 : f32
}) : (tensor<4x128xf32, #blocked>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%previous_rstd_75 = arith.divf %previous_rstd_74, %cst_12 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%previous_rstd_76 = arith.addf %previous_rstd_75, %cst_11 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%previous_rstd_77 = math.rsqrt %previous_rstd_76 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%shift = arith.cmpi sge, %mask_19, %cst_2 : tensor<1x128xi32, #blocked>
%shift_78 = tt.broadcast %shift : tensor<1x128xi1, #blocked> -> tensor<4x128xi1, #blocked>
%shift_79 = arith.andi %x_previous_64, %shift_78 : tensor<4x128xi1, #blocked>
%y_80 = tt.expand_dims %previous_rstd_77 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
%y_81 = tt.broadcast %y_80 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
%y_82 = arith.mulf %x_previous_73, %y_81 : tensor<4x128xf32, #blocked>
%y_83 = arith.select %shift_79, %y_82, %y_60 : tensor<4x128xi1, #blocked>, tensor<4x128xf32, #blocked>
%output_ptrs = arith.muli %x, %cst_3 : tensor<4x1xi32, #blocked>
%output_ptrs_84 = tt.expand_dims %logical_head {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%output_ptrs_85 = arith.muli %output_ptrs_84, %cst_5 : tensor<4x1xi32, #blocked>
%output_ptrs_86 = arith.addi %output_ptrs, %output_ptrs_85 : tensor<4x1xi32, #blocked>
%output_ptrs_87 = tt.broadcast %output_ptrs_86 : tensor<4x1xi32, #blocked> -> tensor<4x128xi32, #blocked>
%output_ptrs_88 = arith.addi %output_ptrs_87, %x_31 : tensor<4x128xi32, #blocked>
%0 = arith.cmpi slt, %x_27, %cst_4 : tensor<4x1xi32, #blocked>
%1 = tt.broadcast %0 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%2 = arith.andi %mask_23, %1 : tensor<4x128xi1, #blocked>
%3 = tt.splat %out_q : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%4 = tt.addptr %3, %output_ptrs_88 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%5 = arith.truncf %y_83 : tensor<4x128xf32, #blocked> to tensor<4x128xbf16, #blocked>
tt.store %4, %5, %2 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%6 = arith.cmpi sge, %x_27, %cst_4 : tensor<4x1xi32, #blocked>
%7 = tt.broadcast %6 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%8 = arith.andi %mask_23, %7 : tensor<4x128xi1, #blocked>
%9 = tt.splat %out_k : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%10 = tt.addptr %9, %output_ptrs_88 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
tt.store %10, %5, %8 : tensor<4x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#

private def referencePTXTemplate : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry _qk_norm_rope_forward_kernel(
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_0,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_1,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_2,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_3,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_4,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_5,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_6
)
.reqntid 64
{
.reg .pred %p<11>;
.reg .b16 %rs<41>;
.reg .b32 %r<155>;
.reg .b64 %rd<20>;
$L__func_begin0:
ld.param.b64 %rd7, [_qk_norm_rope_forward_kernel_param_0];
ld.param.b64 %rd8, [_qk_norm_rope_forward_kernel_param_1];
$L__tmp0:
mov.u32 %r22, %ctaid.x;
shl.b32 %r23, %r22, 2;
ld.param.b64 %rd9, [_qk_norm_rope_forward_kernel_param_2];
ld.param.b64 %rd10, [_qk_norm_rope_forward_kernel_param_3];
mov.u32 %r24, %tid.x;
ld.param.b64 %rd11, [_qk_norm_rope_forward_kernel_param_4];
bfe.u32 %r25, %r24, 4, 2;
or.b32 %r26, %r25, %r23;
setp.lt.s32 %p1, %r26, @ROWS@;
shl.b32 %r27, %r24, 3;
and.b32 %r28, %r27, 120;
mul.hi.s32 %r29, %r26, 715827883;
shr.u32 %r30, %r29, 31;
shr.s32 %r31, %r29, 1;
add.s32 %r32, %r31, %r30;
mul.lo.s32 %r33, %r32, 12;
sub.s32 %r34, %r26, %r33;
mul.lo.s32 %r35, %r32, 2304;
mad.wide.s32 %rd12, %r35, 2, %rd7;
shl.b32 %r36, %r34, 7;
mul.wide.s32 %rd13, %r36, 2;
add.s64 %rd14, %rd12, %rd13;
mul.wide.u32 %rd15, %r28, 2;
add.s64 %rd1, %rd14, %rd15;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r37, %rs1;
cvt.f32.bf16 %r38, %rs2;
mov.b32 {%rs3, %rs4}, %r2;
cvt.f32.bf16 %r39, %rs4;
cvt.f32.bf16 %r40, %rs3;
mov.b32 {%rs5, %rs6}, %r3;
cvt.f32.bf16 %r41, %rs6;
cvt.f32.bf16 %r42, %rs5;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r43, %rs8;
cvt.f32.bf16 %r44, %rs7;
mul.f32 %r45, %r38, %r38;
$L__tmp1:
fma.rn.f32 %r46, %r37, %r37, %r45;
fma.rn.f32 %r47, %r40, %r40, %r46;
fma.rn.f32 %r48, %r39, %r39, %r47;
fma.rn.f32 %r49, %r42, %r42, %r48;
fma.rn.f32 %r50, %r41, %r41, %r49;
fma.rn.f32 %r51, %r44, %r44, %r50;
fma.rn.f32 %r52, %r43, %r43, %r51;
$L__tmp2:
shfl.sync.bfly.b32 %r53, %r52, 8, 31, -1;
$L__tmp3:
add.f32 %r54, %r52, %r53;
$L__tmp4:
shfl.sync.bfly.b32 %r55, %r54, 4, 31, -1;
$L__tmp5:
add.f32 %r56, %r54, %r55;
$L__tmp6:
shfl.sync.bfly.b32 %r57, %r56, 2, 31, -1;
$L__tmp7:
add.f32 %r58, %r56, %r57;
$L__tmp8:
shfl.sync.bfly.b32 %r59, %r58, 1, 31, -1;
$L__tmp9:
add.f32 %r60, %r58, %r59;
mov.b32 %r61, 0f43000000;
$L__tmp10:
div.full.f32 %r62, %r60, %r61;
add.f32 %r63, %r62, 0f34000000;
rsqrt.approx.ftz.f32 %r64, %r63;
shl.b32 %r65, %r32, 7;
or.b32 %r66, %r65, %r28;
mul.wide.s32 %rd16, %r66, 2;
add.s64 %rd2, %rd8, %rd16;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
add.s64 %rd3, %rd9, %rd16;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
setp.gt.s32 %p5, %r34, 5;
setp.gt.s32 %p6, %r26, 11;
and.pred %p7, %p6, %p5;
max.s32 %r67, %r32, 1;
and.pred %p2, %p1, %p7;
mad.lo.s32 %r68, %r67, 2304, -2304;
mad.wide.s32 %rd17, %r68, 2, %rd7;
add.s64 %rd18, %rd17, %rd13;
add.s64 %rd4, %rd18, %rd15;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs9, %rs10}, %r14;
cvt.f32.bf16 %r69, %rs9;
cvt.f32.bf16 %r70, %rs10;
mov.b32 {%rs11, %rs12}, %r15;
cvt.f32.bf16 %r71, %rs12;
cvt.f32.bf16 %r72, %rs11;
mov.b32 {%rs13, %rs14}, %r16;
cvt.f32.bf16 %r73, %rs14;
cvt.f32.bf16 %r74, %rs13;
mov.b32 {%rs15, %rs16}, %r17;
cvt.f32.bf16 %r75, %rs16;
cvt.f32.bf16 %r76, %rs15;
mul.f32 %r77, %r70, %r70;
$L__tmp11:
fma.rn.f32 %r78, %r69, %r69, %r77;
fma.rn.f32 %r79, %r72, %r72, %r78;
fma.rn.f32 %r80, %r71, %r71, %r79;
fma.rn.f32 %r81, %r74, %r74, %r80;
fma.rn.f32 %r82, %r73, %r73, %r81;
fma.rn.f32 %r83, %r76, %r76, %r82;
fma.rn.f32 %r84, %r75, %r75, %r83;
$L__tmp12:
shfl.sync.bfly.b32 %r85, %r84, 8, 31, -1;
$L__tmp13:
add.f32 %r86, %r84, %r85;
$L__tmp14:
shfl.sync.bfly.b32 %r87, %r86, 4, 31, -1;
$L__tmp15:
add.f32 %r88, %r86, %r87;
$L__tmp16:
shfl.sync.bfly.b32 %r89, %r88, 2, 31, -1;
$L__tmp17:
add.f32 %r90, %r88, %r89;
$L__tmp18:
shfl.sync.bfly.b32 %r91, %r90, 1, 31, -1;
$L__tmp19:
add.f32 %r92, %r90, %r91;
$L__tmp20:
div.full.f32 %r93, %r92, %r61;
add.f32 %r94, %r93, 0f34000000;
rsqrt.approx.ftz.f32 %r95, %r94;
mul.f32 %r96, %r64, %r44;
mul.f32 %r97, %r64, %r43;
mul.f32 %r98, %r64, %r42;
mul.f32 %r99, %r64, %r41;
mul.f32 %r100, %r64, %r40;
mul.f32 %r101, %r64, %r39;
mul.f32 %r102, %r64, %r37;
mul.f32 %r103, %r64, %r38;
cvt.u16.u32 %rs17, %r34;
mul.lo.s16 %rs18, %rs17, 43;
shr.u16 %rs19, %rs18, 15;
shr.u16 %rs20, %rs18, 8;
add.s16 %rs21, %rs20, %rs19;
mul.lo.s16 %rs22, %rs21, 6;
sub.s16 %rs23, %rs17, %rs22;
setp.gt.u32 %p8, %r28, 63;
and.pred %p9, %p8, %p7;
mul.f32 %r104, %r95, %r70;
mul.f32 %r105, %r95, %r69;
mul.f32 %r106, %r95, %r71;
mul.f32 %r107, %r95, %r72;
mul.f32 %r108, %r95, %r73;
mul.f32 %r109, %r95, %r74;
mul.f32 %r110, %r95, %r75;
mul.f32 %r111, %r95, %r76;
mul.lo.s32 %r112, %r32, 768;
cvt.s16.s8 %rs24, %rs23;
mad.wide.s16 %r113, %rs24, 128, %r112;
or.b32 %r114, %r113, %r28;
setp.lt.s32 %p10, %r34, 6;
and.pred %p3, %p1, %p10;
mul.wide.s32 %rd19, %r114, 2;
add.s64 %rd5, %rd10, %rd19;
mov.b32 {%rs25, %rs26}, %r6;
cvt.f32.bf16 %r115, %rs25;
cvt.f32.bf16 %r116, %rs26;
mov.b32 {%rs27, %rs28}, %r10;
cvt.f32.bf16 %r117, %rs28;
cvt.f32.bf16 %r118, %rs27;
mul.f32 %r119, %r103, %r118;
mul.f32 %r120, %r102, %r117;
fma.rn.f32 %r121, %r103, %r116, %r120;
fma.rn.f32 %r122, %r102, %r115, %r119;
selp.f32 %r123, %r105, %r122, %p9;
selp.f32 %r124, %r104, %r121, %p9;
cvt.rn.bf16x2.f32 %r18, %r124, %r123;
mov.b32 {%rs29, %rs30}, %r7;
cvt.f32.bf16 %r125, %rs29;
cvt.f32.bf16 %r126, %rs30;
mov.b32 {%rs31, %rs32}, %r11;
cvt.f32.bf16 %r127, %rs32;
cvt.f32.bf16 %r128, %rs31;
mul.f32 %r129, %r101, %r128;
mul.f32 %r130, %r100, %r127;
fma.rn.f32 %r131, %r101, %r126, %r130;
fma.rn.f32 %r132, %r100, %r125, %r129;
selp.f32 %r133, %r107, %r132, %p9;
selp.f32 %r134, %r106, %r131, %p9;
cvt.rn.bf16x2.f32 %r19, %r134, %r133;
mov.b32 {%rs33, %rs34}, %r8;
cvt.f32.bf16 %r135, %rs33;
cvt.f32.bf16 %r136, %rs34;
mov.b32 {%rs35, %rs36}, %r12;
cvt.f32.bf16 %r137, %rs36;
cvt.f32.bf16 %r138, %rs35;
mul.f32 %r139, %r99, %r138;
mul.f32 %r140, %r98, %r137;
fma.rn.f32 %r141, %r99, %r136, %r140;
fma.rn.f32 %r142, %r98, %r135, %r139;
selp.f32 %r143, %r109, %r142, %p9;
selp.f32 %r144, %r108, %r141, %p9;
cvt.rn.bf16x2.f32 %r20, %r144, %r143;
mov.b32 {%rs37, %rs38}, %r9;
cvt.f32.bf16 %r145, %rs37;
cvt.f32.bf16 %r146, %rs38;
mov.b32 {%rs39, %rs40}, %r13;
cvt.f32.bf16 %r147, %rs40;
cvt.f32.bf16 %r148, %rs39;
mul.f32 %r149, %r97, %r148;
mul.f32 %r150, %r96, %r147;
fma.rn.f32 %r151, %r97, %r146, %r150;
fma.rn.f32 %r152, %r96, %r145, %r149;
selp.f32 %r153, %r111, %r152, %p9;
selp.f32 %r154, %r110, %r151, %p9;
cvt.rn.bf16x2.f32 %r21, %r154, %r153;
@%p3 st.global.v4.b32 [ %rd5 + 0 ], { %r18, %r19, %r20, %r21 };
and.pred %p4, %p1, %p5;
add.s64 %rd6, %rd11, %rd19;
@%p4 st.global.v4.b32 [ %rd6 + 0 ], { %r18, %r19, %r20, %r21 };
ret;
$L__tmp21:
$L__func_end0:
}
"#

private def shiftedIRTemplate : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [2, 16, 1], warpsPerCTA = [2, 1, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_qk_norm_rope_forward_kernel(%qk: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<6> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<0> : tensor<4x1xi32, #blocked>
%cst_1 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<6> : tensor<4x1xi32, #blocked>
%cst_3 = arith.constant dense<64> : tensor<1x128xi32, #blocked>
%cst_4 = arith.constant dense<true> : tensor<1x128xi1, #blocked>
%cst_5 = arith.constant dense<1> : tensor<4x1xi32, #blocked>
%cst_6 = arith.constant dense<@TOKENS@> : tensor<4x1xi32, #blocked>
%cst_7 = arith.constant dense<768> : tensor<4x128xi32, #blocked>
%cst_8 = arith.constant dense<128> : tensor<4x1xi32, #blocked>
%cst_9 = arith.constant dense<2304> : tensor<4x1xi32, #blocked>
%cst_10 = arith.constant dense<12> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_11 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
%cst_12 = arith.constant dense<@ROWS@> : tensor<4x1xi32, #blocked>
%cst_13 = arith.constant dense<0.000000e+00> : tensor<4x128xbf16, #blocked>
%cst_14 = arith.constant dense<1.1920929E-7> : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_15 = arith.constant dense<1.280000e+02> : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m = tt.get_program_id x : i32
%offs_m_16 = arith.muli %offs_m, 4 : i32
%offs_m_17 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_18 = tt.splat %offs_m_16 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_19 = arith.addi %offs_m_18, %offs_m_17 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%mask = tt.expand_dims %offs_m_19 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%mask_20 = arith.cmpi slt, %mask, %cst_12 : tensor<4x1xi32, #blocked>
%mask_21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%mask_22 = tt.expand_dims %mask_21 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%mask_23 = arith.cmpi slt, %mask_22, %cst_11 : tensor<1x128xi32, #blocked>
%mask_24 = tt.broadcast %mask_20 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%mask_25 = tt.broadcast %mask_23 : tensor<1x128xi1, #blocked> -> tensor<4x128xi1, #blocked>
%mask_26 = arith.andi %mask_24, %mask_25 : tensor<4x128xi1, #blocked>
%token = arith.divsi %offs_m_19, %cst_10 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%input_head = arith.remsi %offs_m_19, %cst_10 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%x = tt.expand_dims %token {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%x_27 = arith.muli %x, %cst_9 : tensor<4x1xi32, #blocked>
%x_28 = tt.splat %qk : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%x_29 = tt.addptr %x_28, %x_27 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_30 = tt.expand_dims %input_head {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%x_31 = arith.muli %x_30, %cst_8 : tensor<4x1xi32, #blocked>
%x_32 = tt.addptr %x_29, %x_31 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%x_33 = tt.broadcast %x_32 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_34 = tt.broadcast %mask_22 : tensor<1x128xi32, #blocked> -> tensor<4x128xi32, #blocked>
%x_35 = tt.addptr %x_33, %x_34 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%x_36 = tt.load %x_35, %mask_26, %cst_13 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%x_flip = tt.reshape %x_36 : tensor<4x128xbf16, #blocked> -> tensor<4x64x2xbf16, #blocked1>
%y = tt.bitcast %x_flip : tensor<4x64x2xbf16, #blocked1> -> tensor<4x64x2xi16, #blocked1>
%y_37 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_69: i16, %y_70: i16):
%y_71 = arith.xori %y_69, %y_70 : i16
tt.reduce.return %y_71 : i16
}) : (tensor<4x64x2xi16, #blocked1>) -> tensor<4x64xi16, #ttg.slice<{dim = 2, parent = #blocked1}>>
%y_38 = tt.expand_dims %y_37 {axis = 2 : i32} : tensor<4x64xi16, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<4x64x1xi16, #blocked1>
%y_39 = tt.broadcast %y_38 : tensor<4x64x1xi16, #blocked1> -> tensor<4x64x2xi16, #blocked1>
%y_40 = arith.xori %y, %y_39 : tensor<4x64x2xi16, #blocked1>
%x_41 = tt.bitcast %y_40 : tensor<4x64x2xi16, #blocked1> -> tensor<4x64x2xbf16, #blocked1>
%x_flip_42 = tt.reshape %x_41 : tensor<4x64x2xbf16, #blocked1> -> tensor<4x128xbf16, #blocked>
%x_43 = arith.extf %x_36 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%x_flip_44 = arith.extf %x_flip_42 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%rstd = arith.mulf %x_43, %x_43 : tensor<4x128xf32, #blocked>
%rstd_45 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_69: f32, %rstd_70: f32):
%rstd_71 = arith.addf %rstd_69, %rstd_70 : f32
tt.reduce.return %rstd_71 : f32
}) : (tensor<4x128xf32, #blocked>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_46 = arith.divf %rstd_45, %cst_15 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_47 = arith.addf %rstd_46, %cst_14 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rstd_48 = math.rsqrt %rstd_47 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
%logical_head = arith.remsi %input_head, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%factor_ptrs = arith.muli %x, %cst_8 : tensor<4x1xi32, #blocked>
%factor_ptrs_49 = tt.broadcast %factor_ptrs : tensor<4x1xi32, #blocked> -> tensor<4x128xi32, #blocked>
%factor_ptrs_50 = arith.addi %factor_ptrs_49, %x_34 : tensor<4x128xi32, #blocked>
%f1 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%f1_51 = tt.addptr %f1, %factor_ptrs_50 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%f1_52 = tt.load %f1_51, %mask_26, %cst_13 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%f1_53 = arith.extf %f1_52 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%f2 = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%f2_54 = tt.addptr %f2, %factor_ptrs_50 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%f2_55 = tt.load %f2_54, %mask_26, %cst_13 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%f2_56 = arith.extf %f2_55 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%y_57 = tt.expand_dims %rstd_48 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
%y_58 = tt.broadcast %y_57 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
%y_59 = arith.mulf %x_43, %y_58 : tensor<4x128xf32, #blocked>
%y_60 = arith.mulf %f1_53, %y_59 : tensor<4x128xf32, #blocked>
%y_61 = arith.mulf %x_flip_44, %y_58 : tensor<4x128xf32, #blocked>
%y_62 = arith.mulf %f2_56, %y_61 : tensor<4x128xf32, #blocked>
%y_63 = arith.addf %y_60, %y_62 : tensor<4x128xf32, #blocked>
%output_ptrs = arith.muli %x, %cst_1 : tensor<4x1xi32, #blocked>
%output_ptrs_64 = tt.expand_dims %logical_head {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%output_ptrs_65 = arith.muli %output_ptrs_64, %cst_8 : tensor<4x1xi32, #blocked>
%output_ptrs_66 = arith.addi %output_ptrs, %output_ptrs_65 : tensor<4x1xi32, #blocked>
%output_ptrs_67 = tt.broadcast %output_ptrs_66 : tensor<4x1xi32, #blocked> -> tensor<4x128xi32, #blocked>
%output_ptrs_68 = arith.addi %output_ptrs_67, %x_34 : tensor<4x128xi32, #blocked>
%0 = arith.cmpi slt, %x_30, %cst_2 : tensor<4x1xi32, #blocked>
%1 = tt.broadcast %0 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%2 = arith.andi %mask_26, %1 : tensor<4x128xi1, #blocked>
%3 = tt.splat %out_q : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%4 = tt.addptr %3, %output_ptrs_68 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%5 = arith.truncf %y_63 : tensor<4x128xf32, #blocked> to tensor<4x128xbf16, #blocked>
tt.store %4, %5, %2 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%key = arith.cmpi sge, %x_30, %cst_2 : tensor<4x1xi32, #blocked>
%stationary = arith.cmpi sge, %mask_22, %cst_3 : tensor<1x128xi32, #blocked>
%6 = tt.broadcast %key : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%7 = arith.andi %mask_26, %6 : tensor<4x128xi1, #blocked>
%8 = arith.xori %stationary, %cst_4 : tensor<1x128xi1, #blocked>
%9 = arith.cmpi eq, %x, %cst_0 : tensor<4x1xi32, #blocked>
%10 = tt.broadcast %8 : tensor<1x128xi1, #blocked> -> tensor<4x128xi1, #blocked>
%11 = tt.broadcast %9 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%12 = arith.ori %10, %11 : tensor<4x128xi1, #blocked>
%13 = arith.andi %7, %12 : tensor<4x128xi1, #blocked>
%14 = tt.splat %out_k : !tt.ptr<bf16> -> tensor<4x128x!tt.ptr<bf16>, #blocked>
%15 = tt.addptr %14, %output_ptrs_68 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
tt.store %15, %5, %13 : tensor<4x128x!tt.ptr<bf16>, #blocked>
%16 = tt.broadcast %stationary : tensor<1x128xi1, #blocked> -> tensor<4x128xi1, #blocked>
%17 = arith.andi %7, %16 : tensor<4x128xi1, #blocked>
%18 = arith.addi %x, %cst_5 : tensor<4x1xi32, #blocked>
%19 = arith.cmpi slt, %18, %cst_6 : tensor<4x1xi32, #blocked>
%20 = tt.broadcast %19 : tensor<4x1xi1, #blocked> -> tensor<4x128xi1, #blocked>
%21 = arith.andi %17, %20 : tensor<4x128xi1, #blocked>
%22 = tt.addptr %15, %cst_7 : tensor<4x128x!tt.ptr<bf16>, #blocked>, tensor<4x128xi32, #blocked>
%23 = arith.truncf %y_59 : tensor<4x128xf32, #blocked> to tensor<4x128xbf16, #blocked>
tt.store %22, %23, %21 : tensor<4x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#

private def shiftedPTXTemplate : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry _qk_norm_rope_forward_kernel(
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_0,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_1,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_2,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_3,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_4,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_5,
.param .u64 .ptr .global .align 1 _qk_norm_rope_forward_kernel_param_6
)
.reqntid 64
{
.reg .pred %p<14>;
.reg .b16 %rs<33>;
.reg .b32 %r<111>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd7, [_qk_norm_rope_forward_kernel_param_0];
ld.param.b64 %rd8, [_qk_norm_rope_forward_kernel_param_1];
$L__tmp0:
mov.u32 %r22, %ctaid.x;
shl.b32 %r23, %r22, 2;
ld.param.b64 %rd9, [_qk_norm_rope_forward_kernel_param_2];
ld.param.b64 %rd10, [_qk_norm_rope_forward_kernel_param_3];
mov.u32 %r24, %tid.x;
ld.param.b64 %rd11, [_qk_norm_rope_forward_kernel_param_4];
bfe.u32 %r25, %r24, 4, 2;
or.b32 %r26, %r25, %r23;
setp.lt.s32 %p1, %r26, @ROWS@;
shl.b32 %r27, %r24, 3;
and.b32 %r28, %r27, 120;
mul.hi.s32 %r29, %r26, 715827883;
shr.u32 %r30, %r29, 31;
shr.s32 %r31, %r29, 1;
add.s32 %r32, %r31, %r30;
mul.lo.s32 %r33, %r32, 12;
sub.s32 %r34, %r26, %r33;
mul.lo.s32 %r35, %r32, 2304;
mad.wide.s32 %rd12, %r35, 2, %rd7;
shl.b32 %r36, %r34, 7;
mad.wide.s32 %rd13, %r36, 2, %rd12;
mad.wide.u32 %rd1, %r28, 2, %rd13;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r37, %rs1;
cvt.f32.bf16 %r38, %rs2;
mov.b32 {%rs3, %rs4}, %r2;
cvt.f32.bf16 %r39, %rs4;
cvt.f32.bf16 %r40, %rs3;
mov.b32 {%rs5, %rs6}, %r3;
cvt.f32.bf16 %r41, %rs6;
cvt.f32.bf16 %r42, %rs5;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r43, %rs8;
cvt.f32.bf16 %r44, %rs7;
mul.f32 %r45, %r38, %r38;
$L__tmp1:
fma.rn.f32 %r46, %r37, %r37, %r45;
fma.rn.f32 %r47, %r40, %r40, %r46;
fma.rn.f32 %r48, %r39, %r39, %r47;
fma.rn.f32 %r49, %r42, %r42, %r48;
fma.rn.f32 %r50, %r41, %r41, %r49;
fma.rn.f32 %r51, %r44, %r44, %r50;
fma.rn.f32 %r52, %r43, %r43, %r51;
$L__tmp2:
shfl.sync.bfly.b32 %r53, %r52, 8, 31, -1;
$L__tmp3:
add.f32 %r54, %r52, %r53;
$L__tmp4:
shfl.sync.bfly.b32 %r55, %r54, 4, 31, -1;
$L__tmp5:
add.f32 %r56, %r54, %r55;
$L__tmp6:
shfl.sync.bfly.b32 %r57, %r56, 2, 31, -1;
$L__tmp7:
add.f32 %r58, %r56, %r57;
$L__tmp8:
shfl.sync.bfly.b32 %r59, %r58, 1, 31, -1;
$L__tmp9:
add.f32 %r60, %r58, %r59;
mov.b32 %r61, 0f43000000;
$L__tmp10:
div.full.f32 %r62, %r60, %r61;
add.f32 %r63, %r62, 0f34000000;
rsqrt.approx.ftz.f32 %r64, %r63;
cvt.u16.u32 %rs9, %r34;
mul.lo.s16 %rs10, %rs9, 43;
shr.u16 %rs11, %rs10, 15;
shr.u16 %rs12, %rs10, 8;
add.s16 %rs13, %rs12, %rs11;
mul.lo.s16 %rs14, %rs13, 6;
sub.s16 %rs15, %rs9, %rs14;
shl.b32 %r65, %r32, 7;
or.b32 %r66, %r65, %r28;
mul.wide.s32 %rd14, %r66, 2;
add.s64 %rd2, %rd8, %rd14;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
add.s64 %rd3, %rd9, %rd14;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mul.f32 %r67, %r64, %r38;
mul.f32 %r68, %r64, %r37;
mul.f32 %r69, %r64, %r39;
mul.f32 %r70, %r64, %r40;
mul.f32 %r71, %r64, %r41;
mul.f32 %r72, %r64, %r42;
mul.f32 %r73, %r64, %r43;
mul.f32 %r74, %r64, %r44;
mul.lo.s32 %r75, %r32, 768;
cvt.s16.s8 %rs16, %rs15;
mad.wide.s16 %r76, %rs16, 128, %r75;
or.b32 %r77, %r76, %r28;
setp.lt.s32 %p5, %r34, 6;
and.pred %p2, %p1, %p5;
mul.wide.s32 %rd15, %r77, 2;
add.s64 %rd4, %rd10, %rd15;
mov.b32 {%rs17, %rs18}, %r6;
cvt.f32.bf16 %r78, %rs18;
cvt.f32.bf16 %r79, %rs17;
mov.b32 {%rs19, %rs20}, %r10;
cvt.f32.bf16 %r80, %rs19;
cvt.f32.bf16 %r81, %rs20;
mul.f32 %r82, %r68, %r81;
mul.f32 %r83, %r67, %r80;
fma.rn.f32 %r84, %r68, %r79, %r83;
fma.rn.f32 %r85, %r67, %r78, %r82;
cvt.rn.bf16x2.f32 %r14, %r85, %r84;
mov.b32 {%rs21, %rs22}, %r7;
cvt.f32.bf16 %r86, %rs22;
cvt.f32.bf16 %r87, %rs21;
mov.b32 {%rs23, %rs24}, %r11;
cvt.f32.bf16 %r88, %rs23;
cvt.f32.bf16 %r89, %rs24;
mul.f32 %r90, %r70, %r89;
mul.f32 %r91, %r69, %r88;
fma.rn.f32 %r92, %r70, %r87, %r91;
fma.rn.f32 %r93, %r69, %r86, %r90;
cvt.rn.bf16x2.f32 %r15, %r93, %r92;
mov.b32 {%rs25, %rs26}, %r8;
cvt.f32.bf16 %r94, %rs26;
cvt.f32.bf16 %r95, %rs25;
mov.b32 {%rs27, %rs28}, %r12;
cvt.f32.bf16 %r96, %rs27;
cvt.f32.bf16 %r97, %rs28;
mul.f32 %r98, %r72, %r97;
mul.f32 %r99, %r71, %r96;
fma.rn.f32 %r100, %r72, %r95, %r99;
fma.rn.f32 %r101, %r71, %r94, %r98;
cvt.rn.bf16x2.f32 %r16, %r101, %r100;
mov.b32 {%rs29, %rs30}, %r9;
cvt.f32.bf16 %r102, %rs30;
cvt.f32.bf16 %r103, %rs29;
mov.b32 {%rs31, %rs32}, %r13;
cvt.f32.bf16 %r104, %rs31;
cvt.f32.bf16 %r105, %rs32;
mul.f32 %r106, %r74, %r105;
mul.f32 %r107, %r73, %r104;
fma.rn.f32 %r108, %r74, %r103, %r107;
fma.rn.f32 %r109, %r73, %r102, %r106;
cvt.rn.bf16x2.f32 %r17, %r109, %r108;
@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r14, %r15, %r16, %r17 };
setp.gt.s32 %p6, %r34, 5;
setp.gt.u32 %p7, %r28, 63;
setp.lt.u32 %p8, %r28, 64;
add.s32 %r110, %r26, 11;
setp.lt.u32 %p9, %r110, 23;
or.pred %p10, %p9, %p8;
and.pred %p11, %p6, %p10;
and.pred %p3, %p1, %p11;
add.s64 %rd5, %rd11, %rd15;
@%p3 st.global.v4.b32 [ %rd5 + 0 ], { %r14, %r15, %r16, %r17 };
setp.lt.s32 %p12, %r26, @LAST@;
and.pred %p13, %p7, %p12;
and.pred %p4, %p13, %p6;
add.s64 %rd6, %rd5, 1536;
cvt.rn.bf16x2.f32 %r18, %r67, %r68;
cvt.rn.bf16x2.f32 %r19, %r69, %r70;
cvt.rn.bf16x2.f32 %r20, %r71, %r72;
cvt.rn.bf16x2.f32 %r21, %r73, %r74;
@%p4 st.global.v4.b32 [ %rd6 + 0 ], { %r18, %r19, %r20, %r21 };
ret;
$L__tmp11:
$L__func_end0:
}
"#

structure Artifacts where
  referenceIR : String
  referencePTX : String
  shiftedIR : String
  shiftedPTX : String
  deriving BEq

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

private def resolveLength (n : Length) (s : String) : String :=
  ((s.replace "@ROWS@" (toString (12 * n.tokens))).replace "@LAST@" (toString (12 * n.tokens - 12))).replace
    "@TOKENS@" (toString n.tokens)

def matchesArtifacts (n : Length) (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines (resolveLength n referenceIRTemplate)) &&
    QKVTTIR.normalizePTX a.referencePTX == lines (resolveLength n referencePTXTemplate) &&
    QKVTTIR.normalizeTTIR a.shiftedIR == some (lines (resolveLength n shiftedIRTemplate)) &&
    QKVTTIR.normalizePTX a.shiftedPTX == lines (resolveLength n shiftedPTXTemplate)

structure Certificate (n : Length) (a : Artifacts) : Type where
  matched : matchesArtifacts n a = true
  gridCoverage : ∀ p : Point n, cell (owner p) = p
  gridUnique : ∀ (p q : Thread n), cell p = cell q → p = q
  threadCoordinates : ∀ t : Thread n,
    compiledCoordinates t = ((cell t).token.val, (cell t).plane, (cell t).head.val, (cell t).channel.val)
  masks : ∀ t : Thread n, placementMasks t
  bounds : ∀ (p : Point n) (d : Channel),
    qkOffset p d + 1 < 2 * n.tokens * 2304 ∧
      factorOffset p d + 1 < 2 * n.tokens * 128 ∧
      outputOffset p + 1 < 2 * n.tokens * 768 ∧ 2 * n.tokens * 2304 < 2^31
  addrUnique : ∀ (p q : Point n), p.plane = q.plane → outputOffset p = outputOffset q → p = q
  writerUnique : ∀ (dst lhs rhs : Point n) (p q : Bool) (x y : Word),
    store p lhs = some (dst, x) → store q rhs = some (dst, y) → lhs = rhs ∧ p = q
  stores : ∀ (op : Arithmetic) (read : Inputs) (p : Point n),
    emittedWrite op read p = referenceWrite op read p

def certify (n : Length) (a : Artifacts) : Option (Certificate n a) :=
  if h : matchesArtifacts n a = true then
    some ⟨h, owner_covers, cell_owner_unique, compiled_coordinates_exact,
      compiled_placement_masks, address_bounds, output_address_unique, store_source_unique,
      fun op read p => writes_equal op read p⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.RopeShift
