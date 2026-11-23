import models.NanoGPT.Track1.AnvilNormalization
import models.NanoGPT.Track1.AnvilTraceBinding

/-! Recognize the complete Hopper graphs and assembly for each of the three
normalization launches. This keeps the divisor selector, vector address,
full divide and BF16 stores in the checked program. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilNormalization
open AnvilEpilogue (Bank)
open AnvilTrace (Dispatch Normalization)

private def ttgirBody : Bank → String
  | .qk => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_normalize_gram_inputs(%X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c-64_i32 = arith.constant -64 : i32
%tile = tt.get_program_id x : i32
%matrix = tt.get_program_id y : i32
%prepared = arith.cmpi slt, %tile, 96 : i32
%selector = arith.cmpi sge, %tile, 96 : i32
%selector_0 = arith.extui %selector : i1 to i32
%stride = arith.muli %selector_0, %c-64_i32 : i32
%stride_1 = arith.addi %stride, 96 : i32
%divisor = arith.muli %matrix, 2 : i32
%divisor_2 = tt.addptr %Divisors, %divisor : !tt.ptr<f32>, i32
%divisor_3 = tt.addptr %divisor_2, %selector_0 : !tt.ptr<f32>, i32
%divisor_4 = tt.load %divisor_3 : !tt.ptr<f32>
%local_tile = arith.muli %selector_0, 96 : i32
%local_tile_5 = arith.subi %tile, %local_tile : i32
%offset = arith.muli %stride_1, 2048 : i32
%offset_6 = arith.muli %matrix, %offset : i32
%offset_7 = arith.muli %local_tile_5, 2048 : i32
%offset_8 = arith.addi %offset_6, %offset_7 : i32
%base = arith.select %prepared, %X, %A : !tt.ptr<bf16>
%base_9 = tt.addptr %base, %offset_8 : !tt.ptr<bf16>, i32
%ptr = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked>
%ptr_10 = tt.splat %base_9 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>, #blocked>
%ptr_11 = tt.addptr %ptr_10, %ptr : tensor<2048x!tt.ptr<bf16>, #blocked>, tensor<2048xi32, #blocked>
%x = tt.load %ptr_11 : tensor<2048x!tt.ptr<bf16>, #blocked>
%x_12 = arith.extf %x : tensor<2048xbf16, #blocked> to tensor<2048xf32, #blocked>
%0 = tt.splat %divisor_4 : f32 -> tensor<2048xf32, #blocked>
%1 = arith.divf %x_12, %0 : tensor<2048xf32, #blocked>
%2 = arith.truncf %1 : tensor<2048xf32, #blocked> to tensor<2048xbf16, #blocked>
tt.store %ptr_11, %2 : tensor<2048x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#
  | .vo => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_normalize_gram_inputs(%X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%tile = tt.get_program_id x : i32
%matrix = tt.get_program_id y : i32
%prepared = arith.cmpi slt, %tile, 288 : i32
%selector = arith.cmpi sge, %tile, 288 : i32
%selector_0 = arith.extui %selector : i1 to i32
%divisor = arith.muli %matrix, 2 : i32
%divisor_1 = tt.addptr %Divisors, %divisor : !tt.ptr<f32>, i32
%divisor_2 = tt.addptr %divisor_1, %selector_0 : !tt.ptr<f32>, i32
%divisor_3 = tt.load %divisor_2 : !tt.ptr<f32>
%local_tile = arith.muli %selector_0, 288 : i32
%local_tile_4 = arith.subi %tile, %local_tile : i32
%offset = arith.muli %matrix, 589824 : i32
%offset_5 = arith.muli %local_tile_4, 2048 : i32
%offset_6 = arith.addi %offset, %offset_5 : i32
%base = arith.select %prepared, %X, %A : !tt.ptr<bf16>
%base_7 = tt.addptr %base, %offset_6 : !tt.ptr<bf16>, i32
%ptr = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked>
%ptr_8 = tt.splat %base_7 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>, #blocked>
%ptr_9 = tt.addptr %ptr_8, %ptr : tensor<2048x!tt.ptr<bf16>, #blocked>, tensor<2048xi32, #blocked>
%x = tt.load %ptr_9 : tensor<2048x!tt.ptr<bf16>, #blocked>
%x_10 = arith.extf %x : tensor<2048xbf16, #blocked> to tensor<2048xf32, #blocked>
%0 = tt.splat %divisor_3 : f32 -> tensor<2048xf32, #blocked>
%1 = arith.divf %x_10, %0 : tensor<2048xf32, #blocked>
%2 = arith.truncf %1 : tensor<2048xf32, #blocked> to tensor<2048xbf16, #blocked>
tt.store %ptr_9, %2 : tensor<2048x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#
  | .mlp => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_normalize_gram_inputs(%X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %A: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c-768_i32 = arith.constant -768 : i32
%tile = tt.get_program_id x : i32
%matrix = tt.get_program_id y : i32
%prepared = arith.cmpi slt, %tile, 1056 : i32
%selector = arith.cmpi sge, %tile, 1056 : i32
%selector_0 = arith.extui %selector : i1 to i32
%stride = arith.muli %selector_0, %c-768_i32 : i32
%stride_1 = arith.addi %stride, 1056 : i32
%divisor = arith.muli %matrix, 2 : i32
%divisor_2 = tt.addptr %Divisors, %divisor : !tt.ptr<f32>, i32
%divisor_3 = tt.addptr %divisor_2, %selector_0 : !tt.ptr<f32>, i32
%divisor_4 = tt.load %divisor_3 : !tt.ptr<f32>
%local_tile = arith.muli %selector_0, 1056 : i32
%local_tile_5 = arith.subi %tile, %local_tile : i32
%offset = arith.muli %stride_1, 2048 : i32
%offset_6 = arith.muli %matrix, %offset : i32
%offset_7 = arith.muli %local_tile_5, 2048 : i32
%offset_8 = arith.addi %offset_6, %offset_7 : i32
%base = arith.select %prepared, %X, %A : !tt.ptr<bf16>
%base_9 = tt.addptr %base, %offset_8 : !tt.ptr<bf16>, i32
%ptr = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked>
%ptr_10 = tt.splat %base_9 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>, #blocked>
%ptr_11 = tt.addptr %ptr_10, %ptr : tensor<2048x!tt.ptr<bf16>, #blocked>, tensor<2048xi32, #blocked>
%x = tt.load %ptr_11 : tensor<2048x!tt.ptr<bf16>, #blocked>
%x_12 = arith.extf %x : tensor<2048xbf16, #blocked> to tensor<2048xf32, #blocked>
%0 = tt.splat %divisor_4 : f32 -> tensor<2048xf32, #blocked>
%1 = arith.divf %x_12, %0 : tensor<2048xf32, #blocked>
%2 = arith.truncf %1 : tensor<2048xf32, #blocked> to tensor<2048xbf16, #blocked>
tt.store %ptr_11, %2 : tensor<2048x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#

private def ptxBody : Bank → String
  | .qk => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _normalize_gram_inputs(
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_0,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_1,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_2,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_3,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_4
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<17>;
.reg .b32 %r<62>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd4, [_normalize_gram_inputs_param_0];
ld.param.b64 %rd5, [_normalize_gram_inputs_param_1];
$L__tmp0:
mov.u32 %r18, %ctaid.x;
ld.param.b64 %rd6, [_normalize_gram_inputs_param_2];
mov.u32 %r19, %ctaid.y;
setp.lt.u32 %p1, %r18, 96;
setp.gt.u32 %p2, %r18, 95;
shl.b32 %r20, %r19, 1;
mad.wide.u32 %rd7, %r20, 4, %rd6;
selp.b32 %r21, 1, 0, %p2;
mad.wide.u32 %rd1, %r21, 4, %rd7;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
selp.b32 %r22, 2097056, 0, %p2;
add.s32 %r23, %r22, %r18;
selp.b32 %r24, 65536, 196608, %p2;
shl.b32 %r25, %r23, 11;
mad.lo.s32 %r26, %r24, %r19, %r25;
selp.b64 %rd8, %rd4, %rd5, %p1;
mad.wide.s32 %rd9, %r26, 2, %rd8;
mov.u32 %r27, %tid.x;
shl.b32 %r28, %r27, 3;
and.b32 %r29, %r28, 1016;
mad.wide.u32 %rd2, %r29, 2, %rd9;
add.s64 %rd3, %rd2, 2048;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.b32 {%rs9, %rs10}, %r6;
mov.b32 {%rs11, %rs12}, %r7;
mov.b32 {%rs13, %rs14}, %r8;
mov.b32 {%rs15, %rs16}, %r9;
cvt.f32.bf16 %r30, %rs1;
cvt.f32.bf16 %r31, %rs2;
cvt.f32.bf16 %r32, %rs3;
cvt.f32.bf16 %r33, %rs4;
cvt.f32.bf16 %r34, %rs5;
cvt.f32.bf16 %r35, %rs6;
cvt.f32.bf16 %r36, %rs7;
cvt.f32.bf16 %r37, %rs8;
cvt.f32.bf16 %r38, %rs9;
cvt.f32.bf16 %r39, %rs10;
cvt.f32.bf16 %r40, %rs11;
cvt.f32.bf16 %r41, %rs12;
cvt.f32.bf16 %r42, %rs13;
cvt.f32.bf16 %r43, %rs14;
cvt.f32.bf16 %r44, %rs15;
cvt.f32.bf16 %r45, %rs16;
div.full.f32 %r46, %r30, %r1;
div.full.f32 %r47, %r31, %r1;
div.full.f32 %r48, %r32, %r1;
div.full.f32 %r49, %r33, %r1;
div.full.f32 %r50, %r34, %r1;
div.full.f32 %r51, %r35, %r1;
div.full.f32 %r52, %r36, %r1;
div.full.f32 %r53, %r37, %r1;
div.full.f32 %r54, %r38, %r1;
div.full.f32 %r55, %r39, %r1;
div.full.f32 %r56, %r40, %r1;
div.full.f32 %r57, %r41, %r1;
div.full.f32 %r58, %r42, %r1;
div.full.f32 %r59, %r43, %r1;
div.full.f32 %r60, %r44, %r1;
div.full.f32 %r61, %r45, %r1;
cvt.rn.bf16x2.f32 %r10, %r47, %r46;
cvt.rn.bf16x2.f32 %r11, %r49, %r48;
cvt.rn.bf16x2.f32 %r12, %r51, %r50;
cvt.rn.bf16x2.f32 %r13, %r53, %r52;
cvt.rn.bf16x2.f32 %r14, %r55, %r54;
cvt.rn.bf16x2.f32 %r15, %r57, %r56;
cvt.rn.bf16x2.f32 %r16, %r59, %r58;
cvt.rn.bf16x2.f32 %r17, %r61, %r60;
st.global.v4.b32 [ %rd2 + 0 ], { %r10, %r11, %r12, %r13 };
st.global.v4.b32 [ %rd3 + 0 ], { %r14, %r15, %r16, %r17 };
ret;
$L__tmp1:
$L__func_end0:
}
"#
  | .vo => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _normalize_gram_inputs(
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_0,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_1,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_2,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_3,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_4
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<17>;
.reg .b32 %r<61>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd4, [_normalize_gram_inputs_param_0];
ld.param.b64 %rd5, [_normalize_gram_inputs_param_1];
$L__tmp0:
mov.u32 %r18, %ctaid.x;
ld.param.b64 %rd6, [_normalize_gram_inputs_param_2];
mov.u32 %r19, %ctaid.y;
setp.lt.u32 %p1, %r18, 288;
setp.gt.u32 %p2, %r18, 287;
shl.b32 %r20, %r19, 1;
mad.wide.u32 %rd7, %r20, 4, %rd6;
selp.b32 %r21, 1, 0, %p2;
mad.wide.u32 %rd1, %r21, 4, %rd7;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
selp.b32 %r22, 2096864, 0, %p2;
add.s32 %r23, %r22, %r18;
shl.b32 %r24, %r23, 11;
mad.lo.s32 %r25, %r19, 589824, %r24;
selp.b64 %rd8, %rd4, %rd5, %p1;
mad.wide.s32 %rd9, %r25, 2, %rd8;
mov.u32 %r26, %tid.x;
shl.b32 %r27, %r26, 3;
and.b32 %r28, %r27, 1016;
mad.wide.u32 %rd2, %r28, 2, %rd9;
add.s64 %rd3, %rd2, 2048;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.b32 {%rs9, %rs10}, %r6;
mov.b32 {%rs11, %rs12}, %r7;
mov.b32 {%rs13, %rs14}, %r8;
mov.b32 {%rs15, %rs16}, %r9;
cvt.f32.bf16 %r29, %rs1;
cvt.f32.bf16 %r30, %rs2;
cvt.f32.bf16 %r31, %rs3;
cvt.f32.bf16 %r32, %rs4;
cvt.f32.bf16 %r33, %rs5;
cvt.f32.bf16 %r34, %rs6;
cvt.f32.bf16 %r35, %rs7;
cvt.f32.bf16 %r36, %rs8;
cvt.f32.bf16 %r37, %rs9;
cvt.f32.bf16 %r38, %rs10;
cvt.f32.bf16 %r39, %rs11;
cvt.f32.bf16 %r40, %rs12;
cvt.f32.bf16 %r41, %rs13;
cvt.f32.bf16 %r42, %rs14;
cvt.f32.bf16 %r43, %rs15;
cvt.f32.bf16 %r44, %rs16;
div.full.f32 %r45, %r29, %r1;
div.full.f32 %r46, %r30, %r1;
div.full.f32 %r47, %r31, %r1;
div.full.f32 %r48, %r32, %r1;
div.full.f32 %r49, %r33, %r1;
div.full.f32 %r50, %r34, %r1;
div.full.f32 %r51, %r35, %r1;
div.full.f32 %r52, %r36, %r1;
div.full.f32 %r53, %r37, %r1;
div.full.f32 %r54, %r38, %r1;
div.full.f32 %r55, %r39, %r1;
div.full.f32 %r56, %r40, %r1;
div.full.f32 %r57, %r41, %r1;
div.full.f32 %r58, %r42, %r1;
div.full.f32 %r59, %r43, %r1;
div.full.f32 %r60, %r44, %r1;
cvt.rn.bf16x2.f32 %r10, %r46, %r45;
cvt.rn.bf16x2.f32 %r11, %r48, %r47;
cvt.rn.bf16x2.f32 %r12, %r50, %r49;
cvt.rn.bf16x2.f32 %r13, %r52, %r51;
cvt.rn.bf16x2.f32 %r14, %r54, %r53;
cvt.rn.bf16x2.f32 %r15, %r56, %r55;
cvt.rn.bf16x2.f32 %r16, %r58, %r57;
cvt.rn.bf16x2.f32 %r17, %r60, %r59;
st.global.v4.b32 [ %rd2 + 0 ], { %r10, %r11, %r12, %r13 };
st.global.v4.b32 [ %rd3 + 0 ], { %r14, %r15, %r16, %r17 };
ret;
$L__tmp1:
$L__func_end0:
}
"#
  | .mlp => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _normalize_gram_inputs(
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_0,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_1,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_2,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_3,
.param .u64 .ptr .global .align 1 _normalize_gram_inputs_param_4
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<17>;
.reg .b32 %r<62>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd4, [_normalize_gram_inputs_param_0];
ld.param.b64 %rd5, [_normalize_gram_inputs_param_1];
$L__tmp0:
mov.u32 %r18, %ctaid.x;
ld.param.b64 %rd6, [_normalize_gram_inputs_param_2];
mov.u32 %r19, %ctaid.y;
setp.lt.u32 %p1, %r18, 1056;
setp.gt.u32 %p2, %r18, 1055;
shl.b32 %r20, %r19, 1;
mad.wide.u32 %rd7, %r20, 4, %rd6;
selp.b32 %r21, 1, 0, %p2;
mad.wide.u32 %rd1, %r21, 4, %rd7;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
selp.b32 %r22, 2096096, 0, %p2;
add.s32 %r23, %r22, %r18;
selp.b32 %r24, 589824, 2162688, %p2;
shl.b32 %r25, %r23, 11;
mad.lo.s32 %r26, %r24, %r19, %r25;
selp.b64 %rd8, %rd4, %rd5, %p1;
mad.wide.s32 %rd9, %r26, 2, %rd8;
mov.u32 %r27, %tid.x;
shl.b32 %r28, %r27, 3;
and.b32 %r29, %r28, 1016;
mad.wide.u32 %rd2, %r29, 2, %rd9;
add.s64 %rd3, %rd2, 2048;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.b32 {%rs9, %rs10}, %r6;
mov.b32 {%rs11, %rs12}, %r7;
mov.b32 {%rs13, %rs14}, %r8;
mov.b32 {%rs15, %rs16}, %r9;
cvt.f32.bf16 %r30, %rs1;
cvt.f32.bf16 %r31, %rs2;
cvt.f32.bf16 %r32, %rs3;
cvt.f32.bf16 %r33, %rs4;
cvt.f32.bf16 %r34, %rs5;
cvt.f32.bf16 %r35, %rs6;
cvt.f32.bf16 %r36, %rs7;
cvt.f32.bf16 %r37, %rs8;
cvt.f32.bf16 %r38, %rs9;
cvt.f32.bf16 %r39, %rs10;
cvt.f32.bf16 %r40, %rs11;
cvt.f32.bf16 %r41, %rs12;
cvt.f32.bf16 %r42, %rs13;
cvt.f32.bf16 %r43, %rs14;
cvt.f32.bf16 %r44, %rs15;
cvt.f32.bf16 %r45, %rs16;
div.full.f32 %r46, %r30, %r1;
div.full.f32 %r47, %r31, %r1;
div.full.f32 %r48, %r32, %r1;
div.full.f32 %r49, %r33, %r1;
div.full.f32 %r50, %r34, %r1;
div.full.f32 %r51, %r35, %r1;
div.full.f32 %r52, %r36, %r1;
div.full.f32 %r53, %r37, %r1;
div.full.f32 %r54, %r38, %r1;
div.full.f32 %r55, %r39, %r1;
div.full.f32 %r56, %r40, %r1;
div.full.f32 %r57, %r41, %r1;
div.full.f32 %r58, %r42, %r1;
div.full.f32 %r59, %r43, %r1;
div.full.f32 %r60, %r44, %r1;
div.full.f32 %r61, %r45, %r1;
cvt.rn.bf16x2.f32 %r10, %r47, %r46;
cvt.rn.bf16x2.f32 %r11, %r49, %r48;
cvt.rn.bf16x2.f32 %r12, %r51, %r50;
cvt.rn.bf16x2.f32 %r13, %r53, %r52;
cvt.rn.bf16x2.f32 %r14, %r55, %r54;
cvt.rn.bf16x2.f32 %r15, %r57, %r56;
cvt.rn.bf16x2.f32 %r16, %r59, %r58;
cvt.rn.bf16x2.f32 %r17, %r61, %r60;
st.global.v4.b32 [ %rd2 + 0 ], { %r10, %r11, %r12, %r13 };
st.global.v4.b32 [ %rd3 + 0 ], { %r14, %r15, %r16, %r17 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def fitsArtifacts (bank : Bank) (ttgir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttgir == some (lines (ttgirBody bank)) &&
    QKVTTIR.normalizePTX ptx == lines (ptxBody bank)

structure Program (bank : Bank) where
  ttgir : String
  ptx : String
  checked : fitsArtifacts bank ttgir ptx = true

def Program.check (bank : Bank) (ttgir ptx : String) : Option (Program bank) :=
  if h : fitsArtifacts bank ttgir ptx = true then some ⟨ttgir, ptx, h⟩ else none

def Program.run {bank : Bank} (program : Program bank) (trace : Dispatch)
    (input : AnvilTrace.Input) : AnvilTrace.Output bank :=
  if fitsArtifacts bank program.ttgir program.ptx then packedNormalize trace bank input
  else AnvilTrace.normalize trace bank input

theorem Program.run_eq {bank : Bank} (program : Program bank) (trace : Dispatch)
    (input : AnvilTrace.Input) :
    program.run trace input = AnvilTrace.normalize trace bank input := by
  simp only [Program.run, program.checked, if_true, packed_normalize_eq]

abbrev Programs := (bank : Bank) → Program bank

def Programs.run (programs : Programs) (trace : AnvilTrace.Programs) : Normalization :=
  fun bank input => (programs bank).run trace.run input

theorem Programs.run_eq (programs : Programs) (trace : AnvilTrace.Programs) :
    programs.run trace = AnvilTrace.normalize AnvilTrace.original := by
  funext bank input
  change (programs bank).run trace.run input = _
  rw [Program.run_eq, AnvilTrace.Programs.run_eq]

#axiomsClean Program.run_eq
#axiomsClean Programs.run_eq

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilNormalization
