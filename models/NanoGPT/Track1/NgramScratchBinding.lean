import models.NanoGPT.Track1.NgramScratch
import models.NanoGPT.Track1.NgramAdam
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The final consumers of the n-gram gradient scratch. The recognized
Hopper artifacts include the original value paths and the source stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramScratch

structure Artifacts where
  peerIR : String
  peerPTX : String
  ownIR : String
  ownPTX : String
  deriving BEq

private def peerIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_ngram_gradient_packet(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %wire: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %peer_rows: i32, %lo: i32, %own_rows: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<4x256xf16, #blocked>
%cst_2 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_3 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row = tt.get_program_id x : i32
%row_4 = arith.muli %row, 4 : i32
%row_5 = tt.splat %row_4 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_6 = arith.addi %row_5, %lanes : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_7 = arith.cmpi slt, %row_6, %selected : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_8 = tt.splat %own_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_9 = arith.select %selected_7, %cst, %selected_8 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_10 = arith.addi %row_6, %selected_9 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%live = tt.expand_dims %row_6 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%live_11 = tt.splat %peer_rows : i32 -> tensor<4x1xi32, #blocked>
%live_12 = arith.cmpi slt, %live, %live_11 : tensor<4x1xi32, #blocked>
%source_row = tt.expand_dims %selected_10 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%source_row_13 = arith.muli %source_row, %cst_0 : tensor<4x1xi32, #blocked>
%source_row_14 = tt.splat %src : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked>
%source_row_15 = tt.addptr %source_row_14, %source_row_13 : tensor<4x1x!tt.ptr<f16>, #blocked>, tensor<4x1xi32, #blocked>
%dest_row = arith.muli %live, %cst_0 : tensor<4x1xi32, #blocked>
%dest_row_16 = tt.splat %wire : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%dest_row_17 = tt.addptr %dest_row_16, %dest_row : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%values = tt.expand_dims %columns {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%values_18 = tt.broadcast %source_row_15 : tensor<4x1x!tt.ptr<f16>, #blocked> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%values_19 = tt.broadcast %values : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%values_20 = tt.addptr %values_18, %values_19 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%values_21 = tt.broadcast %live_12 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%values_22 = tt.load %values_20, %values_21, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%values_23 = arith.extf %values_22 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%converted = tt.elementwise_inline_asm "cvt.rn.bf16.f32 $0, $1;" {constraints = "=h,f", packed_element = 1 : i32, pure = true} %values_23 : tensor<4x256xf32, #blocked> -> tensor<4x256xbf16, #blocked>
%0 = tt.broadcast %dest_row_17 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %values_19 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %1, %converted, %values_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.store %values_20, %cst_1, %values_21 : tensor<4x256x!tt.ptr<f16>, #blocked>
%cols = arith.addi %columns, %cst_2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%values_24 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%values_25 = tt.broadcast %values_24 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%values_26 = tt.addptr %values_18, %values_25 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%values_27 = tt.load %values_26, %values_21, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%values_28 = arith.extf %values_27 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%converted_29 = tt.elementwise_inline_asm "cvt.rn.bf16.f32 $0, $1;" {constraints = "=h,f", packed_element = 1 : i32, pure = true} %values_28 : tensor<4x256xf32, #blocked> -> tensor<4x256xbf16, #blocked>
%2 = tt.addptr %0, %values_25 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %2, %converted_29, %values_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.store %values_26, %cst_1, %values_21 : tensor<4x256x!tt.ptr<f16>, #blocked>
%cols_30 = arith.addi %columns, %cst_3 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%values_31 = tt.expand_dims %cols_30 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%values_32 = tt.broadcast %values_31 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%values_33 = tt.addptr %values_18, %values_32 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%values_34 = tt.load %values_33, %values_21, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%values_35 = arith.extf %values_34 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%converted_36 = tt.elementwise_inline_asm "cvt.rn.bf16.f32 $0, $1;" {constraints = "=h,f", packed_element = 1 : i32, pure = true} %values_35 : tensor<4x256xf32, #blocked> -> tensor<4x256xbf16, #blocked>
%3 = tt.addptr %0, %values_32 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %3, %converted_36, %values_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.store %values_33, %cst_1, %values_21 : tensor<4x256x!tt.ptr<f16>, #blocked>
tt.return
}
}"#

private def peerPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _ngram_gradient_packet(
.param .u64 .ptr .global .align 1 _ngram_gradient_packet_param_0,
.param .u64 .ptr .global .align 1 _ngram_gradient_packet_param_1,
.param .u32 _ngram_gradient_packet_param_2,
.param .u32 _ngram_gradient_packet_param_3,
.param .u32 _ngram_gradient_packet_param_4,
.param .u64 .ptr .global .align 1 _ngram_gradient_packet_param_5,
.param .u64 .ptr .global .align 1 _ngram_gradient_packet_param_6
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<49>;
.reg .b32 %r<64>;
.reg .b64 %rd<12>;
$L__func_begin0:
ld.param.b64 %rd7, [_ngram_gradient_packet_param_0];
ld.param.b64 %rd8, [_ngram_gradient_packet_param_1];
$L__tmp0:
mov.u32 %r50, %tid.x;
ld.param.b32 %r51, [_ngram_gradient_packet_param_2];
bfe.u32 %r52, %r50, 5, 2;
ld.param.b32 %r53, [_ngram_gradient_packet_param_3];
ld.param.b32 %r54, [_ngram_gradient_packet_param_4];
mov.u32 %r55, %ctaid.x;
shl.b32 %r56, %r55, 2;
or.b32 %r57, %r52, %r56;
setp.lt.s32 %p2, %r57, %r53;
selp.b32 %r58, 0, %r54, %p2;
add.s32 %r59, %r58, %r57;
shl.b32 %r60, %r50, 3;
and.b32 %r61, %r60, 248;
setp.lt.s32 %p1, %r57, %r51;
mul.lo.s32 %r62, %r59, 768;
mad.wide.s32 %rd9, %r62, 2, %rd7;
mul.lo.s32 %r63, %r57, 768;
mad.wide.s32 %rd10, %r63, 2, %rd8;
mul.wide.u32 %rd11, %r61, 2;
add.s64 %rd1, %rd9, %rd11;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs25, %rs26}, %r1;
mov.b32 {%rs27, %rs28}, %r2;
mov.b32 {%rs29, %rs30}, %r3;
mov.b32 {%rs31, %rs32}, %r4;
cvt.f32.f16 %r6, %rs25;
cvt.f32.f16 %r7, %rs26;
cvt.f32.f16 %r8, %rs27;
cvt.f32.f16 %r9, %rs28;
cvt.f32.f16 %r10, %rs29;
cvt.f32.f16 %r11, %rs30;
cvt.f32.f16 %r12, %rs31;
cvt.f32.f16 %r13, %rs32;
cvt.rn.bf16.f32 %rs1, %r6;
cvt.rn.bf16.f32 %rs2, %r7;
cvt.rn.bf16.f32 %rs3, %r8;
cvt.rn.bf16.f32 %rs4, %r9;
cvt.rn.bf16.f32 %rs5, %r10;
cvt.rn.bf16.f32 %rs6, %r11;
cvt.rn.bf16.f32 %rs7, %r12;
cvt.rn.bf16.f32 %rs8, %r13;
add.s64 %rd2, %rd10, %rd11;
mov.b32 %r14, {%rs1, %rs2};
mov.b32 %r15, {%rs3, %rs4};
mov.b32 %r16, {%rs5, %rs6};
mov.b32 %r17, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd2 + 0 ], { %r14, %r15, %r16, %r17 };
@%p1 st.global.v4.b32 [ %rd1 + 0 ], { %r5, %r5, %r5, %r5 };
add.s64 %rd3, %rd1, 512;
mov.u32 %r18, %r5;
mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd3 + 0 ];
mov.b32 {%rs33, %rs34}, %r18;
mov.b32 {%rs35, %rs36}, %r19;
mov.b32 {%rs37, %rs38}, %r20;
mov.b32 {%rs39, %rs40}, %r21;
cvt.f32.f16 %r22, %rs33;
cvt.f32.f16 %r23, %rs34;
cvt.f32.f16 %r24, %rs35;
cvt.f32.f16 %r25, %rs36;
cvt.f32.f16 %r26, %rs37;
cvt.f32.f16 %r27, %rs38;
cvt.f32.f16 %r28, %rs39;
cvt.f32.f16 %r29, %rs40;
cvt.rn.bf16.f32 %rs9, %r22;
cvt.rn.bf16.f32 %rs10, %r23;
cvt.rn.bf16.f32 %rs11, %r24;
cvt.rn.bf16.f32 %rs12, %r25;
cvt.rn.bf16.f32 %rs13, %r26;
cvt.rn.bf16.f32 %rs14, %r27;
cvt.rn.bf16.f32 %rs15, %r28;
cvt.rn.bf16.f32 %rs16, %r29;
add.s64 %rd4, %rd2, 512;
mov.b32 %r30, {%rs9, %rs10};
mov.b32 %r31, {%rs11, %rs12};
mov.b32 %r32, {%rs13, %rs14};
mov.b32 %r33, {%rs15, %rs16};
@%p1 st.global.v4.b32 [ %rd4 + 0 ], { %r30, %r31, %r32, %r33 };
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r5, %r5, %r5, %r5 };
add.s64 %rd5, %rd1, 1024;
mov.u32 %r34, %r5;
mov.u32 %r35, %r5;
mov.u32 %r36, %r5;
mov.u32 %r37, %r5;
@%p1 ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd5 + 0 ];
mov.b32 {%rs41, %rs42}, %r34;
mov.b32 {%rs43, %rs44}, %r35;
mov.b32 {%rs45, %rs46}, %r36;
mov.b32 {%rs47, %rs48}, %r37;
cvt.f32.f16 %r38, %rs41;
cvt.f32.f16 %r39, %rs42;
cvt.f32.f16 %r40, %rs43;
cvt.f32.f16 %r41, %rs44;
cvt.f32.f16 %r42, %rs45;
cvt.f32.f16 %r43, %rs46;
cvt.f32.f16 %r44, %rs47;
cvt.f32.f16 %r45, %rs48;
cvt.rn.bf16.f32 %rs17, %r38;
cvt.rn.bf16.f32 %rs18, %r39;
cvt.rn.bf16.f32 %rs19, %r40;
cvt.rn.bf16.f32 %rs20, %r41;
cvt.rn.bf16.f32 %rs21, %r42;
cvt.rn.bf16.f32 %rs22, %r43;
cvt.rn.bf16.f32 %rs23, %r44;
cvt.rn.bf16.f32 %rs24, %r45;
add.s64 %rd6, %rd2, 1024;
mov.b32 %r46, {%rs17, %rs18};
mov.b32 %r47, {%rs19, %rs20};
mov.b32 %r48, {%rs21, %rs22};
mov.b32 %r49, {%rs23, %rs24};
@%p1 st.global.v4.b32 [ %rd6 + 0 ], { %r46, %r47, %r48, %r49 };
@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r5, %r5, %r5, %r5 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def ownIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_mapped_rows_kernel(%source: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %indices: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dest: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row_map: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %count: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<768> : tensor<4x1xi64, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<4x256xf16, #blocked>
%cst_2 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_3 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%row = tt.get_program_id x : i32
%row_4 = arith.muli %row, 4 : i32
%row_5 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_6 = tt.splat %row_4 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_7 = arith.addi %row_6, %row_5 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_8 = arith.cmpi slt, %row_7, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%indices_9 = tt.splat %indices : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%indices_10 = tt.addptr %indices_9, %row_7 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%indices_11 = tt.load %indices_10, %live_8, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%entry = tt.splat %row_map : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%entry_12 = tt.addptr %entry, %indices_11 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%entry_13 = tt.load %entry_12, %live_8, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%src = tt.expand_dims %row_7 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%src_14 = arith.extsi %src : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%src_15 = arith.muli %src_14, %cst_0 : tensor<4x1xi64, #blocked>
%src_16 = tt.splat %source : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked>
%src_17 = tt.addptr %src_16, %src_15 : tensor<4x1x!tt.ptr<f16>, #blocked>, tensor<4x1xi64, #blocked>
%src_18 = tt.expand_dims %columns {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%src_19 = tt.broadcast %src_17 : tensor<4x1x!tt.ptr<f16>, #blocked> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%src_20 = tt.broadcast %src_18 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%src_21 = tt.addptr %src_19, %src_20 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%dst = tt.expand_dims %entry_13 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%dst_22 = arith.extsi %dst : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%dst_23 = arith.muli %dst_22, %cst_0 : tensor<4x1xi64, #blocked>
%dst_24 = tt.splat %dest : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%dst_25 = tt.addptr %dst_24, %dst_23 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi64, #blocked>
%dst_26 = tt.broadcast %dst_25 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%dst_27 = tt.addptr %dst_26, %src_20 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%words = tt.expand_dims %live_8 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%words_28 = tt.broadcast %words : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%words_29 = tt.load %src_21, %words_28, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%words_30 = arith.extf %words_29 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%0 = arith.truncf %words_30 : tensor<4x256xf32, #blocked> to tensor<4x256xbf16, #blocked>
%1 = tt.atomic_rmw fadd, relaxed, gpu, %dst_27, %0, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
tt.store %src_21, %cst_1, %words_28 : tensor<4x256x!tt.ptr<f16>, #blocked>
%words_31 = tt.addptr %src_21, %cst_2 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%words_32 = tt.load %words_31, %words_28, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%words_33 = arith.extf %words_32 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%2 = tt.addptr %dst_27, %cst_2 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%3 = arith.truncf %words_33 : tensor<4x256xf32, #blocked> to tensor<4x256xbf16, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %2, %3, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
tt.store %words_31, %cst_1, %words_28 : tensor<4x256x!tt.ptr<f16>, #blocked>
%words_34 = tt.addptr %src_21, %cst_3 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%words_35 = tt.load %words_34, %words_28, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%words_36 = arith.extf %words_35 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%5 = tt.addptr %dst_27, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%6 = arith.truncf %words_36 : tensor<4x256xf32, #blocked> to tensor<4x256xbf16, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %5, %6, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
tt.store %words_34, %cst_1, %words_28 : tensor<4x256x!tt.ptr<f16>, #blocked>
tt.return
}
}"#

private def ownPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_mapped_rows_kernel(
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_0,
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_1,
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_2,
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_3,
.param .u32 _scatter_mapped_rows_kernel_param_4,
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_5,
.param .u64 .ptr .global .align 1 _scatter_mapped_rows_kernel_param_6
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<73>;
.reg .b32 %r<48>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd9, [_scatter_mapped_rows_kernel_param_0];
ld.param.b64 %rd10, [_scatter_mapped_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r16, %ctaid.x;
shl.b32 %r17, %r16, 2;
ld.param.b64 %rd11, [_scatter_mapped_rows_kernel_param_2];
ld.param.b64 %rd12, [_scatter_mapped_rows_kernel_param_3];
mov.u32 %r18, %tid.x;
ld.param.b32 %r19, [_scatter_mapped_rows_kernel_param_4];
bfe.u32 %r20, %r18, 5, 2;
or.b32 %r21, %r20, %r17;
shl.b32 %r22, %r18, 3;
and.b32 %r23, %r22, 248;
setp.lt.s32 %p1, %r21, %r19;
mad.wide.s32 %rd1, %r21, 4, %rd10;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mad.wide.s32 %rd2, %r1, 4, %rd12;
mov.u32 %r2, 0x0;
@%p1 ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.s32 %rd13, %r21, 1536, %rd9;
mul.wide.u32 %rd14, %r23, 2;
add.s64 %rd3, %rd13, %rd14;
mad.wide.s32 %rd15, %r2, 1536, %rd11;
add.s64 %rd4, %rd15, %rd14;
mov.b32 %r7, 0;
mov.u32 %r3, %r7;
mov.u32 %r4, %r7;
mov.u32 %r5, %r7;
mov.u32 %r6, %r7;
@%p1 ld.global.v4.b32 { %r3, %r4, %r5, %r6 }, [ %rd3 + 0 ];
mov.b32 {%rs49, %rs50}, %r3;
mov.b32 {%rs51, %rs52}, %r4;
mov.b32 {%rs53, %rs54}, %r5;
mov.b32 {%rs55, %rs56}, %r6;
cvt.f32.f16 %r24, %rs49;
cvt.f32.f16 %r25, %rs50;
cvt.f32.f16 %r26, %rs51;
cvt.f32.f16 %r27, %rs52;
cvt.f32.f16 %r28, %rs53;
cvt.f32.f16 %r29, %rs54;
cvt.f32.f16 %r30, %rs55;
cvt.f32.f16 %r31, %rs56;
cvt.rn.bf16.f32 %rs9, %r24;
cvt.rn.bf16.f32 %rs10, %r25;
cvt.rn.bf16.f32 %rs11, %r26;
cvt.rn.bf16.f32 %rs12, %r27;
cvt.rn.bf16.f32 %rs13, %r28;
cvt.rn.bf16.f32 %rs14, %r29;
cvt.rn.bf16.f32 %rs15, %r30;
cvt.rn.bf16.f32 %rs16, %r31;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
mov.u16 %rs3, 0x0;
mov.u16 %rs4, 0x0;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
mov.u16 %rs7, 0x0;
mov.u16 %rs8, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs1, %rs2, %rs3, %rs4, %rs5, %rs6, %rs7, %rs8 }, [ %rd4 + 0 ], { %rs9, %rs10, %rs11, %rs12, %rs13, %rs14, %rs15, %rs16 };
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r7, %r7, %r7, %r7 };
add.s64 %rd5, %rd3, 512;
mov.u32 %r8, %r7;
mov.u32 %r9, %r7;
mov.u32 %r10, %r7;
mov.u32 %r11, %r7;
@%p1 ld.global.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd5 + 0 ];
mov.b32 {%rs57, %rs58}, %r8;
mov.b32 {%rs59, %rs60}, %r9;
mov.b32 {%rs61, %rs62}, %r10;
mov.b32 {%rs63, %rs64}, %r11;
cvt.f32.f16 %r32, %rs57;
cvt.f32.f16 %r33, %rs58;
cvt.f32.f16 %r34, %rs59;
cvt.f32.f16 %r35, %rs60;
cvt.f32.f16 %r36, %rs61;
cvt.f32.f16 %r37, %rs62;
cvt.f32.f16 %r38, %rs63;
cvt.f32.f16 %r39, %rs64;
add.s64 %rd6, %rd4, 512;
cvt.rn.bf16.f32 %rs25, %r32;
cvt.rn.bf16.f32 %rs26, %r33;
cvt.rn.bf16.f32 %rs27, %r34;
cvt.rn.bf16.f32 %rs28, %r35;
cvt.rn.bf16.f32 %rs29, %r36;
cvt.rn.bf16.f32 %rs30, %r37;
cvt.rn.bf16.f32 %rs31, %r38;
cvt.rn.bf16.f32 %rs32, %r39;
mov.u16 %rs17, 0x0;
mov.u16 %rs18, 0x0;
mov.u16 %rs19, 0x0;
mov.u16 %rs20, 0x0;
mov.u16 %rs21, 0x0;
mov.u16 %rs22, 0x0;
mov.u16 %rs23, 0x0;
mov.u16 %rs24, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs17, %rs18, %rs19, %rs20, %rs21, %rs22, %rs23, %rs24 }, [ %rd6 + 0 ], { %rs25, %rs26, %rs27, %rs28, %rs29, %rs30, %rs31, %rs32 };
@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r7, %r7, %r7, %r7 };
add.s64 %rd7, %rd3, 1024;
mov.u32 %r12, %r7;
mov.u32 %r13, %r7;
mov.u32 %r14, %r7;
mov.u32 %r15, %r7;
@%p1 ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd7 + 0 ];
mov.b32 {%rs65, %rs66}, %r12;
mov.b32 {%rs67, %rs68}, %r13;
mov.b32 {%rs69, %rs70}, %r14;
mov.b32 {%rs71, %rs72}, %r15;
cvt.f32.f16 %r40, %rs65;
cvt.f32.f16 %r41, %rs66;
cvt.f32.f16 %r42, %rs67;
cvt.f32.f16 %r43, %rs68;
cvt.f32.f16 %r44, %rs69;
cvt.f32.f16 %r45, %rs70;
cvt.f32.f16 %r46, %rs71;
cvt.f32.f16 %r47, %rs72;
add.s64 %rd8, %rd4, 1024;
cvt.rn.bf16.f32 %rs41, %r40;
cvt.rn.bf16.f32 %rs42, %r41;
cvt.rn.bf16.f32 %rs43, %r42;
cvt.rn.bf16.f32 %rs44, %r43;
cvt.rn.bf16.f32 %rs45, %r44;
cvt.rn.bf16.f32 %rs46, %r45;
cvt.rn.bf16.f32 %rs47, %r46;
cvt.rn.bf16.f32 %rs48, %r47;
mov.u16 %rs33, 0x0;
mov.u16 %rs34, 0x0;
mov.u16 %rs35, 0x0;
mov.u16 %rs36, 0x0;
mov.u16 %rs37, 0x0;
mov.u16 %rs38, 0x0;
mov.u16 %rs39, 0x0;
mov.u16 %rs40, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs33, %rs34, %rs35, %rs36, %rs37, %rs38, %rs39, %rs40 }, [ %rd8 + 0 ], { %rs41, %rs42, %rs43, %rs44, %rs45, %rs46, %rs47, %rs48 };
@%p1 st.global.v4.b32 [ %rd7 + 0 ], { %r7, %r7, %r7, %r7 };
ret;
$L__tmp1:
$L__func_end0:
}"#

def recognized (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.peerIR == some (peerIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.peerPTX == peerPTX.splitOn "\n" &&
    QKVTTIR.normalizeTTIR a.ownIR == some (ownIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.ownPTX == ownPTX.splitOn "\n"

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  packet : ∀ fused input, (send fused input).1 = SparsePackets.ngramDirect input
  retainsOwn : ∀ cut before i col,
    clearPeer cut before (SparsePackets.ownSlot cut i).val col = before (SparsePackets.ownSlot cut i).val col
  clearPeerWords : ∀ cut before i col, clearPeer cut before (SparsePackets.peerSlot cut i).val col = 0
  clearOwnWords : ∀ cut before i col, clearOwn cut before (SparsePackets.ownSlot cut i).val col = 0
  cycles : ∀ {capacity} (events : List (Event capacity)) scratch, Clean capacity scratch →
    (execute events scratch).1 = events.map reference ∧ Clean capacity (execute events scratch).2

def certify (a : Artifacts) : Option (Certificate a) :=
  if matched : recognized a then
    some ⟨matched, send_payload, owned_slice_live, peer_slot_erased, own_slot_erased, reuse_cycles⟩
  else none

/-- The checked owner program reads a lane's eight adjacent input words
before clearing them. Project each vector atomic onto its individual
channel while keeping the merge's recorded scalar order. -/
def Certificate.ownedWords {a : Artifacts} (_checked : Certificate a) : NgramAdam.OwnedWords :=
  fun p => if recognized a then SparsePackets.rowTiled p.ownGradient else p.ownGradient

theorem Certificate.ownedWords_eq {a : Artifacts} (checked : Certificate a) :
    checked.ownedWords = NgramAdam.Pull.ownGradient := by
  funext p slot col
  simp only [Certificate.ownedWords, checked.matched, if_true, SparsePackets.row_tiled_exact]

#axiomsClean Certificate.ownedWords_eq
#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.NgramScratch
