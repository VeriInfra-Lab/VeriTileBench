import models.NanoGPT.Track1.SampledInverse
import models.NanoGPT.Track1.SampledGatherBinding

/-! Normalized SM90 TTGIR and PTX for the fused sampled gather. The two
branches take disjoint buffers: one emits both FP8 byte layouts, and the
other searches the last class per block, sums the selected warp's
prefix, and clears and publishes the inverse map with a CTA barrier. The output contract is pointwise over both branches. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SampledInverse
open SampledRows (Count ClassID Point)

private def graph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_gather_candidate_rows_and_inverse(%weights: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %candidates: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %rows: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %transposed: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %inverse: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c-1_i32 = arith.constant -1 : i32
%cst = arith.constant dense<16> : tensor<32xi32, #blocked>
%cst_0 = arith.constant dense<@P@> : tensor<32xi32, #blocked>
%cst_1 = arith.constant dense<50304> : tensor<32xi64, #blocked>
%cst_2 = arith.constant dense<@P@> : tensor<256xi32, #blocked>
%cst_3 = arith.constant dense<50304> : tensor<256xi64, #blocked>
%cst_4 = arith.constant dense<-1> : tensor<256xi32, #blocked>
%cst_5 = arith.constant dense<768> : tensor<128x1xi32, #blocked1>
%cst_6 = arith.constant dense<@P@> : tensor<1x128xi32, #blocked2>
%pid = tt.get_program_id x : i32
%0 = arith.cmpi slt, %pid, 197 : i32
scf.if %0 {
%start = arith.muli %pid, 256 : i32
%stop = arith.addi %start, 256 : i32
%stop_7 = arith.minsi %stop, 50304 : i32
%hi:2 = scf.while (%lo_26 = 0, %hi_27 = @B@ ) : (i32, i32) -> (i32, i32) {
%9 = arith.cmpi slt, %lo_26, %hi_27 : i32
scf.condition(%9) %lo_26, %hi_27 : i32, i32
} do {
^bb0(%lo_26: i32, %hi_27: i32):
%mid = arith.subi %hi_27, %lo_26 : i32
%mid_28 = arith.divsi %mid, 2 : i32
%mid_29 = arith.addi %lo_26, %mid_28 : i32
%before = arith.addi %mid_29, 1 : i32
%before_30 = arith.muli %before, 16 : i32
%before_31 = tt.addptr %candidates, %before_30 : !tt.ptr<i64>, i32
%before_32 = tt.addptr %before_31, %c-1_i32 : !tt.ptr<i64>, i32
%before_33 = tt.load %before_32 : !tt.ptr<i64>
%before_34 = arith.trunci %before_33 : i64 to i32
%before_35 = arith.cmpi slt, %before_34, %start : i32
%lo_36 = arith.select %before_35, %before, %lo_26 : i32
%hi_37 = arith.select %before_35, %hi_27, %mid_29 : i32
scf.yield %lo_36, %hi_37 : i32, i32
}
%within = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
%index = arith.muli %hi#0, 16 : i32
%index_8 = tt.splat %index : i32 -> tensor<32xi32, #blocked>
%index_9 = arith.addi %index_8, %within : tensor<32xi32, #blocked>
%value = tt.splat %candidates : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #blocked>
%value_10 = tt.addptr %value, %index_9 : tensor<32x!tt.ptr<i64>, #blocked>, tensor<32xi32, #blocked>
%value_11 = arith.cmpi slt, %within, %cst : tensor<32xi32, #blocked>
%value_12 = arith.cmpi slt, %index_9, %cst_0 : tensor<32xi32, #blocked>
%value_13 = arith.andi %value_11, %value_12 : tensor<32xi1, #blocked>
%value_14 = tt.load %value_10, %value_13, %cst_1 : tensor<32x!tt.ptr<i64>, #blocked>
%value_15 = arith.trunci %value_14 : tensor<32xi64, #blocked> to tensor<32xi32, #blocked>
%lo = tt.splat %start : i32 -> tensor<32xi32, #blocked>
%lo_16 = arith.cmpi slt, %value_15, %lo : tensor<32xi32, #blocked>
%lo_17 = arith.extui %lo_16 : tensor<32xi1, #blocked> to tensor<32xi32, #blocked>
%lo_18 = "tt.reduce"(%lo_17) <{axis = 0 : i32}> ({
^bb0(%lo_26: i32, %lo_27: i32):
%lo_28 = arith.addi %lo_26, %lo_27 : i32
tt.reduce.return %lo_28 : i32
}) : (tensor<32xi32, #blocked>) -> i32
%lo_19 = arith.addi %index, %lo_18 : i32
%lane = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%cls = tt.splat %start : i32 -> tensor<256xi32, #blocked>
%cls_20 = arith.addi %cls, %lane : tensor<256xi32, #blocked>
%slot = tt.splat %lo_19 : i32 -> tensor<256xi32, #blocked>
%slot_21 = arith.addi %slot, %lane : tensor<256xi32, #blocked>
%vocab = tt.splat %candidates : !tt.ptr<i64> -> tensor<256x!tt.ptr<i64>, #blocked>
%vocab_22 = tt.addptr %vocab, %slot_21 : tensor<256x!tt.ptr<i64>, #blocked>, tensor<256xi32, #blocked>
%vocab_23 = arith.cmpi slt, %slot_21, %cst_2 : tensor<256xi32, #blocked>
%vocab_24 = tt.load %vocab_22, %vocab_23, %cst_3 : tensor<256x!tt.ptr<i64>, #blocked>
%vocab_25 = arith.trunci %vocab_24 : tensor<256xi64, #blocked> to tensor<256xi32, #blocked>
%1 = tt.splat %inverse : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked>
%2 = tt.addptr %1, %cls_20 : tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>
%3 = tt.splat %stop_7 : i32 -> tensor<256xi32, #blocked>
%4 = arith.cmpi slt, %cls_20, %3 : tensor<256xi32, #blocked>
tt.store %2, %cst_4, %4 : tensor<256x!tt.ptr<i32>, #blocked>
gpu.barrier
%5 = tt.addptr %1, %vocab_25 : tensor<256x!tt.ptr<i32>, #blocked>, tensor<256xi32, #blocked>
%6 = arith.cmpi sge, %vocab_25, %cls : tensor<256xi32, #blocked>
%7 = arith.cmpi slt, %vocab_25, %3 : tensor<256xi32, #blocked>
%8 = arith.andi %6, %7 : tensor<256xi1, #blocked>
tt.store %5, %slot_21, %8 : tensor<256x!tt.ptr<i32>, #blocked>
} else {
%tile = arith.subi %pid, 197 : i32
%1 = arith.divsi %tile, 6 : i32
%2 = arith.remsi %tile, 6 : i32
%start_m = arith.muli %1, 128 : i32
%start_n = arith.muli %2, 128 : i32
%rm = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%vocab = tt.addptr %candidates, %start_m : !tt.ptr<i64>, i32
%vocab_7 = tt.splat %vocab : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>
%vocab_8 = tt.addptr %vocab_7, %rm : tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%vocab_9 = tt.load %vocab_8 : tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked1}>>
%vocab_10 = arith.trunci %vocab_9 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%tile_11 = tt.expand_dims %vocab_10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%tile_12 = arith.muli %tile_11, %cst_5 : tensor<128x1xi32, #blocked1>
%tile_13 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%tile_14 = tt.splat %start_n : i32 -> tensor<1x128xi32, #blocked1>
%tile_15 = arith.addi %tile_14, %tile_13 : tensor<1x128xi32, #blocked1>
%tile_16 = tt.broadcast %tile_12 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%tile_17 = tt.broadcast %tile_15 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%tile_18 = arith.addi %tile_16, %tile_17 : tensor<128x128xi32, #blocked1>
%tile_19 = tt.splat %weights : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
%tile_20 = tt.addptr %tile_19, %tile_18 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
%tile_21 = tt.load %tile_20 : tensor<128x128x!tt.ptr<i8>, #blocked1>
%3 = arith.muli %1, 98304 : i32
%4 = arith.addi %3, %start_n : i32
%5 = tt.addptr %rows, %4 : !tt.ptr<i8>, i32
%6 = tt.expand_dims %rm {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%7 = arith.muli %6, %cst_5 : tensor<128x1xi32, #blocked1>
%8 = tt.broadcast %7 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%9 = tt.broadcast %tile_13 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%10 = arith.addi %8, %9 : tensor<128x128xi32, #blocked1>
%11 = tt.splat %5 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
%12 = tt.addptr %11, %10 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
tt.store %12, %tile_21 : tensor<128x128x!tt.ptr<i8>, #blocked1>
%moved = ttg.convert_layout %tile_21 : tensor<128x128xi8, #blocked1> -> tensor<128x128xi8, #blocked2>
%tm = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%tn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%13 = arith.muli %2, @128P@ : i32
%14 = arith.addi %13, %start_m : i32
%15 = tt.addptr %transposed, %14 : !tt.ptr<i8>, i32
%16 = tt.expand_dims %tn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
%17 = arith.muli %16, %cst_6 : tensor<1x128xi32, #blocked2>
%18 = tt.expand_dims %tm {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
%19 = tt.broadcast %17 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
%20 = tt.broadcast %18 : tensor<128x1xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
%21 = arith.addi %19, %20 : tensor<128x128xi32, #blocked2>
%22 = tt.splat %15 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked2>
%23 = tt.addptr %22, %21 : tensor<128x128x!tt.ptr<i8>, #blocked2>, tensor<128x128xi32, #blocked2>
tt.store %23, %moved : tensor<128x128x!tt.ptr<i8>, #blocked2>
}
tt.return
}
}
"#

private def assembly : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _gather_candidate_rows_and_inverse(
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_0,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_1,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_2,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_3,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_4,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_5,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_and_inverse_param_6
)
.reqntid 256
{
.reg .pred %p<13>;
.reg .b32 %r<143>;
.reg .b64 %rd<47>;
$L__func_begin0:
ld.param.b64 %rd2, [_gather_candidate_rows_and_inverse_param_1];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
setp.gt.u32 %p1, %r1, 196;
@%p1 bra $L__BB0_4;
ld.param.b64 %rd5, [_gather_candidate_rows_and_inverse_param_4];
$L__tmp1:
shl.b32 %r2, %r1, 8;
mov.b32 %r142, 0;
mov.b32 %r141, @B@;
$L__BB0_2:
sub.s32 %r118, %r141, %r142;
shr.u32 %r119, %r118, 31;
add.s32 %r120, %r118, %r119;
shr.s32 %r121, %r120, 1;
add.s32 %r122, %r121, %r142;
add.s32 %r123, %r122, 1;
shl.b32 %r124, %r123, 4;
mad.wide.s32 %rd38, %r124, 8, %rd2;
add.s64 %rd37, %rd38, -8;
mov.u64 %rd36, 0x0;
ld.global.b64 { %rd36 }, [ %rd37 + 0 ];
cvt.u32.u64 %r125, %rd36;
setp.gt.s32 %p2, %r2, %r125;
selp.b32 %r142, %r123, %r142, %p2;
selp.b32 %r141, %r141, %r122, %p2;
setp.lt.s32 %p3, %r142, %r141;
@%p3 bra $L__BB0_2;
min.u32 %r128, %r2, 50048;
add.s32 %r129, %r128, 256;
mov.u32 %r130, %tid.x;
and.b32 %r131, %r130, 255;
and.b32 %r132, %r130, 31;
shl.b32 %r133, %r142, 4;
add.s32 %r134, %r133, %r132;
mad.wide.s32 %rd40, %r134, 8, %rd2;
setp.lt.u32 %p8, %r132, 16;
setp.lt.s32 %p9, %r134, @P@;
and.pred %p4, %p8, %p9;
mov.u64 %rd39, 0xc480;
@%p4 ld.global.b64 { %rd39 }, [ %rd40 + 0 ];
cvt.u32.u64 %r135, %rd39;
setp.gt.s32 %p10, %r2, %r135;
selp.b32 %r136, 1, 0, %p10;
mov.b32 %r126, -1;
$L__tmp2:
redux.sync.add.s32 %r137, %r136, %r126;
$L__tmp3:
or.b32 %r138, %r2, %r131;
add.s32 %r139, %r133, %r131;
add.s32 %r127, %r139, %r137;
mad.wide.s32 %rd42, %r127, 8, %rd2;
setp.lt.s32 %p5, %r127, @P@;
mov.u64 %rd41, 0xc480;
@%p5 ld.global.b64 { %rd41 }, [ %rd42 + 0 ];
cvt.u32.u64 %r140, %rd41;
mad.wide.u32 %rd43, %r138, 4, %rd5;
setp.lt.u32 %p6, %r138, %r129;
@%p6 st.global.b32 [ %rd43 + 0 ], { %r126 };
bar.sync 0;
shl.b64 %rd45, %rd41, 32;
shr.s64 %rd46, %rd45, 30;
add.s64 %rd44, %rd5, %rd46;
setp.le.s32 %p11, %r2, %r140;
setp.gt.s32 %p12, %r129, %r140;
and.pred %p7, %p11, %p12;
@%p7 st.global.b32 [ %rd44 + 0 ], { %r127 };
$L__tmp4:
bra.uni $L__BB0_5;
$L__BB0_4:
ld.param.b64 %rd4, [_gather_candidate_rows_and_inverse_param_3];
ld.param.b64 %rd3, [_gather_candidate_rows_and_inverse_param_2];
ld.param.b64 %rd1, [_gather_candidate_rows_and_inverse_param_0];
add.s32 %r35, %r1, -197;
mul.hi.u32 %r36, %r35, -1431655765;
shr.u32 %r37, %r36, 2;
mul.lo.s32 %r38, %r37, 6;
sub.s32 %r39, %r35, %r38;
$L__tmp5:
shl.b32 %r40, %r37, 7;
shl.b32 %r41, %r39, 7;
mov.u32 %r42, %tid.x;
shr.u32 %r43, %r42, 1;
and.b32 %r44, %r43, 124;
shl.b32 %r45, %r42, 4;
and.b32 %r46, %r45, 112;
mad.wide.s32 %rd24, %r40, 8, %rd2;
mad.wide.u32 %rd8, %r44, 8, %rd24;
add.s64 %rd11, %rd8, 16;
mov.u64 %rd6, 0x0;
mov.u64 %rd7, 0x0;
ld.global.v2.b64 { %rd6, %rd7 }, [ %rd8 + 0 ];
mov.u64 %rd9, 0x0;
mov.u64 %rd10, 0x0;
ld.global.v2.b64 { %rd9, %rd10 }, [ %rd11 + 0 ];
cvt.u32.u64 %r47, %rd6;
cvt.u32.u64 %r48, %rd7;
cvt.u32.u64 %r49, %rd9;
cvt.u32.u64 %r50, %rd10;
or.b32 %r51, %r41, %r46;
mad.lo.s32 %r52, %r47, 768, %r51;
mad.lo.s32 %r53, %r48, 768, %r51;
mad.lo.s32 %r54, %r49, 768, %r51;
mad.lo.s32 %r55, %r50, 768, %r51;
cvt.s64.s32 %rd25, %r52;
add.s64 %rd12, %rd1, %rd25;
cvt.s64.s32 %rd26, %r53;
add.s64 %rd13, %rd1, %rd26;
cvt.s64.s32 %rd27, %r54;
add.s64 %rd14, %rd1, %rd27;
cvt.s64.s32 %rd28, %r55;
add.s64 %rd15, %rd1, %rd28;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
ld.global.v4.b32 { %r3, %r4, %r5, %r6 }, [ %rd12 + 0 ];
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd13 + 0 ];
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd14 + 0 ];
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd15 + 0 ];
mul.lo.s32 %r56, %r37, 98304;
or.b32 %r57, %r56, %r41;
cvt.s64.s32 %rd29, %r57;
add.s64 %rd30, %rd3, %rd29;
mul.lo.s32 %r58, %r44, 768;
or.b32 %r59, %r58, %r46;
add.s32 %r60, %r59, 1536;
cvt.u64.u32 %rd31, %r59;
add.s64 %rd16, %rd30, %rd31;
add.s64 %rd17, %rd16, 768;
cvt.u64.u32 %rd32, %r60;
add.s64 %rd18, %rd30, %rd32;
add.s64 %rd19, %rd16, 2304;
st.global.v4.b32 [ %rd16 + 0 ], { %r3, %r4, %r5, %r6 };
st.global.v4.b32 [ %rd17 + 0 ], { %r7, %r8, %r9, %r10 };
st.global.v4.b32 [ %rd18 + 0 ], { %r11, %r12, %r13, %r14 };
st.global.v4.b32 [ %rd19 + 0 ], { %r15, %r16, %r17, %r18 };
and.b32 %r61, %r45, 4080;
mov.b32 %r62, global_smem;
add.s32 %r63, %r62, %r61;
st.shared.v4.b32 [%r63], {%r3, %r4, %r5, %r6};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r64, %r65, %r66, %r67}, [%r63];
bar.warp.sync -1;
st.shared.v4.b32 [%r63], {%r7, %r8, %r9, %r10};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r68, %r69, %r70, %r71}, [%r63];
bar.warp.sync -1;
st.shared.v4.b32 [%r63], {%r11, %r12, %r13, %r14};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r72, %r73, %r74, %r75}, [%r63];
bar.warp.sync -1;
st.shared.v4.b32 [%r63], {%r15, %r16, %r17, %r18};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r76, %r77, %r78, %r79}, [%r63];
and.b32 %r80, %r43, 112;
shl.b32 %r81, %r42, 2;
and.b32 %r82, %r81, 124;
mad.lo.s32 %r83, %r39, @128P@, %r40;
cvt.s64.s32 %rd33, %r83;
add.s64 %rd34, %rd4, %rd33;
mul.lo.s32 %r84, %r82, @P@;
or.b32 %r85, %r84, %r80;
cvt.u64.u32 %rd35, %r85;
add.s64 %rd20, %rd34, %rd35;
add.s64 %rd21, %rd20, @P@;
add.s64 %rd22, %rd20, @2P@;
add.s64 %rd23, %rd20, @3P@;
prmt.b32 %r86, %r64, %r68, 0x40U;
prmt.b32 %r87, %r86, %r72, 0x410U;
prmt.b32 %r19, %r87, %r76, 0x4210U;
prmt.b32 %r88, %r65, %r69, 0x40U;
prmt.b32 %r89, %r88, %r73, 0x410U;
prmt.b32 %r20, %r89, %r77, 0x4210U;
prmt.b32 %r90, %r66, %r70, 0x40U;
prmt.b32 %r91, %r90, %r74, 0x410U;
prmt.b32 %r21, %r91, %r78, 0x4210U;
prmt.b32 %r92, %r67, %r71, 0x40U;
prmt.b32 %r93, %r92, %r75, 0x410U;
prmt.b32 %r22, %r93, %r79, 0x4210U;
st.global.v4.b32 [ %rd20 + 0 ], { %r19, %r20, %r21, %r22 };
prmt.b32 %r94, %r64, %r68, 0x51U;
prmt.b32 %r95, %r94, %r72, 0x510U;
prmt.b32 %r23, %r95, %r76, 0x5210U;
prmt.b32 %r96, %r65, %r69, 0x51U;
prmt.b32 %r97, %r96, %r73, 0x510U;
prmt.b32 %r24, %r97, %r77, 0x5210U;
prmt.b32 %r98, %r66, %r70, 0x51U;
prmt.b32 %r99, %r98, %r74, 0x510U;
prmt.b32 %r25, %r99, %r78, 0x5210U;
prmt.b32 %r100, %r67, %r71, 0x51U;
prmt.b32 %r101, %r100, %r75, 0x510U;
prmt.b32 %r26, %r101, %r79, 0x5210U;
st.global.v4.b32 [ %rd21 + 0 ], { %r23, %r24, %r25, %r26 };
prmt.b32 %r102, %r64, %r68, 0x62U;
prmt.b32 %r103, %r102, %r72, 0x610U;
prmt.b32 %r27, %r103, %r76, 0x6210U;
prmt.b32 %r104, %r65, %r69, 0x62U;
prmt.b32 %r105, %r104, %r73, 0x610U;
prmt.b32 %r28, %r105, %r77, 0x6210U;
prmt.b32 %r106, %r66, %r70, 0x62U;
prmt.b32 %r107, %r106, %r74, 0x610U;
prmt.b32 %r29, %r107, %r78, 0x6210U;
prmt.b32 %r108, %r67, %r71, 0x62U;
prmt.b32 %r109, %r108, %r75, 0x610U;
prmt.b32 %r30, %r109, %r79, 0x6210U;
st.global.v4.b32 [ %rd22 + 0 ], { %r27, %r28, %r29, %r30 };
prmt.b32 %r110, %r64, %r68, 0x73U;
prmt.b32 %r111, %r110, %r72, 0x710U;
prmt.b32 %r31, %r111, %r76, 0x7210U;
prmt.b32 %r112, %r65, %r69, 0x73U;
prmt.b32 %r113, %r112, %r73, 0x710U;
prmt.b32 %r32, %r113, %r77, 0x7210U;
prmt.b32 %r114, %r66, %r70, 0x73U;
prmt.b32 %r115, %r114, %r74, 0x710U;
prmt.b32 %r33, %r115, %r78, 0x7210U;
prmt.b32 %r116, %r67, %r71, 0x73U;
prmt.b32 %r117, %r116, %r75, 0x710U;
prmt.b32 %r34, %r117, %r79, 0x7210U;
st.global.v4.b32 [ %rd23 + 0 ], { %r31, %r32, %r33, %r34 };
$L__tmp6:
$L__BB0_5:
ret;
$L__tmp7:
$L__func_end0:
}
"#

def matchesArtifacts (c : Count) (ttgir ptx : String) : Bool :=
  let limit := s!"%c{c.rows / 16}_i32"
  let ttgir := ttgir.replace (limit ++ ")") (limit ++ " )")
  QKVTTIR.normalizeTTIR ttgir == some (SampledRows.artifactLines c graph) &&
    QKVTTIR.normalizePTX ptx == SampledRows.artifactLines c assembly

structure Certificate (c : Count) (ttgir ptx : String) : Type where
  matched : matchesArtifacts c ttgir ptx = true
  gatherTiles : ∀ row col, col < 6 →
    197 ≤ packPID row col ∧ (packPID row col - 197) / 6 = row ∧ (packPID row col - 197) % 6 = col
  searchRank : ∀ (s : Candidates c) t, blockedLower s t = lower s t
  inverseTiles : ∀ (s : Candidates c) t (i : Fin c.rows),
    start t ≤ (s.ids i).val ∧ (s.ids i).val < stop t →
      ∃ lane : Lane, blockedLower s t + lane.val = i.val
  clearTiles : ∀ t (cls : ClassID),
    (∃ lane : Lane, start t + lane.val = cls.val ∧ cls.val < stop t) ↔ t = owner cls
  gatherStores : ∀ (bank : Nat → UInt8) (s : Candidates c) (p : Point c),
    SampledRows.Vector.observe bank s.ids p = SampledRows.reference c bank s.ids p.row p.column
  inverseMap : ∀ s : Candidates c, blockedInverse s = Densify.inverseMap s.ids

def certify (c : Count) (ttgir ptx : String) : Option (Certificate c ttgir ptx) :=
  if h : matchesArtifacts c ttgir ptx = true then
    some ⟨h, gather_pid_exact, blocked_lower_exact,
      fun s t i h => by rw [blocked_lower_exact]; exact covers_candidates s t i h,
      clear_owner, fun bank s p => SampledRows.Vector.observe_equal bank s.ids p, blocked_inverse_exact⟩
  else none

structure Program (c : Count) where
  ttgir : String
  ptx : String
  checked : Certificate c ttgir ptx

def Program.check (c : Count) (ttgir ptx : String) : Option (Program c) := do
  return ⟨ttgir, ptx, ← certify c ttgir ptx⟩

def Program.run {c : Count} (program : Program c) (bank : Nat → UInt8)
    (candidates : Candidates c) : Option (Prepared c) :=
  if matchesArtifacts c program.ttgir program.ptx then
    some (SampledRows.Vector.observe bank candidates.ids, blockedInverse candidates)
  else none

theorem Program.run_eq {c : Count} (program : Program c) (bank : Nat → UInt8) (candidates : Candidates c) :
    program.run bank candidates = some (referenceGather true c bank candidates) := by
  simp only [Program.run, program.checked.matched, if_true, Option.some.injEq, referenceGather]
  exact Prod.ext (funext (program.checked.gatherStores bank candidates)) (program.checked.inverseMap candidates)

abbrev Programs := (c : Count) → Program c

def Programs.run (programs : Programs) : Gather := fun hopper c bank candidates =>
  if hopper then ((programs c).run bank candidates).getD (fun _ => ((0,0), (0,0)), fun _ => none)
  else referenceGather false c bank candidates

theorem Programs.run_eq (programs : Programs) : programs.run = referenceGather := by
  funext hopper c bank candidates
  cases hopper
  · rfl
  · simp only [Programs.run, if_true, Program.run_eq, Option.getD_some]

#axiomsClean certify
#axiomsClean Program.run_eq
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.SampledInverse
