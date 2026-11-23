import models.NanoGPT.Track1.RowExchange
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Hopper code for the shard gather and cache writes. The word
loads, widened offsets, predicates and assertions are part of the check. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SparseLanding

structure Artifacts where
  serveIR : String
  servePTX : String
  landIR : String
  landPTX : String
  deriving BEq

private def serveIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_serve_rows(%shard: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %asked: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %served: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %count: i32, %first: i32, %shard_rows: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<true> : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<768> : tensor<4x1xi64, #blocked>
%cst_2 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_3 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16, #blocked>
%cst_4 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_5 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%row = tt.get_program_id x : i32
%row_6 = arith.muli %row, 4 : i32
%row_7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_8 = tt.splat %row_6 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_9 = arith.addi %row_8, %row_7 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_10 = arith.cmpi slt, %row_9, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local = tt.splat %asked : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_11 = tt.addptr %local, %row_9 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_12 = tt.load %local_11, %live_10, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_13 = tt.splat %first : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_14 = arith.subi %local_12, %local_13 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid = arith.cmpi sge, %local_14, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_15 = tt.splat %shard_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_16 = arith.cmpi slt, %local_14, %valid_15 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_17 = arith.andi %valid, %valid_16 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%0 = arith.xori %live_10, %cst_0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%1 = arith.ori %valid_17, %0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
tt.assert %1, "requested shard row out of bounds" : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%lanes = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%source = tt.expand_dims %local_14 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%source_18 = arith.extsi %source : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%source_19 = arith.muli %source_18, %cst_1 : tensor<4x1xi64, #blocked>
%source_20 = tt.splat %shard : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%source_21 = tt.addptr %source_20, %source_19 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi64, #blocked>
%destination = tt.expand_dims %row_9 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%destination_22 = arith.muli %destination, %cst_2 : tensor<4x1xi32, #blocked>
%destination_23 = tt.splat %served : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%destination_24 = tt.addptr %destination_23, %destination_22 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%bits = arith.andi %live_10, %valid_17 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_25 = tt.expand_dims %bits {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%bits_26 = tt.expand_dims %lanes {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_27 = tt.broadcast %source_21 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%bits_28 = tt.broadcast %bits_26 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_29 = tt.addptr %bits_27, %bits_28 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_30 = tt.broadcast %bits_25 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%bits_31 = tt.load %bits_29, %bits_30, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%2 = tt.expand_dims %live_10 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%3 = tt.broadcast %destination_24 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%4 = tt.addptr %3, %bits_28 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%5 = tt.broadcast %2 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %4, %bits_31, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%cols = arith.addi %lanes, %cst_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_32 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_33 = tt.broadcast %bits_32 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_34 = tt.addptr %bits_27, %bits_33 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_35 = tt.load %bits_34, %bits_30, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%6 = tt.addptr %3, %bits_33 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %6, %bits_35, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%cols_36 = arith.addi %lanes, %cst_5 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_37 = tt.expand_dims %cols_36 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_38 = tt.broadcast %bits_37 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_39 = tt.addptr %bits_27, %bits_38 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_40 = tt.load %bits_39, %bits_30, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%7 = tt.addptr %3, %bits_38 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %7, %bits_40, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.return
}
}"#

private def servePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.extern .func __assertfail
(
.param .b64 __assertfail_param_0,
.param .b64 __assertfail_param_1,
.param .b32 __assertfail_param_2,
.param .b64 __assertfail_param_3,
.param .b64 __assertfail_param_4
)
.noreturn;
.global .align 1 .b8 assertFunc_0[8] = {117, 110, 107, 110, 111, 119, 110};
.global .align 1 .b8 assertMessage_0[34] = {114, 101, 113, 117, 101, 115, 116, 101, 100, 32, 115, 104, 97, 114, 100, 32, 114, 111, 119, 32, 111, 117, 116, 32, 111, 102, 32, 98, 111, 117, 110, 100, 115};
.visible .entry _serve_rows(
.param .u64 .ptr .global .align 1 _serve_rows_param_0,
.param .u64 .ptr .global .align 1 _serve_rows_param_1,
.param .u64 .ptr .global .align 1 _serve_rows_param_2,
.param .u32 _serve_rows_param_3,
.param .u32 _serve_rows_param_4,
.param .u32 _serve_rows_param_5,
.param .u64 .ptr .global .align 1 _serve_rows_param_6,
.param .u64 .ptr .global .align 1 _serve_rows_param_7
)
.reqntid 128
{
.reg .pred %p<9>;
.reg .b32 %r<27>;
.reg .b64 %rd<20>;
$L__func_begin0:
ld.param.b32 %r4, [_serve_rows_param_3];
ld.param.b64 %rd4, [_serve_rows_param_1];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
shl.b32 %r7, %r6, 2;
mov.u32 %r1, %tid.x;
ld.param.b32 %r8, [_serve_rows_param_4];
bfe.u32 %r9, %r1, 5, 2;
ld.param.b32 %r10, [_serve_rows_param_5];
or.b32 %r2, %r9, %r7;
setp.lt.s32 %p2, %r2, %r4;
mad.wide.s32 %rd3, %r2, 4, %rd4;
mov.u32 %r5, 0x0;
@%p2 ld.global.b32 { %r5 }, [ %rd3 + 0 ];
sub.s32 %r3, %r5, %r8;
setp.gt.s32 %p3, %r3, -1;
setp.lt.s32 %p4, %r3, %r10;
and.pred %p1, %p3, %p4;
not.pred %p5, %p1;
and.pred %p6, %p2, %p5;
not.pred %p7, %p6;
@%p7 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd2, [_serve_rows_param_2];
ld.param.b64 %rd1, [_serve_rows_param_0];
bar.sync 0;
shl.b32 %r24, %r1, 3;
and.b32 %r25, %r24, 248;
mad.wide.s32 %rd11, %r3, 1536, %rd1;
mul.lo.s32 %r26, %r2, 768;
mad.wide.s32 %rd12, %r26, 2, %rd2;
and.pred %p8, %p2, %p1;
mul.wide.u32 %rd13, %r25, 2;
add.s64 %rd5, %rd11, %rd13;
mov.b32 %r11, 0;
mov.u32 %r12, %r11;
mov.u32 %r13, %r11;
mov.u32 %r14, %r11;
mov.u32 %r15, %r11;
@%p8 ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd5 + 0 ];
add.s64 %rd6, %rd12, %rd13;
@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r12, %r13, %r14, %r15 };
add.s64 %rd7, %rd5, 512;
mov.u32 %r16, %r11;
mov.u32 %r17, %r11;
mov.u32 %r18, %r11;
mov.u32 %r19, %r11;
@%p8 ld.global.v4.b32 { %r16, %r17, %r18, %r19 }, [ %rd7 + 0 ];
add.s64 %rd8, %rd6, 512;
@%p2 st.global.v4.b32 [ %rd8 + 0 ], { %r16, %r17, %r18, %r19 };
add.s64 %rd9, %rd5, 1024;
mov.u32 %r20, %r11;
mov.u32 %r21, %r11;
mov.u32 %r22, %r11;
mov.u32 %r23, %r11;
@%p8 ld.global.v4.b32 { %r20, %r21, %r22, %r23 }, [ %rd9 + 0 ];
add.s64 %rd10, %rd6, 1024;
@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r20, %r21, %r22, %r23 };
ret;
$L__BB0_1:
{
.param .b64 param0;
.param .b64 param1;
.param .b32 param2;
.param .b64 param3;
.param .b64 param4;
mov.b64 %rd14, assertFunc_0;
cvta.global.u64 %rd15, %rd14;
st.param.b64 [param3], %rd15;
mov.b64 %rd16, assertFile_0;
cvta.global.u64 %rd17, %rd16;
st.param.b64 [param1], %rd17;
mov.b64 %rd18, assertMessage_0;
cvta.global.u64 %rd19, %rd18;
st.param.b64 [param0], %rd19;
st.param.b64 [param4], 1;
st.param.b32 [param2], 21;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp1:
$L__func_end0:
}"#

private def landIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_land_ngram(%shard: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %received: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %cache: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %count: i32, %lo: i32, %hi: i32, %first: i32, %shard_rows: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<true> : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<768> : tensor<4x1xi64, #blocked>
%cst_2 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_3 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16, #blocked>
%cst_4 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_5 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%row = tt.get_program_id x : i32
%row_6 = arith.muli %row, 4 : i32
%row_7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_8 = tt.splat %row_6 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_9 = arith.addi %row_8, %row_7 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_10 = arith.cmpi slt, %row_9, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%owned = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%owned_11 = arith.cmpi sge, %row_9, %owned : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%owned_12 = tt.splat %hi : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%owned_13 = arith.cmpi slt, %row_9, %owned_12 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%owned_14 = arith.andi %owned_11, %owned_13 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%local = arith.andi %live_10, %owned_14 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_15 = tt.splat %want : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_16 = tt.addptr %local_15, %row_9 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_17 = tt.load %local_16, %local, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_18 = tt.splat %first : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_19 = arith.subi %local_17, %local_18 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid = arith.cmpi sge, %local_19, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_20 = tt.splat %shard_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_21 = arith.cmpi slt, %local_19, %valid_20 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%valid_22 = arith.andi %valid, %valid_21 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%0 = arith.xori %local, %cst_0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%1 = arith.ori %valid_22, %0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
tt.assert %1, "requested cache row out of bounds" : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%wire = arith.cmpi sge, %row_9, %owned_12 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%wire_23 = arith.subi %hi, %lo : i32
%wire_24 = tt.splat %wire_23 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%wire_25 = arith.select %wire, %wire_24, %cst : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%wire_26 = arith.subi %row_9, %wire_25 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%source = tt.splat %shard : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_27 = tt.splat %received : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_28 = arith.select %owned_14, %source, %source_27 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%index = arith.select %owned_14, %local_19, %wire_26 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lanes = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%src = tt.expand_dims %source_28 {axis = 1 : i32} : tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%src_29 = tt.expand_dims %index {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%src_30 = arith.extsi %src_29 : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%src_31 = arith.muli %src_30, %cst_1 : tensor<4x1xi64, #blocked>
%src_32 = tt.addptr %src, %src_31 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi64, #blocked>
%destination = tt.expand_dims %row_9 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%destination_33 = arith.muli %destination, %cst_2 : tensor<4x1xi32, #blocked>
%destination_34 = tt.splat %cache : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%destination_35 = tt.addptr %destination_34, %destination_33 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%bits = arith.xori %owned_14, %cst_0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_36 = arith.ori %bits, %valid_22 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_37 = arith.andi %live_10, %bits_36 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_38 = tt.expand_dims %bits_37 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%bits_39 = tt.expand_dims %lanes {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_40 = tt.broadcast %src_32 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%bits_41 = tt.broadcast %bits_39 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_42 = tt.addptr %bits_40, %bits_41 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_43 = tt.broadcast %bits_38 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%bits_44 = tt.load %bits_42, %bits_43, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%2 = tt.expand_dims %live_10 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%3 = tt.broadcast %destination_35 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%4 = tt.addptr %3, %bits_41 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%5 = tt.broadcast %2 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %4, %bits_44, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%cols = arith.addi %lanes, %cst_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_45 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_46 = tt.broadcast %bits_45 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_47 = tt.addptr %bits_40, %bits_46 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_48 = tt.load %bits_47, %bits_43, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%6 = tt.addptr %3, %bits_46 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %6, %bits_48, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%cols_49 = arith.addi %lanes, %cst_5 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_50 = tt.expand_dims %cols_49 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_51 = tt.broadcast %bits_50 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_52 = tt.addptr %bits_40, %bits_51 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_53 = tt.load %bits_52, %bits_43, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%7 = tt.addptr %3, %bits_51 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %7, %bits_53, %5 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.return
}
}"#

private def landPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.extern .func __assertfail
(
.param .b64 __assertfail_param_0,
.param .b64 __assertfail_param_1,
.param .b32 __assertfail_param_2,
.param .b64 __assertfail_param_3,
.param .b64 __assertfail_param_4
)
.noreturn;
.global .align 1 .b8 assertFunc_0[8] = {117, 110, 107, 110, 111, 119, 110};
.global .align 1 .b8 assertMessage_0[34] = {114, 101, 113, 117, 101, 115, 116, 101, 100, 32, 99, 97, 99, 104, 101, 32, 114, 111, 119, 32, 111, 117, 116, 32, 111, 102, 32, 98, 111, 117, 110, 100, 115};
.visible .entry _land_ngram(
.param .u64 .ptr .global .align 1 _land_ngram_param_0,
.param .u64 .ptr .global .align 1 _land_ngram_param_1,
.param .u64 .ptr .global .align 1 _land_ngram_param_2,
.param .u64 .ptr .global .align 1 _land_ngram_param_3,
.param .u32 _land_ngram_param_4,
.param .u32 _land_ngram_param_5,
.param .u32 _land_ngram_param_6,
.param .u32 _land_ngram_param_7,
.param .u32 _land_ngram_param_8,
.param .u64 .ptr .global .align 1 _land_ngram_param_9,
.param .u64 .ptr .global .align 1 _land_ngram_param_10
)
.reqntid 128
{
.reg .pred %p<15>;
.reg .b32 %r<33>;
.reg .b64 %rd<22>;
$L__func_begin0:
ld.param.b32 %r6, [_land_ngram_param_6];
ld.param.b32 %r5, [_land_ngram_param_5];
ld.param.b32 %r4, [_land_ngram_param_4];
$L__tmp0:
mov.u32 %r8, %ctaid.x;
shl.b32 %r9, %r8, 2;
ld.param.b64 %rd5, [_land_ngram_param_2];
mov.u32 %r1, %tid.x;
bfe.u32 %r10, %r1, 5, 2;
or.b32 %r2, %r10, %r9;
setp.lt.s32 %p4, %r2, %r4;
ld.param.b32 %r11, [_land_ngram_param_7];
setp.ge.s32 %p5, %r2, %r5;
ld.param.b32 %r12, [_land_ngram_param_8];
setp.lt.s32 %p6, %r2, %r6;
and.pred %p1, %p5, %p6;
and.pred %p3, %p4, %p1;
mad.wide.s32 %rd4, %r2, 4, %rd5;
mov.u32 %r7, 0x0;
@%p3 ld.global.b32 { %r7 }, [ %rd4 + 0 ];
sub.s32 %r3, %r7, %r11;
setp.gt.s32 %p7, %r3, -1;
setp.lt.s32 %p8, %r3, %r12;
and.pred %p2, %p7, %p8;
not.pred %p9, %p2;
and.pred %p10, %p3, %p9;
not.pred %p11, %p10;
@%p11 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd3, [_land_ngram_param_3];
ld.param.b64 %rd2, [_land_ngram_param_1];
ld.param.b64 %rd1, [_land_ngram_param_0];
bar.sync 0;
sub.s32 %r26, %r5, %r6;
selp.b32 %r27, 0, %r26, %p6;
add.s32 %r28, %r27, %r2;
selp.b64 %rd12, %rd1, %rd2, %p1;
selp.b32 %r29, %r3, %r28, %p1;
shl.b32 %r30, %r1, 3;
and.b32 %r31, %r30, 248;
mad.wide.s32 %rd13, %r29, 1536, %rd12;
mul.lo.s32 %r32, %r2, 768;
mad.wide.s32 %rd14, %r32, 2, %rd3;
not.pred %p13, %p1;
or.pred %p14, %p2, %p13;
and.pred %p12, %p4, %p14;
mul.wide.u32 %rd15, %r31, 2;
add.s64 %rd6, %rd13, %rd15;
mov.b32 %r13, 0;
mov.u32 %r14, %r13;
mov.u32 %r15, %r13;
mov.u32 %r16, %r13;
mov.u32 %r17, %r13;
@%p12 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd6 + 0 ];
add.s64 %rd7, %rd14, %rd15;
@%p4 st.global.v4.b32 [ %rd7 + 0 ], { %r14, %r15, %r16, %r17 };
add.s64 %rd8, %rd6, 512;
mov.u32 %r18, %r13;
mov.u32 %r19, %r13;
mov.u32 %r20, %r13;
mov.u32 %r21, %r13;
@%p12 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd8 + 0 ];
add.s64 %rd9, %rd7, 512;
@%p4 st.global.v4.b32 [ %rd9 + 0 ], { %r18, %r19, %r20, %r21 };
add.s64 %rd10, %rd6, 1024;
mov.u32 %r22, %r13;
mov.u32 %r23, %r13;
mov.u32 %r24, %r13;
mov.u32 %r25, %r13;
@%p12 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd10 + 0 ];
add.s64 %rd11, %rd7, 1024;
@%p4 st.global.v4.b32 [ %rd11 + 0 ], { %r22, %r23, %r24, %r25 };
ret;
$L__BB0_1:
{
.param .b64 param0;
.param .b64 param1;
.param .b32 param2;
.param .b64 param3;
.param .b64 param4;
mov.b64 %rd16, assertFunc_0;
cvta.global.u64 %rd17, %rd16;
st.param.b64 [param3], %rd17;
mov.b64 %rd18, assertFile_0;
cvta.global.u64 %rd19, %rd18;
st.param.b64 [param1], %rd19;
mov.b64 %rd20, assertMessage_0;
cvta.global.u64 %rd21, %rd20;
st.param.b64 [param0], %rd21;
st.param.b64 [param4], 1;
st.param.b32 [param2], 41;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp1:
$L__func_end0:
}"#

private def rowPTX (s : String) : List String :=
  (QKVTTIR.normalizePTX s).filter (!·.startsWith ".global .align 1 .b8 assertFile_")

def recognized (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.serveIR == some (serveIR.splitOn "\n") &&
    QKVTTIR.normalizeTTIR a.landIR == some (landIR.splitOn "\n") &&
    rowPTX a.servePTX == servePTX.splitOn "\n" &&
    rowPTX a.landPTX == landPTX.splitOn "\n"

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  ports : direct = staged
  cacheTail : ∀ (s : LandInput) (row : Nat) (col : SparsePackets.Channel),
    s.cut.rows ≤ row → directLand s row col = some (s.previous row col)
  address : ∀ {rows} (row : Fin rows) (col : SparsePackets.Channel),
    rows < 2 ^ 31 →
      (Int64.ofNat (row.val * 768 + col.val)).toInt = row.val * 768 + col.val
  servedWords : ∀ wanted banks owner i col,
    directServe (RowExchange.serveInput wanted banks owner) i col = (RowExchange.served wanted banks owner col)[i]?
  landedWords : ∀ wanted banks (rank : Fin 8) previous slot col,
    slot < (wanted rank).size →
      directLand (RowExchange.landingInput wanted banks rank previous) slot col =
        some (RowExchange.globalWord banks ((wanted rank).read slot) col)

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then
    some ⟨h, kernels_equal, cache_tail_exact, shard_address_i64,
      RowExchange.served_packet_word, RowExchange.landed_cache_word⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.SparseLanding
