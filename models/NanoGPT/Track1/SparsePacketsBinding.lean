import models.NanoGPT.Track1.RowGradientExchange
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Hopper code for the packet builders and sent value-row clear.
Address arithmetic, tail masks, the device assertion and conversion instructions are retained. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SparsePackets

structure Artifacts where
  valueIR : String
  valuePTX : String
  valueWireIR : String
  valueWirePTX : String
  ngramIR : String
  ngramPTX : String
  clearIR : String
  clearPTX : String
  deriving BEq

private def valueIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_value_gradient_packets(%src: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %own: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %wire: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %own_rows: i32, %peer_rows: i32, %lo: i32, %source_rows: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<true> : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<0> : tensor<4x256xi16, #blocked>
%cst_3 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_4 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%block = tt.get_program_id x : i32
%lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%own_blocks = arith.addi %own_rows, 3 : i32
%own_blocks_5 = arith.divui %own_blocks, 4 : i32
%is_own = arith.cmpi slt, %block, %own_blocks_5 : i32
%dst = arith.select %is_own, %own, %wire : !tt.ptr<i16>
%row = arith.select %is_own, 0, %own_blocks_5 : i32
%row_6 = arith.subi %block, %row : i32
%row_7 = arith.muli %row_6, 4 : i32
%row_8 = tt.splat %row_7 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_9 = arith.addi %row_8, %lanes : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = arith.select %is_own, %own_rows, %peer_rows : i32
%live_10 = tt.splat %live : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_11 = arith.cmpi slt, %row_9, %live_10 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_12 = arith.cmpi slt, %row_9, %selected : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_13 = tt.splat %own_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_14 = arith.select %selected_12, %cst, %selected_13 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_15 = tt.splat %is_own : i1 -> tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_16 = arith.select %selected_15, %selected, %selected_14 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_17 = arith.addi %row_9, %selected_16 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%source = tt.splat %ids : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_18 = tt.addptr %source, %selected_17 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_19 = tt.load %source_18, %live_11, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded = arith.cmpi sge, %source_19, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_20 = tt.splat %source_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_21 = arith.cmpi slt, %source_19, %bounded_20 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_22 = arith.andi %bounded, %bounded_21 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%0 = arith.xori %live_11, %cst_0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%1 = arith.ori %bounded_22, %0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
tt.assert %1, "value gradient row out of bounds" : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%source_row = tt.expand_dims %source_19 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%source_row_23 = arith.muli %source_row, %cst_1 : tensor<4x1xi32, #blocked>
%source_row_24 = tt.splat %src : !tt.ptr<i16> -> tensor<4x1x!tt.ptr<i16>, #blocked>
%source_row_25 = tt.addptr %source_row_24, %source_row_23 : tensor<4x1x!tt.ptr<i16>, #blocked>, tensor<4x1xi32, #blocked>
%dest_row = tt.expand_dims %row_9 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%dest_row_26 = arith.muli %dest_row, %cst_1 : tensor<4x1xi32, #blocked>
%dest_row_27 = tt.splat %dst : !tt.ptr<i16> -> tensor<4x1x!tt.ptr<i16>, #blocked>
%dest_row_28 = tt.addptr %dest_row_27, %dest_row_26 : tensor<4x1x!tt.ptr<i16>, #blocked>, tensor<4x1xi32, #blocked>
%bits = arith.andi %live_11, %bounded_22 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_29 = tt.expand_dims %bits {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%bits_30 = tt.expand_dims %columns {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_31 = tt.broadcast %source_row_25 : tensor<4x1x!tt.ptr<i16>, #blocked> -> tensor<4x256x!tt.ptr<i16>, #blocked>
%bits_32 = tt.broadcast %bits_30 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_33 = tt.addptr %bits_31, %bits_32 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_34 = tt.broadcast %bits_29 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%bits_35 = tt.load %bits_33, %bits_34, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%2 = tt.expand_dims %live_11 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%3 = tt.broadcast %dest_row_28 : tensor<4x1x!tt.ptr<i16>, #blocked> -> tensor<4x256x!tt.ptr<i16>, #blocked>
%4 = tt.addptr %3, %bits_32 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%5 = tt.broadcast %2 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %4, %bits_35, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
%cols = arith.addi %columns, %cst_3 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_36 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_37 = tt.broadcast %bits_36 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_38 = tt.addptr %bits_31, %bits_37 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_39 = tt.load %bits_38, %bits_34, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%6 = tt.addptr %3, %bits_37 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %6, %bits_39, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
%cols_40 = arith.addi %columns, %cst_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_41 = tt.expand_dims %cols_40 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_42 = tt.broadcast %bits_41 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_43 = tt.addptr %bits_31, %bits_42 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_44 = tt.load %bits_43, %bits_34, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%7 = tt.addptr %3, %bits_42 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %7, %bits_44, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
tt.return
}
}"#

private def valuePTX : String := r#".version 8.7
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
.global .align 1 .b8 assertMessage_0[33] = {118, 97, 108, 117, 101, 32, 103, 114, 97, 100, 105, 101, 110, 116, 32, 114, 111, 119, 32, 111, 117, 116, 32, 111, 102, 32, 98, 111, 117, 110, 100, 115};
.visible .entry _value_gradient_packets(
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_0,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_1,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_2,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_3,
.param .u32 _value_gradient_packets_param_4,
.param .u32 _value_gradient_packets_param_5,
.param .u32 _value_gradient_packets_param_6,
.param .u32 _value_gradient_packets_param_7,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_8,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_9
)
.reqntid 128
{
.reg .pred %p<11>;
.reg .b32 %r<35>;
.reg .b64 %rd<22>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
ld.param.b64 %rd5, [_value_gradient_packets_param_1];
mov.u32 %r2, %tid.x;
bfe.u32 %r7, %r2, 5, 2;
ld.param.b32 %r8, [_value_gradient_packets_param_4];
add.s32 %r9, %r8, 3;
ld.param.b32 %r10, [_value_gradient_packets_param_5];
shr.u32 %r3, %r9, 2;
ld.param.b32 %r11, [_value_gradient_packets_param_6];
setp.lt.u32 %p3, %r1, %r3;
ld.param.b32 %r12, [_value_gradient_packets_param_7];
sub.s32 %r13, %r1, %r3;
min.u32 %r14, %r1, %r13;
shl.b32 %r15, %r14, 2;
or.b32 %r4, %r15, %r7;
selp.b32 %r5, %r8, %r10, %p3;
setp.lt.s32 %p2, %r4, %r5;
setp.lt.s32 %p4, %r4, %r11;
selp.b32 %r16, 0, %r8, %p4;
selp.b32 %r17, %r11, %r16, %p3;
add.s32 %r18, %r17, %r4;
mad.wide.s32 %rd4, %r18, 4, %rd5;
mov.u32 %r6, 0x0;
@%p2 ld.global.b32 { %r6 }, [ %rd4 + 0 ];
setp.gt.s32 %p5, %r6, -1;
setp.lt.s32 %p6, %r6, %r12;
and.pred %p1, %p5, %p6;
not.pred %p7, %p1;
and.pred %p8, %p2, %p7;
not.pred %p9, %p8;
@%p9 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd3, [_value_gradient_packets_param_3];
ld.param.b64 %rd2, [_value_gradient_packets_param_2];
ld.param.b64 %rd1, [_value_gradient_packets_param_0];
selp.b64 %rd12, %rd2, %rd3, %p3;
bar.sync 0;
shl.b32 %r31, %r2, 3;
and.b32 %r32, %r31, 248;
mul.lo.s32 %r33, %r6, 768;
mad.wide.s32 %rd13, %r33, 2, %rd1;
mul.lo.s32 %r34, %r4, 768;
mad.wide.s32 %rd14, %r34, 2, %rd12;
and.pred %p10, %p2, %p1;
mul.wide.u32 %rd15, %r32, 2;
add.s64 %rd6, %rd13, %rd15;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
mov.u32 %r21, 0x0;
mov.u32 %r22, 0x0;
@%p10 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd6 + 0 ];
add.s64 %rd7, %rd14, %rd15;
@%p2 st.global.v4.b32 [ %rd7 + 0 ], { %r19, %r20, %r21, %r22 };
add.s64 %rd8, %rd6, 512;
mov.u32 %r23, 0x0;
mov.u32 %r24, 0x0;
mov.u32 %r25, 0x0;
mov.u32 %r26, 0x0;
@%p10 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd8 + 0 ];
add.s64 %rd9, %rd7, 512;
@%p2 st.global.v4.b32 [ %rd9 + 0 ], { %r23, %r24, %r25, %r26 };
add.s64 %rd10, %rd6, 1024;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
mov.u32 %r30, 0x0;
@%p10 ld.global.v4.b32 { %r27, %r28, %r29, %r30 }, [ %rd10 + 0 ];
add.s64 %rd11, %rd7, 1024;
@%p2 st.global.v4.b32 [ %rd11 + 0 ], { %r27, %r28, %r29, %r30 };
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
st.param.b32 [param2], 36;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp0:
$L__func_end0:
}"#

private def ngramIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
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
%cols = arith.addi %columns, %cst_2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%values_24 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%values_25 = tt.broadcast %values_24 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%values_26 = tt.addptr %values_18, %values_25 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%values_27 = tt.load %values_26, %values_21, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%values_28 = arith.extf %values_27 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%converted_29 = tt.elementwise_inline_asm "cvt.rn.bf16.f32 $0, $1;" {constraints = "=h,f", packed_element = 1 : i32, pure = true} %values_28 : tensor<4x256xf32, #blocked> -> tensor<4x256xbf16, #blocked>
%2 = tt.addptr %0, %values_25 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %2, %converted_29, %values_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%cols_30 = arith.addi %columns, %cst_3 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%values_31 = tt.expand_dims %cols_30 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%values_32 = tt.broadcast %values_31 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%values_33 = tt.addptr %values_18, %values_32 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%values_34 = tt.load %values_33, %values_21, %cst_1 : tensor<4x256x!tt.ptr<f16>, #blocked>
%values_35 = arith.extf %values_34 : tensor<4x256xf16, #blocked> to tensor<4x256xf32, #blocked>
%converted_36 = tt.elementwise_inline_asm "cvt.rn.bf16.f32 $0, $1;" {constraints = "=h,f", packed_element = 1 : i32, pure = true} %values_35 : tensor<4x256xf32, #blocked> -> tensor<4x256xbf16, #blocked>
%3 = tt.addptr %0, %values_32 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %3, %converted_36, %values_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.return
}
}"#

private def ngramPTX : String := r#".version 8.7
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
ret;
$L__tmp1:
$L__func_end0:
}"#

private def clearIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_clear_value_want(%source: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %count: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_1 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_2 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%cst_3 = arith.constant dense<0.000000e+00> : tensor<4x256xf16, #blocked>
%slot = tt.get_program_id x : i32
%slot_4 = arith.muli %slot, 4 : i32
%slot_5 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_6 = tt.splat %slot_4 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_7 = arith.addi %slot_6, %slot_5 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_8 = arith.cmpi slt, %slot_7, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row = tt.splat %ids : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_9 = tt.addptr %row, %slot_7 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_10 = tt.load %row_9, %live_8, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%offsets = tt.expand_dims %row_10 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%offsets_11 = arith.muli %offsets, %cst_0 : tensor<4x1xi32, #blocked>
%offsets_12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offsets_13 = tt.expand_dims %offsets_12 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%offsets_14 = tt.broadcast %offsets_11 : tensor<4x1xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_15 = tt.broadcast %offsets_13 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_16 = arith.addi %offsets_14, %offsets_15 : tensor<4x256xi32, #blocked>
%0 = tt.expand_dims %live_8 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%1 = tt.splat %source : !tt.ptr<f16> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %offsets_16 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%3 = tt.broadcast %0 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %2, %cst_3, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_17 = arith.addi %offsets_16, %cst_1 : tensor<4x256xi32, #blocked>
%4 = tt.addptr %1, %offsets_17 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %4, %cst_3, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_18 = arith.addi %offsets_16, %cst_2 : tensor<4x256xi32, #blocked>
%5 = tt.addptr %1, %offsets_18 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %5, %cst_3, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
tt.return
}
}"#

private def clearPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _clear_value_want(
.param .u64 .ptr .global .align 1 _clear_value_want_param_0,
.param .u64 .ptr .global .align 1 _clear_value_want_param_1,
.param .u32 _clear_value_want_param_2,
.param .u64 .ptr .global .align 1 _clear_value_want_param_3,
.param .u64 .ptr .global .align 1 _clear_value_want_param_4
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b32 %r<15>;
.reg .b64 %rd<7>;
$L__func_begin0:
ld.param.b64 %rd5, [_clear_value_want_param_0];
ld.param.b64 %rd6, [_clear_value_want_param_1];
$L__tmp0:
mov.u32 %r3, %ctaid.x;
shl.b32 %r4, %r3, 2;
ld.param.b32 %r5, [_clear_value_want_param_2];
mov.u32 %r6, %tid.x;
bfe.u32 %r7, %r6, 5, 2;
or.b32 %r8, %r7, %r4;
setp.lt.s32 %p1, %r8, %r5;
mad.wide.s32 %rd1, %r8, 4, %rd6;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mul.lo.s32 %r9, %r1, 768;
shl.b32 %r10, %r6, 3;
and.b32 %r11, %r10, 248;
or.b32 %r12, %r9, %r11;
mad.wide.s32 %rd2, %r12, 2, %rd5;
mov.b32 %r2, 0;
@%p1 st.global.v4.b32 [ %rd2 + 0 ], { %r2, %r2, %r2, %r2 };
add.s32 %r13, %r12, 256;
mad.wide.s32 %rd3, %r13, 2, %rd5;
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r2, %r2, %r2, %r2 };
add.s32 %r14, %r12, 512;
mad.wide.s32 %rd4, %r14, 2, %rd5;
@%p1 st.global.v4.b32 [ %rd4 + 0 ], { %r2, %r2, %r2, %r2 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def valueWireIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_value_gradient_packets(%src: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %own: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %wire: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %own_rows: i32, %peer_rows: i32, %lo: i32, %source_rows: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<true> : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<0> : tensor<4x256xi16, #blocked>
%cst_3 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_4 = arith.constant dense<512> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%block = tt.get_program_id x : i32
%lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row = arith.muli %block, 4 : i32
%row_5 = tt.splat %row : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_6 = arith.addi %row_5, %lanes : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = tt.splat %peer_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_7 = arith.cmpi slt, %row_6, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_8 = arith.cmpi slt, %row_6, %selected : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_9 = tt.splat %own_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_10 = arith.select %selected_8, %cst, %selected_9 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%selected_11 = arith.addi %row_6, %selected_10 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%source = tt.splat %ids : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_12 = tt.addptr %source, %selected_11 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%source_13 = tt.load %source_12, %live_7, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded = arith.cmpi sge, %source_13, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_14 = tt.splat %source_rows : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_15 = arith.cmpi slt, %source_13, %bounded_14 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%bounded_16 = arith.andi %bounded, %bounded_15 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%0 = arith.xori %live_7, %cst_0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%1 = arith.ori %bounded_16, %0 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
tt.assert %1, "value gradient row out of bounds" : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%columns = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%source_row = tt.expand_dims %source_13 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%source_row_17 = arith.muli %source_row, %cst_1 : tensor<4x1xi32, #blocked>
%source_row_18 = tt.splat %src : !tt.ptr<i16> -> tensor<4x1x!tt.ptr<i16>, #blocked>
%source_row_19 = tt.addptr %source_row_18, %source_row_17 : tensor<4x1x!tt.ptr<i16>, #blocked>, tensor<4x1xi32, #blocked>
%dest_row = tt.expand_dims %row_6 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%dest_row_20 = arith.muli %dest_row, %cst_1 : tensor<4x1xi32, #blocked>
%dest_row_21 = tt.splat %wire : !tt.ptr<i16> -> tensor<4x1x!tt.ptr<i16>, #blocked>
%dest_row_22 = tt.addptr %dest_row_21, %dest_row_20 : tensor<4x1x!tt.ptr<i16>, #blocked>, tensor<4x1xi32, #blocked>
%bits = arith.andi %live_7, %bounded_16 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%bits_23 = tt.expand_dims %bits {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%bits_24 = tt.expand_dims %columns {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_25 = tt.broadcast %source_row_19 : tensor<4x1x!tt.ptr<i16>, #blocked> -> tensor<4x256x!tt.ptr<i16>, #blocked>
%bits_26 = tt.broadcast %bits_24 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_27 = tt.addptr %bits_25, %bits_26 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_28 = tt.broadcast %bits_23 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%bits_29 = tt.load %bits_27, %bits_28, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%2 = tt.expand_dims %live_7 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%3 = tt.broadcast %dest_row_22 : tensor<4x1x!tt.ptr<i16>, #blocked> -> tensor<4x256x!tt.ptr<i16>, #blocked>
%4 = tt.addptr %3, %bits_26 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%5 = tt.broadcast %2 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %4, %bits_29, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
%cols = arith.addi %columns, %cst_3 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_30 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_31 = tt.broadcast %bits_30 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_32 = tt.addptr %bits_25, %bits_31 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_33 = tt.load %bits_32, %bits_28, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%6 = tt.addptr %3, %bits_31 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %6, %bits_33, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
%cols_34 = arith.addi %columns, %cst_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%bits_35 = tt.expand_dims %cols_34 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%bits_36 = tt.broadcast %bits_35 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%bits_37 = tt.addptr %bits_25, %bits_36 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
%bits_38 = tt.load %bits_37, %bits_28, %cst_2 : tensor<4x256x!tt.ptr<i16>, #blocked>
%7 = tt.addptr %3, %bits_36 : tensor<4x256x!tt.ptr<i16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %7, %bits_38, %5 : tensor<4x256x!tt.ptr<i16>, #blocked>
tt.return
}
}"#

private def valueWirePTX : String := r#".version 8.7
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
.global .align 1 .b8 assertMessage_0[33] = {118, 97, 108, 117, 101, 32, 103, 114, 97, 100, 105, 101, 110, 116, 32, 114, 111, 119, 32, 111, 117, 116, 32, 111, 102, 32, 98, 111, 117, 110, 100, 115};
.visible .entry _value_gradient_packets(
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_0,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_1,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_2,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_3,
.param .u32 _value_gradient_packets_param_4,
.param .u32 _value_gradient_packets_param_5,
.param .u32 _value_gradient_packets_param_6,
.param .u32 _value_gradient_packets_param_7,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_8,
.param .u64 .ptr .global .align 1 _value_gradient_packets_param_9
)
.reqntid 128
{
.reg .pred %p<10>;
.reg .b32 %r<29>;
.reg .b64 %rd<20>;
$L__func_begin0:
ld.param.b32 %r3, [_value_gradient_packets_param_5];
ld.param.b64 %rd4, [_value_gradient_packets_param_1];
$L__tmp0:
mov.u32 %r5, %ctaid.x;
mov.u32 %r1, %tid.x;
bfe.u32 %r6, %r1, 5, 2;
ld.param.b32 %r7, [_value_gradient_packets_param_4];
shl.b32 %r8, %r5, 2;
or.b32 %r2, %r6, %r8;
ld.param.b32 %r9, [_value_gradient_packets_param_6];
setp.lt.s32 %p2, %r2, %r3;
ld.param.b32 %r10, [_value_gradient_packets_param_7];
setp.lt.s32 %p3, %r2, %r9;
selp.b32 %r11, 0, %r7, %p3;
add.s32 %r12, %r11, %r2;
mad.wide.s32 %rd3, %r12, 4, %rd4;
mov.u32 %r4, 0x0;
@%p2 ld.global.b32 { %r4 }, [ %rd3 + 0 ];
setp.gt.s32 %p4, %r4, -1;
setp.lt.s32 %p5, %r4, %r10;
and.pred %p1, %p4, %p5;
not.pred %p6, %p1;
and.pred %p7, %p2, %p6;
not.pred %p8, %p7;
@%p8 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd2, [_value_gradient_packets_param_3];
ld.param.b64 %rd1, [_value_gradient_packets_param_0];
bar.sync 0;
shl.b32 %r25, %r1, 3;
and.b32 %r26, %r25, 248;
mul.lo.s32 %r27, %r4, 768;
mad.wide.s32 %rd11, %r27, 2, %rd1;
mul.lo.s32 %r28, %r2, 768;
mad.wide.s32 %rd12, %r28, 2, %rd2;
and.pred %p9, %p2, %p1;
mul.wide.u32 %rd13, %r26, 2;
add.s64 %rd5, %rd11, %rd13;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
@%p9 ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd5 + 0 ];
add.s64 %rd6, %rd12, %rd13;
@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r13, %r14, %r15, %r16 };
add.s64 %rd7, %rd5, 512;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
@%p9 ld.global.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd7 + 0 ];
add.s64 %rd8, %rd6, 512;
@%p2 st.global.v4.b32 [ %rd8 + 0 ], { %r17, %r18, %r19, %r20 };
add.s64 %rd9, %rd5, 1024;
mov.u32 %r21, 0x0;
mov.u32 %r22, 0x0;
mov.u32 %r23, 0x0;
mov.u32 %r24, 0x0;
@%p9 ld.global.v4.b32 { %r21, %r22, %r23, %r24 }, [ %rd9 + 0 ];
add.s64 %rd10, %rd6, 1024;
@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r21, %r22, %r23, %r24 };
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
st.param.b32 [param2], 36;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp1:
$L__func_end0:
}"#

private def packetPTX (s : String) : List String :=
  (QKVTTIR.normalizePTX s).filter (!·.startsWith ".global .align 1 .b8 assertFile_")

def recognized (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.valueIR == some (valueIR.splitOn "\n") &&
    QKVTTIR.normalizeTTIR a.valueWireIR == some (valueWireIR.splitOn "\n") &&
    QKVTTIR.normalizeTTIR a.ngramIR == some (ngramIR.splitOn "\n") &&
    QKVTTIR.normalizeTTIR a.clearIR == some (clearIR.splitOn "\n") &&
    packetPTX a.valuePTX == valuePTX.splitOn "\n" &&
    packetPTX a.valueWirePTX == valueWirePTX.splitOn "\n" &&
    packetPTX a.ngramPTX == ngramPTX.splitOn "\n" &&
    packetPTX a.clearPTX == clearPTX.splitOn "\n"

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  value : ∀ s, valueDirect s = valueStaged s
  deferredValue : ∀ s, valueDeferred s = valueStaged s
  ngram : ∀ s, ngramDirect s = ngramStaged s
  ngramWire : ∀ want gradient source col,
    RowGradientExchange.wire want gradient source col =
      (RowGradientExchange.wireSlots want source).map (fun slot => wireCast (gradient slot col))
  clear : ∀ s, clearDirect s = cleared s
  distinct : ∀ (s : ValueInput) (a b : Fin s.cut.rows) (x y : Channel),
    (s.want a).val * 768 + x.val = (s.want b).val * 768 + y.val → a = b ∧ x = y
  bounds : ∀ {rows} (r : Fin rows) (col : Channel), r.val * 768 + col.val < rows * 768
  launched : ∀ (State : Type) (r : Runtime State) prepareNgram (s : State),
    launch r direct prepareNgram s = launch r staged prepareNgram s

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then
    some ⟨h, value_packets_equal, value_deferred_exact, ngram_packets_equal, RowGradientExchange.packed_wire_exact,
      clear_exact, fun s _ _ _ _ => clear_destinations_unique s, address_range, fun _ => launch_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.SparsePackets
