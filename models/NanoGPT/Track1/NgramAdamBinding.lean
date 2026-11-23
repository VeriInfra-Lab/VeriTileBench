import models.NanoGPT.Track1.NgramAdam
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Hopper artifacts for the index packer, Adam row kernels
and scatter kernels used at this site. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramAdam

structure Code where
  ir : String
  ptx : String
  deriving BEq

structure Artifacts where
  pack : Code
  prepare : Code
  update : Code
  updateZero : Code
  scatterOwn : Code
  scatterPeer : Code
  scatterPeerTiled : Code
  deriving BEq

private def prepareIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_prepare_entries(%want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %asked: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %entries: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %peer_rows: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %entry_of_row: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %grad: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %own: i32, %count: i32, %lo: i32, %first: i32) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<4x1xi64, #blocked>
%cst_0 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_1 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16, #blocked>
%cst_3 = arith.constant dense<true> : tensor<4xi1, #blocked1>
%cst_4 = arith.constant dense<0> : tensor<4xi32, #blocked1>
%slot = tt.get_program_id x : i32
%slot_5 = arith.muli %slot, 4 : i32
%slot_6 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked1>
%slot_8 = tt.splat %slot_5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_9 = tt.splat %slot_5 : i32 -> tensor<4xi32, #blocked1>
%slot_10 = arith.addi %slot_8, %slot_6 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_11 = arith.addi %slot_9, %slot_7 : tensor<4xi32, #blocked1>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_12 = tt.splat %count : i32 -> tensor<4xi32, #blocked1>
%live_13 = arith.cmpi slt, %slot_10, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_14 = arith.cmpi slt, %slot_11, %live_12 : tensor<4xi32, #blocked1>
%local = tt.splat %own : i32 -> tensor<4xi32, #blocked1>
%local_15 = arith.cmpi slt, %slot_11, %local : tensor<4xi32, #blocked1>
%peer_slot = arith.subi %slot_11, %local : tensor<4xi32, #blocked1>
%source = tt.splat %want : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%source_16 = tt.splat %asked : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%source_17 = arith.select %local_15, %source, %source_16 : tensor<4xi1, #blocked1>, tensor<4x!tt.ptr<i32>, #blocked1>
%index = tt.splat %lo : i32 -> tensor<4xi32, #blocked1>
%index_18 = arith.addi %index, %slot_11 : tensor<4xi32, #blocked1>
%index_19 = arith.select %local_15, %index_18, %peer_slot : tensor<4xi1, #blocked1>, tensor<4xi32, #blocked1>
%row = tt.addptr %source_17, %index_19 : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
%row_20 = tt.load %row, %live_14, %cst_4 : tensor<4x!tt.ptr<i32>, #blocked1>
%row_21 = tt.splat %first : i32 -> tensor<4xi32, #blocked1>
%row_22 = arith.subi %row_20, %row_21 : tensor<4xi32, #blocked1>
%0 = tt.splat %entries : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%1 = tt.addptr %0, %slot_11 : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
tt.store %1, %row_22, %live_14 : tensor<4x!tt.ptr<i32>, #blocked1>
%2 = tt.splat %peer_rows : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%3 = tt.addptr %2, %peer_slot : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
%4 = arith.xori %local_15, %cst_3 : tensor<4xi1, #blocked1>
%5 = arith.andi %live_14, %4 : tensor<4xi1, #blocked1>
tt.store %3, %row_22, %5 : tensor<4x!tt.ptr<i32>, #blocked1>
%6 = tt.splat %entry_of_row : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%7 = tt.addptr %6, %row_22 : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
%8 = tt.atomic_rmw exch, relaxed, gpu, %7, %slot_11, %live_14 : (tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>, tensor<4xi1, #blocked1>) -> tensor<4xi32, #blocked1>
%offset = tt.expand_dims %slot_10 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%offset_23 = arith.extsi %offset : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%offset_24 = arith.muli %offset_23, %cst : tensor<4x1xi64, #blocked>
%offset_25 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offset_26 = tt.expand_dims %offset_25 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%offset_27 = arith.extsi %offset_26 : tensor<1x256xi32, #blocked> to tensor<1x256xi64, #blocked>
%offset_28 = tt.broadcast %offset_24 : tensor<4x1xi64, #blocked> -> tensor<4x256xi64, #blocked>
%offset_29 = tt.broadcast %offset_27 : tensor<1x256xi64, #blocked> -> tensor<4x256xi64, #blocked>
%offset_30 = arith.addi %offset_28, %offset_29 : tensor<4x256xi64, #blocked>
%9 = tt.splat %grad : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%10 = tt.addptr %9, %offset_30 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi64, #blocked>
%11 = tt.expand_dims %live_13 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%12 = tt.broadcast %11 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %10, %cst_2, %12 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%13 = tt.addptr %10, %cst_0 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %13, %cst_2, %12 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%14 = tt.addptr %10, %cst_1 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %14, %cst_2, %12 : tensor<4x256x!tt.ptr<bf16>, #blocked>
tt.return
}
}"#

private def preparePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _prepare_entries(
.param .u64 .ptr .global .align 1 _prepare_entries_param_0,
.param .u64 .ptr .global .align 1 _prepare_entries_param_1,
.param .u64 .ptr .global .align 1 _prepare_entries_param_2,
.param .u64 .ptr .global .align 1 _prepare_entries_param_3,
.param .u64 .ptr .global .align 1 _prepare_entries_param_4,
.param .u64 .ptr .global .align 1 _prepare_entries_param_5,
.param .u32 _prepare_entries_param_6,
.param .u32 _prepare_entries_param_7,
.param .u32 _prepare_entries_param_8,
.param .u32 _prepare_entries_param_9,
.param .u64 .ptr .global .align 1 _prepare_entries_param_10,
.param .u64 .ptr .global .align 1 _prepare_entries_param_11
)
.reqntid 128
{
.reg .pred %p<8>;
.reg .b32 %r<22>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd8, [_prepare_entries_param_0];
ld.param.b64 %rd9, [_prepare_entries_param_1];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
shl.b32 %r7, %r6, 2;
ld.param.b64 %rd10, [_prepare_entries_param_2];
ld.param.b64 %rd11, [_prepare_entries_param_3];
mov.u32 %r8, %tid.x;
ld.param.b64 %rd12, [_prepare_entries_param_4];
bfe.u32 %r9, %r8, 5, 2;
ld.param.b64 %rd13, [_prepare_entries_param_5];
and.b32 %r10, %r8, 3;
ld.param.b32 %r11, [_prepare_entries_param_6];
or.b32 %r12, %r9, %r7;
ld.param.b32 %r13, [_prepare_entries_param_7];
or.b32 %r4, %r7, %r10;
ld.param.b32 %r14, [_prepare_entries_param_8];
setp.lt.s32 %p4, %r12, %r13;
ld.param.b32 %r15, [_prepare_entries_param_9];
setp.lt.s32 %p1, %r4, %r13;
setp.ge.s32 %p5, %r4, %r11;
sub.s32 %r16, %r4, %r11;
selp.b64 %rd14, %rd9, %rd8, %p5;
add.s32 %r17, %r14, %r4;
selp.b32 %r18, %r16, %r17, %p5;
mad.wide.s32 %rd1, %r18, 4, %rd14;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
sub.s32 %r2, %r1, %r15;
mad.wide.s32 %rd2, %r4, 4, %rd10;
and.b32 %r19, %r8, 124;
setp.eq.b32 %p6, %r19, 0;
and.pred %p2, %p6, %p1;
@%p2 st.global.b32 [ %rd2 + 0 ], { %r2 };
mad.wide.s32 %rd3, %r16, 4, %rd11;
and.pred %p7, %p5, %p1;
and.pred %p3, %p6, %p7;
@%p3 st.global.b32 [ %rd3 + 0 ], { %r2 };
mad.wide.s32 %rd4, %r2, 4, %rd12;
mov.u32 %r3, 0x0;
@%p2 atom.global.gpu.relaxed.exch.b32 %r3, [ %rd4 + 0 ], %r4;
shl.b32 %r20, %r8, 3;
and.b32 %r21, %r20, 248;
mad.wide.s32 %rd15, %r12, 1536, %rd13;
mad.wide.u32 %rd5, %r21, 2, %rd15;
mov.b32 %r5, 0;
@%p4 st.global.v4.b32 [ %rd5 + 0 ], { %r5, %r5, %r5, %r5 };
add.s64 %rd6, %rd5, 512;
@%p4 st.global.v4.b32 [ %rd6 + 0 ], { %r5, %r5, %r5, %r5 };
add.s64 %rd7, %rd5, 1024;
@%p4 st.global.v4.b32 [ %rd7 + 0 ], { %r5, %r5, %r5, %r5 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def packIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_pack_entries(%want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %asked: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %entries: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %peer_rows: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %own: i32, %count: i32, %lo: i32, %first: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<128xi32, #blocked>
%cst_0 = arith.constant dense<true> : tensor<128xi1, #blocked>
%slot = tt.get_program_id x : i32
%slot_1 = arith.muli %slot, 128 : i32
%slot_2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
%slot_3 = tt.splat %slot_1 : i32 -> tensor<128xi32, #blocked>
%slot_4 = arith.addi %slot_3, %slot_2 : tensor<128xi32, #blocked>
%live = tt.splat %count : i32 -> tensor<128xi32, #blocked>
%live_5 = arith.cmpi slt, %slot_4, %live : tensor<128xi32, #blocked>
%local = tt.splat %own : i32 -> tensor<128xi32, #blocked>
%local_6 = arith.cmpi slt, %slot_4, %local : tensor<128xi32, #blocked>
%peer_slot = arith.subi %slot_4, %local : tensor<128xi32, #blocked>
%source = tt.splat %want : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
%source_7 = tt.splat %asked : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
%source_8 = arith.select %local_6, %source, %source_7 : tensor<128xi1, #blocked>, tensor<128x!tt.ptr<i32>, #blocked>
%index = tt.splat %lo : i32 -> tensor<128xi32, #blocked>
%index_9 = arith.addi %index, %slot_4 : tensor<128xi32, #blocked>
%index_10 = arith.select %local_6, %index_9, %peer_slot : tensor<128xi1, #blocked>, tensor<128xi32, #blocked>
%row = tt.addptr %source_8, %index_10 : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
%row_11 = tt.load %row, %live_5, %cst : tensor<128x!tt.ptr<i32>, #blocked>
%row_12 = tt.splat %first : i32 -> tensor<128xi32, #blocked>
%row_13 = arith.subi %row_11, %row_12 : tensor<128xi32, #blocked>
%0 = tt.splat %entries : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
%1 = tt.addptr %0, %slot_4 : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
tt.store %1, %row_13, %live_5 : tensor<128x!tt.ptr<i32>, #blocked>
%2 = tt.splat %peer_rows : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
%3 = tt.addptr %2, %peer_slot : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
%4 = arith.xori %local_6, %cst_0 : tensor<128xi1, #blocked>
%5 = arith.andi %live_5, %4 : tensor<128xi1, #blocked>
tt.store %3, %row_13, %5 : tensor<128x!tt.ptr<i32>, #blocked>
tt.return
}
}"#

private def packPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _pack_entries(
.param .u64 .ptr .global .align 1 _pack_entries_param_0,
.param .u64 .ptr .global .align 1 _pack_entries_param_1,
.param .u64 .ptr .global .align 1 _pack_entries_param_2,
.param .u64 .ptr .global .align 1 _pack_entries_param_3,
.param .u32 _pack_entries_param_4,
.param .u32 _pack_entries_param_5,
.param .u32 _pack_entries_param_6,
.param .u32 _pack_entries_param_7,
.param .u64 .ptr .global .align 1 _pack_entries_param_8,
.param .u64 .ptr .global .align 1 _pack_entries_param_9
)
.reqntid 128
{
.reg .pred %p<4>;
.reg .b32 %r<15>;
.reg .b64 %rd<9>;
$L__func_begin0:
ld.param.b64 %rd4, [_pack_entries_param_0];
ld.param.b64 %rd5, [_pack_entries_param_1];
$L__tmp0:
mov.u32 %r3, %ctaid.x;
shl.b32 %r4, %r3, 7;
ld.param.b64 %rd6, [_pack_entries_param_2];
ld.param.b64 %rd7, [_pack_entries_param_3];
mov.u32 %r5, %tid.x;
and.b32 %r6, %r5, 127;
ld.param.b32 %r7, [_pack_entries_param_4];
or.b32 %r8, %r4, %r6;
ld.param.b32 %r9, [_pack_entries_param_5];
setp.lt.s32 %p1, %r8, %r9;
ld.param.b32 %r10, [_pack_entries_param_6];
setp.ge.s32 %p3, %r8, %r7;
ld.param.b32 %r11, [_pack_entries_param_7];
sub.s32 %r12, %r8, %r7;
selp.b64 %rd8, %rd5, %rd4, %p3;
add.s32 %r13, %r10, %r8;
selp.b32 %r14, %r12, %r13, %p3;
mad.wide.s32 %rd1, %r14, 4, %rd8;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
sub.s32 %r2, %r1, %r11;
mad.wide.s32 %rd2, %r8, 4, %rd6;
@%p1 st.global.b32 [ %rd2 + 0 ], { %r2 };
mad.wide.s32 %rd3, %r12, 4, %rd7;
and.pred %p2, %p3, %p1;
@%p2 st.global.b32 [ %rd3 + 0 ], { %r2 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def updateIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_adam_rows_kernel(%P: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %LAST_EVENT: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %BETA2_HISTORY: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %G: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROWS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %ENTRY_OF_ROW: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %event: i32, %beta2: f32, %eps: f32, %step_size: f32, %decay: f32, %grad_mul: f32, %sq_mul: f32) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xbf16, #blocked>
%c768_i64 = arith.constant 768 : i64
%cst_1 = arith.constant 7.680000e+02 : f32
%cst_2 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%pid = tt.get_program_id x : i32
%offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%mask = arith.cmpi slt, %offs, %cst : tensor<1024xi32, #blocked>
%row = tt.addptr %ROWS, %pid : !tt.ptr<i32>, i32
%row_3 = tt.load %row : !tt.ptr<i32>
%row64 = arith.extsi %row_3 : i32 to i64
%last = tt.addptr %LAST_EVENT, %row_3 : !tt.ptr<i32>, i32
%last_4 = tt.load %last : !tt.ptr<i32>
%commit = tt.addptr %ENTRY_OF_ROW, %row_3 : !tt.ptr<i32>, i32
%commit_5 = tt.load %commit : !tt.ptr<i32>
%commit_6 = arith.cmpi eq, %commit_5, %pid : i32
%missed = arith.subi %event, 1 : i32
%missed_7 = arith.subi %missed, %last_4 : i32
%missed_8 = arith.select %commit_6, %missed_7, 0 : i32
%v = tt.addptr %V, %row64 : !tt.ptr<f32>, i64
%v_9 = tt.load %v : !tt.ptr<f32>
%v_10 = arith.addi %last_4, 1 : i32
%v_11 = scf.for %i = 0 to %missed_8 step 1 iter_args(%v_12 = %v_9) -> (f32) : i32 {
%v_13 = arith.addi %v_10, %i : i32
%v_14 = tt.addptr %BETA2_HISTORY, %v_13 : !tt.ptr<f32>, i32
%v_15 = tt.load %v_14 : !tt.ptr<f32>
%v_16 = arith.mulf %v_12, %v_15 : f32
scf.yield %v_16 : f32
}
scf.if %commit_6 {
%g = arith.extsi %pid : i32 to i64
%g_12 = arith.muli %g, %c768_i64 : i64
%g_13 = arith.extsi %offs : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
%g_14 = tt.splat %g_12 : i64 -> tensor<1024xi64, #blocked>
%g_15 = arith.addi %g_14, %g_13 : tensor<1024xi64, #blocked>
%g_16 = tt.splat %G : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%g_17 = tt.addptr %g_16, %g_15 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi64, #blocked>
%g_18 = tt.load %g_17, %mask, %cst_0 : tensor<1024x!tt.ptr<bf16>, #blocked>
%g_19 = arith.extf %g_18 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%p = arith.muli %row64, %c768_i64 : i64
%p_20 = tt.splat %p : i64 -> tensor<1024xi64, #blocked>
%p_21 = arith.addi %p_20, %g_13 : tensor<1024xi64, #blocked>
%p_22 = tt.splat %P : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%p_23 = tt.addptr %p_22, %p_21 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi64, #blocked>
%p_24 = tt.load %p_23, %mask, %cst_0 : tensor<1024x!tt.ptr<bf16>, #blocked>
%p_25 = arith.extf %p_24 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%m = tt.splat %grad_mul : f32 -> tensor<1024xf32, #blocked>
%m_26 = arith.mulf %g_19, %m : tensor<1024xf32, #blocked>
%v_27 = arith.mulf %v_11, %beta2 : f32
%v_28 = arith.mulf %g_19, %g_19 : tensor<1024xf32, #blocked>
%v_29 = "tt.reduce"(%v_28) <{axis = 0 : i32}> ({
^bb0(%v_45: f32, %v_46: f32):
%v_47 = arith.addf %v_45, %v_46 : f32
tt.reduce.return %v_47 : f32
}) : (tensor<1024xf32, #blocked>) -> f32
%v_30 = arith.divf %v_29, %cst_1 : f32
%v_31 = arith.mulf %v_30, %sq_mul : f32
%v_32 = arith.addf %v_27, %v_31 : f32
%u = math.sqrt %v_32 : f32
%u_33 = arith.addf %u, %eps : f32
%u_34 = tt.splat %u_33 : f32 -> tensor<1024xf32, #blocked>
%u_35 = arith.divf %m_26, %u_34 : tensor<1024xf32, #blocked>
%u_36 = tt.splat %step_size : f32 -> tensor<1024xf32, #blocked>
%u_37 = arith.mulf %u_35, %u_36 : tensor<1024xf32, #blocked>
%u_38 = arith.mulf %u_37, %p_25 : tensor<1024xf32, #blocked>
%u_39 = arith.cmpf ogt, %u_38, %cst_2 : tensor<1024xf32, #blocked>
%u_40 = tt.splat %decay : f32 -> tensor<1024xf32, #blocked>
%u_41 = arith.mulf %p_25, %u_40 : tensor<1024xf32, #blocked>
%u_42 = arith.select %u_39, %u_41, %cst_2 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%u_43 = arith.addf %u_37, %u_42 : tensor<1024xf32, #blocked>
%p_44 = arith.subf %p_25, %u_43 : tensor<1024xf32, #blocked>
tt.store %v, %v_32 : !tt.ptr<f32>
%0 = arith.truncf %p_44 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %p_23, %0, %mask : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.store %last, %event : !tt.ptr<i32>
}
tt.return
}
}"#

private def updatePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.global .align 1 .b8 _$_str_$_2[17] = {95, 95, 67, 85, 68, 65, 95, 80, 82, 69, 67, 95, 83, 81, 82, 84};
.visible .entry _adam_rows_kernel(
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_0,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_1,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_2,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_3,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_4,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_5,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_6,
.param .u32 _adam_rows_kernel_param_7,
.param .f32 _adam_rows_kernel_param_8,
.param .f32 _adam_rows_kernel_param_9,
.param .f32 _adam_rows_kernel_param_10,
.param .f32 _adam_rows_kernel_param_11,
.param .f32 _adam_rows_kernel_param_12,
.param .f32 _adam_rows_kernel_param_13,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_14,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_15
)
.reqntid 128
{
.reg .pred %p<21>;
.reg .b16 %rs<17>;
.reg .b32 %r<153>;
.reg .b64 %rd<22>;
$L__func_begin0:
ld.param.b32 %r40, [_adam_rows_kernel_param_7];
ld.param.b64 %rd8, [_adam_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r16, %ctaid.x;
ld.param.b64 %rd9, [_adam_rows_kernel_param_2];
ld.param.b64 %rd10, [_adam_rows_kernel_param_5];
ld.param.b64 %rd11, [_adam_rows_kernel_param_6];
cvt.u64.u32 %rd1, %r16;
mad.wide.u32 %rd6, %r16, 4, %rd10;
mov.u32 %r13, 0x0;
ld.global.b32 { %r13 }, [ %rd6 + 0 ];
mul.wide.s32 %rd12, %r13, 4;
add.s64 %rd17, %rd9, %rd12;
mov.u32 %r14, 0x0;
ld.global.b32 { %r14 }, [ %rd17 + 0 ];
add.s64 %rd7, %rd11, %rd12;
mov.u32 %r15, 0x0;
ld.global.b32 { %r15 }, [ %rd7 + 0 ];
setp.ne.b32 %p1, %r15, %r16;
not.b32 %r18, %r14;
add.s32 %r6, %r40, %r18;
add.s64 %rd16, %rd8, %rd12;
mov.u32 %r152, 0x0;
ld.global.b32 { %r152 }, [ %rd16 + 0 ];
setp.lt.s32 %p2, %r6, 1;
or.pred %p3, %p1, %p2;
@%p3 bra $L__BB0_3;
ld.param.b64 %rd4, [_adam_rows_kernel_param_3];
cvt.u64.u32 %rd21, %r6;
add.s32 %r151, %r14, 1;
$L__BB0_2:
mad.wide.s32 %rd13, %r151, 4, %rd4;
mov.u32 %r19, 0x0;
ld.global.b32 { %r19 }, [ %rd13 + 0 ];
mul.f32 %r152, %r152, %r19;
add.s64 %rd21, %rd21, -1;
add.s32 %r151, %r151, 1;
setp.ne.b64 %p4, %rd21, 0;
@%p4 bra $L__BB0_2;
$L__BB0_3:
cvt.u32.u64 %r20, %rd1;
setp.ne.b32 %p5, %r15, %r20;
@%p5 bra $L__BB0_5;
mov.u32 %r1, %tid.x;
shl.b32 %r17, %r1, 3;
ld.param.b32 %r12, [_adam_rows_kernel_param_13];
ld.param.b32 %r11, [_adam_rows_kernel_param_12];
ld.param.b32 %r10, [_adam_rows_kernel_param_11];
ld.param.b32 %r9, [_adam_rows_kernel_param_10];
ld.param.b32 %r8, [_adam_rows_kernel_param_9];
ld.param.b32 %r7, [_adam_rows_kernel_param_8];
ld.param.b64 %rd5, [_adam_rows_kernel_param_4];
ld.param.b64 %rd3, [_adam_rows_kernel_param_0];
and.b32 %r2, %r1, 127;
and.b32 %r3, %r1, 31;
bfe.u32 %r4, %r1, 5, 2;
and.b32 %r5, %r17, 1016;
cvt.s64.s32 %rd2, %r13;
setp.lt.u32 %p6, %r5, 768;
mad.lo.s64 %rd18, %rd1, 1536, %rd5;
mul.wide.u32 %rd19, %r5, 2;
add.s64 %rd14, %rd18, %rd19;
mov.b32 %r25, 0;
mov.u32 %r21, %r25;
mov.u32 %r22, %r25;
mov.u32 %r23, %r25;
mov.u32 %r24, %r25;
@%p6 ld.global.v4.b32 { %r21, %r22, %r23, %r24 }, [ %rd14 + 0 ];
mov.b32 {%rs1, %rs2}, %r21;
mov.b32 {%rs3, %rs4}, %r22;
mov.b32 {%rs5, %rs6}, %r23;
mov.b32 {%rs7, %rs8}, %r24;
cvt.f32.bf16 %r41, %rs1;
cvt.f32.bf16 %r42, %rs2;
cvt.f32.bf16 %r43, %rs3;
cvt.f32.bf16 %r44, %rs4;
cvt.f32.bf16 %r45, %rs5;
cvt.f32.bf16 %r46, %rs6;
cvt.f32.bf16 %r47, %rs7;
cvt.f32.bf16 %r48, %rs8;
mad.lo.s64 %rd20, %rd2, 1536, %rd3;
add.s64 %rd15, %rd20, %rd19;
mov.u32 %r26, %r25;
mov.u32 %r27, %r25;
mov.u32 %r28, %r25;
mov.u32 %r29, %r25;
@%p6 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd15 + 0 ];
mov.b32 {%rs9, %rs10}, %r26;
cvt.f32.bf16 %r49, %rs9;
cvt.f32.bf16 %r50, %rs10;
mov.b32 {%rs11, %rs12}, %r27;
cvt.f32.bf16 %r51, %rs11;
cvt.f32.bf16 %r52, %rs12;
mov.b32 {%rs13, %rs14}, %r28;
cvt.f32.bf16 %r53, %rs13;
cvt.f32.bf16 %r54, %rs14;
mov.b32 {%rs15, %rs16}, %r29;
cvt.f32.bf16 %r55, %rs15;
cvt.f32.bf16 %r56, %rs16;
mul.f32 %r57, %r11, %r41;
mul.f32 %r58, %r11, %r42;
mul.f32 %r59, %r11, %r43;
mul.f32 %r60, %r11, %r44;
mul.f32 %r61, %r11, %r45;
mul.f32 %r62, %r11, %r46;
mul.f32 %r63, %r11, %r47;
mul.f32 %r64, %r11, %r48;
mul.f32 %r65, %r42, %r42;
$L__tmp1:
fma.rn.f32 %r66, %r41, %r41, %r65;
fma.rn.f32 %r67, %r43, %r43, %r66;
fma.rn.f32 %r68, %r44, %r44, %r67;
fma.rn.f32 %r69, %r45, %r45, %r68;
fma.rn.f32 %r70, %r46, %r46, %r69;
fma.rn.f32 %r71, %r47, %r47, %r70;
fma.rn.f32 %r72, %r48, %r48, %r71;
$L__tmp2:
shfl.sync.bfly.b32 %r73, %r72, 16, 31, -1;
$L__tmp3:
add.f32 %r74, %r72, %r73;
$L__tmp4:
shfl.sync.bfly.b32 %r75, %r74, 8, 31, -1;
$L__tmp5:
add.f32 %r76, %r74, %r75;
$L__tmp6:
shfl.sync.bfly.b32 %r77, %r76, 4, 31, -1;
$L__tmp7:
add.f32 %r78, %r76, %r77;
$L__tmp8:
shfl.sync.bfly.b32 %r79, %r78, 2, 31, -1;
$L__tmp9:
add.f32 %r80, %r78, %r79;
$L__tmp10:
shfl.sync.bfly.b32 %r81, %r80, 1, 31, -1;
$L__tmp11:
add.f32 %r31, %r80, %r81;
$L__tmp12:
setp.eq.b32 %p7, %r3, 0;
shl.b32 %r82, %r4, 2;
mov.b32 %r83, global_smem;
add.s32 %r30, %r83, %r82;
@%p7 st.shared.b32 [ %r30 + 0 ], %r31;
bar.sync 0;
setp.lt.u32 %p8, %r2, 4;
shl.b32 %r84, %r2, 2;
add.s32 %r33, %r83, %r84;
@%p8 ld.shared.b32 %r32, [ %r33 + 0 ];
shfl.sync.bfly.b32 %r85, %r32, 2, 31, -1;
$L__tmp13:
add.f32 %r86, %r32, %r85;
$L__tmp14:
shfl.sync.bfly.b32 %r87, %r86, 1, 31, -1;
$L__tmp15:
add.f32 %r34, %r86, %r87;
$L__tmp16:
and.b32 %r88, %r1, 3;
setp.eq.b32 %p11, %r88, 0;
and.pred %p9, %p8, %p11;
@%p9 st.shared.b32 [ %r33 + 0 ], %r34;
bar.sync 0;
ld.shared.b32 %r89, [global_smem];
mov.b32 %r90, 0f44400000;
$L__tmp17:
div.full.f32 %r91, %r89, %r90;
mul.f32 %r92, %r12, %r91;
fma.rn.f32 %r35, %r7, %r152, %r92;
sqrt.approx.ftz.f32 %r93, %r35;
add.f32 %r94, %r8, %r93;
div.full.f32 %r95, %r57, %r94;
div.full.f32 %r96, %r58, %r94;
div.full.f32 %r97, %r59, %r94;
div.full.f32 %r98, %r60, %r94;
div.full.f32 %r99, %r61, %r94;
div.full.f32 %r100, %r62, %r94;
div.full.f32 %r101, %r63, %r94;
div.full.f32 %r102, %r64, %r94;
mul.f32 %r103, %r10, %r50;
mul.f32 %r104, %r10, %r49;
mul.f32 %r105, %r10, %r52;
mul.f32 %r106, %r10, %r51;
mul.f32 %r107, %r10, %r54;
mul.f32 %r108, %r10, %r53;
mul.f32 %r109, %r10, %r56;
mul.f32 %r110, %r10, %r55;
setp.lt.u32 %p12, %r2, 32;
and.pred %p10, %p7, %p12;
@%p10 st.global.b32 [ %rd16 + 0 ], { %r35 };
mul.f32 %r111, %r9, %r96;
mul.f32 %r112, %r9, %r95;
mul.f32 %r113, %r112, %r49;
mul.f32 %r114, %r111, %r50;
setp.gt.f32 %p13, %r114, 0f00000000;
setp.gt.f32 %p14, %r113, 0f00000000;
selp.f32 %r115, %r104, 0f00000000, %p14;
selp.f32 %r116, %r103, 0f00000000, %p13;
fma.rn.f32 %r117, %r9, %r96, %r116;
fma.rn.f32 %r118, %r9, %r95, %r115;
sub.f32 %r119, %r49, %r118;
sub.f32 %r120, %r50, %r117;
cvt.rn.bf16x2.f32 %r36, %r120, %r119;
mul.f32 %r121, %r9, %r98;
mul.f32 %r122, %r9, %r97;
mul.f32 %r123, %r122, %r51;
mul.f32 %r124, %r121, %r52;
setp.gt.f32 %p15, %r124, 0f00000000;
setp.gt.f32 %p16, %r123, 0f00000000;
selp.f32 %r125, %r106, 0f00000000, %p16;
selp.f32 %r126, %r105, 0f00000000, %p15;
fma.rn.f32 %r127, %r9, %r98, %r126;
fma.rn.f32 %r128, %r9, %r97, %r125;
sub.f32 %r129, %r51, %r128;
sub.f32 %r130, %r52, %r127;
cvt.rn.bf16x2.f32 %r37, %r130, %r129;
mul.f32 %r131, %r9, %r100;
mul.f32 %r132, %r9, %r99;
mul.f32 %r133, %r132, %r53;
mul.f32 %r134, %r131, %r54;
setp.gt.f32 %p17, %r134, 0f00000000;
setp.gt.f32 %p18, %r133, 0f00000000;
selp.f32 %r135, %r108, 0f00000000, %p18;
selp.f32 %r136, %r107, 0f00000000, %p17;
fma.rn.f32 %r137, %r9, %r100, %r136;
fma.rn.f32 %r138, %r9, %r99, %r135;
sub.f32 %r139, %r53, %r138;
sub.f32 %r140, %r54, %r137;
cvt.rn.bf16x2.f32 %r38, %r140, %r139;
mul.f32 %r141, %r9, %r102;
mul.f32 %r142, %r9, %r101;
mul.f32 %r143, %r142, %r55;
mul.f32 %r144, %r141, %r56;
setp.gt.f32 %p19, %r144, 0f00000000;
setp.gt.f32 %p20, %r143, 0f00000000;
selp.f32 %r145, %r110, 0f00000000, %p20;
selp.f32 %r146, %r109, 0f00000000, %p19;
fma.rn.f32 %r147, %r9, %r102, %r146;
fma.rn.f32 %r148, %r9, %r101, %r145;
sub.f32 %r149, %r55, %r148;
sub.f32 %r150, %r56, %r147;
cvt.rn.bf16x2.f32 %r39, %r150, %r149;
@%p6 st.global.v4.b32 [ %rd15 + 0 ], { %r36, %r37, %r38, %r39 };
@%p10 st.global.b32 [ %rd17 + 0 ], { %r40 };
$L__BB0_5:
ret;
$L__tmp18:
$L__func_end0:
}"#

private def scatterOwnIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_add_rows_kernel(%SRC: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %IDX: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROW_MAP: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c768_i64 = arith.constant 768 : i64
%c256_i64 = arith.constant 256 : i64
%c512_i64 = arith.constant 512 : i64
%cst = arith.constant dense<true> : tensor<256xi1, #blocked>
%pid = tt.get_program_id x : i32
%o64 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%o64_0 = arith.extsi %o64 : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%drow = tt.addptr %IDX, %pid : !tt.ptr<i32>, i32
%drow_1 = tt.load %drow : !tt.ptr<i32>
%drow_2 = tt.addptr %ROW_MAP, %drow_1 : !tt.ptr<i32>, i32
%drow_3 = tt.load %drow_2 : !tt.ptr<i32>
%sbase = arith.extsi %pid : i32 to i64
%sbase_4 = arith.muli %sbase, %c768_i64 : i64
%dbase = arith.extsi %drow_3 : i32 to i64
%dbase_5 = arith.muli %dbase, %c768_i64 : i64
%val = tt.splat %sbase_4 : i64 -> tensor<256xi64, #blocked>
%val_6 = arith.addi %val, %o64_0 : tensor<256xi64, #blocked>
%val_7 = tt.splat %SRC : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_8 = tt.addptr %val_7, %val_6 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_9 = tt.load %val_8 : tensor<256x!tt.ptr<f16>, #blocked>
%val_10 = arith.extf %val_9 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
%0 = tt.splat %dbase_5 : i64 -> tensor<256xi64, #blocked>
%1 = arith.addi %0, %o64_0 : tensor<256xi64, #blocked>
%2 = tt.splat %DST : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%4 = arith.truncf %val_10 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %3, %4, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
%val_11 = arith.addi %sbase_4, %c256_i64 : i64
%val_12 = tt.splat %val_11 : i64 -> tensor<256xi64, #blocked>
%val_13 = arith.addi %val_12, %o64_0 : tensor<256xi64, #blocked>
%val_14 = tt.addptr %val_7, %val_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_15 = tt.load %val_14 : tensor<256x!tt.ptr<f16>, #blocked>
%val_16 = arith.extf %val_15 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
%6 = arith.addi %dbase_5, %c256_i64 : i64
%7 = tt.splat %6 : i64 -> tensor<256xi64, #blocked>
%8 = arith.addi %7, %o64_0 : tensor<256xi64, #blocked>
%9 = tt.addptr %2, %8 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%10 = arith.truncf %val_16 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
%11 = tt.atomic_rmw fadd, relaxed, gpu, %9, %10, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
%val_17 = arith.addi %sbase_4, %c512_i64 : i64
%val_18 = tt.splat %val_17 : i64 -> tensor<256xi64, #blocked>
%val_19 = arith.addi %val_18, %o64_0 : tensor<256xi64, #blocked>
%val_20 = tt.addptr %val_7, %val_19 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_21 = tt.load %val_20 : tensor<256x!tt.ptr<f16>, #blocked>
%val_22 = arith.extf %val_21 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
%12 = arith.addi %dbase_5, %c512_i64 : i64
%13 = tt.splat %12 : i64 -> tensor<256xi64, #blocked>
%14 = arith.addi %13, %o64_0 : tensor<256xi64, #blocked>
%15 = tt.addptr %2, %14 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%16 = arith.truncf %val_22 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
%17 = tt.atomic_rmw fadd, relaxed, gpu, %15, %16, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
tt.return
}
}"#

private def scatterOwnPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_add_rows_kernel(
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_0,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_1,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_2,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_3,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_4,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_5
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<19>;
.reg .b32 %r<16>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd9, [_scatter_add_rows_kernel_param_0];
ld.param.b64 %rd10, [_scatter_add_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
ld.param.b64 %rd11, [_scatter_add_rows_kernel_param_2];
mov.u32 %r7, %tid.x;
shl.b32 %r8, %r7, 1;
ld.param.b64 %rd12, [_scatter_add_rows_kernel_param_3];
and.b32 %r9, %r8, 254;
mad.wide.u32 %rd1, %r6, 4, %rd10;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mad.wide.s32 %rd2, %r1, 4, %rd12;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.u32 %rd13, %r6, 1536, %rd9;
mul.wide.u32 %rd14, %r9, 2;
add.s64 %rd3, %rd13, %rd14;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd3 + 0 ];
mov.b32 {%rs13, %rs14}, %r3;
cvt.f32.f16 %r10, %rs13;
cvt.f32.f16 %r11, %rs14;
mad.wide.s32 %rd15, %r2, 1536, %rd11;
add.s64 %rd4, %rd15, %rd14;
cvt.rn.bf16.f32 %rs3, %r10;
cvt.rn.bf16.f32 %rs4, %r11;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs1, %rs2 }, [ %rd4 + 0 ], { %rs3, %rs4 };
add.s64 %rd5, %rd3, 512;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd5 + 0 ];
mov.b32 {%rs15, %rs16}, %r4;
cvt.f32.f16 %r12, %rs15;
cvt.f32.f16 %r13, %rs16;
add.s64 %rd6, %rd4, 512;
cvt.rn.bf16.f32 %rs7, %r12;
cvt.rn.bf16.f32 %rs8, %r13;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs5, %rs6 }, [ %rd6 + 0 ], { %rs7, %rs8 };
add.s64 %rd7, %rd3, 1024;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd7 + 0 ];
mov.b32 {%rs17, %rs18}, %r5;
cvt.f32.f16 %r14, %rs17;
cvt.f32.f16 %r15, %rs18;
add.s64 %rd8, %rd4, 1024;
cvt.rn.bf16.f32 %rs11, %r14;
cvt.rn.bf16.f32 %rs12, %r15;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs9, %rs10 }, [ %rd8 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def scatterPeerIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_add_rows_kernel(%SRC: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %IDX: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROW_MAP: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c768_i64 = arith.constant 768 : i64
%c256_i64 = arith.constant 256 : i64
%c512_i64 = arith.constant 512 : i64
%cst = arith.constant dense<true> : tensor<256xi1, #blocked>
%pid = tt.get_program_id x : i32
%o64 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%o64_0 = arith.extsi %o64 : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%drow = tt.addptr %IDX, %pid : !tt.ptr<i32>, i32
%drow_1 = tt.load %drow : !tt.ptr<i32>
%drow_2 = tt.addptr %ROW_MAP, %drow_1 : !tt.ptr<i32>, i32
%drow_3 = tt.load %drow_2 : !tt.ptr<i32>
%sbase = arith.extsi %pid : i32 to i64
%sbase_4 = arith.muli %sbase, %c768_i64 : i64
%dbase = arith.extsi %drow_3 : i32 to i64
%dbase_5 = arith.muli %dbase, %c768_i64 : i64
%val = tt.splat %sbase_4 : i64 -> tensor<256xi64, #blocked>
%val_6 = arith.addi %val, %o64_0 : tensor<256xi64, #blocked>
%val_7 = tt.splat %SRC : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%val_8 = tt.addptr %val_7, %val_6 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_9 = tt.load %val_8 : tensor<256x!tt.ptr<bf16>, #blocked>
%0 = tt.splat %dbase_5 : i64 -> tensor<256xi64, #blocked>
%1 = arith.addi %0, %o64_0 : tensor<256xi64, #blocked>
%2 = tt.splat %DST : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %3, %val_9, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
%val_10 = arith.addi %sbase_4, %c256_i64 : i64
%val_11 = tt.splat %val_10 : i64 -> tensor<256xi64, #blocked>
%val_12 = arith.addi %val_11, %o64_0 : tensor<256xi64, #blocked>
%val_13 = tt.addptr %val_7, %val_12 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_14 = tt.load %val_13 : tensor<256x!tt.ptr<bf16>, #blocked>
%5 = arith.addi %dbase_5, %c256_i64 : i64
%6 = tt.splat %5 : i64 -> tensor<256xi64, #blocked>
%7 = arith.addi %6, %o64_0 : tensor<256xi64, #blocked>
%8 = tt.addptr %2, %7 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%9 = tt.atomic_rmw fadd, relaxed, gpu, %8, %val_14, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
%val_15 = arith.addi %sbase_4, %c512_i64 : i64
%val_16 = tt.splat %val_15 : i64 -> tensor<256xi64, #blocked>
%val_17 = arith.addi %val_16, %o64_0 : tensor<256xi64, #blocked>
%val_18 = tt.addptr %val_7, %val_17 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_19 = tt.load %val_18 : tensor<256x!tt.ptr<bf16>, #blocked>
%10 = arith.addi %dbase_5, %c512_i64 : i64
%11 = tt.splat %10 : i64 -> tensor<256xi64, #blocked>
%12 = arith.addi %11, %o64_0 : tensor<256xi64, #blocked>
%13 = tt.addptr %2, %12 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%14 = tt.atomic_rmw fadd, relaxed, gpu, %13, %val_19, %cst : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
tt.return
}
}"#

private def scatterPeerPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_add_rows_kernel(
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_0,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_1,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_2,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_3,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_4,
.param .u64 .ptr .global .align 1 _scatter_add_rows_kernel_param_5
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<13>;
.reg .b32 %r<10>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd9, [_scatter_add_rows_kernel_param_0];
ld.param.b64 %rd10, [_scatter_add_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
ld.param.b64 %rd11, [_scatter_add_rows_kernel_param_2];
mov.u32 %r7, %tid.x;
shl.b32 %r8, %r7, 1;
ld.param.b64 %rd12, [_scatter_add_rows_kernel_param_3];
and.b32 %r9, %r8, 254;
mad.wide.u32 %rd1, %r6, 4, %rd10;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mad.wide.s32 %rd2, %r1, 4, %rd12;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.u32 %rd13, %r6, 1536, %rd9;
mul.wide.u32 %rd14, %r9, 2;
add.s64 %rd3, %rd13, %rd14;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd3 + 0 ];
mov.b32 {%rs3, %rs4}, %r3;
mad.wide.s32 %rd15, %r2, 1536, %rd11;
add.s64 %rd4, %rd15, %rd14;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs1, %rs2 }, [ %rd4 + 0 ], { %rs3, %rs4 };
add.s64 %rd5, %rd3, 512;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd5 + 0 ];
mov.b32 {%rs7, %rs8}, %r4;
add.s64 %rd6, %rd4, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs5, %rs6 }, [ %rd6 + 0 ], { %rs7, %rs8 };
add.s64 %rd7, %rd3, 1024;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd7 + 0 ];
mov.b32 {%rs11, %rs12}, %r5;
add.s64 %rd8, %rd4, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.bf16 { %rs9, %rs10 }, [ %rd8 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def updateZeroIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_adam_rows_kernel(%P: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %LAST_EVENT: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %BETA2_HISTORY: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %G: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROWS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %ENTRY_OF_ROW: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %event: i32, %beta2: f32, %eps: f32, %step_size: f32, %decay: f32, %grad_mul: f32, %sq_mul: f32) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xbf16, #blocked>
%c768_i64 = arith.constant 768 : i64
%cst_1 = arith.constant 7.680000e+02 : f32
%cst_2 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%pid = tt.get_program_id x : i32
%offs = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%mask = arith.cmpi slt, %offs, %cst : tensor<1024xi32, #blocked>
%row = tt.addptr %ROWS, %pid : !tt.ptr<i32>, i32
%row_3 = tt.load %row : !tt.ptr<i32>
%row64 = arith.extsi %row_3 : i32 to i64
%commit = tt.addptr %ENTRY_OF_ROW, %row_3 : !tt.ptr<i32>, i32
%commit_4 = tt.load %commit : !tt.ptr<i32>
%commit_5 = arith.cmpi eq, %commit_4, %pid : i32
scf.if %commit_5 {
%last = tt.addptr %LAST_EVENT, %row_3 : !tt.ptr<i32>, i32
%last_6 = tt.load %last : !tt.ptr<i32>
%v = tt.addptr %V, %row64 : !tt.ptr<f32>, i64
%v_7 = tt.load %v : !tt.ptr<f32>
%end = arith.subi %event, 1 : i32
%last_8 = tt.bitcast %v_7 : f32 -> i32
%last_9 = arith.cmpi eq, %last_8, 0 : i32
%last_10 = arith.select %last_9, %end, %last_6 : i32
%0 = arith.subi %end, %last_10 : i32
%v_11 = arith.addi %last_10, 1 : i32
%v_12 = scf.for %i = 0 to %0 step 1 iter_args(%v_46 = %v_7) -> (f32) : i32 {
%v_47 = arith.addi %v_11, %i : i32
%v_48 = tt.addptr %BETA2_HISTORY, %v_47 : !tt.ptr<f32>, i32
%v_49 = tt.load %v_48 : !tt.ptr<f32>
%v_50 = arith.mulf %v_46, %v_49 : f32
scf.yield %v_50 : f32
}
%g = arith.extsi %pid : i32 to i64
%g_13 = arith.muli %g, %c768_i64 : i64
%g_14 = arith.extsi %offs : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
%g_15 = tt.splat %g_13 : i64 -> tensor<1024xi64, #blocked>
%g_16 = arith.addi %g_15, %g_14 : tensor<1024xi64, #blocked>
%g_17 = tt.splat %G : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%g_18 = tt.addptr %g_17, %g_16 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi64, #blocked>
%g_19 = tt.load %g_18, %mask, %cst_0 : tensor<1024x!tt.ptr<bf16>, #blocked>
%g_20 = arith.extf %g_19 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%p = arith.muli %row64, %c768_i64 : i64
%p_21 = tt.splat %p : i64 -> tensor<1024xi64, #blocked>
%p_22 = arith.addi %p_21, %g_14 : tensor<1024xi64, #blocked>
%p_23 = tt.splat %P : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%p_24 = tt.addptr %p_23, %p_22 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi64, #blocked>
%p_25 = tt.load %p_24, %mask, %cst_0 : tensor<1024x!tt.ptr<bf16>, #blocked>
%p_26 = arith.extf %p_25 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%m = tt.splat %grad_mul : f32 -> tensor<1024xf32, #blocked>
%m_27 = arith.mulf %g_20, %m : tensor<1024xf32, #blocked>
%v_28 = arith.mulf %v_12, %beta2 : f32
%v_29 = arith.mulf %g_20, %g_20 : tensor<1024xf32, #blocked>
%v_30 = "tt.reduce"(%v_29) <{axis = 0 : i32}> ({
^bb0(%v_46: f32, %v_47: f32):
%v_48 = arith.addf %v_46, %v_47 : f32
tt.reduce.return %v_48 : f32
}) : (tensor<1024xf32, #blocked>) -> f32
%v_31 = arith.divf %v_30, %cst_1 : f32
%v_32 = arith.mulf %v_31, %sq_mul : f32
%v_33 = arith.addf %v_28, %v_32 : f32
%u = math.sqrt %v_33 : f32
%u_34 = arith.addf %u, %eps : f32
%u_35 = tt.splat %u_34 : f32 -> tensor<1024xf32, #blocked>
%u_36 = arith.divf %m_27, %u_35 : tensor<1024xf32, #blocked>
%u_37 = tt.splat %step_size : f32 -> tensor<1024xf32, #blocked>
%u_38 = arith.mulf %u_36, %u_37 : tensor<1024xf32, #blocked>
%u_39 = arith.mulf %u_38, %p_26 : tensor<1024xf32, #blocked>
%u_40 = arith.cmpf ogt, %u_39, %cst_2 : tensor<1024xf32, #blocked>
%u_41 = tt.splat %decay : f32 -> tensor<1024xf32, #blocked>
%u_42 = arith.mulf %p_26, %u_41 : tensor<1024xf32, #blocked>
%u_43 = arith.select %u_40, %u_42, %cst_2 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%u_44 = arith.addf %u_38, %u_43 : tensor<1024xf32, #blocked>
%p_45 = arith.subf %p_26, %u_44 : tensor<1024xf32, #blocked>
tt.store %v, %v_33 : !tt.ptr<f32>
%1 = arith.truncf %p_45 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %p_24, %1, %mask : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.store %last, %event : !tt.ptr<i32>
}
tt.return
}
}"#

private def updateZeroPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.global .align 1 .b8 _$_str_$_2[17] = {95, 95, 67, 85, 68, 65, 95, 80, 82, 69, 67, 95, 83, 81, 82, 84};
.visible .entry _adam_rows_kernel(
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_0,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_1,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_2,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_3,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_4,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_5,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_6,
.param .u32 _adam_rows_kernel_param_7,
.param .f32 _adam_rows_kernel_param_8,
.param .f32 _adam_rows_kernel_param_9,
.param .f32 _adam_rows_kernel_param_10,
.param .f32 _adam_rows_kernel_param_11,
.param .f32 _adam_rows_kernel_param_12,
.param .f32 _adam_rows_kernel_param_13,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_14,
.param .u64 .ptr .global .align 1 _adam_rows_kernel_param_15
)
.reqntid 128
{
.reg .pred %p<20>;
.reg .b16 %rs<17>;
.reg .b32 %r<153>;
.reg .b64 %rd<22>;
$L__func_begin0:
mov.u32 %r16, %ctaid.x;
ld.param.b64 %rd10, [_adam_rows_kernel_param_5];
ld.param.b64 %rd11, [_adam_rows_kernel_param_6];
mad.wide.u32 %rd8, %r16, 4, %rd10;
mov.u32 %r14, 0x0;
ld.global.b32 { %r14 }, [ %rd8 + 0 ];
mad.wide.s32 %rd9, %r14, 4, %rd11;
mov.u32 %r15, 0x0;
ld.global.b32 { %r15 }, [ %rd9 + 0 ];
setp.ne.b32 %p1, %r15, %r16;
@%p1 bra $L__BB0_5;
ld.param.b32 %r13, [_adam_rows_kernel_param_13];
ld.param.b32 %r12, [_adam_rows_kernel_param_12];
ld.param.b32 %r11, [_adam_rows_kernel_param_11];
ld.param.b32 %r10, [_adam_rows_kernel_param_10];
ld.param.b32 %r9, [_adam_rows_kernel_param_9];
ld.param.b32 %r8, [_adam_rows_kernel_param_8];
ld.param.b32 %r40, [_adam_rows_kernel_param_7];
ld.param.b64 %rd7, [_adam_rows_kernel_param_4];
ld.param.b64 %rd5, [_adam_rows_kernel_param_2];
ld.param.b64 %rd4, [_adam_rows_kernel_param_1];
ld.param.b64 %rd3, [_adam_rows_kernel_param_0];
mov.u32 %r1, %tid.x;
and.b32 %r2, %r1, 127;
and.b32 %r3, %r1, 31;
bfe.u32 %r4, %r1, 5, 2;
shl.b32 %r17, %r1, 3;
and.b32 %r5, %r17, 1016;
cvt.u64.u32 %rd1, %r16;
cvt.s64.s32 %rd2, %r14;
shl.b64 %rd12, %rd2, 2;
add.s64 %rd17, %rd5, %rd12;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd17 + 0 ];
add.s64 %rd16, %rd4, %rd12;
mov.u32 %r152, 0x0;
ld.global.b32 { %r152 }, [ %rd16 + 0 ];
add.s32 %r19, %r40, -1;
setp.eq.b32 %p2, %r152, 0;
selp.b32 %r6, %r19, %r18, %p2;
sub.s32 %r7, %r19, %r6;
setp.lt.s32 %p3, %r7, 1;
@%p3 bra $L__BB0_4;
ld.param.b64 %rd6, [_adam_rows_kernel_param_3];
cvt.u64.u32 %rd21, %r7;
add.s32 %r151, %r6, 1;
$L__BB0_3:
mad.wide.s32 %rd13, %r151, 4, %rd6;
mov.u32 %r20, 0x0;
ld.global.b32 { %r20 }, [ %rd13 + 0 ];
mul.f32 %r152, %r152, %r20;
add.s64 %rd21, %rd21, -1;
add.s32 %r151, %r151, 1;
setp.ne.b64 %p4, %rd21, 0;
@%p4 bra $L__BB0_3;
$L__BB0_4:
setp.lt.u32 %p5, %r5, 768;
mad.lo.s64 %rd18, %rd1, 1536, %rd7;
mul.wide.u32 %rd19, %r5, 2;
add.s64 %rd14, %rd18, %rd19;
mov.b32 %r25, 0;
mov.u32 %r21, %r25;
mov.u32 %r22, %r25;
mov.u32 %r23, %r25;
mov.u32 %r24, %r25;
@%p5 ld.global.v4.b32 { %r21, %r22, %r23, %r24 }, [ %rd14 + 0 ];
mov.b32 {%rs1, %rs2}, %r21;
mov.b32 {%rs3, %rs4}, %r22;
mov.b32 {%rs5, %rs6}, %r23;
mov.b32 {%rs7, %rs8}, %r24;
cvt.f32.bf16 %r41, %rs1;
cvt.f32.bf16 %r42, %rs2;
cvt.f32.bf16 %r43, %rs3;
cvt.f32.bf16 %r44, %rs4;
cvt.f32.bf16 %r45, %rs5;
cvt.f32.bf16 %r46, %rs6;
cvt.f32.bf16 %r47, %rs7;
cvt.f32.bf16 %r48, %rs8;
mad.lo.s64 %rd20, %rd2, 1536, %rd3;
add.s64 %rd15, %rd20, %rd19;
mov.u32 %r26, %r25;
mov.u32 %r27, %r25;
mov.u32 %r28, %r25;
mov.u32 %r29, %r25;
@%p5 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd15 + 0 ];
mov.b32 {%rs9, %rs10}, %r26;
cvt.f32.bf16 %r49, %rs9;
cvt.f32.bf16 %r50, %rs10;
mov.b32 {%rs11, %rs12}, %r27;
cvt.f32.bf16 %r51, %rs11;
cvt.f32.bf16 %r52, %rs12;
mov.b32 {%rs13, %rs14}, %r28;
cvt.f32.bf16 %r53, %rs13;
cvt.f32.bf16 %r54, %rs14;
mov.b32 {%rs15, %rs16}, %r29;
cvt.f32.bf16 %r55, %rs15;
cvt.f32.bf16 %r56, %rs16;
mul.f32 %r57, %r12, %r41;
mul.f32 %r58, %r12, %r42;
mul.f32 %r59, %r12, %r43;
mul.f32 %r60, %r12, %r44;
mul.f32 %r61, %r12, %r45;
mul.f32 %r62, %r12, %r46;
mul.f32 %r63, %r12, %r47;
mul.f32 %r64, %r12, %r48;
mul.f32 %r65, %r42, %r42;
$L__tmp0:
fma.rn.f32 %r66, %r41, %r41, %r65;
fma.rn.f32 %r67, %r43, %r43, %r66;
fma.rn.f32 %r68, %r44, %r44, %r67;
fma.rn.f32 %r69, %r45, %r45, %r68;
fma.rn.f32 %r70, %r46, %r46, %r69;
fma.rn.f32 %r71, %r47, %r47, %r70;
fma.rn.f32 %r72, %r48, %r48, %r71;
$L__tmp1:
shfl.sync.bfly.b32 %r73, %r72, 16, 31, -1;
$L__tmp2:
add.f32 %r74, %r72, %r73;
$L__tmp3:
shfl.sync.bfly.b32 %r75, %r74, 8, 31, -1;
$L__tmp4:
add.f32 %r76, %r74, %r75;
$L__tmp5:
shfl.sync.bfly.b32 %r77, %r76, 4, 31, -1;
$L__tmp6:
add.f32 %r78, %r76, %r77;
$L__tmp7:
shfl.sync.bfly.b32 %r79, %r78, 2, 31, -1;
$L__tmp8:
add.f32 %r80, %r78, %r79;
$L__tmp9:
shfl.sync.bfly.b32 %r81, %r80, 1, 31, -1;
$L__tmp10:
add.f32 %r31, %r80, %r81;
$L__tmp11:
setp.eq.b32 %p6, %r3, 0;
shl.b32 %r82, %r4, 2;
mov.b32 %r83, global_smem;
add.s32 %r30, %r83, %r82;
@%p6 st.shared.b32 [ %r30 + 0 ], %r31;
bar.sync 0;
setp.lt.u32 %p7, %r2, 4;
shl.b32 %r84, %r2, 2;
add.s32 %r33, %r83, %r84;
@%p7 ld.shared.b32 %r32, [ %r33 + 0 ];
shfl.sync.bfly.b32 %r85, %r32, 2, 31, -1;
$L__tmp12:
add.f32 %r86, %r32, %r85;
$L__tmp13:
shfl.sync.bfly.b32 %r87, %r86, 1, 31, -1;
$L__tmp14:
add.f32 %r34, %r86, %r87;
$L__tmp15:
and.b32 %r88, %r1, 3;
setp.eq.b32 %p10, %r88, 0;
and.pred %p8, %p7, %p10;
@%p8 st.shared.b32 [ %r33 + 0 ], %r34;
bar.sync 0;
ld.shared.b32 %r89, [global_smem];
mov.b32 %r90, 0f44400000;
$L__tmp16:
div.full.f32 %r91, %r89, %r90;
mul.f32 %r92, %r13, %r91;
fma.rn.f32 %r35, %r8, %r152, %r92;
sqrt.approx.ftz.f32 %r93, %r35;
add.f32 %r94, %r9, %r93;
div.full.f32 %r95, %r57, %r94;
div.full.f32 %r96, %r58, %r94;
div.full.f32 %r97, %r59, %r94;
div.full.f32 %r98, %r60, %r94;
div.full.f32 %r99, %r61, %r94;
div.full.f32 %r100, %r62, %r94;
div.full.f32 %r101, %r63, %r94;
div.full.f32 %r102, %r64, %r94;
mul.f32 %r103, %r11, %r50;
mul.f32 %r104, %r11, %r49;
mul.f32 %r105, %r11, %r52;
mul.f32 %r106, %r11, %r51;
mul.f32 %r107, %r11, %r54;
mul.f32 %r108, %r11, %r53;
mul.f32 %r109, %r11, %r56;
mul.f32 %r110, %r11, %r55;
setp.lt.u32 %p11, %r2, 32;
and.pred %p9, %p6, %p11;
@%p9 st.global.b32 [ %rd16 + 0 ], { %r35 };
mul.f32 %r111, %r10, %r96;
mul.f32 %r112, %r10, %r95;
mul.f32 %r113, %r112, %r49;
mul.f32 %r114, %r111, %r50;
setp.gt.f32 %p12, %r114, 0f00000000;
setp.gt.f32 %p13, %r113, 0f00000000;
selp.f32 %r115, %r104, 0f00000000, %p13;
selp.f32 %r116, %r103, 0f00000000, %p12;
fma.rn.f32 %r117, %r10, %r96, %r116;
fma.rn.f32 %r118, %r10, %r95, %r115;
sub.f32 %r119, %r49, %r118;
sub.f32 %r120, %r50, %r117;
cvt.rn.bf16x2.f32 %r36, %r120, %r119;
mul.f32 %r121, %r10, %r98;
mul.f32 %r122, %r10, %r97;
mul.f32 %r123, %r122, %r51;
mul.f32 %r124, %r121, %r52;
setp.gt.f32 %p14, %r124, 0f00000000;
setp.gt.f32 %p15, %r123, 0f00000000;
selp.f32 %r125, %r106, 0f00000000, %p15;
selp.f32 %r126, %r105, 0f00000000, %p14;
fma.rn.f32 %r127, %r10, %r98, %r126;
fma.rn.f32 %r128, %r10, %r97, %r125;
sub.f32 %r129, %r51, %r128;
sub.f32 %r130, %r52, %r127;
cvt.rn.bf16x2.f32 %r37, %r130, %r129;
mul.f32 %r131, %r10, %r100;
mul.f32 %r132, %r10, %r99;
mul.f32 %r133, %r132, %r53;
mul.f32 %r134, %r131, %r54;
setp.gt.f32 %p16, %r134, 0f00000000;
setp.gt.f32 %p17, %r133, 0f00000000;
selp.f32 %r135, %r108, 0f00000000, %p17;
selp.f32 %r136, %r107, 0f00000000, %p16;
fma.rn.f32 %r137, %r10, %r100, %r136;
fma.rn.f32 %r138, %r10, %r99, %r135;
sub.f32 %r139, %r53, %r138;
sub.f32 %r140, %r54, %r137;
cvt.rn.bf16x2.f32 %r38, %r140, %r139;
mul.f32 %r141, %r10, %r102;
mul.f32 %r142, %r10, %r101;
mul.f32 %r143, %r142, %r55;
mul.f32 %r144, %r141, %r56;
setp.gt.f32 %p18, %r144, 0f00000000;
setp.gt.f32 %p19, %r143, 0f00000000;
selp.f32 %r145, %r110, 0f00000000, %p19;
selp.f32 %r146, %r109, 0f00000000, %p18;
fma.rn.f32 %r147, %r10, %r102, %r146;
fma.rn.f32 %r148, %r10, %r101, %r145;
sub.f32 %r149, %r55, %r148;
sub.f32 %r150, %r56, %r147;
cvt.rn.bf16x2.f32 %r39, %r150, %r149;
@%p5 st.global.v4.b32 [ %rd15 + 0 ], { %r36, %r37, %r38, %r39 };
@%p9 st.global.b32 [ %rd17 + 0 ], { %r40 };
$L__BB0_5:
ret;
$L__tmp17:
$L__func_end0:
}"#

private def scatterPeerTiledIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_mapped_rows_kernel(%source: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %indices: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dest: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row_map: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %count: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<768> : tensor<4x1xi64, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16, #blocked>
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
%src_16 = tt.splat %source : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%src_17 = tt.addptr %src_16, %src_15 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi64, #blocked>
%src_18 = tt.expand_dims %columns {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%src_19 = tt.broadcast %src_17 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%src_20 = tt.broadcast %src_18 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%src_21 = tt.addptr %src_19, %src_20 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%dst = tt.expand_dims %entry_13 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%dst_22 = arith.extsi %dst : tensor<4x1xi32, #blocked> to tensor<4x1xi64, #blocked>
%dst_23 = arith.muli %dst_22, %cst_0 : tensor<4x1xi64, #blocked>
%dst_24 = tt.splat %dest : !tt.ptr<bf16> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%dst_25 = tt.addptr %dst_24, %dst_23 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi64, #blocked>
%dst_26 = tt.broadcast %dst_25 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%dst_27 = tt.addptr %dst_26, %src_20 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%words = tt.expand_dims %live_8 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%words_28 = tt.broadcast %words : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%words_29 = tt.load %src_21, %words_28, %cst_1 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%0 = tt.atomic_rmw fadd, relaxed, gpu, %dst_27, %words_29, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
%words_30 = tt.addptr %src_21, %cst_2 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%words_31 = tt.load %words_30, %words_28, %cst_1 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %dst_27, %cst_2 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%2 = tt.atomic_rmw fadd, relaxed, gpu, %1, %words_31, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
%words_32 = tt.addptr %src_21, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%words_33 = tt.load %words_32, %words_28, %cst_1 : tensor<4x256x!tt.ptr<bf16>, #blocked>
%3 = tt.addptr %dst_27, %cst_3 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %3, %words_33, %words_28 : (tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xbf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xbf16, #blocked>
tt.return
}
}"#

private def scatterPeerTiledPTX : String := r#".version 8.7
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
.reg .b16 %rs<49>;
.reg .b32 %r<24>;
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
mov.b32 {%rs9, %rs10}, %r3;
mov.b32 {%rs11, %rs12}, %r4;
mov.b32 {%rs13, %rs14}, %r5;
mov.b32 {%rs15, %rs16}, %r6;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
mov.u16 %rs3, 0x0;
mov.u16 %rs4, 0x0;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
mov.u16 %rs7, 0x0;
mov.u16 %rs8, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs1, %rs2, %rs3, %rs4, %rs5, %rs6, %rs7, %rs8 }, [ %rd4 + 0 ], { %rs9, %rs10, %rs11, %rs12, %rs13, %rs14, %rs15, %rs16 };
add.s64 %rd5, %rd3, 512;
mov.u32 %r8, %r7;
mov.u32 %r9, %r7;
mov.u32 %r10, %r7;
mov.u32 %r11, %r7;
@%p1 ld.global.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd5 + 0 ];
mov.b32 {%rs25, %rs26}, %r8;
mov.b32 {%rs27, %rs28}, %r9;
mov.b32 {%rs29, %rs30}, %r10;
mov.b32 {%rs31, %rs32}, %r11;
add.s64 %rd6, %rd4, 512;
mov.u16 %rs17, 0x0;
mov.u16 %rs18, 0x0;
mov.u16 %rs19, 0x0;
mov.u16 %rs20, 0x0;
mov.u16 %rs21, 0x0;
mov.u16 %rs22, 0x0;
mov.u16 %rs23, 0x0;
mov.u16 %rs24, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs17, %rs18, %rs19, %rs20, %rs21, %rs22, %rs23, %rs24 }, [ %rd6 + 0 ], { %rs25, %rs26, %rs27, %rs28, %rs29, %rs30, %rs31, %rs32 };
add.s64 %rd7, %rd3, 1024;
mov.u32 %r12, %r7;
mov.u32 %r13, %r7;
mov.u32 %r14, %r7;
mov.u32 %r15, %r7;
@%p1 ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd7 + 0 ];
mov.b32 {%rs41, %rs42}, %r12;
mov.b32 {%rs43, %rs44}, %r13;
mov.b32 {%rs45, %rs46}, %r14;
mov.b32 {%rs47, %rs48}, %r15;
add.s64 %rd8, %rd4, 1024;
mov.u16 %rs33, 0x0;
mov.u16 %rs34, 0x0;
mov.u16 %rs35, 0x0;
mov.u16 %rs36, 0x0;
mov.u16 %rs37, 0x0;
mov.u16 %rs38, 0x0;
mov.u16 %rs39, 0x0;
mov.u16 %rs40, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.bf16 { %rs33, %rs34, %rs35, %rs36, %rs37, %rs38, %rs39, %rs40 }, [ %rd8 + 0 ], { %rs41, %rs42, %rs43, %rs44, %rs45, %rs46, %rs47, %rs48 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def recognizesCode (code : Code) (ir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR code.ir == some (ir.splitOn "\n") &&
    QKVTTIR.normalizePTX code.ptx == ptx.splitOn "\n"

def recognized (a : Artifacts) : Bool :=
  recognizesCode a.pack packIR packPTX && recognizesCode a.prepare prepareIR preparePTX &&
    recognizesCode a.update updateIR updatePTX && recognizesCode a.updateZero updateZeroIR updateZeroPTX &&
    recognizesCode a.scatterOwn scatterOwnIR scatterOwnPTX && recognizesCode a.scatterPeer scatterPeerIR scatterPeerPTX &&
    recognizesCode a.scatterPeerTiled scatterPeerTiledIR scatterPeerTiledPTX

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  indices : (AdamEntries.fused : AdamEntries.Pack) = AdamEntries.staged
  preparation : fusedPrepare = stagedPrepare
  update : ∀ p (s : Schedule p) native launch previous,
    eventPrepared p (fusedPrepare p s) s native launch previous =
      event p (AdamEntries.staged p.ids) s native launch previous
  updateZero : ∀ p (s : Schedule p) native launch previous,
    MomentReplay.PositiveHistory previous.betaHistory (launch.event - 1) →
    eventPreparedWith rowUpdateSkippingZero p (fusedPrepare p s) s native launch previous =
      event p (AdamEntries.staged p.ids) s native launch previous
  disjointRows : ∀ {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p)
    (a b : Fin p.ids.count) (row : Int32), issuedRow rows (claim rows s) a = some row →
    issuedRow rows (claim rows s) b = some row → a = b
  persistentUpdate : ∀ p (a b : Schedule p),
    a.localAtomics = b.localAtomics → a.peerAtomics = b.peerAtomics →
    ∀ native launch previous,
      (eventPrepared p (fusedPrepare p a) a native launch previous).state =
        (event p (AdamEntries.staged p.ids) b native launch previous).state

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then
    some ⟨h, AdamEntries.pack_equal, prepare_equal, prepared_event_equal,
      (fun p s native launch previous valid =>
        (prepared_skip_exact p _ s native launch previous valid).trans
          (prepared_event_equal p s native launch previous)), issued_rows_disjoint, prepared_state_equal⟩
  else none

def Certificate.prepareKernel {a : Artifacts} (_checked : Certificate a) : Prepare :=
  if recognized a then fusedPrepare else stagedPrepare

theorem Certificate.prepareKernel_eq {a : Artifacts} (checked : Certificate a) :
    checked.prepareKernel = fusedPrepare := by
  simp only [Certificate.prepareKernel, checked.matched, if_true]

def Certificate.rowKernel {a : Artifacts} (_checked : Certificate a) : RowKernel :=
  if recognized a then selectedRowUpdate else rowUpdate

theorem Certificate.rowKernel_eq {a : Artifacts} (checked : Certificate a) :
    checked.rowKernel = selectedRowUpdate := by
  simp only [Certificate.rowKernel, checked.matched, if_true]

/-- The tiled warp reads the same wire word for each slot and channel.
The ordered merge still visits every peer occurrence of a repeated ID. -/
def Certificate.peerWords {a : Artifacts} (_checked : Certificate a) : PeerWords :=
  fun p => if recognized a then SparsePackets.rowTiled p.peerGradient else p.peerGradient

theorem Certificate.peerWords_eq {a : Artifacts} (checked : Certificate a) :
    checked.peerWords = Pull.peerGradient := by
  funext p slot col
  simp only [Certificate.peerWords, checked.matched, if_true, SparsePackets.row_tiled_exact]

#axiomsClean Certificate.prepareKernel_eq
#axiomsClean Certificate.rowKernel_eq
#axiomsClean Certificate.peerWords_eq
#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.NgramAdam
