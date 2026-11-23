import models.NanoGPT.Track1.MomentReplay
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The pinned scalar replay and the Hopper warp replay use the same
plain FP32 multiply. The normalized artifacts retain the loop bounds,
warp reduction, lane predicates, and the load and store addresses. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MomentReplay

structure Artifacts where
  referenceIR : String
  referencePTX : String
  fallbackIR : String
  fallbackPTX : String
  groupedIR : String
  groupedPTX : String
  groupedZeroIR : String
  groupedZeroPTX : String
  deriving BEq

private def referenceIR : String := r#"module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_adam_rows_kernel(%P: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %LAST_EVENT: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %BETA2_HISTORY: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %G: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROWS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %ENTRY_OF_ROW: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %event: i32, %beta2: f32, %eps: f32, %step_size: f32, %decay: f32, %grad_mul: f32, %sq_mul: f32) attributes {noinline = false} {
%pid = tt.get_program_id x : i32
%row = tt.addptr %ROWS, %pid : !tt.ptr<i32>, i32
%row_0 = tt.load %row : !tt.ptr<i32>
%row64 = arith.extsi %row_0 : i32 to i64
%last = tt.addptr %LAST_EVENT, %row_0 : !tt.ptr<i32>, i32
%last_1 = tt.load %last : !tt.ptr<i32>
%missed = arith.subi %event, %last_1 : i32
%v = tt.addptr %V, %row64 : !tt.ptr<f32>, i64
%v_2 = tt.load %v : !tt.ptr<f32>
%v_3 = arith.addi %last_1, 1 : i32
%v_4 = scf.for %i = 0 to %missed step 1 iter_args(%v_5 = %v_2) -> (f32) : i32 {
%v_6 = arith.addi %v_3, %i : i32
%v_7 = tt.addptr %BETA2_HISTORY, %v_6 : !tt.ptr<f32>, i32
%v_8 = tt.load %v_7 : !tt.ptr<f32>
%v_9 = arith.mulf %v_5, %v_8 : f32
scf.yield %v_9 : f32
}
tt.store %v, %v_4 : !tt.ptr<f32>
tt.store %last, %event : !tt.ptr<i32>
tt.return
}
}"#

private def referencePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
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
.reg .pred %p<4>;
.reg .b32 %r<11>;
.reg .b64 %rd<11>;
$L__func_begin0:
ld.param.b32 %r6, [_adam_rows_kernel_param_7];
ld.param.b64 %rd3, [_adam_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r4, %ctaid.x;
ld.param.b64 %rd4, [_adam_rows_kernel_param_2];
ld.param.b64 %rd5, [_adam_rows_kernel_param_5];
mad.wide.u32 %rd2, %r4, 4, %rd5;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mul.wide.s32 %rd6, %r2, 4;
add.s64 %rd9, %rd4, %rd6;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd9 + 0 ];
sub.s32 %r1, %r6, %r3;
add.s64 %rd8, %rd3, %rd6;
mov.u32 %r10, 0x0;
ld.global.b32 { %r10 }, [ %rd8 + 0 ];
setp.lt.s32 %p1, %r1, 1;
@%p1 bra $L__BB0_3;
ld.param.b64 %rd1, [_adam_rows_kernel_param_3];
cvt.u64.u32 %rd10, %r1;
add.s32 %r9, %r3, 1;
$L__BB0_2:
mad.wide.s32 %rd7, %r9, 4, %rd1;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd7 + 0 ];
mul.f32 %r10, %r10, %r5;
add.s64 %rd10, %rd10, -1;
add.s32 %r9, %r9, 1;
setp.ne.b64 %p2, %rd10, 0;
@%p2 bra $L__BB0_2;
$L__BB0_3:
mov.u32 %r7, %tid.x;
and.b32 %r8, %r7, 127;
setp.eq.b32 %p3, %r8, 0;
@%p3 st.global.b32 [ %rd8 + 0 ], { %r10 };
@%p3 st.global.b32 [ %rd9 + 0 ], { %r6 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def groupedIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_replay_moments(%moment: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stamps: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %history: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %count: i32, %offset: i32, %first: i32, %event: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<32xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
%slot = tt.get_program_id x : i32
%slot_1 = arith.muli %slot, 32 : i32
%slot_2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
%slot_3 = tt.splat %slot_1 : i32 -> tensor<32xi32, #blocked>
%slot_4 = arith.addi %slot_3, %slot_2 : tensor<32xi32, #blocked>
%live = tt.splat %count : i32 -> tensor<32xi32, #blocked>
%live_5 = arith.cmpi slt, %slot_4, %live : tensor<32xi32, #blocked>
%row = tt.addptr %want, %offset : !tt.ptr<i32>, i32
%row_6 = tt.splat %row : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
%row_7 = tt.addptr %row_6, %slot_4 : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
%row_8 = tt.load %row_7, %live_5, %cst : tensor<32x!tt.ptr<i32>, #blocked>
%row_9 = tt.splat %first : i32 -> tensor<32xi32, #blocked>
%row_10 = arith.subi %row_8, %row_9 : tensor<32xi32, #blocked>
%last = tt.splat %stamps : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
%last_11 = tt.addptr %last, %row_10 : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
%last_12 = tt.splat %event : i32 -> tensor<32xi32, #blocked>
%last_13 = tt.load %last_11, %live_5, %last_12 : tensor<32x!tt.ptr<i32>, #blocked>
%due = arith.cmpi slt, %last_13, %last_12 : tensor<32xi32, #blocked>
%due_14 = arith.andi %live_5, %due : tensor<32xi1, #blocked>
%start = arith.minsi %last_13, %last_12 : tensor<32xi32, #blocked>
%start_15 = "tt.reduce"(%start) <{axis = 0 : i32}> ({
^bb0(%start_19: i32, %start_20: i32):
%start_21 = arith.minsi %start_19, %start_20 : i32
tt.reduce.return %start_21 : i32
}) : (tensor<32xi32, #blocked>) -> i32
%value = tt.splat %moment : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
%value_16 = tt.addptr %value, %row_10 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
%value_17 = tt.load %value_16, %due_14, %cst_0 : tensor<32x!tt.ptr<f32>, #blocked>
%0 = arith.addi %start_15, 1 : i32
%1 = arith.addi %event, 1 : i32
%value_18 = scf.for %e = %0 to %1 step 1 iter_args(%value_19 = %value_17) -> (tensor<32xf32, #blocked>) : i32 {
%beta = tt.addptr %history, %e : !tt.ptr<f32>, i32
%beta_20 = tt.load %beta : !tt.ptr<f32>
%value_21 = tt.splat %e : i32 -> tensor<32xi32, #blocked>
%value_22 = arith.cmpi slt, %last_13, %value_21 : tensor<32xi32, #blocked>
%value_23 = tt.splat %beta_20 : f32 -> tensor<32xf32, #blocked>
%value_24 = arith.mulf %value_19, %value_23 : tensor<32xf32, #blocked>
%value_25 = arith.select %value_22, %value_24, %value_19 : tensor<32xi1, #blocked>, tensor<32xf32, #blocked>
scf.yield %value_25 : tensor<32xf32, #blocked>
}
tt.store %value_16, %value_18, %due_14 : tensor<32x!tt.ptr<f32>, #blocked>
%2 = arith.cmpi ne, %last_13, %last_12 : tensor<32xi32, #blocked>
%3 = arith.andi %live_5, %2 : tensor<32xi1, #blocked>
tt.store %last_11, %last_12, %3 : tensor<32x!tt.ptr<i32>, #blocked>
tt.return
}
}"#

private def groupedPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _replay_moments(
.param .u64 .ptr .global .align 1 _replay_moments_param_0,
.param .u64 .ptr .global .align 1 _replay_moments_param_1,
.param .u64 .ptr .global .align 1 _replay_moments_param_2,
.param .u64 .ptr .global .align 1 _replay_moments_param_3,
.param .u32 _replay_moments_param_4,
.param .u32 _replay_moments_param_5,
.param .u32 _replay_moments_param_6,
.param .u32 _replay_moments_param_7,
.param .u64 .ptr .global .align 1 _replay_moments_param_8,
.param .u64 .ptr .global .align 1 _replay_moments_param_9
)
.reqntid 32
{
.reg .pred %p<9>;
.reg .b32 %r<23>;
.reg .b64 %rd<13>;
$L__func_begin0:
ld.param.b32 %r20, [_replay_moments_param_7];
ld.param.b32 %r4, [_replay_moments_param_4];
ld.param.b64 %rd4, [_replay_moments_param_0];
ld.param.b64 %rd5, [_replay_moments_param_1];
$L__tmp0:
mov.u32 %r8, %ctaid.x;
shl.b32 %r9, %r8, 5;
ld.param.b64 %rd6, [_replay_moments_param_3];
mov.u32 %r10, %tid.x;
and.b32 %r11, %r10, 31;
or.b32 %r1, %r9, %r11;
ld.param.b32 %r12, [_replay_moments_param_5];
setp.lt.s32 %p1, %r1, %r4;
ld.param.b32 %r13, [_replay_moments_param_6];
mad.wide.s32 %rd7, %r12, 4, %rd6;
mad.wide.s32 %rd3, %r1, 4, %rd7;
mov.u32 %r5, 0x0;
@%p1 ld.global.b32 { %r5 }, [ %rd3 + 0 ];
sub.s32 %r14, %r5, %r13;
mul.wide.s32 %rd8, %r14, 4;
add.s64 %rd10, %rd5, %rd8;
mov.u32 %r6, %r20;
@%p1 ld.global.b32 { %r6 }, [ %rd10 + 0 ];
setp.lt.s32 %p2, %r6, %r20;
and.pred %p6, %p1, %p2;
min.s32 %r15, %r6, %r20;
mov.b32 %r16, -1;
$L__tmp1:
redux.sync.min.s32 %r2, %r15, %r16;
$L__tmp2:
add.s64 %rd9, %rd4, %rd8;
mov.b32 %r7, 0;
mov.u32 %r22, %r7;
@%p6 ld.global.b32 { %r22 }, [ %rd9 + 0 ];
add.s32 %r17, %r20, 1;
add.s32 %r3, %r2, 1;
setp.ge.s32 %p3, %r3, %r17;
@%p3 bra $L__BB0_3;
ld.param.b64 %rd2, [_replay_moments_param_2];
cvt.s64.s32 %rd12, %r3;
cvt.s64.s32 %rd1, %r6;
sub.s32 %r21, %r20, %r2;
mad.wide.s32 %rd11, %r3, 4, %rd2;
$L__BB0_2:
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd11 + 0 ];
setp.gt.s64 %p4, %rd12, %rd1;
mul.f32 %r19, %r22, %r18;
selp.f32 %r22, %r19, %r22, %p4;
add.s64 %rd12, %rd12, 1;
add.s32 %r21, %r21, -1;
add.s64 %rd11, %rd11, 4;
setp.ne.b32 %p5, %r21, 0;
@%p5 bra $L__BB0_2;
$L__BB0_3:
@%p6 st.global.b32 [ %rd9 + 0 ], { %r22 };
setp.ne.b32 %p8, %r6, %r20;
and.pred %p7, %p1, %p8;
@%p7 st.global.b32 [ %rd10 + 0 ], { %r20 };
ret;
$L__tmp3:
$L__func_end0:
}"#

private def groupedZeroIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_replay_moments(%moment: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %stamps: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %history: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %count: i32, %offset: i32, %first: i32, %event: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<32xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32, #blocked>
%slot = tt.get_program_id x : i32
%slot_1 = arith.muli %slot, 32 : i32
%slot_2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
%slot_3 = tt.splat %slot_1 : i32 -> tensor<32xi32, #blocked>
%slot_4 = arith.addi %slot_3, %slot_2 : tensor<32xi32, #blocked>
%live = tt.splat %count : i32 -> tensor<32xi32, #blocked>
%live_5 = arith.cmpi slt, %slot_4, %live : tensor<32xi32, #blocked>
%row = tt.addptr %want, %offset : !tt.ptr<i32>, i32
%row_6 = tt.splat %row : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
%row_7 = tt.addptr %row_6, %slot_4 : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
%row_8 = tt.load %row_7, %live_5, %cst : tensor<32x!tt.ptr<i32>, #blocked>
%row_9 = tt.splat %first : i32 -> tensor<32xi32, #blocked>
%row_10 = arith.subi %row_8, %row_9 : tensor<32xi32, #blocked>
%last = tt.splat %stamps : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
%last_11 = tt.addptr %last, %row_10 : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
%last_12 = tt.splat %event : i32 -> tensor<32xi32, #blocked>
%last_13 = tt.load %last_11, %live_5, %last_12 : tensor<32x!tt.ptr<i32>, #blocked>
%due = arith.cmpi slt, %last_13, %last_12 : tensor<32xi32, #blocked>
%due_14 = arith.andi %live_5, %due : tensor<32xi1, #blocked>
%value = tt.splat %moment : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
%value_15 = tt.addptr %value, %row_10 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
%value_16 = tt.load %value_15, %due_14, %cst_0 : tensor<32x!tt.ptr<f32>, #blocked>
%due_17 = tt.bitcast %value_16 : tensor<32xf32, #blocked> -> tensor<32xi32, #blocked>
%due_18 = arith.cmpi ne, %due_17, %cst : tensor<32xi32, #blocked>
%due_19 = arith.andi %due_14, %due_18 : tensor<32xi1, #blocked>
%replay_from = arith.select %due_19, %last_13, %last_12 : tensor<32xi1, #blocked>, tensor<32xi32, #blocked>
%start = "tt.reduce"(%replay_from) <{axis = 0 : i32}> ({
^bb0(%start_21: i32, %start_22: i32):
%start_23 = arith.minsi %start_21, %start_22 : i32
tt.reduce.return %start_23 : i32
}) : (tensor<32xi32, #blocked>) -> i32
%0 = arith.addi %start, 1 : i32
%1 = arith.addi %event, 1 : i32
%value_20 = scf.for %e = %0 to %1 step 1 iter_args(%value_21 = %value_16) -> (tensor<32xf32, #blocked>) : i32 {
%beta = tt.addptr %history, %e : !tt.ptr<f32>, i32
%beta_22 = tt.load %beta : !tt.ptr<f32>
%value_23 = tt.splat %e : i32 -> tensor<32xi32, #blocked>
%value_24 = arith.cmpi slt, %replay_from, %value_23 : tensor<32xi32, #blocked>
%value_25 = tt.splat %beta_22 : f32 -> tensor<32xf32, #blocked>
%value_26 = arith.mulf %value_21, %value_25 : tensor<32xf32, #blocked>
%value_27 = arith.select %value_24, %value_26, %value_21 : tensor<32xi1, #blocked>, tensor<32xf32, #blocked>
scf.yield %value_27 : tensor<32xf32, #blocked>
}
tt.store %value_15, %value_20, %due_19 : tensor<32x!tt.ptr<f32>, #blocked>
%2 = arith.cmpi ne, %last_13, %last_12 : tensor<32xi32, #blocked>
%3 = arith.andi %live_5, %2 : tensor<32xi1, #blocked>
tt.store %last_11, %last_12, %3 : tensor<32x!tt.ptr<i32>, #blocked>
tt.return
}
}"#

private def groupedZeroPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _replay_moments(
.param .u64 .ptr .global .align 1 _replay_moments_param_0,
.param .u64 .ptr .global .align 1 _replay_moments_param_1,
.param .u64 .ptr .global .align 1 _replay_moments_param_2,
.param .u64 .ptr .global .align 1 _replay_moments_param_3,
.param .u32 _replay_moments_param_4,
.param .u32 _replay_moments_param_5,
.param .u32 _replay_moments_param_6,
.param .u32 _replay_moments_param_7,
.param .u64 .ptr .global .align 1 _replay_moments_param_8,
.param .u64 .ptr .global .align 1 _replay_moments_param_9
)
.reqntid 32
{
.reg .pred %p<11>;
.reg .b32 %r<23>;
.reg .b64 %rd<13>;
$L__func_begin0:
ld.param.b32 %r20, [_replay_moments_param_7];
ld.param.b32 %r5, [_replay_moments_param_4];
ld.param.b64 %rd4, [_replay_moments_param_0];
ld.param.b64 %rd5, [_replay_moments_param_1];
$L__tmp0:
mov.u32 %r9, %ctaid.x;
shl.b32 %r10, %r9, 5;
ld.param.b64 %rd6, [_replay_moments_param_3];
mov.u32 %r11, %tid.x;
and.b32 %r12, %r11, 31;
or.b32 %r1, %r10, %r12;
ld.param.b32 %r13, [_replay_moments_param_5];
setp.lt.s32 %p1, %r1, %r5;
ld.param.b32 %r14, [_replay_moments_param_6];
mad.wide.s32 %rd7, %r13, 4, %rd6;
mad.wide.s32 %rd3, %r1, 4, %rd7;
mov.u32 %r6, 0x0;
@%p1 ld.global.b32 { %r6 }, [ %rd3 + 0 ];
sub.s32 %r15, %r6, %r14;
mul.wide.s32 %rd8, %r15, 4;
add.s64 %rd10, %rd5, %rd8;
mov.u32 %r7, %r20;
@%p1 ld.global.b32 { %r7 }, [ %rd10 + 0 ];
setp.lt.s32 %p3, %r7, %r20;
and.pred %p2, %p1, %p3;
add.s64 %rd9, %rd4, %rd8;
mov.b32 %r8, 0;
mov.u32 %r22, %r8;
@%p2 ld.global.b32 { %r22 }, [ %rd9 + 0 ];
setp.ne.b32 %p4, %r22, 0;
and.pred %p8, %p2, %p4;
selp.b32 %r2, %r7, %r20, %p8;
mov.b32 %r16, -1;
$L__tmp1:
redux.sync.min.s32 %r3, %r2, %r16;
$L__tmp2:
add.s32 %r17, %r20, 1;
add.s32 %r4, %r3, 1;
setp.ge.s32 %p5, %r4, %r17;
@%p5 bra $L__BB0_3;
ld.param.b64 %rd2, [_replay_moments_param_2];
cvt.s64.s32 %rd12, %r4;
cvt.s64.s32 %rd1, %r2;
sub.s32 %r21, %r20, %r3;
mad.wide.s32 %rd11, %r4, 4, %rd2;
$L__BB0_2:
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd11 + 0 ];
setp.gt.s64 %p6, %rd12, %rd1;
mul.f32 %r19, %r22, %r18;
selp.f32 %r22, %r19, %r22, %p6;
add.s64 %rd12, %rd12, 1;
add.s32 %r21, %r21, -1;
add.s64 %rd11, %rd11, 4;
setp.ne.b32 %p7, %r21, 0;
@%p7 bra $L__BB0_2;
$L__BB0_3:
@%p8 st.global.b32 [ %rd9 + 0 ], { %r22 };
setp.ne.b32 %p10, %r7, %r20;
and.pred %p9, %p1, %p10;
@%p9 st.global.b32 [ %rd10 + 0 ], { %r20 };
ret;
$L__tmp3:
$L__func_end0:
}"#

private def fallbackIR : String := r#"module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_adam_rows_kernel(%P: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %LAST_EVENT: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %BETA2_HISTORY: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %G: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROWS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %ENTRY_OF_ROW: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %event: i32, %beta2: f32, %eps: f32, %step_size: f32, %decay: f32, %grad_mul: f32, %sq_mul: f32) attributes {noinline = false} {
%pid = tt.get_program_id x : i32
%row = tt.addptr %ROWS, %pid : !tt.ptr<i32>, i32
%row_0 = tt.load %row : !tt.ptr<i32>
%row64 = arith.extsi %row_0 : i32 to i64
%last = tt.addptr %LAST_EVENT, %row_0 : !tt.ptr<i32>, i32
%last_1 = tt.load %last : !tt.ptr<i32>
%v = tt.addptr %V, %row64 : !tt.ptr<f32>, i64
%v_2 = tt.load %v : !tt.ptr<f32>
%0 = arith.subi %event, %last_1 : i32
%v_3 = arith.addi %last_1, 1 : i32
%v_4 = scf.for %i = 0 to %0 step 1 iter_args(%v_5 = %v_2) -> (f32) : i32 {
%v_6 = arith.addi %v_3, %i : i32
%v_7 = tt.addptr %BETA2_HISTORY, %v_6 : !tt.ptr<f32>, i32
%v_8 = tt.load %v_7 : !tt.ptr<f32>
%v_9 = arith.mulf %v_5, %v_8 : f32
scf.yield %v_9 : f32
}
tt.store %v, %v_4 : !tt.ptr<f32>
tt.store %last, %event : !tt.ptr<i32>
tt.return
}
}"#

private def fallbackPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
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
.reg .pred %p<4>;
.reg .b32 %r<11>;
.reg .b64 %rd<11>;
$L__func_begin0:
ld.param.b32 %r6, [_adam_rows_kernel_param_7];
ld.param.b64 %rd3, [_adam_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r4, %ctaid.x;
ld.param.b64 %rd4, [_adam_rows_kernel_param_2];
ld.param.b64 %rd5, [_adam_rows_kernel_param_5];
mad.wide.u32 %rd2, %r4, 4, %rd5;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mul.wide.s32 %rd6, %r2, 4;
add.s64 %rd9, %rd4, %rd6;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd9 + 0 ];
add.s64 %rd8, %rd3, %rd6;
mov.u32 %r10, 0x0;
ld.global.b32 { %r10 }, [ %rd8 + 0 ];
sub.s32 %r1, %r6, %r3;
setp.lt.s32 %p1, %r1, 1;
@%p1 bra $L__BB0_3;
ld.param.b64 %rd1, [_adam_rows_kernel_param_3];
cvt.u64.u32 %rd10, %r1;
add.s32 %r9, %r3, 1;
$L__BB0_2:
mad.wide.s32 %rd7, %r9, 4, %rd1;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd7 + 0 ];
mul.f32 %r10, %r10, %r5;
add.s64 %rd10, %rd10, -1;
add.s32 %r9, %r9, 1;
setp.ne.b64 %p2, %rd10, 0;
@%p2 bra $L__BB0_2;
$L__BB0_3:
mov.u32 %r7, %tid.x;
and.b32 %r8, %r7, 127;
setp.eq.b32 %p3, %r8, 0;
@%p3 st.global.b32 [ %rd8 + 0 ], { %r10 };
@%p3 st.global.b32 [ %rd9 + 0 ], { %r6 };
ret;
$L__tmp1:
$L__func_end0:
}"#

def recognized (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (referenceIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.referencePTX == referencePTX.splitOn "\n" &&
    QKVTTIR.normalizeTTIR a.fallbackIR == some (fallbackIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.fallbackPTX == fallbackPTX.splitOn "\n" &&
    QKVTTIR.normalizeTTIR a.groupedIR == some (groupedIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.groupedPTX == groupedPTX.splitOn "\n" &&
    QKVTTIR.normalizeTTIR a.groupedZeroIR == some (groupedZeroIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.groupedZeroPTX == groupedZeroPTX.splitOn "\n"

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  outputs : grouped = staged
  zeroOutputs : ∀ (s : Input), PositiveHistory s.beta s.event → groupedZero s = staged s
  preserves : ∀ (s : Input) (row : Nat),
    (∀ slot, slot ∈ List.range (s.hi - s.lo) → rowAt s slot ≠ some row) →
    grouped s row = s.old row

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then some ⟨h, replay_equal, grouped_zero_exact, unchanged_outside⟩ else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.MomentReplay
