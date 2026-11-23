import models.NanoGPT.Track1.ValueAdamClaimedClear
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The value-embedding scatter records the rows that need scaling and
the rows to clear before the following event. The clear resets flags only
after the CTA has consumed them. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

structure Code where
  ir : String
  ptx : String
  deriving BEq

structure Artifacts where
  reference : Code
  direct : Code
  marked : Code
  own : Code
  ownMarked : Code
  scale : Code
  clear : Code
  deriving BEq

private def referenceIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_add_rows_kernel(%SRC: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %IDX: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ROW_MAP: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c768_i64 = arith.constant 768 : i64
%c256_i64 = arith.constant 256 : i64
%c512_i64 = arith.constant 512 : i64
%cst = arith.constant dense<true> : tensor<256xi1, #blocked>
%pid = tt.get_program_id x : i32
%o64 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%o64_0 = arith.extsi %o64 : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%drow = tt.addptr %IDX, %pid : !tt.ptr<i32>, i32
%drow_1 = tt.load %drow : !tt.ptr<i32>
%sbase = arith.extsi %pid : i32 to i64
%sbase_2 = arith.muli %sbase, %c768_i64 : i64
%dbase = arith.extsi %drow_1 : i32 to i64
%dbase_3 = arith.muli %dbase, %c768_i64 : i64
%val = tt.splat %sbase_2 : i64 -> tensor<256xi64, #blocked>
%val_4 = arith.addi %val, %o64_0 : tensor<256xi64, #blocked>
%val_5 = tt.splat %SRC : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_6 = tt.addptr %val_5, %val_4 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_7 = tt.load %val_6 : tensor<256x!tt.ptr<f16>, #blocked>
%0 = tt.splat %dbase_3 : i64 -> tensor<256xi64, #blocked>
%1 = arith.addi %0, %o64_0 : tensor<256xi64, #blocked>
%2 = tt.splat %DST : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %3, %val_7, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%val_8 = arith.addi %sbase_2, %c256_i64 : i64
%val_9 = tt.splat %val_8 : i64 -> tensor<256xi64, #blocked>
%val_10 = arith.addi %val_9, %o64_0 : tensor<256xi64, #blocked>
%val_11 = tt.addptr %val_5, %val_10 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_12 = tt.load %val_11 : tensor<256x!tt.ptr<f16>, #blocked>
%5 = arith.addi %dbase_3, %c256_i64 : i64
%6 = tt.splat %5 : i64 -> tensor<256xi64, #blocked>
%7 = arith.addi %6, %o64_0 : tensor<256xi64, #blocked>
%8 = tt.addptr %2, %7 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%9 = tt.atomic_rmw fadd, relaxed, gpu, %8, %val_12, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%val_13 = arith.addi %sbase_2, %c512_i64 : i64
%val_14 = tt.splat %val_13 : i64 -> tensor<256xi64, #blocked>
%val_15 = arith.addi %val_14, %o64_0 : tensor<256xi64, #blocked>
%val_16 = tt.addptr %val_5, %val_15 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_17 = tt.load %val_16 : tensor<256x!tt.ptr<f16>, #blocked>
%10 = arith.addi %dbase_3, %c512_i64 : i64
%11 = tt.splat %10 : i64 -> tensor<256xi64, #blocked>
%12 = arith.addi %11, %o64_0 : tensor<256xi64, #blocked>
%13 = tt.addptr %2, %12 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%14 = tt.atomic_rmw fadd, relaxed, gpu, %13, %val_17, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.return
}
}"#

private def referencePTX : String := r#".version 8.7
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
.reg .b32 %r<9>;
.reg .b64 %rd<14>;
$L__func_begin0:
ld.param.b64 %rd8, [_scatter_add_rows_kernel_param_0];
ld.param.b64 %rd9, [_scatter_add_rows_kernel_param_1];
$L__tmp0:
mov.u32 %r5, %ctaid.x;
ld.param.b64 %rd10, [_scatter_add_rows_kernel_param_2];
mov.u32 %r6, %tid.x;
shl.b32 %r7, %r6, 1;
and.b32 %r8, %r7, 254;
mad.wide.u32 %rd1, %r5, 4, %rd9;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mad.wide.u32 %rd11, %r5, 1536, %rd8;
mul.wide.u32 %rd12, %r8, 2;
add.s64 %rd2, %rd11, %rd12;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mov.b32 {%rs3, %rs4}, %r2;
mad.wide.s32 %rd13, %r1, 1536, %rd10;
add.s64 %rd3, %rd13, %rd12;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd4 + 0 ];
mov.b32 {%rs7, %rs8}, %r3;
add.s64 %rd5, %rd3, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd6 + 0 ];
mov.b32 {%rs11, %rs12}, %r4;
add.s64 %rd7, %rd3, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def directIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_value_rows(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %offset: i32, %first: i32, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %claim_start: i32) attributes {noinline = false} {
%cst = arith.constant dense<256> : tensor<256xi64, #blocked>
%cst_0 = arith.constant dense<512> : tensor<256xi64, #blocked>
%cst_1 = arith.constant dense<true> : tensor<256xi1, #blocked>
%c768_i64 = arith.constant 768 : i64
%slot = tt.get_program_id x : i32
%row = tt.addptr %ids, %offset : !tt.ptr<i32>, i32
%row_2 = tt.addptr %row, %slot : !tt.ptr<i32>, i32
%row_3 = tt.load %row_2 : !tt.ptr<i32>
%row_4 = arith.subi %row_3, %first : i32
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%col_5 = arith.extsi %col : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%src_base = arith.extsi %slot : i32 to i64
%src_base_6 = arith.muli %src_base, %c768_i64 : i64
%dst_base = arith.extsi %row_4 : i32 to i64
%dst_base_7 = arith.muli %dst_base, %c768_i64 : i64
%val = tt.addptr %src, %src_base_6 : !tt.ptr<f16>, i64
%val_8 = tt.splat %val : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_9 = tt.addptr %val_8, %col_5 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_10 = tt.load %val_9 : tensor<256x!tt.ptr<f16>, #blocked>
%0 = tt.addptr %dst, %dst_base_7 : !tt.ptr<f16>, i64
%1 = tt.splat %0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %col_5 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%3 = tt.atomic_rmw fadd, relaxed, gpu, %2, %val_10, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%index = arith.addi %col_5, %cst : tensor<256xi64, #blocked>
%val_11 = tt.addptr %val_8, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_12 = tt.load %val_11 : tensor<256x!tt.ptr<f16>, #blocked>
%4 = tt.addptr %1, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %4, %val_12, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%index_13 = arith.addi %col_5, %cst_0 : tensor<256xi64, #blocked>
%val_14 = tt.addptr %val_8, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_15 = tt.load %val_14 : tensor<256x!tt.ptr<f16>, #blocked>
%6 = tt.addptr %1, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %6, %val_15, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.return
}
}"#

private def directPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_value_rows(
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_0,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_1,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_2,
.param .u32 _scatter_value_rows_param_3,
.param .u32 _scatter_value_rows_param_4,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_5,
.param .u32 _scatter_value_rows_param_6,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_7,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_8
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<13>;
.reg .b32 %r<12>;
.reg .b64 %rd<15>;
$L__func_begin0:
ld.param.b64 %rd8, [_scatter_value_rows_param_0];
$L__tmp0:
mov.u32 %r5, %ctaid.x;
ld.param.b64 %rd9, [_scatter_value_rows_param_1];
ld.param.b64 %rd10, [_scatter_value_rows_param_2];
ld.param.b32 %r6, [_scatter_value_rows_param_3];
mad.wide.s32 %rd11, %r6, 4, %rd9;
ld.param.b32 %r7, [_scatter_value_rows_param_4];
mad.wide.u32 %rd1, %r5, 4, %rd11;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
sub.s32 %r8, %r1, %r7;
mov.u32 %r9, %tid.x;
shl.b32 %r10, %r9, 1;
and.b32 %r11, %r10, 254;
mad.wide.u32 %rd12, %r5, 1536, %rd8;
mul.wide.u32 %rd13, %r11, 2;
add.s64 %rd2, %rd12, %rd13;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mov.b32 {%rs3, %rs4}, %r2;
mad.wide.s32 %rd14, %r8, 1536, %rd10;
add.s64 %rd3, %rd14, %rd13;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd4 + 0 ];
mov.b32 {%rs7, %rs8}, %r3;
add.s64 %rd5, %rd3, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd6 + 0 ];
mov.b32 {%rs11, %rs12}, %r4;
add.s64 %rd7, %rd3, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def markedIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_value_rows(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %offset: i32, %first: i32, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %claim_start: i32) attributes {noinline = false} {
%cst = arith.constant dense<256> : tensor<256xi64, #blocked>
%cst_0 = arith.constant dense<512> : tensor<256xi64, #blocked>
%cst_1 = arith.constant dense<true> : tensor<256xi1, #blocked>
%true = arith.constant true
%c768_i64 = arith.constant 768 : i64
%slot = tt.get_program_id x : i32
%row = tt.addptr %ids, %offset : !tt.ptr<i32>, i32
%row_2 = tt.addptr %row, %slot : !tt.ptr<i32>, i32
%row_3 = tt.load %row_2 : !tt.ptr<i32>
%row_4 = arith.subi %row_3, %first : i32
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%col_5 = arith.extsi %col : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%src_base = arith.extsi %slot : i32 to i64
%src_base_6 = arith.muli %src_base, %c768_i64 : i64
%dst_base = arith.extsi %row_4 : i32 to i64
%dst_base_7 = arith.muli %dst_base, %c768_i64 : i64
%val = tt.addptr %src, %src_base_6 : !tt.ptr<f16>, i64
%val_8 = tt.splat %val : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_9 = tt.addptr %val_8, %col_5 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_10 = tt.load %val_9 : tensor<256x!tt.ptr<f16>, #blocked>
%0 = tt.addptr %dst, %dst_base_7 : !tt.ptr<f16>, i64
%1 = tt.splat %0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %col_5 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%3 = tt.atomic_rmw fadd, relaxed, gpu, %2, %val_10, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%index = arith.addi %col_5, %cst : tensor<256xi64, #blocked>
%val_11 = tt.addptr %val_8, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_12 = tt.load %val_11 : tensor<256x!tt.ptr<f16>, #blocked>
%4 = tt.addptr %1, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %4, %val_12, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%index_13 = arith.addi %col_5, %cst_0 : tensor<256xi64, #blocked>
%val_14 = tt.addptr %val_8, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_15 = tt.load %val_14 : tensor<256x!tt.ptr<f16>, #blocked>
%6 = tt.addptr %1, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %6, %val_15, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%8 = tt.addptr %touched, %row_4 : !tt.ptr<i32>, i32
%9 = arith.addi %claim_start, %slot : i32
%10 = arith.addi %9, 1 : i32
%11 = tt.atomic_rmw max, relaxed, gpu, %8, %10, %true : (!tt.ptr<i32>, i32, i1) -> i32
tt.return
}
}"#

private def markedPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_value_rows(
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_0,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_1,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_2,
.param .u32 _scatter_value_rows_param_3,
.param .u32 _scatter_value_rows_param_4,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_5,
.param .u32 _scatter_value_rows_param_6,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_7,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_8
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<13>;
.reg .b32 %r<17>;
.reg .b64 %rd<17>;
$L__func_begin0:
ld.param.b64 %rd9, [_scatter_value_rows_param_0];
$L__tmp0:
mov.u32 %r7, %ctaid.x;
ld.param.b64 %rd10, [_scatter_value_rows_param_1];
ld.param.b64 %rd11, [_scatter_value_rows_param_2];
ld.param.b32 %r8, [_scatter_value_rows_param_3];
mad.wide.s32 %rd12, %r8, 4, %rd10;
ld.param.b32 %r9, [_scatter_value_rows_param_4];
mad.wide.u32 %rd1, %r7, 4, %rd12;
ld.param.b64 %rd13, [_scatter_value_rows_param_5];
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
ld.param.b32 %r10, [_scatter_value_rows_param_6];
sub.s32 %r11, %r1, %r9;
mov.u32 %r12, %tid.x;
shl.b32 %r13, %r12, 1;
and.b32 %r14, %r13, 254;
mad.wide.u32 %rd14, %r7, 1536, %rd9;
mul.wide.u32 %rd15, %r14, 2;
add.s64 %rd2, %rd14, %rd15;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mov.b32 {%rs3, %rs4}, %r2;
mad.wide.s32 %rd16, %r11, 1536, %rd11;
add.s64 %rd3, %rd16, %rd15;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd4 + 0 ];
mov.b32 {%rs7, %rs8}, %r3;
add.s64 %rd5, %rd3, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd6 + 0 ];
mov.b32 {%rs11, %rs12}, %r4;
add.s64 %rd7, %rd3, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
mad.wide.s32 %rd8, %r11, 4, %rd13;
add.s32 %r15, %r7, %r10;
add.s32 %r6, %r15, 1;
and.b32 %r16, %r12, 127;
setp.eq.b32 %p2, %r16, 0;
mov.u32 %r5, 0x0;
@%p2 atom.global.gpu.relaxed.max.s32 %r5, [ %rd8 + 0 ], %r6;
ret;
$L__tmp1:
$L__func_end0:
}"#

private def scaleIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scale_claimed(%grad: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %asked: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %own: i32, %count: i32, %lo: i32, %first: i32) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<1> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_3 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%cst_4 = arith.constant dense<0.000000e+00> : tensor<4x256xf16, #blocked>
%slot = tt.get_program_id x : i32
%slot_5 = arith.muli %slot, 4 : i32
%slot_6 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_7 = tt.splat %slot_5 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_8 = arith.addi %slot_7, %slot_6 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_9 = arith.cmpi slt, %slot_8, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local = tt.splat %own : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_10 = arith.cmpi slt, %slot_8, %local : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids = tt.splat %want : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids_11 = tt.splat %asked : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids_12 = arith.select %local_10, %ids, %ids_11 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%index = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_13 = arith.addi %index, %slot_8 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_14 = arith.subi %slot_8, %local : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_15 = arith.select %local_10, %index_13, %index_14 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row = tt.addptr %ids_12, %index_15 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_16 = tt.load %row, %live_9, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_17 = tt.splat %first : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_18 = arith.subi %row_16, %row_17 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim = tt.splat %touched : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim_19 = tt.addptr %claim, %row_18 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim_20 = tt.load %claim_19, %live_9, %cst : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%use = arith.addi %slot_8, %cst_0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%use_21 = arith.cmpi eq, %claim_20, %use : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%use_22 = arith.andi %live_9, %use_21 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%offsets = tt.expand_dims %row_18 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%offsets_23 = arith.muli %offsets, %cst_1 : tensor<4x1xi32, #blocked>
%offsets_24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offsets_25 = tt.expand_dims %offsets_24 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%offsets_26 = tt.broadcast %offsets_23 : tensor<4x1xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_27 = tt.broadcast %offsets_25 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_28 = arith.addi %offsets_26, %offsets_27 : tensor<4x256xi32, #blocked>
%value = tt.expand_dims %use_22 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%value_29 = tt.splat %grad : !tt.ptr<f16> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%value_30 = tt.addptr %value_29, %offsets_28 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%value_31 = tt.broadcast %value : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
%value_32 = tt.load %value_30, %value_31, %cst_4 : tensor<4x256x!tt.ptr<f16>, #blocked>
%0 = tt.elementwise_inline_asm "{ .reg .f32 wide; cvt.f32.f16 wide, $1; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 $0, wide; }" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %value_32 : tensor<4x256xf16, #blocked> -> tensor<4x256xf16, #blocked>
tt.store %value_30, %0, %value_31 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_33 = arith.addi %offsets_28, %cst_2 : tensor<4x256xi32, #blocked>
%value_34 = tt.addptr %value_29, %offsets_33 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%value_35 = tt.load %value_34, %value_31, %cst_4 : tensor<4x256x!tt.ptr<f16>, #blocked>
%1 = tt.elementwise_inline_asm "{ .reg .f32 wide; cvt.f32.f16 wide, $1; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 $0, wide; }" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %value_35 : tensor<4x256xf16, #blocked> -> tensor<4x256xf16, #blocked>
tt.store %value_34, %1, %value_31 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_36 = arith.addi %offsets_28, %cst_3 : tensor<4x256xi32, #blocked>
%value_37 = tt.addptr %value_29, %offsets_36 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%value_38 = tt.load %value_37, %value_31, %cst_4 : tensor<4x256x!tt.ptr<f16>, #blocked>
%2 = tt.elementwise_inline_asm "{ .reg .f32 wide; cvt.f32.f16 wide, $1; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 $0, wide; }" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %value_38 : tensor<4x256xf16, #blocked> -> tensor<4x256xf16, #blocked>
tt.store %value_37, %2, %value_31 : tensor<4x256x!tt.ptr<f16>, #blocked>
tt.return
}
}"#

private def scalePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scale_claimed(
.param .u64 .ptr .global .align 1 _scale_claimed_param_0,
.param .u64 .ptr .global .align 1 _scale_claimed_param_1,
.param .u64 .ptr .global .align 1 _scale_claimed_param_2,
.param .u64 .ptr .global .align 1 _scale_claimed_param_3,
.param .u32 _scale_claimed_param_4,
.param .u32 _scale_claimed_param_5,
.param .u32 _scale_claimed_param_6,
.param .u32 _scale_claimed_param_7,
.param .u64 .ptr .global .align 1 _scale_claimed_param_8,
.param .u64 .ptr .global .align 1 _scale_claimed_param_9
)
.reqntid 128
{
.reg .pred %p<5>;
.reg .b16 %rs<49>;
.reg .b32 %r<48>;
.reg .b64 %rd<11>;
$L__func_begin0:
ld.param.b64 %rd6, [_scale_claimed_param_0];
ld.param.b64 %rd7, [_scale_claimed_param_1];
$L__tmp0:
mov.u32 %r28, %ctaid.x;
shl.b32 %r29, %r28, 2;
ld.param.b64 %rd8, [_scale_claimed_param_2];
ld.param.b64 %rd9, [_scale_claimed_param_3];
mov.u32 %r30, %tid.x;
ld.param.b32 %r31, [_scale_claimed_param_4];
bfe.u32 %r32, %r30, 5, 2;
ld.param.b32 %r33, [_scale_claimed_param_5];
or.b32 %r34, %r32, %r29;
ld.param.b32 %r35, [_scale_claimed_param_6];
setp.lt.s32 %p1, %r34, %r33;
ld.param.b32 %r36, [_scale_claimed_param_7];
setp.lt.s32 %p3, %r34, %r31;
selp.b64 %rd10, %rd8, %rd9, %p3;
neg.s32 %r37, %r31;
selp.b32 %r38, %r35, %r37, %p3;
add.s32 %r39, %r38, %r34;
mad.wide.s32 %rd1, %r39, 4, %rd10;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
sub.s32 %r40, %r1, %r36;
mad.wide.s32 %rd2, %r40, 4, %rd7;
mov.u32 %r2, 0x0;
@%p1 ld.global.b32 { %r2 }, [ %rd2 + 0 ];
add.s32 %r41, %r34, 1;
setp.eq.b32 %p4, %r2, %r41;
and.pred %p2, %p1, %p4;
$L__tmp1:
mul.lo.s32 %r42, %r40, 768;
shl.b32 %r43, %r30, 3;
and.b32 %r44, %r43, 248;
or.b32 %r45, %r42, %r44;
mad.wide.s32 %rd3, %r45, 2, %rd6;
mov.b32 %r7, 0;
mov.u32 %r3, %r7;
mov.u32 %r4, %r7;
mov.u32 %r5, %r7;
mov.u32 %r6, %r7;
@%p2 ld.global.v4.b32 { %r3, %r4, %r5, %r6 }, [ %rd3 + 0 ];
mov.b32 {%rs2, %rs4}, %r3;
mov.b32 {%rs6, %rs8}, %r4;
mov.b32 {%rs10, %rs12}, %r5;
mov.b32 {%rs14, %rs16}, %r6;
$L__tmp2:
{ .reg .f32 wide; cvt.f32.f16 wide, %rs2; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs1, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs4; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs3, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs6; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs5, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs8; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs7, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs10; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs9, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs12; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs11, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs14; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs13, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs16; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs15, wide; }
$L__tmp3:
mov.b32 %r8, {%rs1, %rs3};
mov.b32 %r9, {%rs5, %rs7};
mov.b32 %r10, {%rs9, %rs11};
mov.b32 %r11, {%rs13, %rs15};
@%p2 st.global.v4.b32 [ %rd3 + 0 ], { %r8, %r9, %r10, %r11 };
add.s32 %r46, %r45, 256;
mad.wide.s32 %rd4, %r46, 2, %rd6;
mov.u32 %r12, %r7;
mov.u32 %r13, %r7;
mov.u32 %r14, %r7;
mov.u32 %r15, %r7;
@%p2 ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd4 + 0 ];
mov.b32 {%rs18, %rs20}, %r12;
mov.b32 {%rs22, %rs24}, %r13;
mov.b32 {%rs26, %rs28}, %r14;
mov.b32 {%rs30, %rs32}, %r15;
$L__tmp4:
{ .reg .f32 wide; cvt.f32.f16 wide, %rs18; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs17, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs20; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs19, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs22; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs21, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs24; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs23, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs26; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs25, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs28; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs27, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs30; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs29, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs32; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs31, wide; }
$L__tmp5:
mov.b32 %r16, {%rs17, %rs19};
mov.b32 %r17, {%rs21, %rs23};
mov.b32 %r18, {%rs25, %rs27};
mov.b32 %r19, {%rs29, %rs31};
@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r16, %r17, %r18, %r19 };
add.s32 %r47, %r45, 512;
mad.wide.s32 %rd5, %r47, 2, %rd6;
mov.u32 %r20, %r7;
mov.u32 %r21, %r7;
mov.u32 %r22, %r7;
mov.u32 %r23, %r7;
@%p2 ld.global.v4.b32 { %r20, %r21, %r22, %r23 }, [ %rd5 + 0 ];
mov.b32 {%rs34, %rs36}, %r20;
mov.b32 {%rs38, %rs40}, %r21;
mov.b32 {%rs42, %rs44}, %r22;
mov.b32 {%rs46, %rs48}, %r23;
$L__tmp6:
{ .reg .f32 wide; cvt.f32.f16 wide, %rs34; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs33, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs36; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs35, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs38; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs37, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs40; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs39, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs42; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs41, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs44; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs43, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs46; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs45, wide; }
{ .reg .f32 wide; cvt.f32.f16 wide, %rs48; mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 %rs47, wide; }
$L__tmp7:
mov.b32 %r24, {%rs33, %rs35};
mov.b32 %r25, {%rs37, %rs39};
mov.b32 %r26, {%rs41, %rs43};
mov.b32 %r27, {%rs45, %rs47};
@%p2 st.global.v4.b32 [ %rd5 + 0 ], { %r24, %r25, %r26, %r27 };
ret;
$L__tmp8:
$L__func_end0:
}"#

private def clearIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_clear_claimed(%grad: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %want: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %asked: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %own: i32, %count: i32, %lo: i32, %first: i32) attributes {noinline = false} {
%cst = arith.constant dense<1> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_1 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_2 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%cst_3 = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_4 = arith.constant dense<0> : tensor<4xi32, #blocked1>
%cst_5 = arith.constant dense<0.000000e+00> : tensor<4x256xf16, #blocked>
%cst_6 = arith.constant dense<1> : tensor<4xi32, #blocked1>
%slot = tt.get_program_id x : i32
%slot_7 = arith.muli %slot, 4 : i32
%slot_8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_9 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked1>
%slot_10 = tt.splat %slot_7 : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_11 = tt.splat %slot_7 : i32 -> tensor<4xi32, #blocked1>
%slot_12 = arith.addi %slot_10, %slot_8 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%slot_13 = arith.addi %slot_11, %slot_9 : tensor<4xi32, #blocked1>
%live = tt.splat %count : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_14 = tt.splat %count : i32 -> tensor<4xi32, #blocked1>
%live_15 = arith.cmpi slt, %slot_12, %live : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%live_16 = arith.cmpi slt, %slot_13, %live_14 : tensor<4xi32, #blocked1>
%local = tt.splat %own : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_17 = tt.splat %own : i32 -> tensor<4xi32, #blocked1>
%local_18 = arith.cmpi slt, %slot_12, %local : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%local_19 = arith.cmpi slt, %slot_13, %local_17 : tensor<4xi32, #blocked1>
%ids = tt.splat %want : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids_20 = tt.splat %want : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%ids_21 = tt.splat %asked : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids_22 = tt.splat %asked : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%ids_23 = arith.select %local_18, %ids, %ids_21 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%ids_24 = arith.select %local_19, %ids_20, %ids_22 : tensor<4xi1, #blocked1>, tensor<4x!tt.ptr<i32>, #blocked1>
%index = tt.splat %lo : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_25 = tt.splat %lo : i32 -> tensor<4xi32, #blocked1>
%index_26 = arith.addi %index, %slot_12 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_27 = arith.addi %index_25, %slot_13 : tensor<4xi32, #blocked1>
%index_28 = arith.subi %slot_12, %local : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_29 = arith.subi %slot_13, %local_17 : tensor<4xi32, #blocked1>
%index_30 = arith.select %local_18, %index_26, %index_28 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%index_31 = arith.select %local_19, %index_27, %index_29 : tensor<4xi1, #blocked1>, tensor<4xi32, #blocked1>
%row = tt.addptr %ids_23, %index_30 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_32 = tt.addptr %ids_24, %index_31 : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
%row_33 = tt.load %row, %live_15, %cst_3 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_34 = tt.load %row_32, %live_16, %cst_4 : tensor<4x!tt.ptr<i32>, #blocked1>
%row_35 = tt.splat %first : i32 -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_36 = tt.splat %first : i32 -> tensor<4xi32, #blocked1>
%row_37 = arith.subi %row_33, %row_35 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%row_38 = arith.subi %row_34, %row_36 : tensor<4xi32, #blocked1>
%claim = tt.splat %touched : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim_39 = tt.splat %touched : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked1>
%claim_40 = tt.addptr %claim, %row_37 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim_41 = tt.addptr %claim_39, %row_38 : tensor<4x!tt.ptr<i32>, #blocked1>, tensor<4xi32, #blocked1>
%claim_42 = tt.load %claim_40, %live_15, %cst_3 : tensor<4x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked}>>
%claim_43 = tt.load %claim_41, %live_16, %cst_4 : tensor<4x!tt.ptr<i32>, #blocked1>
%use = arith.addi %slot_12, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%use_44 = arith.addi %slot_13, %cst_6 : tensor<4xi32, #blocked1>
%use_45 = arith.cmpi eq, %claim_42, %use : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%use_46 = arith.cmpi eq, %claim_43, %use_44 : tensor<4xi32, #blocked1>
%use_47 = arith.andi %live_15, %use_45 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>
%use_48 = arith.andi %live_16, %use_46 : tensor<4xi1, #blocked1>
%offsets = tt.expand_dims %row_37 {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%offsets_49 = arith.muli %offsets, %cst_0 : tensor<4x1xi32, #blocked>
%offsets_50 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offsets_51 = tt.expand_dims %offsets_50 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%offsets_52 = tt.broadcast %offsets_49 : tensor<4x1xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_53 = tt.broadcast %offsets_51 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%offsets_54 = arith.addi %offsets_52, %offsets_53 : tensor<4x256xi32, #blocked>
%0 = tt.expand_dims %use_47 {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi1, #blocked>
%1 = tt.splat %grad : !tt.ptr<f16> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %offsets_54 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%3 = tt.broadcast %0 : tensor<4x1xi1, #blocked> -> tensor<4x256xi1, #blocked>
tt.store %2, %cst_5, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_55 = arith.addi %offsets_54, %cst_1 : tensor<4x256xi32, #blocked>
%4 = tt.addptr %1, %offsets_55 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %4, %cst_5, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
%offsets_56 = arith.addi %offsets_54, %cst_2 : tensor<4x256xi32, #blocked>
%5 = tt.addptr %1, %offsets_56 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
tt.store %5, %cst_5, %3 : tensor<4x256x!tt.ptr<f16>, #blocked>
gpu.barrier
tt.store %claim_41, %cst_4, %use_48 : tensor<4x!tt.ptr<i32>, #blocked1>
tt.return
}
}"#

private def clearPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _clear_claimed(
.param .u64 .ptr .global .align 1 _clear_claimed_param_0,
.param .u64 .ptr .global .align 1 _clear_claimed_param_1,
.param .u64 .ptr .global .align 1 _clear_claimed_param_2,
.param .u64 .ptr .global .align 1 _clear_claimed_param_3,
.param .u32 _clear_claimed_param_4,
.param .u32 _clear_claimed_param_5,
.param .u32 _clear_claimed_param_6,
.param .u32 _clear_claimed_param_7,
.param .u64 .ptr .global .align 1 _clear_claimed_param_8,
.param .u64 .ptr .global .align 1 _clear_claimed_param_9
)
.reqntid 128
{
.reg .pred %p<11>;
.reg .b32 %r<33>;
.reg .b64 %rd<14>;
$L__func_begin0:
ld.param.b64 %rd8, [_clear_claimed_param_0];
ld.param.b64 %rd9, [_clear_claimed_param_1];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
shl.b32 %r7, %r6, 2;
ld.param.b64 %rd10, [_clear_claimed_param_2];
ld.param.b64 %rd11, [_clear_claimed_param_3];
mov.u32 %r8, %tid.x;
ld.param.b32 %r9, [_clear_claimed_param_4];
bfe.u32 %r10, %r8, 5, 2;
ld.param.b32 %r11, [_clear_claimed_param_5];
and.b32 %r12, %r8, 3;
ld.param.b32 %r13, [_clear_claimed_param_6];
or.b32 %r14, %r10, %r7;
ld.param.b32 %r15, [_clear_claimed_param_7];
or.b32 %r16, %r7, %r12;
setp.lt.s32 %p1, %r14, %r11;
setp.lt.s32 %p2, %r16, %r11;
setp.lt.s32 %p5, %r14, %r9;
setp.lt.s32 %p6, %r16, %r9;
selp.b64 %rd12, %rd10, %rd11, %p5;
selp.b64 %rd13, %rd10, %rd11, %p6;
neg.s32 %r17, %r9;
selp.b32 %r18, %r13, %r17, %p5;
add.s32 %r19, %r18, %r14;
selp.b32 %r20, %r13, %r17, %p6;
add.s32 %r21, %r20, %r16;
mad.wide.s32 %rd1, %r19, 4, %rd12;
mad.wide.s32 %rd2, %r21, 4, %rd13;
mov.u32 %r1, 0x0;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mov.u32 %r2, 0x0;
@%p2 ld.global.b32 { %r2 }, [ %rd2 + 0 ];
sub.s32 %r22, %r1, %r15;
sub.s32 %r23, %r2, %r15;
mad.wide.s32 %rd3, %r22, 4, %rd9;
mad.wide.s32 %rd4, %r23, 4, %rd9;
mov.u32 %r3, 0x0;
@%p1 ld.global.b32 { %r3 }, [ %rd3 + 0 ];
mov.u32 %r4, 0x0;
@%p2 ld.global.b32 { %r4 }, [ %rd4 + 0 ];
add.s32 %r24, %r14, 1;
add.s32 %r25, %r16, 1;
setp.eq.b32 %p7, %r3, %r24;
setp.eq.b32 %p8, %r4, %r25;
and.pred %p3, %p1, %p7;
$L__tmp1:
mul.lo.s32 %r26, %r22, 768;
shl.b32 %r27, %r8, 3;
and.b32 %r28, %r27, 248;
or.b32 %r29, %r26, %r28;
mad.wide.s32 %rd5, %r29, 2, %rd8;
mov.b32 %r5, 0;
@%p3 st.global.v4.b32 [ %rd5 + 0 ], { %r5, %r5, %r5, %r5 };
add.s32 %r30, %r29, 256;
mad.wide.s32 %rd6, %r30, 2, %rd8;
@%p3 st.global.v4.b32 [ %rd6 + 0 ], { %r5, %r5, %r5, %r5 };
add.s32 %r31, %r29, 512;
mad.wide.s32 %rd7, %r31, 2, %rd8;
@%p3 st.global.v4.b32 [ %rd7 + 0 ], { %r5, %r5, %r5, %r5 };
bar.sync 0;
and.b32 %r32, %r8, 124;
setp.eq.b32 %p9, %r32, 0;
and.pred %p10, %p9, %p8;
and.pred %p4, %p2, %p10;
@%p4 st.global.b32 [ %rd4 + 0 ], { %r5 };
ret;
$L__tmp2:
$L__func_end0:
}"#

private def ownIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_value_rows(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %offset: i32, %first: i32, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %claim_start: i32) attributes {noinline = false} {
%cst = arith.constant dense<256> : tensor<256xi64, #blocked>
%cst_0 = arith.constant dense<512> : tensor<256xi64, #blocked>
%cst_1 = arith.constant dense<true> : tensor<256xi1, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<256xf16, #blocked>
%c768_i64 = arith.constant 768 : i64
%slot = tt.get_program_id x : i32
%row = tt.addptr %ids, %offset : !tt.ptr<i32>, i32
%row_3 = tt.addptr %row, %slot : !tt.ptr<i32>, i32
%row_4 = tt.load %row_3 : !tt.ptr<i32>
%row_5 = arith.subi %row_4, %first : i32
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%col_6 = arith.extsi %col : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%src_base = arith.extsi %row_5 : i32 to i64
%src_base_7 = arith.muli %src_base, %c768_i64 : i64
%val = tt.addptr %src, %src_base_7 : !tt.ptr<f16>, i64
%val_8 = tt.splat %val : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_9 = tt.addptr %val_8, %col_6 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_10 = tt.load %val_9 : tensor<256x!tt.ptr<f16>, #blocked>
%0 = tt.addptr %dst, %src_base_7 : !tt.ptr<f16>, i64
%1 = tt.splat %0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %col_6 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%3 = tt.atomic_rmw fadd, relaxed, gpu, %2, %val_10, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_9, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
%index = arith.addi %col_6, %cst : tensor<256xi64, #blocked>
%val_11 = tt.addptr %val_8, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_12 = tt.load %val_11 : tensor<256x!tt.ptr<f16>, #blocked>
%4 = tt.addptr %1, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %4, %val_12, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_11, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
%index_13 = arith.addi %col_6, %cst_0 : tensor<256xi64, #blocked>
%val_14 = tt.addptr %val_8, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_15 = tt.load %val_14 : tensor<256x!tt.ptr<f16>, #blocked>
%6 = tt.addptr %1, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %6, %val_15, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_14, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
tt.return
}
}"#

private def ownPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_value_rows(
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_0,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_1,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_2,
.param .u32 _scatter_value_rows_param_3,
.param .u32 _scatter_value_rows_param_4,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_5,
.param .u32 _scatter_value_rows_param_6,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_7,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_8
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<13>;
.reg .b32 %r<13>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd8, [_scatter_value_rows_param_0];
$L__tmp0:
mov.u32 %r6, %ctaid.x;
ld.param.b64 %rd9, [_scatter_value_rows_param_1];
ld.param.b64 %rd10, [_scatter_value_rows_param_2];
ld.param.b32 %r7, [_scatter_value_rows_param_3];
mad.wide.s32 %rd11, %r7, 4, %rd9;
ld.param.b32 %r8, [_scatter_value_rows_param_4];
mad.wide.u32 %rd1, %r6, 4, %rd11;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
sub.s32 %r9, %r1, %r8;
mov.u32 %r10, %tid.x;
shl.b32 %r11, %r10, 1;
and.b32 %r12, %r11, 254;
mul.wide.s32 %rd12, %r9, 1536;
add.s64 %rd13, %rd8, %rd12;
mul.wide.u32 %rd14, %r12, 2;
add.s64 %rd2, %rd13, %rd14;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mov.b32 {%rs3, %rs4}, %r2;
add.s64 %rd15, %rd10, %rd12;
add.s64 %rd3, %rd15, %rd14;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
mov.b32 %r3, 0;
st.global.b32 [ %rd2 + 0 ], { %r3 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd4 + 0 ];
mov.b32 {%rs7, %rs8}, %r4;
add.s64 %rd5, %rd3, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
st.global.b32 [ %rd4 + 0 ], { %r3 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd6 + 0 ];
mov.b32 {%rs11, %rs12}, %r5;
add.s64 %rd7, %rd3, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
st.global.b32 [ %rd6 + 0 ], { %r3 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def ownMarkedIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_value_rows(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %offset: i32, %first: i32, %touched: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %claim_start: i32) attributes {noinline = false} {
%cst = arith.constant dense<256> : tensor<256xi64, #blocked>
%cst_0 = arith.constant dense<512> : tensor<256xi64, #blocked>
%cst_1 = arith.constant dense<true> : tensor<256xi1, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<256xf16, #blocked>
%true = arith.constant true
%c768_i64 = arith.constant 768 : i64
%slot = tt.get_program_id x : i32
%row = tt.addptr %ids, %offset : !tt.ptr<i32>, i32
%row_3 = tt.addptr %row, %slot : !tt.ptr<i32>, i32
%row_4 = tt.load %row_3 : !tt.ptr<i32>
%row_5 = arith.subi %row_4, %first : i32
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%col_6 = arith.extsi %col : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
%src_base = arith.extsi %row_5 : i32 to i64
%src_base_7 = arith.muli %src_base, %c768_i64 : i64
%val = tt.addptr %src, %src_base_7 : !tt.ptr<f16>, i64
%val_8 = tt.splat %val : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%val_9 = tt.addptr %val_8, %col_6 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_10 = tt.load %val_9 : tensor<256x!tt.ptr<f16>, #blocked>
%0 = tt.addptr %dst, %src_base_7 : !tt.ptr<f16>, i64
%1 = tt.splat %0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%2 = tt.addptr %1, %col_6 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%3 = tt.atomic_rmw fadd, relaxed, gpu, %2, %val_10, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_9, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
%index = arith.addi %col_6, %cst : tensor<256xi64, #blocked>
%val_11 = tt.addptr %val_8, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_12 = tt.load %val_11 : tensor<256x!tt.ptr<f16>, #blocked>
%4 = tt.addptr %1, %index : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %4, %val_12, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_11, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
%index_13 = arith.addi %col_6, %cst_0 : tensor<256xi64, #blocked>
%val_14 = tt.addptr %val_8, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%val_15 = tt.load %val_14 : tensor<256x!tt.ptr<f16>, #blocked>
%6 = tt.addptr %1, %index_13 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %6, %val_15, %cst_1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.store %val_14, %cst_2 : tensor<256x!tt.ptr<f16>, #blocked>
%8 = tt.addptr %touched, %row_5 : !tt.ptr<i32>, i32
%9 = arith.addi %claim_start, %slot : i32
%10 = arith.addi %9, 1 : i32
%11 = tt.atomic_rmw max, relaxed, gpu, %8, %10, %true : (!tt.ptr<i32>, i32, i1) -> i32
tt.return
}
}"#

private def ownMarkedPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_value_rows(
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_0,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_1,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_2,
.param .u32 _scatter_value_rows_param_3,
.param .u32 _scatter_value_rows_param_4,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_5,
.param .u32 _scatter_value_rows_param_6,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_7,
.param .u64 .ptr .global .align 1 _scatter_value_rows_param_8
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<13>;
.reg .b32 %r<18>;
.reg .b64 %rd<18>;
$L__func_begin0:
ld.param.b64 %rd9, [_scatter_value_rows_param_0];
$L__tmp0:
mov.u32 %r8, %ctaid.x;
ld.param.b64 %rd10, [_scatter_value_rows_param_1];
ld.param.b64 %rd11, [_scatter_value_rows_param_2];
ld.param.b32 %r9, [_scatter_value_rows_param_3];
mad.wide.s32 %rd12, %r9, 4, %rd10;
ld.param.b32 %r10, [_scatter_value_rows_param_4];
mad.wide.u32 %rd1, %r8, 4, %rd12;
ld.param.b64 %rd13, [_scatter_value_rows_param_5];
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
ld.param.b32 %r11, [_scatter_value_rows_param_6];
sub.s32 %r12, %r1, %r10;
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
and.b32 %r15, %r14, 254;
mul.wide.s32 %rd14, %r12, 1536;
add.s64 %rd15, %rd9, %rd14;
mul.wide.u32 %rd16, %r15, 2;
add.s64 %rd2, %rd15, %rd16;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mov.b32 {%rs3, %rs4}, %r2;
add.s64 %rd17, %rd11, %rd14;
add.s64 %rd3, %rd17, %rd16;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
mov.b32 %r3, 0;
st.global.b32 [ %rd2 + 0 ], { %r3 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd4 + 0 ];
mov.b32 {%rs7, %rs8}, %r4;
add.s64 %rd5, %rd3, 512;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
st.global.b32 [ %rd4 + 0 ], { %r3 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd6 + 0 ];
mov.b32 {%rs11, %rs12}, %r5;
add.s64 %rd7, %rd3, 1024;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
st.global.b32 [ %rd6 + 0 ], { %r3 };
mad.wide.s32 %rd8, %r12, 4, %rd13;
add.s32 %r16, %r8, %r11;
add.s32 %r7, %r16, 1;
and.b32 %r17, %r13, 127;
setp.eq.b32 %p2, %r17, 0;
mov.u32 %r6, 0x0;
@%p2 atom.global.gpu.relaxed.max.s32 %r6, [ %rd8 + 0 ], %r7;
ret;
$L__tmp1:
$L__func_end0:
}"#

private def recognizesCode (c : Code) (ir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR c.ir == some (ir.splitOn "\n") &&
    QKVTTIR.normalizePTX c.ptx == ptx.splitOn "\n"

def recognized (a : Artifacts) : Bool :=
  recognizesCode a.reference referenceIR referencePTX &&
    recognizesCode a.direct directIR directPTX &&
    recognizesCode a.marked markedIR markedPTX &&
    recognizesCode a.own ownIR ownPTX &&
    recognizesCode a.ownMarked ownMarkedIR ownMarkedPTX &&
    recognizesCode a.scale scaleIR scalePTX &&
    recognizesCode a.clear clearIR clearPTX

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  normalization : clearSelected = clearDense
  update : ∀ p (s : Schedule p) arithmetic launch memory,
    event selected p (AdamEntries.fused p.ids) s arithmetic launch memory =
      event dense p (AdamEntries.staged p.ids) s arithmetic launch memory
  sourceUpdate : ∀ (s : SourceSlice) (peer : Fin s.asked.size → SparsePackets.Channel → Half)
      (schedule : (p : Pull) → Schedule p) ar launch memory,
    let candidate := s.deferredPull peer
    let reference := s.stagedPull peer
    (SparsePackets.valueOwnedCleared s.input (SparsePackets.valuePeersCleared s.input),
      event (clearSelected memory.prior) candidate (AdamEntries.fused candidate.ids)
        (schedule candidate) ar launch memory) =
    (SparsePackets.cleared s.input,
      event dense reference (AdamEntries.staged reference.ids) (schedule reference) ar launch memory)

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then some ⟨h, workspace_normalization_equal, event_equal, source_event_exact⟩ else none

def Certificate.normalizer {a : Artifacts} (_checked : Certificate a) : WorkspaceNormalize :=
  if recognized a then clearIndexedWorkspace else clearDense

theorem Certificate.normalizer_eq {a : Artifacts} (checked : Certificate a) :
    checked.normalizer = clearSelected := by
  simp only [Certificate.normalizer, checked.matched, if_true, clear_indexed_workspace_equal]

#axiomsClean Certificate.normalizer_eq
#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
