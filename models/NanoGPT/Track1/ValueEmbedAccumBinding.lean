import models.NanoGPT.Track1.ValueEmbedAccum
import models.NanoGPT.Track1.QKVTTIRBinding

/-! SM90a output for the original per-plane launch and the fused
four-warp launch, with the ordinary and debug variants. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueEmbedAccum

structure Code where
  ir : String
  ptx : String
  deriving BEq

structure Artifacts where
  reference : Code
  four : Code
  referenceDebug : Code
  fourDebug : Code
  deriving BEq

private def referenceIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_row_scatter_add_kernel(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %token_ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<true> : tensor<256xi1, #blocked>
%pid = tt.get_program_id x : i32
%token = tt.addptr %token_ids, %pid : !tt.ptr<i32>, i32
%token_0 = tt.load %token evictionPolicy = evict_last : !tt.ptr<i32>
%o = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%src_base = arith.muli %pid, 768 : i32
%dst_base = arith.muli %token_0, 768 : i32
%value = tt.splat %src_base : i32 -> tensor<256xi32, #blocked>
%value_1 = arith.addi %value, %o : tensor<256xi32, #blocked>
%value_2 = tt.splat %src : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%value_3 = tt.addptr %value_2, %value_1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_4 = tt.load %value_3 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_5 = arith.extf %value_4 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%0 = tt.splat %dst_base : i32 -> tensor<256xi32, #blocked>
%1 = arith.addi %0, %o : tensor<256xi32, #blocked>
%2 = tt.splat %out : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%4 = arith.truncf %value_5 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %3, %4, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%value_6 = arith.addi %src_base, 256 : i32
%value_7 = tt.splat %value_6 : i32 -> tensor<256xi32, #blocked>
%value_8 = arith.addi %value_7, %o : tensor<256xi32, #blocked>
%value_9 = tt.addptr %value_2, %value_8 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_10 = tt.load %value_9 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_11 = arith.extf %value_10 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%6 = arith.addi %dst_base, 256 : i32
%7 = tt.splat %6 : i32 -> tensor<256xi32, #blocked>
%8 = arith.addi %7, %o : tensor<256xi32, #blocked>
%9 = tt.addptr %2, %8 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%10 = arith.truncf %value_11 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%11 = tt.atomic_rmw fadd, relaxed, gpu, %9, %10, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%value_12 = arith.addi %src_base, 512 : i32
%value_13 = tt.splat %value_12 : i32 -> tensor<256xi32, #blocked>
%value_14 = arith.addi %value_13, %o : tensor<256xi32, #blocked>
%value_15 = tt.addptr %value_2, %value_14 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_16 = tt.load %value_15 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_17 = arith.extf %value_16 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%12 = arith.addi %dst_base, 512 : i32
%13 = tt.splat %12 : i32 -> tensor<256xi32, #blocked>
%14 = arith.addi %13, %o : tensor<256xi32, #blocked>
%15 = tt.addptr %2, %14 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%16 = arith.truncf %value_17 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%17 = tt.atomic_rmw fadd, relaxed, gpu, %15, %16, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.return
}
}"#

private def referencePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _row_scatter_add_kernel(
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_0,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_1,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_2,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_3,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_4
)
.reqntid 128
{
.reg .pred %p<2>;
.reg .b16 %rs<19>;
.reg .b32 %r<23>;
.reg .b64 %rd<15>;
$L__func_begin0:
ld.param.b64 %rd12, [_row_scatter_add_kernel_param_0];
ld.param.b64 %rd13, [_row_scatter_add_kernel_param_1];
$L__tmp0:
mov.u32 %r5, %ctaid.x;
ld.param.b64 %rd14, [_row_scatter_add_kernel_param_2];
mad.wide.u32 %rd1, %r5, 4, %rd13;
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
mov.u32 %r6, %tid.x;
shl.b32 %r7, %r6, 1;
and.b32 %r8, %r7, 254;
mul.lo.s32 %r9, %r5, 768;
mul.lo.s32 %r10, %r1, 768;
or.b32 %r11, %r8, %r9;
mad.wide.s32 %rd3, %r11, 2, %rd12;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd4, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r2 }, [ %rd3 + 0 ], %rd4;
mov.b32 {%rs13, %rs14}, %r2;
cvt.f32.bf16 %r12, %rs13;
cvt.f32.bf16 %r13, %rs14;
or.b32 %r14, %r10, %r8;
mad.wide.s32 %rd5, %r14, 2, %rd14;
cvt.rn.f16.f32 %rs3, %r12;
cvt.rn.f16.f32 %rs4, %r13;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd5 + 0 ], { %rs3, %rs4 };
add.s32 %r15, %r11, 256;
mad.wide.s32 %rd6, %r15, 2, %rd12;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd7, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r3 }, [ %rd6 + 0 ], %rd7;
mov.b32 {%rs15, %rs16}, %r3;
cvt.f32.bf16 %r16, %rs15;
cvt.f32.bf16 %r17, %rs16;
add.s32 %r18, %r14, 256;
mad.wide.s32 %rd8, %r18, 2, %rd14;
cvt.rn.f16.f32 %rs7, %r16;
cvt.rn.f16.f32 %rs8, %r17;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd8 + 0 ], { %rs7, %rs8 };
add.s32 %r19, %r11, 512;
mad.wide.s32 %rd9, %r19, 2, %rd12;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd10, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r4 }, [ %rd9 + 0 ], %rd10;
mov.b32 {%rs17, %rs18}, %r4;
cvt.f32.bf16 %r20, %rs17;
cvt.f32.bf16 %r21, %rs18;
add.s32 %r22, %r14, 512;
mad.wide.s32 %rd11, %r22, 2, %rd14;
cvt.rn.f16.f32 %rs11, %r20;
cvt.rn.f16.f32 %rs12, %r21;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd11 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def fourIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_four_planes(%g0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %token_ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<50304> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_3 = arith.constant dense<true> : tensor<4x256xi1, #blocked>
%cst_4 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_5 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%pid = tt.get_program_id x : i32
%token = tt.addptr %token_ids, %pid : !tt.ptr<i32>, i32
%token_6 = tt.load %token evictionPolicy = evict_last : !tt.ptr<i32>
%plane = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%g = arith.cmpi slt, %plane, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_7 = arith.cmpi eq, %plane, %cst_0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_8 = tt.splat %g0 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_9 = tt.splat %g1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_10 = arith.select %g_7, %g_8, %g_9 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_11 = arith.cmpi eq, %plane, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_12 = tt.splat %g2 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_13 = tt.splat %g3 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_14 = arith.select %g_11, %g_12, %g_13 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_15 = arith.select %g, %g_10, %g_14 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source = tt.expand_dims %g_15 {axis = 1 : i32} : tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%source_16 = arith.muli %pid, 768 : i32
%source_17 = tt.splat %source_16 : i32 -> tensor<4x1xi32, #blocked>
%source_18 = tt.addptr %source, %source_17 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%source_19 = tt.expand_dims %col {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%source_20 = tt.broadcast %source_18 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%source_21 = tt.broadcast %source_19 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%source_22 = tt.addptr %source_20, %source_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%target = tt.expand_dims %plane {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%target_23 = arith.muli %target, %cst_1 : tensor<4x1xi32, #blocked>
%target_24 = tt.splat %token_6 : i32 -> tensor<4x1xi32, #blocked>
%target_25 = arith.addi %target_23, %target_24 : tensor<4x1xi32, #blocked>
%target_26 = arith.muli %target_25, %cst_2 : tensor<4x1xi32, #blocked>
%target_27 = tt.splat %out : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked>
%target_28 = tt.addptr %target_27, %target_26 : tensor<4x1x!tt.ptr<f16>, #blocked>, tensor<4x1xi32, #blocked>
%target_29 = tt.broadcast %target_28 : tensor<4x1x!tt.ptr<f16>, #blocked> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%target_30 = tt.addptr %target_29, %source_21 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%value = tt.load %source_22 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_31 = arith.extf %value : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%0 = arith.truncf %value_31 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%1 = tt.atomic_rmw fadd, relaxed, gpu, %target_30, %0, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
%value_32 = tt.addptr %source_22, %cst_4 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%value_33 = tt.load %value_32 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_34 = arith.extf %value_33 : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%2 = tt.addptr %target_30, %cst_4 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%3 = arith.truncf %value_34 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %2, %3, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
%value_35 = tt.addptr %source_22, %cst_5 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%value_36 = tt.load %value_35 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_37 = arith.extf %value_36 : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%5 = tt.addptr %target_30, %cst_5 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%6 = arith.truncf %value_37 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %5, %6, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
tt.return
}
}"#

private def fourPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _scatter_four_planes(
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_0,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_1,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_2,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_3,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_4,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_5,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_6,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_7
)
.reqntid 128
{
.reg .pred %p<5>;
.reg .b16 %rs<73>;
.reg .b32 %r<46>;
.reg .b64 %rd<24>;
$L__func_begin0:
ld.param.b64 %rd12, [_scatter_four_planes_param_0];
ld.param.b64 %rd13, [_scatter_four_planes_param_1];
$L__tmp0:
mov.u32 %r14, %ctaid.x;
ld.param.b64 %rd14, [_scatter_four_planes_param_2];
ld.param.b64 %rd15, [_scatter_four_planes_param_4];
mad.wide.u32 %rd1, %r14, 4, %rd15;
ld.param.b64 %rd16, [_scatter_four_planes_param_3];
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
ld.param.b64 %rd17, [_scatter_four_planes_param_5];
mov.u32 %r15, %tid.x;
bfe.u32 %r16, %r15, 5, 2;
shl.b32 %r17, %r15, 3;
and.b32 %r18, %r17, 248;
setp.lt.u32 %p2, %r16, 2;
setp.eq.b32 %p3, %r16, 0;
selp.b64 %rd18, %rd12, %rd13, %p3;
setp.eq.b32 %p4, %r16, 2;
selp.b64 %rd19, %rd14, %rd16, %p4;
selp.b64 %rd20, %rd18, %rd19, %p2;
mul.lo.s32 %r19, %r14, 768;
mad.wide.s32 %rd21, %r19, 2, %rd20;
mul.wide.u32 %rd22, %r18, 2;
add.s64 %rd3, %rd21, %rd22;
mad.lo.s32 %r20, %r16, 50304, %r1;
mul.lo.s32 %r21, %r20, 768;
mad.wide.s32 %rd23, %r21, 2, %rd17;
add.s64 %rd5, %rd23, %rd22;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd4, 1.0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd3 + 0 ], %rd4;
mov.b32 {%rs49, %rs50}, %r2;
mov.b32 {%rs51, %rs52}, %r3;
mov.b32 {%rs53, %rs54}, %r4;
mov.b32 {%rs55, %rs56}, %r5;
cvt.f32.bf16 %r22, %rs49;
cvt.f32.bf16 %r23, %rs50;
cvt.f32.bf16 %r24, %rs51;
cvt.f32.bf16 %r25, %rs52;
cvt.f32.bf16 %r26, %rs53;
cvt.f32.bf16 %r27, %rs54;
cvt.f32.bf16 %r28, %rs55;
cvt.f32.bf16 %r29, %rs56;
cvt.rn.f16.f32 %rs9, %r22;
cvt.rn.f16.f32 %rs10, %r23;
cvt.rn.f16.f32 %rs11, %r24;
cvt.rn.f16.f32 %rs12, %r25;
cvt.rn.f16.f32 %rs13, %r26;
cvt.rn.f16.f32 %rs14, %r27;
cvt.rn.f16.f32 %rs15, %r28;
cvt.rn.f16.f32 %rs16, %r29;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
mov.u16 %rs3, 0x0;
mov.u16 %rs4, 0x0;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
mov.u16 %rs7, 0x0;
mov.u16 %rs8, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs1, %rs2, %rs3, %rs4, %rs5, %rs6, %rs7, %rs8 }, [ %rd5 + 0 ], { %rs9, %rs10, %rs11, %rs12, %rs13, %rs14, %rs15, %rs16 };
add.s64 %rd6, %rd3, 512;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd7, 1.0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd6 + 0 ], %rd7;
mov.b32 {%rs57, %rs58}, %r6;
mov.b32 {%rs59, %rs60}, %r7;
mov.b32 {%rs61, %rs62}, %r8;
mov.b32 {%rs63, %rs64}, %r9;
cvt.f32.bf16 %r30, %rs57;
cvt.f32.bf16 %r31, %rs58;
cvt.f32.bf16 %r32, %rs59;
cvt.f32.bf16 %r33, %rs60;
cvt.f32.bf16 %r34, %rs61;
cvt.f32.bf16 %r35, %rs62;
cvt.f32.bf16 %r36, %rs63;
cvt.f32.bf16 %r37, %rs64;
add.s64 %rd8, %rd5, 512;
cvt.rn.f16.f32 %rs25, %r30;
cvt.rn.f16.f32 %rs26, %r31;
cvt.rn.f16.f32 %rs27, %r32;
cvt.rn.f16.f32 %rs28, %r33;
cvt.rn.f16.f32 %rs29, %r34;
cvt.rn.f16.f32 %rs30, %r35;
cvt.rn.f16.f32 %rs31, %r36;
cvt.rn.f16.f32 %rs32, %r37;
mov.u16 %rs17, 0x0;
mov.u16 %rs18, 0x0;
mov.u16 %rs19, 0x0;
mov.u16 %rs20, 0x0;
mov.u16 %rs21, 0x0;
mov.u16 %rs22, 0x0;
mov.u16 %rs23, 0x0;
mov.u16 %rs24, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs17, %rs18, %rs19, %rs20, %rs21, %rs22, %rs23, %rs24 }, [ %rd8 + 0 ], { %rs25, %rs26, %rs27, %rs28, %rs29, %rs30, %rs31, %rs32 };
add.s64 %rd9, %rd3, 1024;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd10, 1.0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd9 + 0 ], %rd10;
mov.b32 {%rs65, %rs66}, %r10;
mov.b32 {%rs67, %rs68}, %r11;
mov.b32 {%rs69, %rs70}, %r12;
mov.b32 {%rs71, %rs72}, %r13;
cvt.f32.bf16 %r38, %rs65;
cvt.f32.bf16 %r39, %rs66;
cvt.f32.bf16 %r40, %rs67;
cvt.f32.bf16 %r41, %rs68;
cvt.f32.bf16 %r42, %rs69;
cvt.f32.bf16 %r43, %rs70;
cvt.f32.bf16 %r44, %rs71;
cvt.f32.bf16 %r45, %rs72;
add.s64 %rd11, %rd5, 1024;
cvt.rn.f16.f32 %rs41, %r38;
cvt.rn.f16.f32 %rs42, %r39;
cvt.rn.f16.f32 %rs43, %r40;
cvt.rn.f16.f32 %rs44, %r41;
cvt.rn.f16.f32 %rs45, %r42;
cvt.rn.f16.f32 %rs46, %r43;
cvt.rn.f16.f32 %rs47, %r44;
cvt.rn.f16.f32 %rs48, %r45;
mov.u16 %rs33, 0x0;
mov.u16 %rs34, 0x0;
mov.u16 %rs35, 0x0;
mov.u16 %rs36, 0x0;
mov.u16 %rs37, 0x0;
mov.u16 %rs38, 0x0;
mov.u16 %rs39, 0x0;
mov.u16 %rs40, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs33, %rs34, %rs35, %rs36, %rs37, %rs38, %rs39, %rs40 }, [ %rd11 + 0 ], { %rs41, %rs42, %rs43, %rs44, %rs45, %rs46, %rs47, %rs48 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def referenceDebugIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_row_scatter_add_kernel(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %token_ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<true> : tensor<256xi1, #blocked>
%pid = tt.get_program_id x : i32
%token = tt.addptr %token_ids, %pid : !tt.ptr<i32>, i32
%token_0 = tt.load %token evictionPolicy = evict_last : !tt.ptr<i32>
%0 = arith.cmpi sge, %token_0, 0 : i32
%1 = arith.cmpi slt, %token_0, 50304 : i32
%2 = arith.andi %0, %1 : i1
tt.assert %2, "token id out of range" : i1
%o = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
%src_base = arith.muli %pid, 768 : i32
%dst_base = arith.muli %token_0, 768 : i32
%value = tt.splat %src_base : i32 -> tensor<256xi32, #blocked>
%value_1 = arith.addi %value, %o : tensor<256xi32, #blocked>
%value_2 = tt.splat %src : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%value_3 = tt.addptr %value_2, %value_1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_4 = tt.load %value_3 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_5 = arith.extf %value_4 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%3 = tt.splat %dst_base : i32 -> tensor<256xi32, #blocked>
%4 = arith.addi %3, %o : tensor<256xi32, #blocked>
%5 = tt.splat %out : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%7 = arith.truncf %value_5 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%8 = tt.atomic_rmw fadd, relaxed, gpu, %6, %7, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%value_6 = arith.addi %src_base, 256 : i32
%value_7 = tt.splat %value_6 : i32 -> tensor<256xi32, #blocked>
%value_8 = arith.addi %value_7, %o : tensor<256xi32, #blocked>
%value_9 = tt.addptr %value_2, %value_8 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_10 = tt.load %value_9 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_11 = arith.extf %value_10 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%9 = arith.addi %dst_base, 256 : i32
%10 = tt.splat %9 : i32 -> tensor<256xi32, #blocked>
%11 = arith.addi %10, %o : tensor<256xi32, #blocked>
%12 = tt.addptr %5, %11 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%13 = arith.truncf %value_11 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%14 = tt.atomic_rmw fadd, relaxed, gpu, %12, %13, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%value_12 = arith.addi %src_base, 512 : i32
%value_13 = tt.splat %value_12 : i32 -> tensor<256xi32, #blocked>
%value_14 = arith.addi %value_13, %o : tensor<256xi32, #blocked>
%value_15 = tt.addptr %value_2, %value_14 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
%value_16 = tt.load %value_15 evictionPolicy = evict_first : tensor<256x!tt.ptr<bf16>, #blocked>
%value_17 = arith.extf %value_16 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%15 = arith.addi %dst_base, 512 : i32
%16 = tt.splat %15 : i32 -> tensor<256xi32, #blocked>
%17 = arith.addi %16, %o : tensor<256xi32, #blocked>
%18 = tt.addptr %5, %17 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
%19 = arith.truncf %value_17 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%20 = tt.atomic_rmw fadd, relaxed, gpu, %18, %19, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.return
}
}"#

private def referenceDebugPTX : String := r#".version 8.7
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
.global .align 1 .b8 assertMessage_0[22] = {116, 111, 107, 101, 110, 32, 105, 100, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101};
.visible .entry _row_scatter_add_kernel(
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_0,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_1,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_2,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_3,
.param .u64 .ptr .global .align 1 _row_scatter_add_kernel_param_4
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<19>;
.reg .b32 %r<21>;
.reg .b64 %rd<21>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
ld.param.b64 %rd5, [_row_scatter_add_kernel_param_1];
mad.wide.u32 %rd4, %r1, 4, %rd5;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd4 + 0 ], %rd3;
setp.lt.u32 %p1, %r2, 50304;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd2, [_row_scatter_add_kernel_param_2];
ld.param.b64 %rd1, [_row_scatter_add_kernel_param_0];
mov.u32 %r6, %tid.x;
shl.b32 %r7, %r6, 1;
and.b32 %r8, %r7, 254;
mul.lo.s32 %r9, %r1, 768;
mul.lo.s32 %r10, %r2, 768;
or.b32 %r11, %r8, %r9;
mad.wide.s32 %rd6, %r11, 2, %rd1;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd7, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r3 }, [ %rd6 + 0 ], %rd7;
mov.b32 {%rs13, %rs14}, %r3;
cvt.f32.bf16 %r12, %rs13;
cvt.f32.bf16 %r13, %rs14;
or.b32 %r14, %r10, %r8;
mad.wide.u32 %rd8, %r14, 2, %rd2;
cvt.rn.f16.f32 %rs3, %r12;
cvt.rn.f16.f32 %rs4, %r13;
mov.pred %p2, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd8 + 0 ], { %rs3, %rs4 };
add.s32 %r15, %r11, 256;
mad.wide.s32 %rd9, %r15, 2, %rd1;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd10, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r4 }, [ %rd9 + 0 ], %rd10;
mov.b32 {%rs15, %rs16}, %r4;
cvt.f32.bf16 %r16, %rs15;
cvt.f32.bf16 %r17, %rs16;
add.s64 %rd11, %rd8, 512;
cvt.rn.f16.f32 %rs7, %r16;
cvt.rn.f16.f32 %rs8, %r17;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd11 + 0 ], { %rs7, %rs8 };
add.s32 %r18, %r11, 512;
mad.wide.s32 %rd12, %r18, 2, %rd1;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd13, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_first.L2::cache_hint.b32 { %r5 }, [ %rd12 + 0 ], %rd13;
mov.b32 {%rs17, %rs18}, %r5;
cvt.f32.bf16 %r19, %rs17;
cvt.f32.bf16 %r20, %rs18;
add.s64 %rd14, %rd8, 1024;
cvt.rn.f16.f32 %rs11, %r19;
cvt.rn.f16.f32 %rs12, %r20;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd14 + 0 ], { %rs11, %rs12 };
ret;
$L__BB0_1:
{
.param .b64 param0;
.param .b64 param1;
.param .b32 param2;
.param .b64 param3;
.param .b64 param4;
mov.b64 %rd15, assertFunc_0;
cvta.global.u64 %rd16, %rd15;
st.param.b64 [param3], %rd16;
mov.b64 %rd17, assertFile_0;
cvta.global.u64 %rd18, %rd17;
st.param.b64 [param1], %rd18;
mov.b64 %rd19, assertMessage_0;
cvta.global.u64 %rd20, %rd19;
st.param.b64 [param0], %rd20;
st.param.b64 [param4], 1;
st.param.b32 [param2], 43;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp0:
$L__func_end0:
}"#

private def fourDebugIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_four_planes(%g0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %g3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %token_ids: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_0 = arith.constant dense<0> : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_1 = arith.constant dense<50304> : tensor<4x1xi32, #blocked>
%cst_2 = arith.constant dense<768> : tensor<4x1xi32, #blocked>
%cst_3 = arith.constant dense<true> : tensor<4x256xi1, #blocked>
%cst_4 = arith.constant dense<256> : tensor<4x256xi32, #blocked>
%cst_5 = arith.constant dense<512> : tensor<4x256xi32, #blocked>
%pid = tt.get_program_id x : i32
%token = tt.addptr %token_ids, %pid : !tt.ptr<i32>, i32
%token_6 = tt.load %token evictionPolicy = evict_last : !tt.ptr<i32>
%0 = arith.cmpi sge, %token_6, 0 : i32
%1 = arith.cmpi slt, %token_6, 50304 : i32
%2 = arith.andi %0, %1 : i1
tt.assert %2, "token id out of range" : i1
%plane = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%col = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%g = arith.cmpi slt, %plane, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_7 = arith.cmpi eq, %plane, %cst_0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_8 = tt.splat %g0 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_9 = tt.splat %g1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_10 = arith.select %g_7, %g_8, %g_9 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_11 = arith.cmpi eq, %plane, %cst : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_12 = tt.splat %g2 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_13 = tt.splat %g3 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_14 = arith.select %g_11, %g_12, %g_13 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%g_15 = arith.select %g, %g_10, %g_14 : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
%source = tt.expand_dims %g_15 {axis = 1 : i32} : tensor<4x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1x!tt.ptr<bf16>, #blocked>
%source_16 = arith.muli %pid, 768 : i32
%source_17 = tt.splat %source_16 : i32 -> tensor<4x1xi32, #blocked>
%source_18 = tt.addptr %source, %source_17 : tensor<4x1x!tt.ptr<bf16>, #blocked>, tensor<4x1xi32, #blocked>
%source_19 = tt.expand_dims %col {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
%source_20 = tt.broadcast %source_18 : tensor<4x1x!tt.ptr<bf16>, #blocked> -> tensor<4x256x!tt.ptr<bf16>, #blocked>
%source_21 = tt.broadcast %source_19 : tensor<1x256xi32, #blocked> -> tensor<4x256xi32, #blocked>
%source_22 = tt.addptr %source_20, %source_21 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%target = tt.expand_dims %plane {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xi32, #blocked>
%target_23 = arith.muli %target, %cst_1 : tensor<4x1xi32, #blocked>
%target_24 = tt.splat %token_6 : i32 -> tensor<4x1xi32, #blocked>
%target_25 = arith.addi %target_23, %target_24 : tensor<4x1xi32, #blocked>
%target_26 = arith.muli %target_25, %cst_2 : tensor<4x1xi32, #blocked>
%target_27 = tt.splat %out : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked>
%target_28 = tt.addptr %target_27, %target_26 : tensor<4x1x!tt.ptr<f16>, #blocked>, tensor<4x1xi32, #blocked>
%target_29 = tt.broadcast %target_28 : tensor<4x1x!tt.ptr<f16>, #blocked> -> tensor<4x256x!tt.ptr<f16>, #blocked>
%target_30 = tt.addptr %target_29, %source_21 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%value = tt.load %source_22 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_31 = arith.extf %value : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%3 = arith.truncf %value_31 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%4 = tt.atomic_rmw fadd, relaxed, gpu, %target_30, %3, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
%value_32 = tt.addptr %source_22, %cst_4 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%value_33 = tt.load %value_32 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_34 = arith.extf %value_33 : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%5 = tt.addptr %target_30, %cst_4 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%6 = arith.truncf %value_34 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%7 = tt.atomic_rmw fadd, relaxed, gpu, %5, %6, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
%value_35 = tt.addptr %source_22, %cst_5 : tensor<4x256x!tt.ptr<bf16>, #blocked>, tensor<4x256xi32, #blocked>
%value_36 = tt.load %value_35 evictionPolicy = evict_first : tensor<4x256x!tt.ptr<bf16>, #blocked>
%value_37 = arith.extf %value_36 : tensor<4x256xbf16, #blocked> to tensor<4x256xf32, #blocked>
%8 = tt.addptr %target_30, %cst_5 : tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xi32, #blocked>
%9 = arith.truncf %value_37 : tensor<4x256xf32, #blocked> to tensor<4x256xf16, #blocked>
%10 = tt.atomic_rmw fadd, relaxed, gpu, %8, %9, %cst_3 : (tensor<4x256x!tt.ptr<f16>, #blocked>, tensor<4x256xf16, #blocked>, tensor<4x256xi1, #blocked>) -> tensor<4x256xf16, #blocked>
tt.return
}
}"#

private def fourDebugPTX : String := r#".version 8.7
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
.global .align 1 .b8 assertMessage_0[22] = {116, 111, 107, 101, 110, 32, 105, 100, 32, 111, 117, 116, 32, 111, 102, 32, 114, 97, 110, 103, 101};
.visible .entry _scatter_four_planes(
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_0,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_1,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_2,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_3,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_4,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_5,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_6,
.param .u64 .ptr .global .align 1 _scatter_four_planes_param_7
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b16 %rs<73>;
.reg .b32 %r<46>;
.reg .b64 %rd<30>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
ld.param.b64 %rd8, [_scatter_four_planes_param_4];
mad.wide.u32 %rd7, %r1, 4, %rd8;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd7 + 0 ], %rd6;
setp.lt.u32 %p1, %r2, 50304;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd5, [_scatter_four_planes_param_5];
ld.param.b64 %rd4, [_scatter_four_planes_param_3];
ld.param.b64 %rd3, [_scatter_four_planes_param_2];
ld.param.b64 %rd2, [_scatter_four_planes_param_1];
ld.param.b64 %rd1, [_scatter_four_planes_param_0];
mov.u32 %r15, %tid.x;
bfe.u32 %r16, %r15, 5, 2;
shl.b32 %r17, %r15, 3;
and.b32 %r18, %r17, 248;
setp.lt.u32 %p3, %r16, 2;
setp.eq.b32 %p4, %r16, 0;
selp.b64 %rd18, %rd1, %rd2, %p4;
setp.eq.b32 %p5, %r16, 2;
selp.b64 %rd19, %rd3, %rd4, %p5;
selp.b64 %rd20, %rd18, %rd19, %p3;
mul.lo.s32 %r19, %r1, 768;
mad.wide.s32 %rd21, %r19, 2, %rd20;
mul.wide.u32 %rd22, %r18, 2;
add.s64 %rd9, %rd21, %rd22;
mad.lo.s32 %r20, %r16, 50304, %r2;
mul.lo.s32 %r21, %r20, 768;
mad.wide.u32 %rd23, %r21, 2, %rd5;
add.s64 %rd11, %rd23, %rd22;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd10, 1.0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r3, %r4, %r5, %r6 }, [ %rd9 + 0 ], %rd10;
mov.b32 {%rs49, %rs50}, %r3;
mov.b32 {%rs51, %rs52}, %r4;
mov.b32 {%rs53, %rs54}, %r5;
mov.b32 {%rs55, %rs56}, %r6;
cvt.f32.bf16 %r22, %rs49;
cvt.f32.bf16 %r23, %rs50;
cvt.f32.bf16 %r24, %rs51;
cvt.f32.bf16 %r25, %rs52;
cvt.f32.bf16 %r26, %rs53;
cvt.f32.bf16 %r27, %rs54;
cvt.f32.bf16 %r28, %rs55;
cvt.f32.bf16 %r29, %rs56;
cvt.rn.f16.f32 %rs9, %r22;
cvt.rn.f16.f32 %rs10, %r23;
cvt.rn.f16.f32 %rs11, %r24;
cvt.rn.f16.f32 %rs12, %r25;
cvt.rn.f16.f32 %rs13, %r26;
cvt.rn.f16.f32 %rs14, %r27;
cvt.rn.f16.f32 %rs15, %r28;
cvt.rn.f16.f32 %rs16, %r29;
mov.pred %p2, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
mov.u16 %rs3, 0x0;
mov.u16 %rs4, 0x0;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
mov.u16 %rs7, 0x0;
mov.u16 %rs8, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs1, %rs2, %rs3, %rs4, %rs5, %rs6, %rs7, %rs8 }, [ %rd11 + 0 ], { %rs9, %rs10, %rs11, %rs12, %rs13, %rs14, %rs15, %rs16 };
add.s64 %rd12, %rd9, 512;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd13, 1.0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd12 + 0 ], %rd13;
mov.b32 {%rs57, %rs58}, %r7;
mov.b32 {%rs59, %rs60}, %r8;
mov.b32 {%rs61, %rs62}, %r9;
mov.b32 {%rs63, %rs64}, %r10;
cvt.f32.bf16 %r30, %rs57;
cvt.f32.bf16 %r31, %rs58;
cvt.f32.bf16 %r32, %rs59;
cvt.f32.bf16 %r33, %rs60;
cvt.f32.bf16 %r34, %rs61;
cvt.f32.bf16 %r35, %rs62;
cvt.f32.bf16 %r36, %rs63;
cvt.f32.bf16 %r37, %rs64;
add.s64 %rd14, %rd11, 512;
cvt.rn.f16.f32 %rs25, %r30;
cvt.rn.f16.f32 %rs26, %r31;
cvt.rn.f16.f32 %rs27, %r32;
cvt.rn.f16.f32 %rs28, %r33;
cvt.rn.f16.f32 %rs29, %r34;
cvt.rn.f16.f32 %rs30, %r35;
cvt.rn.f16.f32 %rs31, %r36;
cvt.rn.f16.f32 %rs32, %r37;
mov.u16 %rs17, 0x0;
mov.u16 %rs18, 0x0;
mov.u16 %rs19, 0x0;
mov.u16 %rs20, 0x0;
mov.u16 %rs21, 0x0;
mov.u16 %rs22, 0x0;
mov.u16 %rs23, 0x0;
mov.u16 %rs24, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs17, %rs18, %rs19, %rs20, %rs21, %rs22, %rs23, %rs24 }, [ %rd14 + 0 ], { %rs25, %rs26, %rs27, %rs28, %rs29, %rs30, %rs31, %rs32 };
add.s64 %rd15, %rd9, 1024;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_first.b64 %rd16, 1.0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
ld.global.L1::evict_first.L2::cache_hint.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd15 + 0 ], %rd16;
mov.b32 {%rs65, %rs66}, %r11;
mov.b32 {%rs67, %rs68}, %r12;
mov.b32 {%rs69, %rs70}, %r13;
mov.b32 {%rs71, %rs72}, %r14;
cvt.f32.bf16 %r38, %rs65;
cvt.f32.bf16 %r39, %rs66;
cvt.f32.bf16 %r40, %rs67;
cvt.f32.bf16 %r41, %rs68;
cvt.f32.bf16 %r42, %rs69;
cvt.f32.bf16 %r43, %rs70;
cvt.f32.bf16 %r44, %rs71;
cvt.f32.bf16 %r45, %rs72;
add.s64 %rd17, %rd11, 1024;
cvt.rn.f16.f32 %rs41, %r38;
cvt.rn.f16.f32 %rs42, %r39;
cvt.rn.f16.f32 %rs43, %r40;
cvt.rn.f16.f32 %rs44, %r41;
cvt.rn.f16.f32 %rs45, %r42;
cvt.rn.f16.f32 %rs46, %r43;
cvt.rn.f16.f32 %rs47, %r44;
cvt.rn.f16.f32 %rs48, %r45;
mov.u16 %rs33, 0x0;
mov.u16 %rs34, 0x0;
mov.u16 %rs35, 0x0;
mov.u16 %rs36, 0x0;
mov.u16 %rs37, 0x0;
mov.u16 %rs38, 0x0;
mov.u16 %rs39, 0x0;
mov.u16 %rs40, 0x0;
@%p2 atom.global.gpu.relaxed.add.noftz.v8.f16 { %rs33, %rs34, %rs35, %rs36, %rs37, %rs38, %rs39, %rs40 }, [ %rd17 + 0 ], { %rs41, %rs42, %rs43, %rs44, %rs45, %rs46, %rs47, %rs48 };
ret;
$L__BB0_1:
{
.param .b64 param0;
.param .b64 param1;
.param .b32 param2;
.param .b64 param3;
.param .b64 param4;
mov.b64 %rd24, assertFunc_0;
cvta.global.u64 %rd25, %rd24;
st.param.b64 [param3], %rd25;
mov.b64 %rd26, assertFile_0;
cvta.global.u64 %rd27, %rd26;
st.param.b64 [param1], %rd27;
mov.b64 %rd28, assertMessage_0;
cvta.global.u64 %rd29, %rd28;
st.param.b64 [param0], %rd29;
st.param.b64 [param4], 1;
st.param.b32 [param2], 12;
call.uni __assertfail, (param0, param1, param2, param3, param4);
}
trap;
$L__tmp0:
$L__func_end0:
}"#

private def recognizesCode (c : Code) (body assembly : String) : Bool :=
  QKVTTIR.normalizeTTIR c.ir == some (body.splitOn "\n") &&
    (QKVTTIR.normalizePTX c.ptx).filter (!·.startsWith ".global .align 1 .b8 assertFile_") ==
      assembly.splitOn "\n"

def recognized (a : Artifacts) : Bool :=
  recognizesCode a.reference referenceIR referencePTX && recognizesCode a.four fourIR fourPTX &&
  recognizesCode a.referenceDebug referenceDebugIR referenceDebugPTX &&
  recognizesCode a.fourDebug fourDebugIR fourDebugPTX

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  equivalent : fused = staged

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : recognized a = true then some ⟨h, accum_equal⟩ else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.ValueEmbedAccum
