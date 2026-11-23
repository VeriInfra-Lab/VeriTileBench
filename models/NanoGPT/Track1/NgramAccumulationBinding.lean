import models.NanoGPT.Track1.NgramAccumulation
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The original row scatter remains the BF16 sink consumer. Match its
Hopper code, including both halfwords of each atomic and the byte strides. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramAccumulation

private def expectedIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_scatter_add_rows_kernel(%SRC: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %IDX: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ROW_MAP: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
%val_5 = tt.splat %SRC : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
%val_6 = tt.addptr %val_5, %val_4 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_7 = tt.load %val_6 : tensor<256x!tt.ptr<bf16>, #blocked>
%val_8 = arith.extf %val_7 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%0 = tt.splat %dbase_3 : i64 -> tensor<256xi64, #blocked>
%1 = arith.addi %0, %o64_0 : tensor<256xi64, #blocked>
%2 = tt.splat %DST : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
%3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%4 = arith.truncf %val_8 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%5 = tt.atomic_rmw fadd, relaxed, gpu, %3, %4, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%val_9 = arith.addi %sbase_2, %c256_i64 : i64
%val_10 = tt.splat %val_9 : i64 -> tensor<256xi64, #blocked>
%val_11 = arith.addi %val_10, %o64_0 : tensor<256xi64, #blocked>
%val_12 = tt.addptr %val_5, %val_11 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_13 = tt.load %val_12 : tensor<256x!tt.ptr<bf16>, #blocked>
%val_14 = arith.extf %val_13 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%6 = arith.addi %dbase_3, %c256_i64 : i64
%7 = tt.splat %6 : i64 -> tensor<256xi64, #blocked>
%8 = arith.addi %7, %o64_0 : tensor<256xi64, #blocked>
%9 = tt.addptr %2, %8 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%10 = arith.truncf %val_14 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%11 = tt.atomic_rmw fadd, relaxed, gpu, %9, %10, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
%val_15 = arith.addi %sbase_2, %c512_i64 : i64
%val_16 = tt.splat %val_15 : i64 -> tensor<256xi64, #blocked>
%val_17 = arith.addi %val_16, %o64_0 : tensor<256xi64, #blocked>
%val_18 = tt.addptr %val_5, %val_17 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi64, #blocked>
%val_19 = tt.load %val_18 : tensor<256x!tt.ptr<bf16>, #blocked>
%val_20 = arith.extf %val_19 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
%12 = arith.addi %dbase_3, %c512_i64 : i64
%13 = tt.splat %12 : i64 -> tensor<256xi64, #blocked>
%14 = arith.addi %13, %o64_0 : tensor<256xi64, #blocked>
%15 = tt.addptr %2, %14 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi64, #blocked>
%16 = arith.truncf %val_20 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
%17 = tt.atomic_rmw fadd, relaxed, gpu, %15, %16, %cst : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
tt.return
}
}"#

private def expectedPTX : String := r#".version 8.7
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
.reg .b32 %r<15>;
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
mov.b32 {%rs13, %rs14}, %r2;
cvt.f32.bf16 %r9, %rs13;
cvt.f32.bf16 %r10, %rs14;
mad.wide.s32 %rd13, %r1, 1536, %rd10;
add.s64 %rd3, %rd13, %rd12;
cvt.rn.f16.f32 %rs3, %r9;
cvt.rn.f16.f32 %rs4, %r10;
mov.pred %p1, -1;
mov.u16 %rs1, 0x0;
mov.u16 %rs2, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs1, %rs2 }, [ %rd3 + 0 ], { %rs3, %rs4 };
add.s64 %rd4, %rd2, 512;
mov.u32 %r3, 0x0;
ld.global.b32 { %r3 }, [ %rd4 + 0 ];
mov.b32 {%rs15, %rs16}, %r3;
cvt.f32.bf16 %r11, %rs15;
cvt.f32.bf16 %r12, %rs16;
add.s64 %rd5, %rd3, 512;
cvt.rn.f16.f32 %rs7, %r11;
cvt.rn.f16.f32 %rs8, %r12;
mov.u16 %rs5, 0x0;
mov.u16 %rs6, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs5, %rs6 }, [ %rd5 + 0 ], { %rs7, %rs8 };
add.s64 %rd6, %rd2, 1024;
mov.u32 %r4, 0x0;
ld.global.b32 { %r4 }, [ %rd6 + 0 ];
mov.b32 {%rs17, %rs18}, %r4;
cvt.f32.bf16 %r13, %rs17;
cvt.f32.bf16 %r14, %rs18;
add.s64 %rd7, %rd3, 1024;
cvt.rn.f16.f32 %rs11, %r13;
cvt.rn.f16.f32 %rs12, %r14;
mov.u16 %rs9, 0x0;
mov.u16 %rs10, 0x0;
@%p1 atom.global.gpu.relaxed.add.noftz.v2.f16 { %rs9, %rs10 }, [ %rd7 + 0 ], { %rs11, %rs12 };
ret;
$L__tmp1:
$L__func_end0:
}"#

def channel (lane : Fin 128) (stripe : Fin 3) (half : Fin 2) : Nat :=
  ((lane.val <<< 1) &&& 254) + 256 * stripe.val + half.val

private theorem thread_offset : ∀ lane : Fin 128, (lane.val <<< 1) &&& 254 = 2 * lane.val := by
  decide +kernel

theorem channel_exact (lane : Fin 128) (stripe : Fin 3) (half : Fin 2) :
    channel lane stripe half = 2 * lane.val + 256 * stripe.val + half.val ∧ channel lane stripe half < 768 := by
  have hl := lane.isLt
  have hs := stripe.isLt
  have hh := half.isLt
  simp only [channel, thread_offset, true_and]
  omega

theorem channel_covered (word : Fin 768) :
    ∃ lane stripe half, channel lane stripe half = word.val := by
  have live := word.isLt
  refine ⟨⟨word.val % 256 / 2, by omega⟩, ⟨word.val / 256, by omega⟩, ⟨word.val % 2, by omega⟩, ?_⟩
  rw [(channel_exact _ _ _).1]
  change 2 * (word.val % 256 / 2) + 256 * (word.val / 256) + word.val % 2 = word.val
  omega

def recognized (ir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ir == some (expectedIR.splitOn "\n") &&
    QKVTTIR.normalizePTX ptx == expectedPTX.splitOn "\n"

structure Certificate (ir ptx : String) : Type where
  matched : recognized ir ptx = true
  slots : ∀ word : Fin 768, ∃ lane stripe half, channel lane stripe half = word.val
  cycle : online = delayed

def certify (ir ptx : String) : Option (Certificate ir ptx) :=
  if matched : recognized ir ptx then some ⟨matched, channel_covered, accumulator_equal⟩ else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.NgramAccumulation
