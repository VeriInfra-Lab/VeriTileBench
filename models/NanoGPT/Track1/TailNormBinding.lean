import models.NanoGPT.Track1.TailNorm
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The Hopper normalization epilogue keeps one mixed-precision
multiply and one round-to-nearest BF16 store per weight. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TailNorm

private def expectedIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_restore_tail_norms(%bank: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %bank_0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %bank_1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %bank_2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %bank_3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ratios: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%local = tt.get_program_id x : i32
%0 = arith.cmpi slt, %local, 9432 : i32
%1:3 = scf.if %0 -> (!tt.ptr<bf16>, i32, i32) {
scf.yield %bank, %local, 0 : !tt.ptr<bf16>, i32, i32
} else {
%5 = arith.cmpi slt, %local, 18864 : i32
%slot:3 = scf.if %5 -> (!tt.ptr<bf16>, i32, i32) {
%local_11 = arith.subi %local, 9432 : i32
scf.yield %bank_0, %local_11, 1 : !tt.ptr<bf16>, i32, i32
} else {
%6 = arith.cmpi slt, %local, 31536 : i32
%slot_11:3 = scf.if %6 -> (!tt.ptr<bf16>, i32, i32) {
%local_12 = arith.subi %local, 18864 : i32
scf.yield %bank_1, %local_12, 2 : !tt.ptr<bf16>, i32, i32
} else {
%7 = arith.cmpi slt, %local, 33840 : i32
%slot_12 = arith.select %7, %bank_2, %bank_3 : !tt.ptr<bf16>
%slot_13 = arith.select %7, 3, 4 : i32
%slot_14 = scf.if %7 -> (i32) {
%local_15 = arith.subi %local, 31536 : i32
scf.yield %local_15 : i32
} else {
%local_15 = arith.subi %local, 33840 : i32
scf.yield %local_15 : i32
}
scf.yield %slot_12, %slot_14, %slot_13 : !tt.ptr<bf16>, i32, i32
}
scf.yield %slot_11#0, %slot_11#1, %slot_11#2 : !tt.ptr<bf16>, i32, i32
}
scf.yield %slot#0, %slot#1, %slot#2 : !tt.ptr<bf16>, i32, i32
}
%lane = arith.muli %1#1, 4096 : i32
%lane_4 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
%lane_5 = tt.splat %lane : i32 -> tensor<4096xi32, #blocked>
%lane_6 = arith.addi %lane_5, %lane_4 : tensor<4096xi32, #blocked>
%ratio = tt.addptr %ratios, %1#2 : !tt.ptr<f32>, i32
%ratio_7 = tt.load %ratio : !tt.ptr<f32>
%value = tt.splat %1#0 : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>, #blocked>
%value_8 = tt.addptr %value, %lane_6 : tensor<4096x!tt.ptr<bf16>, #blocked>, tensor<4096xi32, #blocked>
%value_9 = tt.load %value_8 : tensor<4096x!tt.ptr<bf16>, #blocked>
%value_10 = arith.extf %value_9 : tensor<4096xbf16, #blocked> to tensor<4096xf32, #blocked>
%2 = tt.splat %ratio_7 : f32 -> tensor<4096xf32, #blocked>
%3 = arith.mulf %2, %value_10 : tensor<4096xf32, #blocked>
%4 = arith.truncf %3 : tensor<4096xf32, #blocked> to tensor<4096xbf16, #blocked>
tt.store %value_8, %4 : tensor<4096x!tt.ptr<bf16>, #blocked>
tt.return
}
}"#

private def expectedPTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _restore_tail_norms(
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_0,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_1,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_2,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_3,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_4,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_5,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_6,
.param .u64 .ptr .global .align 1 _restore_tail_norms_param_7
)
.reqntid 256
{
.reg .pred %p<5>;
.reg .b16 %rs<17>;
.reg .b32 %r<57>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd3, [_restore_tail_norms_param_5];
ld.param.b64 %rd8, [_restore_tail_norms_param_0];
$L__tmp0:
mov.u32 %r56, %ctaid.x;
setp.lt.u32 %p1, %r56, 9432;
mov.b64 %rd9, 0;
@%p1 bra $L__BB0_6;
setp.gt.u32 %p2, %r56, 18863;
@%p2 bra $L__BB0_3;
ld.param.b64 %rd8, [_restore_tail_norms_param_1];
add.s32 %r56, %r56, -9432;
mov.b64 %rd9, 1;
bra.uni $L__BB0_6;
$L__BB0_3:
setp.gt.u32 %p3, %r56, 31535;
@%p3 bra $L__BB0_5;
ld.param.b64 %rd8, [_restore_tail_norms_param_2];
add.s32 %r56, %r56, -18864;
mov.b64 %rd9, 2;
bra.uni $L__BB0_6;
$L__BB0_5:
ld.param.b64 %rd2, [_restore_tail_norms_param_4];
ld.param.b64 %rd1, [_restore_tail_norms_param_3];
setp.lt.u32 %p4, %r56, 33840;
selp.b64 %rd8, %rd1, %rd2, %p4;
selp.b64 %rd9, 3, 4, %p4;
selp.b32 %r1, -31536, -33840, %p4;
add.s32 %r56, %r1, %r56;
$L__BB0_6:
shl.b32 %r19, %r56, 12;
mov.u32 %r20, %tid.x;
shl.b32 %r21, %r20, 3;
and.b32 %r22, %r21, 2040;
or.b32 %r23, %r19, %r22;
shl.b64 %rd7, %rd9, 2;
add.s64 %rd4, %rd3, %rd7;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd4 + 0 ];
mad.wide.s32 %rd5, %r23, 2, %rd8;
add.s64 %rd6, %rd5, 4096;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
ld.global.v4.b32 { %r3, %r4, %r5, %r6 }, [ %rd5 + 0 ];
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd6 + 0 ];
mov.b32 {%rs1, %rs2}, %r3;
cvt.f32.bf16 %r24, %rs2;
cvt.f32.bf16 %r25, %rs1;
mul.f32 %r26, %r2, %r25;
mul.f32 %r27, %r2, %r24;
cvt.rn.bf16x2.f32 %r11, %r27, %r26;
mov.b32 {%rs3, %rs4}, %r4;
cvt.f32.bf16 %r28, %rs4;
cvt.f32.bf16 %r29, %rs3;
mul.f32 %r30, %r2, %r29;
mul.f32 %r31, %r2, %r28;
cvt.rn.bf16x2.f32 %r12, %r31, %r30;
mov.b32 {%rs5, %rs6}, %r5;
cvt.f32.bf16 %r32, %rs6;
cvt.f32.bf16 %r33, %rs5;
mul.f32 %r34, %r2, %r33;
mul.f32 %r35, %r2, %r32;
cvt.rn.bf16x2.f32 %r13, %r35, %r34;
mov.b32 {%rs7, %rs8}, %r6;
cvt.f32.bf16 %r36, %rs8;
cvt.f32.bf16 %r37, %rs7;
mul.f32 %r38, %r2, %r37;
mul.f32 %r39, %r2, %r36;
cvt.rn.bf16x2.f32 %r14, %r39, %r38;
mov.b32 {%rs9, %rs10}, %r7;
cvt.f32.bf16 %r40, %rs10;
cvt.f32.bf16 %r41, %rs9;
mul.f32 %r42, %r2, %r41;
mul.f32 %r43, %r2, %r40;
cvt.rn.bf16x2.f32 %r15, %r43, %r42;
mov.b32 {%rs11, %rs12}, %r8;
cvt.f32.bf16 %r44, %rs12;
cvt.f32.bf16 %r45, %rs11;
mul.f32 %r46, %r2, %r45;
mul.f32 %r47, %r2, %r44;
cvt.rn.bf16x2.f32 %r16, %r47, %r46;
mov.b32 {%rs13, %rs14}, %r9;
cvt.f32.bf16 %r48, %rs14;
cvt.f32.bf16 %r49, %rs13;
mul.f32 %r50, %r2, %r49;
mul.f32 %r51, %r2, %r48;
cvt.rn.bf16x2.f32 %r17, %r51, %r50;
mov.b32 {%rs15, %rs16}, %r10;
cvt.f32.bf16 %r52, %rs16;
cvt.f32.bf16 %r53, %rs15;
mul.f32 %r54, %r2, %r53;
mul.f32 %r55, %r2, %r52;
cvt.rn.bf16x2.f32 %r18, %r55, %r54;
st.global.v4.b32 [ %rd5 + 0 ], { %r11, %r12, %r13, %r14 };
st.global.v4.b32 [ %rd6 + 0 ], { %r15, %r16, %r17, %r18 };
ret;
$L__tmp1:
$L__func_end0:
}"#

def recognized (ir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ir == some (expectedIR.splitOn "\n") &&
    QKVTTIR.normalizePTX ptx == expectedPTX.splitOn "\n"

structure Certificate (ir ptx : String) : Type where
  matched : recognized ir ptx = true
  addresses : ∀ cta thread plane word, ptxIndex cta thread plane word < (decode cta.val).1.elements
  routed : ∀ slot tile, tile < slot.elements / tileSize → decode (slot.start + tile) = (slot, tile)
  values : ∀ p slot i, grouped p slot i = reference p slot i

def certify (ir ptx : String) : Option (Certificate ir ptx) :=
  if matched : recognized ir ptx = true then
    some ⟨matched, ptx_index_live, decode_owner, grouped_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.TailNorm
