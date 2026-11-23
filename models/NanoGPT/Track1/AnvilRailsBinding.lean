import models.NanoGPT.Track1.AnvilRails
import models.NanoGPT.Track1.QKVTTIRBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilRails
open VeriTile.Triton.FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 one widen narrow)

private theorem fma_product_commutes (cfg : Config) (x y z : F32) :
    fma cfg x y z = fma cfg y x z := by
  let a := prepare cfg x
  let b := prepare cfg y
  have nan : (a.isNaN || b.isNaN) = (b.isNaN || a.isNaN) := Bool.or_comm _ _
  have invalid :
      ((a.isInf && b.isZero) || (a.isZero && b.isInf)) =
        ((b.isInf && a.isZero) || (b.isZero && a.isInf)) := by
    rw [Bool.and_comm a.isInf b.isZero, Bool.and_comm a.isZero b.isInf, Bool.or_comm]
  have inf : (a.isInf || b.isInf) = (b.isInf || a.isInf) := Bool.or_comm _ _
  have sign : (a.sign != b.sign) = (b.sign != a.sign) := bne_comm
  dsimp only [a, b] at nan invalid inf sign
  simp only [fma, nan, invalid, inf, sign]
  split
  · rfl
  · split
    · rfl
    · split
      · rfl
      · cases (prepare cfg x).decode <;> cases (prepare cfg y).decode <;> cases (prepare cfg z).decode <;>
          simp only [Rat.mul_comm, bne_comm]

private theorem mul_operands_commute (cfg : Config) (x y : F32) :
    mul cfg x y = mul cfg y x := by
  unfold mul
  cases (prepare cfg x).decode <;> cases (prepare cfg y).decode <;>
    simp only [Rat.mul_comm, bne_comm]

/-- One hardware thread owns eight adjacent cells. The PTX uses a 1024-cell
block base, masks the thread's scaled lane number and ORs both offsets. -/
def threadBase (block tid : UInt32) : UInt32 :=
  ((tid <<< (3 : UInt32)) &&& (1016 : UInt32)) ||| (block <<< (10 : UInt32))

theorem thread_base_exact (bank : Bank) (block : Fin (cells bank / blockSize)) (tid : Fin 128) :
    (threadBase (UInt32.ofNat block.val) (UInt32.ofNat tid.val)).toNat =
      1024 * block.val + 8 * tid.val := by
  have hb := block.isLt
  have ht := tid.isLt
  have cap : cells bank / blockSize < 32768 := by cases bank <;> decide
  have bword : (UInt32.ofNat block.val).toNat = block.val :=
    UInt32.toNat_ofNat_of_lt' (by change block.val < 4294967296; omega)
  have tword : (UInt32.ofNat tid.val).toNat = tid.val :=
    UInt32.toNat_ofNat_of_lt' (by change tid.val < 4294967296; omega)
  rw [threadBase, UInt32.toNat_or, UInt32.toNat_and, UInt32.toNat_shiftLeft,
    UInt32.toNat_shiftLeft, bword, tword]
  change ((tid.val <<< 3) % 4294967296 &&& 1016) ||| ((block.val <<< 10) % 4294967296) = _
  have bt : tid.val <<< 3 = 8 * tid.val := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  have bb : block.val <<< 10 = 1024 * block.val := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  have mask : (tid.val <<< 3) &&& 1016 = tid.val <<< 3 := by
    change (tid.val <<< 3) &&& (127 <<< 3) = _
    rw [← Nat.shiftLeft_and_distrib, Nat.and_two_pow_sub_one_of_lt_two_pow (n := 7) ht]
  rw [Nat.mod_eq_of_lt (by rw [bt]; omega), Nat.mod_eq_of_lt (by rw [bb]; omega), mask, Nat.or_comm,
    ← Nat.shiftLeft_add_eq_or_of_lt (i := 10) (by rw [bt]; omega), bt, bb]

def cellLane {bank : Bank} (block : Fin (cells bank / blockSize))
    (tid : Fin 128) (part : Fin 8) : Lane bank :=
  ⟨block, ⟨8 * tid.val + part.val, by
    have ht := tid.isLt
    have hp := part.isLt
    unfold blockSize
    omega⟩⟩

private theorem cell_index_exact {bank : Bank} (block : Fin (cells bank / blockSize))
    (tid : Fin 128) (part : Fin 8) :
    (threadBase (UInt32.ofNat block.val) (UInt32.ofNat tid.val)).toNat + part.val =
      (laneIndex (cellLane block tid part)).val := by
  simp [thread_base_exact, cellLane, laneIndex, blockSize]
  omega

theorem word_offsets {bank : Bank} (block : Fin (cells bank / blockSize))
    (tid : Fin 128) (part : Fin 8) :
    let i := (threadBase (UInt32.ofNat block.val) (UInt32.ofNat tid.val)).toNat + part.val
    let p := laneIndex (cellLane block tid part)
    2 * i = 2 * p.val ∧ 4 * i = 4 * (railOffset false p) ∧
      4 * i + 4 * cells bank = 4 * (railOffset true p) := by
  simp [cell_index_exact, railOffset]
  omega

private def threadOf {bank : Bank} (lane : Lane bank) : Fin 128 :=
  ⟨lane.lane.val / 8, by have h := lane.lane.isLt; change lane.lane.val < 1024 at h; omega⟩

private def partOf {bank : Bank} (lane : Lane bank) : Fin 8 :=
  ⟨lane.lane.val % 8, Nat.mod_lt _ (by decide)⟩

theorem parts_cover {bank : Bank} (lane : Lane bank) :
    cellLane lane.block (threadOf lane) (partOf lane) = lane := by
  rcases lane with ⟨block, ⟨lane, bound⟩⟩
  simp only [cellLane, threadOf, partOf, Lane.mk.injEq, Fin.mk.injEq, true_and]
  omega

def emittedIndex {bank : Bank} (lane : Lane bank) : Fin (cells bank) :=
  ⟨(threadBase (UInt32.ofNat lane.block.val) (UInt32.ofNat (threadOf lane).val)).toNat + (partOf lane).val,
    by rw [cell_index_exact, parts_cover]; exact (laneIndex lane).isLt⟩

theorem emitted_index_eq {bank : Bank} (lane : Lane bank) : emittedIndex lane = laneIndex lane := by
  apply Fin.ext
  simp only [emittedIndex, cell_index_exact, parts_cover]

/-- Operands in the order used by the emitted scalar FMA and multiply.
The arithmetic profile keeps the same rounded values under these two swaps. -/
private def emittedCell {bank : Bank} (input : Input bank) (i : Fin (cells bank)) : F32 × F32 × BF16 :=
  let cfg := Config.ieeeValue
  let grad := widen (input.grad i)
  let fast := orderedLerp (input.fast i) grad (sub cfg one input.fastBeta)
  let prior := input.slow i
  let slow := fma cfg (sub cfg grad prior) (Value.ofBits .fp32 0x3ca3d70a) prior
  let blend := fma cfg input.fastWeight fast (mul cfg slow (sub cfg one input.fastWeight))
  (fast, slow, narrow (orderedLerp grad blend input.momentum))

private theorem slow_word : Driver.anvilSlowWeightBits = 0x3ca3d70a := rfl

theorem emitted_cell_eq {bank : Bank} (input : Input bank) (i : Fin (cells bank)) :
    emittedCell input i = fusedLane input (owner i) := by
  simp only [emittedCell, fusedLane, lane_owner, prepareGrad, fastUpdate, slowUpdate, slow_word]
  rw [fma_product_commutes Config.ieeeValue
    (sub Config.ieeeValue (widen (input.grad i)) (input.slow i)) (Value.ofBits .fp32 0x3ca3d70a)]
  rw [mul_operands_commute Config.ieeeValue
    (fma Config.ieeeValue (Value.ofBits .fp32 0x3ca3d70a)
      (sub Config.ieeeValue (widen (input.grad i)) (input.slow i)) (input.slow i))]
  rfl

def emitted : Dispatch := fun _ input =>
  let outputs := fun i => emittedCell input (emittedIndex (owner i))
  ⟨fun i => (outputs i).1, fun i => (outputs i).2.1, fun i => (outputs i).2.2⟩

theorem emitted_eq : emitted = reference := by
  have same : emitted = fused := by
    funext bank input
    simp only [emitted, fused, emitted_index_eq, lane_owner, emitted_cell_eq]
  exact same.trans fused_equal

private def graph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_prepare_rails(%Grad: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Velocity: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Momentum: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %FastBeta: !tt.ptr<f32>, %FastWeight: !tt.ptr<f32>, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant 1.000000e+00 : f32
%cst_0 = arith.constant 5.000000e-01 : f32
%cst_1 = arith.constant dense<2.000000e-02> : tensor<1024xf32, #blocked>
%i = tt.get_program_id x : i32
%i_2 = arith.muli %i, 1024 : i32
%i_3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%i_4 = tt.splat %i_2 : i32 -> tensor<1024xi32, #blocked>
%i_5 = arith.addi %i_4, %i_3 : tensor<1024xi32, #blocked>
%grad = tt.splat %Grad : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%grad_6 = tt.addptr %grad, %i_5 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%grad_7 = tt.load %grad_6 : tensor<1024x!tt.ptr<bf16>, #blocked>
%grad_8 = arith.extf %grad_7 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%fast = tt.splat %Velocity : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%fast_9 = tt.addptr %fast, %i_5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%fast_10 = tt.load %fast_9 : tensor<1024x!tt.ptr<f32>, #blocked>
%slow = tt.addptr %Velocity, @CELLS@ : !tt.ptr<f32>, i32
%slow_11 = tt.splat %slow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%slow_12 = tt.addptr %slow_11, %i_5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%slow_13 = tt.load %slow_12 : tensor<1024x!tt.ptr<f32>, #blocked>
%momentum = tt.load %Momentum : !tt.ptr<f32>
%beta = tt.load %FastBeta : !tt.ptr<f32>
%weight = tt.load %FastWeight : !tt.ptr<f32>
%fast_14 = arith.subf %cst, %beta : f32
%use_end = math.absf %fast_14 : f32
%use_end_15 = arith.cmpf oge, %use_end, %cst_0 : f32
%scale = arith.subf %fast_14, %cst : f32
%scale_16 = arith.select %use_end_15, %scale, %fast_14 : f32
%base = tt.splat %use_end_15 : i1 -> tensor<1024xi1, #blocked>
%base_17 = arith.select %base, %grad_8, %fast_10 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%fast_18 = arith.subf %grad_8, %fast_10 : tensor<1024xf32, #blocked>
%fast_19 = tt.splat %scale_16 : f32 -> tensor<1024xf32, #blocked>
%fast_20 = math.fma %fast_19, %fast_18, %base_17 : tensor<1024xf32, #blocked>
%slow_21 = arith.subf %grad_8, %slow_13 : tensor<1024xf32, #blocked>
%slow_22 = math.fma %cst_1, %slow_21, %slow_13 : tensor<1024xf32, #blocked>
%blend = arith.subf %cst, %weight : f32
%blend_23 = tt.splat %blend : f32 -> tensor<1024xf32, #blocked>
%blend_24 = arith.mulf %blend_23, %slow_22 : tensor<1024xf32, #blocked>
%blend_25 = tt.splat %weight : f32 -> tensor<1024xf32, #blocked>
%blend_26 = math.fma %blend_25, %fast_20, %blend_24 : tensor<1024xf32, #blocked>
%use_end_27 = math.absf %momentum : f32
%use_end_28 = arith.cmpf oge, %use_end_27, %cst_0 : f32
%scale_29 = arith.subf %momentum, %cst : f32
%scale_30 = arith.select %use_end_28, %scale_29, %momentum : f32
%base_31 = tt.splat %use_end_28 : i1 -> tensor<1024xi1, #blocked>
%base_32 = arith.select %base_31, %blend_26, %grad_8 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%lookahead = arith.subf %blend_26, %grad_8 : tensor<1024xf32, #blocked>
%lookahead_33 = tt.splat %scale_30 : f32 -> tensor<1024xf32, #blocked>
%lookahead_34 = math.fma %lookahead_33, %lookahead, %base_32 : tensor<1024xf32, #blocked>
tt.store %fast_9, %fast_20 : tensor<1024x!tt.ptr<f32>, #blocked>
tt.store %slow_12, %slow_22 : tensor<1024x!tt.ptr<f32>, #blocked>
%0 = tt.splat %X : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %i_5 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%2 = arith.truncf %lookahead_34 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %1, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#

private def assembly : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _prepare_rails(
.param .u64 .ptr .global .align 1 _prepare_rails_param_0,
.param .u64 .ptr .global .align 1 _prepare_rails_param_1,
.param .u64 .ptr .global .align 1 _prepare_rails_param_2,
.param .u64 .ptr .global .align 1 _prepare_rails_param_3,
.param .u64 .ptr .global .align 1 _prepare_rails_param_4,
.param .u64 .ptr .global .align 1 _prepare_rails_param_5,
.param .u64 .ptr .global .align 1 _prepare_rails_param_6,
.param .u64 .ptr .global .align 1 _prepare_rails_param_7
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<9>;
.reg .b32 %r<131>;
.reg .b64 %rd<14>;
$L__func_begin0:
ld.param.b64 %rd10, [_prepare_rails_param_0];
ld.param.b64 %rd11, [_prepare_rails_param_1];
$L__tmp0:
mov.u32 %r44, %ctaid.x;
shl.b32 %r45, %r44, 10;
ld.param.b64 %rd6, [_prepare_rails_param_2];
ld.param.b64 %rd7, [_prepare_rails_param_3];
mov.u32 %r46, %tid.x;
shl.b32 %r47, %r46, 3;
ld.param.b64 %rd8, [_prepare_rails_param_4];
and.b32 %r48, %r47, 1016;
ld.param.b64 %rd12, [_prepare_rails_param_5];
or.b32 %r49, %r48, %r45;
mul.wide.s32 %rd13, %r49, 2;
add.s64 %rd1, %rd10, %rd13;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
mov.b32 {%rs3, %rs4}, %r2;
mov.b32 {%rs5, %rs6}, %r3;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r50, %rs1;
cvt.f32.bf16 %r51, %rs2;
cvt.f32.bf16 %r52, %rs3;
cvt.f32.bf16 %r53, %rs4;
cvt.f32.bf16 %r54, %rs5;
cvt.f32.bf16 %r55, %rs6;
cvt.f32.bf16 %r56, %rs7;
cvt.f32.bf16 %r57, %rs8;
mad.wide.s32 %rd2, %r49, 4, %rd11;
add.s64 %rd3, %rd2, 16;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
add.s64 %rd4, %rd2, @BYTES@;
add.s64 %rd5, %rd2, @BYTES_PLUS_16@;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
ld.global.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd5 + 0 ];
mov.u32 %r21, 0x0;
ld.global.b32 { %r21 }, [ %rd6 + 0 ];
mov.u32 %r22, 0x0;
ld.global.b32 { %r22 }, [ %rd7 + 0 ];
mov.u32 %r23, 0x0;
ld.global.b32 { %r23 }, [ %rd8 + 0 ];
mov.b32 %r58, 0f3F800000;
sub.rn.f32 %r59, %r58, %r22;
$L__tmp1:
abs.f32 %r60, %r59;
setp.ge.f32 %p1, %r60, 0f3F000000;
add.rn.f32 %r61, %r59, 0fBF800000;
selp.f32 %r62, %r61, %r59, %p1;
selp.f32 %r63, %r50, %r5, %p1;
selp.f32 %r64, %r51, %r6, %p1;
selp.f32 %r65, %r52, %r7, %p1;
selp.f32 %r66, %r53, %r8, %p1;
selp.f32 %r67, %r54, %r9, %p1;
selp.f32 %r68, %r55, %r10, %p1;
selp.f32 %r69, %r56, %r11, %p1;
selp.f32 %r70, %r57, %r12, %p1;
sub.rn.f32 %r71, %r50, %r5;
sub.rn.f32 %r72, %r51, %r6;
sub.rn.f32 %r73, %r52, %r7;
sub.rn.f32 %r74, %r53, %r8;
sub.rn.f32 %r75, %r54, %r9;
sub.rn.f32 %r76, %r55, %r10;
sub.rn.f32 %r77, %r56, %r11;
sub.rn.f32 %r78, %r57, %r12;
fma.rn.f32 %r24, %r62, %r71, %r63;
fma.rn.f32 %r25, %r62, %r72, %r64;
fma.rn.f32 %r26, %r62, %r73, %r65;
fma.rn.f32 %r27, %r62, %r74, %r66;
fma.rn.f32 %r28, %r62, %r75, %r67;
fma.rn.f32 %r29, %r62, %r76, %r68;
fma.rn.f32 %r30, %r62, %r77, %r69;
fma.rn.f32 %r31, %r62, %r78, %r70;
$L__tmp2:
sub.rn.f32 %r79, %r50, %r13;
sub.rn.f32 %r80, %r51, %r14;
sub.rn.f32 %r81, %r52, %r15;
sub.rn.f32 %r82, %r53, %r16;
sub.rn.f32 %r83, %r54, %r17;
sub.rn.f32 %r84, %r55, %r18;
sub.rn.f32 %r85, %r56, %r19;
sub.rn.f32 %r86, %r57, %r20;
fma.rn.f32 %r32, %r79, 0f3CA3D70A, %r13;
fma.rn.f32 %r33, %r80, 0f3CA3D70A, %r14;
fma.rn.f32 %r34, %r81, 0f3CA3D70A, %r15;
fma.rn.f32 %r35, %r82, 0f3CA3D70A, %r16;
fma.rn.f32 %r36, %r83, 0f3CA3D70A, %r17;
fma.rn.f32 %r37, %r84, 0f3CA3D70A, %r18;
fma.rn.f32 %r38, %r85, 0f3CA3D70A, %r19;
fma.rn.f32 %r39, %r86, 0f3CA3D70A, %r20;
sub.rn.f32 %r87, %r58, %r23;
mul.rn.f32 %r88, %r32, %r87;
mul.rn.f32 %r89, %r33, %r87;
mul.rn.f32 %r90, %r34, %r87;
mul.rn.f32 %r91, %r35, %r87;
mul.rn.f32 %r92, %r36, %r87;
mul.rn.f32 %r93, %r37, %r87;
mul.rn.f32 %r94, %r38, %r87;
mul.rn.f32 %r95, %r39, %r87;
fma.rn.f32 %r96, %r23, %r24, %r88;
fma.rn.f32 %r97, %r23, %r25, %r89;
fma.rn.f32 %r98, %r23, %r26, %r90;
fma.rn.f32 %r99, %r23, %r27, %r91;
fma.rn.f32 %r100, %r23, %r28, %r92;
fma.rn.f32 %r101, %r23, %r29, %r93;
fma.rn.f32 %r102, %r23, %r30, %r94;
fma.rn.f32 %r103, %r23, %r31, %r95;
$L__tmp3:
abs.f32 %r104, %r21;
setp.ge.f32 %p2, %r104, 0f3F000000;
add.rn.f32 %r105, %r21, 0fBF800000;
selp.f32 %r106, %r105, %r21, %p2;
selp.f32 %r107, %r96, %r50, %p2;
selp.f32 %r108, %r97, %r51, %p2;
selp.f32 %r109, %r98, %r52, %p2;
selp.f32 %r110, %r99, %r53, %p2;
selp.f32 %r111, %r100, %r54, %p2;
selp.f32 %r112, %r101, %r55, %p2;
selp.f32 %r113, %r102, %r56, %p2;
selp.f32 %r114, %r103, %r57, %p2;
sub.rn.f32 %r115, %r96, %r50;
sub.rn.f32 %r116, %r97, %r51;
sub.rn.f32 %r117, %r98, %r52;
sub.rn.f32 %r118, %r99, %r53;
sub.rn.f32 %r119, %r100, %r54;
sub.rn.f32 %r120, %r101, %r55;
sub.rn.f32 %r121, %r102, %r56;
sub.rn.f32 %r122, %r103, %r57;
fma.rn.f32 %r123, %r106, %r115, %r107;
fma.rn.f32 %r124, %r106, %r116, %r108;
fma.rn.f32 %r125, %r106, %r117, %r109;
fma.rn.f32 %r126, %r106, %r118, %r110;
fma.rn.f32 %r127, %r106, %r119, %r111;
fma.rn.f32 %r128, %r106, %r120, %r112;
fma.rn.f32 %r129, %r106, %r121, %r113;
fma.rn.f32 %r130, %r106, %r122, %r114;
$L__tmp4:
st.global.v4.b32 [ %rd2 + 0 ], { %r24, %r25, %r26, %r27 };
st.global.v4.b32 [ %rd3 + 0 ], { %r28, %r29, %r30, %r31 };
st.global.v4.b32 [ %rd4 + 0 ], { %r32, %r33, %r34, %r35 };
st.global.v4.b32 [ %rd5 + 0 ], { %r36, %r37, %r38, %r39 };
add.s64 %rd9, %rd12, %rd13;
cvt.rn.bf16x2.f32 %r40, %r124, %r123;
cvt.rn.bf16x2.f32 %r41, %r126, %r125;
cvt.rn.bf16x2.f32 %r42, %r128, %r127;
cvt.rn.bf16x2.f32 %r43, %r130, %r129;
st.global.v4.b32 [ %rd9 + 0 ], { %r40, %r41, %r42, %r43 };
ret;
$L__tmp5:
$L__func_end0:
}
"#

private def lines (bank : Bank) (code : String) : List String :=
  let text := code.replace "@CELLS@" (toString (cells bank))
    |>.replace "@BYTES@" (toString (4 * cells bank))
    |>.replace "@BYTES_PLUS_16@" (toString (4 * cells bank + 16))
  (text.splitOn "\n").filter (!·.isEmpty)

def fitsArtifacts (bank : Bank) (ttgir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttgir == some (lines bank graph) &&
    QKVTTIR.normalizePTX ptx == lines bank assembly

structure Program (bank : Bank) where
  ttgir : String
  ptx : String
  checked : fitsArtifacts bank ttgir ptx = true

def Program.check (bank : Bank) (ttgir ptx : String) : Option (Program bank) :=
  if h : fitsArtifacts bank ttgir ptx = true then some ⟨ttgir, ptx, h⟩ else none

def Program.run {bank : Bank} (program : Program bank) : Input bank → Output bank := fun input =>
  if input.eligible && fitsArtifacts bank program.ttgir program.ptx then emitted bank input
  else reference bank input

theorem Program.run_eq {bank : Bank} (program : Program bank) : program.run = reference bank := by
  funext input
  simp only [Program.run, program.checked, Bool.and_true, emitted_eq, ite_self]

abbrev Programs := (bank : Bank) → Program bank

def Programs.run (programs : Programs) : Dispatch := fun bank => (programs bank).run

theorem Programs.run_eq (programs : Programs) : programs.run = reference := by
  funext bank
  exact (programs bank).run_eq

#axiomsClean thread_base_exact
#axiomsClean word_offsets
#axiomsClean parts_cover
#axiomsClean emitted_index_eq
#axiomsClean emitted_cell_eq
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.AnvilRails
