import models.NanoGPT.Track1.AnvilCooperative
import models.NanoGPT.Track1.AnvilGramBinding

/-! The checked SM90a cooperative kernel carries the same descriptors
through both phases. Its seed, first Gram and publication protocol are linked to the
training driver alongside the separate Gram and normalization programs. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.Cooperative

open AnvilEpilogue (Bank)

inductive Phase where
  | seed | initial | round

private def seedGraph : Bank → String
  | .qk => r#"
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_prepare_rails_and_seed(%Grad: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Velocity: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Momentum: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %FastBeta: !tt.ptr<f32>, %FastWeight: !tt.ptr<f32>, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<36> : tensor<64xi32, #blocked>
%cst_0 = arith.constant dense<0> : tensor<64xi32, #blocked>
%cst_1 = arith.constant 1.000000e+00 : f32
%cst_2 = arith.constant 5.000000e-01 : f32
%cst_3 = arith.constant dense<2.000000e-02> : tensor<1024xf32, #blocked1>
%lanes = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
%0 = tt.splat %Counter : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
%1 = tt.addptr %0, %lanes : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
%2 = tt.get_program_id x : i32
%3 = arith.cmpi eq, %2, 0 : i32
%4 = arith.cmpi slt, %lanes, %cst : tensor<64xi32, #blocked>
%5 = tt.splat %3 : i1 -> tensor<64xi1, #blocked>
%6 = arith.andi %5, %4 : tensor<64xi1, #blocked>
tt.store %1, %cst_0, %6 : tensor<64x!tt.ptr<i32>, #blocked>
%i = arith.muli %2, 1024 : i32
%i_4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
%i_5 = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
%i_6 = arith.addi %i_5, %i_4 : tensor<1024xi32, #blocked1>
%grad = tt.splat %Grad : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_7 = tt.addptr %grad, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%grad_8 = tt.load %grad_7 : tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_9 = arith.extf %grad_8 : tensor<1024xbf16, #blocked1> to tensor<1024xf32, #blocked1>
%fast = tt.splat %Velocity : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%fast_10 = tt.addptr %fast, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%fast_11 = tt.load %fast_10 : tensor<1024x!tt.ptr<f32>, #blocked1>
%slow = tt.addptr %Velocity, 1179648 : !tt.ptr<f32>, i32
%slow_12 = tt.splat %slow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%slow_13 = tt.addptr %slow_12, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%slow_14 = tt.load %slow_13 : tensor<1024x!tt.ptr<f32>, #blocked1>
%momentum = tt.load %Momentum : !tt.ptr<f32>
%beta = tt.load %FastBeta : !tt.ptr<f32>
%weight = tt.load %FastWeight : !tt.ptr<f32>
%fast_15 = arith.subf %cst_1, %beta : f32
%use_end = math.absf %fast_15 : f32
%use_end_16 = arith.cmpf oge, %use_end, %cst_2 : f32
%scale = arith.subf %fast_15, %cst_1 : f32
%scale_17 = arith.select %use_end_16, %scale, %fast_15 : f32
%base = tt.splat %use_end_16 : i1 -> tensor<1024xi1, #blocked1>
%base_18 = arith.select %base, %grad_9, %fast_11 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%fast_19 = arith.subf %grad_9, %fast_11 : tensor<1024xf32, #blocked1>
%fast_20 = tt.splat %scale_17 : f32 -> tensor<1024xf32, #blocked1>
%fast_21 = math.fma %fast_20, %fast_19, %base_18 : tensor<1024xf32, #blocked1>
%slow_22 = arith.subf %grad_9, %slow_14 : tensor<1024xf32, #blocked1>
%slow_23 = math.fma %cst_3, %slow_22, %slow_14 : tensor<1024xf32, #blocked1>
%blend = arith.subf %cst_1, %weight : f32
%blend_24 = tt.splat %blend : f32 -> tensor<1024xf32, #blocked1>
%blend_25 = arith.mulf %blend_24, %slow_23 : tensor<1024xf32, #blocked1>
%blend_26 = tt.splat %weight : f32 -> tensor<1024xf32, #blocked1>
%blend_27 = math.fma %blend_26, %fast_21, %blend_25 : tensor<1024xf32, #blocked1>
%use_end_28 = math.absf %momentum : f32
%use_end_29 = arith.cmpf oge, %use_end_28, %cst_2 : f32
%scale_30 = arith.subf %momentum, %cst_1 : f32
%scale_31 = arith.select %use_end_29, %scale_30, %momentum : f32
%base_32 = tt.splat %use_end_29 : i1 -> tensor<1024xi1, #blocked1>
%base_33 = arith.select %base_32, %blend_27, %grad_9 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%lookahead = arith.subf %blend_27, %grad_9 : tensor<1024xf32, #blocked1>
%lookahead_34 = tt.splat %scale_31 : f32 -> tensor<1024xf32, #blocked1>
%lookahead_35 = math.fma %lookahead_34, %lookahead, %base_33 : tensor<1024xf32, #blocked1>
tt.store %fast_10, %fast_21 : tensor<1024x!tt.ptr<f32>, #blocked1>
tt.store %slow_13, %slow_23 : tensor<1024x!tt.ptr<f32>, #blocked1>
%7 = tt.splat %X : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%8 = tt.addptr %7, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%9 = arith.truncf %lookahead_35 : tensor<1024xf32, #blocked1> to tensor<1024xbf16, #blocked1>
tt.store %8, %9 : tensor<1024x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
"#
  | .vo => r#"
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_prepare_rails_and_seed(%Grad: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Velocity: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Momentum: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %FastBeta: !tt.ptr<f32>, %FastWeight: !tt.ptr<f32>, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<12> : tensor<64xi32, #blocked>
%cst_0 = arith.constant dense<0> : tensor<64xi32, #blocked>
%cst_1 = arith.constant 1.000000e+00 : f32
%cst_2 = arith.constant 5.000000e-01 : f32
%cst_3 = arith.constant dense<2.000000e-02> : tensor<1024xf32, #blocked1>
%lanes = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
%0 = tt.splat %Counter : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
%1 = tt.addptr %0, %lanes : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
%2 = tt.get_program_id x : i32
%3 = arith.cmpi eq, %2, 0 : i32
%4 = arith.cmpi slt, %lanes, %cst : tensor<64xi32, #blocked>
%5 = tt.splat %3 : i1 -> tensor<64xi1, #blocked>
%6 = arith.andi %5, %4 : tensor<64xi1, #blocked>
tt.store %1, %cst_0, %6 : tensor<64x!tt.ptr<i32>, #blocked>
%i = arith.muli %2, 1024 : i32
%i_4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
%i_5 = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
%i_6 = arith.addi %i_5, %i_4 : tensor<1024xi32, #blocked1>
%grad = tt.splat %Grad : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_7 = tt.addptr %grad, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%grad_8 = tt.load %grad_7 : tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_9 = arith.extf %grad_8 : tensor<1024xbf16, #blocked1> to tensor<1024xf32, #blocked1>
%fast = tt.splat %Velocity : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%fast_10 = tt.addptr %fast, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%fast_11 = tt.load %fast_10 : tensor<1024x!tt.ptr<f32>, #blocked1>
%slow = tt.addptr %Velocity, 1179648 : !tt.ptr<f32>, i32
%slow_12 = tt.splat %slow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%slow_13 = tt.addptr %slow_12, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%slow_14 = tt.load %slow_13 : tensor<1024x!tt.ptr<f32>, #blocked1>
%momentum = tt.load %Momentum : !tt.ptr<f32>
%beta = tt.load %FastBeta : !tt.ptr<f32>
%weight = tt.load %FastWeight : !tt.ptr<f32>
%fast_15 = arith.subf %cst_1, %beta : f32
%use_end = math.absf %fast_15 : f32
%use_end_16 = arith.cmpf oge, %use_end, %cst_2 : f32
%scale = arith.subf %fast_15, %cst_1 : f32
%scale_17 = arith.select %use_end_16, %scale, %fast_15 : f32
%base = tt.splat %use_end_16 : i1 -> tensor<1024xi1, #blocked1>
%base_18 = arith.select %base, %grad_9, %fast_11 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%fast_19 = arith.subf %grad_9, %fast_11 : tensor<1024xf32, #blocked1>
%fast_20 = tt.splat %scale_17 : f32 -> tensor<1024xf32, #blocked1>
%fast_21 = math.fma %fast_20, %fast_19, %base_18 : tensor<1024xf32, #blocked1>
%slow_22 = arith.subf %grad_9, %slow_14 : tensor<1024xf32, #blocked1>
%slow_23 = math.fma %cst_3, %slow_22, %slow_14 : tensor<1024xf32, #blocked1>
%blend = arith.subf %cst_1, %weight : f32
%blend_24 = tt.splat %blend : f32 -> tensor<1024xf32, #blocked1>
%blend_25 = arith.mulf %blend_24, %slow_23 : tensor<1024xf32, #blocked1>
%blend_26 = tt.splat %weight : f32 -> tensor<1024xf32, #blocked1>
%blend_27 = math.fma %blend_26, %fast_21, %blend_25 : tensor<1024xf32, #blocked1>
%use_end_28 = math.absf %momentum : f32
%use_end_29 = arith.cmpf oge, %use_end_28, %cst_2 : f32
%scale_30 = arith.subf %momentum, %cst_1 : f32
%scale_31 = arith.select %use_end_29, %scale_30, %momentum : f32
%base_32 = tt.splat %use_end_29 : i1 -> tensor<1024xi1, #blocked1>
%base_33 = arith.select %base_32, %blend_27, %grad_9 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%lookahead = arith.subf %blend_27, %grad_9 : tensor<1024xf32, #blocked1>
%lookahead_34 = tt.splat %scale_31 : f32 -> tensor<1024xf32, #blocked1>
%lookahead_35 = math.fma %lookahead_34, %lookahead, %base_33 : tensor<1024xf32, #blocked1>
tt.store %fast_10, %fast_21 : tensor<1024x!tt.ptr<f32>, #blocked1>
tt.store %slow_13, %slow_23 : tensor<1024x!tt.ptr<f32>, #blocked1>
%7 = tt.splat %X : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%8 = tt.addptr %7, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%9 = arith.truncf %lookahead_35 : tensor<1024xf32, #blocked1> to tensor<1024xbf16, #blocked1>
tt.store %8, %9 : tensor<1024x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
"#
  | .mlp => r#"
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_prepare_rails_and_seed(%Grad: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Velocity: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Momentum: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %FastBeta: !tt.ptr<f32>, %FastWeight: !tt.ptr<f32>, %X: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<18> : tensor<64xi32, #blocked>
%cst_0 = arith.constant dense<0> : tensor<64xi32, #blocked>
%cst_1 = arith.constant 1.000000e+00 : f32
%cst_2 = arith.constant 5.000000e-01 : f32
%cst_3 = arith.constant dense<2.000000e-02> : tensor<1024xf32, #blocked1>
%lanes = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
%0 = tt.splat %Counter : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
%1 = tt.addptr %0, %lanes : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
%2 = tt.get_program_id x : i32
%3 = arith.cmpi eq, %2, 0 : i32
%4 = arith.cmpi slt, %lanes, %cst : tensor<64xi32, #blocked>
%5 = tt.splat %3 : i1 -> tensor<64xi1, #blocked>
%6 = arith.andi %5, %4 : tensor<64xi1, #blocked>
tt.store %1, %cst_0, %6 : tensor<64x!tt.ptr<i32>, #blocked>
%i = arith.muli %2, 1024 : i32
%i_4 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
%i_5 = tt.splat %i : i32 -> tensor<1024xi32, #blocked1>
%i_6 = arith.addi %i_5, %i_4 : tensor<1024xi32, #blocked1>
%grad = tt.splat %Grad : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_7 = tt.addptr %grad, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%grad_8 = tt.load %grad_7 : tensor<1024x!tt.ptr<bf16>, #blocked1>
%grad_9 = arith.extf %grad_8 : tensor<1024xbf16, #blocked1> to tensor<1024xf32, #blocked1>
%fast = tt.splat %Velocity : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%fast_10 = tt.addptr %fast, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%fast_11 = tt.load %fast_10 : tensor<1024x!tt.ptr<f32>, #blocked1>
%slow = tt.addptr %Velocity, 6488064 : !tt.ptr<f32>, i32
%slow_12 = tt.splat %slow : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
%slow_13 = tt.addptr %slow_12, %i_6 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
%slow_14 = tt.load %slow_13 : tensor<1024x!tt.ptr<f32>, #blocked1>
%momentum = tt.load %Momentum : !tt.ptr<f32>
%beta = tt.load %FastBeta : !tt.ptr<f32>
%weight = tt.load %FastWeight : !tt.ptr<f32>
%fast_15 = arith.subf %cst_1, %beta : f32
%use_end = math.absf %fast_15 : f32
%use_end_16 = arith.cmpf oge, %use_end, %cst_2 : f32
%scale = arith.subf %fast_15, %cst_1 : f32
%scale_17 = arith.select %use_end_16, %scale, %fast_15 : f32
%base = tt.splat %use_end_16 : i1 -> tensor<1024xi1, #blocked1>
%base_18 = arith.select %base, %grad_9, %fast_11 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%fast_19 = arith.subf %grad_9, %fast_11 : tensor<1024xf32, #blocked1>
%fast_20 = tt.splat %scale_17 : f32 -> tensor<1024xf32, #blocked1>
%fast_21 = math.fma %fast_20, %fast_19, %base_18 : tensor<1024xf32, #blocked1>
%slow_22 = arith.subf %grad_9, %slow_14 : tensor<1024xf32, #blocked1>
%slow_23 = math.fma %cst_3, %slow_22, %slow_14 : tensor<1024xf32, #blocked1>
%blend = arith.subf %cst_1, %weight : f32
%blend_24 = tt.splat %blend : f32 -> tensor<1024xf32, #blocked1>
%blend_25 = arith.mulf %blend_24, %slow_23 : tensor<1024xf32, #blocked1>
%blend_26 = tt.splat %weight : f32 -> tensor<1024xf32, #blocked1>
%blend_27 = math.fma %blend_26, %fast_21, %blend_25 : tensor<1024xf32, #blocked1>
%use_end_28 = math.absf %momentum : f32
%use_end_29 = arith.cmpf oge, %use_end_28, %cst_2 : f32
%scale_30 = arith.subf %momentum, %cst_1 : f32
%scale_31 = arith.select %use_end_29, %scale_30, %momentum : f32
%base_32 = tt.splat %use_end_29 : i1 -> tensor<1024xi1, #blocked1>
%base_33 = arith.select %base_32, %blend_27, %grad_9 : tensor<1024xi1, #blocked1>, tensor<1024xf32, #blocked1>
%lookahead = arith.subf %blend_27, %grad_9 : tensor<1024xf32, #blocked1>
%lookahead_34 = tt.splat %scale_31 : f32 -> tensor<1024xf32, #blocked1>
%lookahead_35 = math.fma %lookahead_34, %lookahead, %base_33 : tensor<1024xf32, #blocked1>
tt.store %fast_10, %fast_21 : tensor<1024x!tt.ptr<f32>, #blocked1>
tt.store %slow_13, %slow_23 : tensor<1024x!tt.ptr<f32>, #blocked1>
%7 = tt.splat %X : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked1>
%8 = tt.addptr %7, %i_6 : tensor<1024x!tt.ptr<bf16>, #blocked1>, tensor<1024xi32, #blocked1>
%9 = arith.truncf %lookahead_35 : tensor<1024xf32, #blocked1> to tensor<1024xbf16, #blocked1>
tt.store %8, %9 : tensor<1024x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
"#

private def seedAssembly : Bank → String
  | .qk => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _prepare_rails_and_seed(
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_0,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_1,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_2,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_3,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_4,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_5,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_6,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_7,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_8
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b16 %rs<9>;
.reg .b32 %r<135>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd11, [_prepare_rails_and_seed_param_0];
ld.param.b64 %rd12, [_prepare_rails_and_seed_param_1];
$L__tmp0:
mov.u32 %r45, %tid.x;
and.b32 %r46, %r45, 63;
ld.param.b64 %rd7, [_prepare_rails_and_seed_param_2];
ld.param.b64 %rd8, [_prepare_rails_and_seed_param_3];
ld.param.b64 %rd13, [_prepare_rails_and_seed_param_6];
mad.wide.u32 %rd1, %r46, 4, %rd13;
ld.param.b64 %rd9, [_prepare_rails_and_seed_param_4];
ld.param.b64 %rd14, [_prepare_rails_and_seed_param_5];
mov.u32 %r47, %ctaid.x;
setp.lt.u32 %p2, %r46, 36;
and.b32 %r48, %r45, 64;
or.b32 %r49, %r48, %r47;
setp.eq.b32 %p3, %r49, 0;
and.pred %p1, %p2, %p3;
mov.b32 %r1, 0;
@%p1 st.global.b32 [ %rd1 + 0 ], { %r1 };
$L__tmp1:
shl.b32 %r50, %r47, 10;
shl.b32 %r51, %r45, 3;
and.b32 %r52, %r51, 1016;
or.b32 %r53, %r52, %r50;
mul.wide.s32 %rd15, %r53, 2;
add.s64 %rd2, %rd11, %rd15;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
cvt.f32.bf16 %r54, %rs1;
cvt.f32.bf16 %r55, %rs2;
cvt.f32.bf16 %r56, %rs3;
cvt.f32.bf16 %r57, %rs4;
cvt.f32.bf16 %r58, %rs5;
cvt.f32.bf16 %r59, %rs6;
cvt.f32.bf16 %r60, %rs7;
cvt.f32.bf16 %r61, %rs8;
mad.wide.s32 %rd3, %r53, 4, %rd12;
add.s64 %rd4, %rd3, 16;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd4 + 0 ];
add.s64 %rd5, %rd3, 4718592;
add.s64 %rd6, %rd3, 4718608;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd5 + 0 ];
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
mov.u32 %r21, 0x0;
ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd6 + 0 ];
mov.u32 %r22, 0x0;
ld.global.b32 { %r22 }, [ %rd7 + 0 ];
mov.u32 %r23, 0x0;
ld.global.b32 { %r23 }, [ %rd8 + 0 ];
mov.u32 %r24, 0x0;
ld.global.b32 { %r24 }, [ %rd9 + 0 ];
mov.b32 %r62, 0f3F800000;
sub.rn.f32 %r63, %r62, %r23;
$L__tmp2:
abs.f32 %r64, %r63;
setp.ge.f32 %p4, %r64, 0f3F000000;
add.rn.f32 %r65, %r63, 0fBF800000;
selp.f32 %r66, %r65, %r63, %p4;
selp.f32 %r67, %r54, %r6, %p4;
selp.f32 %r68, %r55, %r7, %p4;
selp.f32 %r69, %r56, %r8, %p4;
selp.f32 %r70, %r57, %r9, %p4;
selp.f32 %r71, %r58, %r10, %p4;
selp.f32 %r72, %r59, %r11, %p4;
selp.f32 %r73, %r60, %r12, %p4;
selp.f32 %r74, %r61, %r13, %p4;
sub.rn.f32 %r75, %r54, %r6;
sub.rn.f32 %r76, %r55, %r7;
sub.rn.f32 %r77, %r56, %r8;
sub.rn.f32 %r78, %r57, %r9;
sub.rn.f32 %r79, %r58, %r10;
sub.rn.f32 %r80, %r59, %r11;
sub.rn.f32 %r81, %r60, %r12;
sub.rn.f32 %r82, %r61, %r13;
fma.rn.f32 %r25, %r66, %r75, %r67;
fma.rn.f32 %r26, %r66, %r76, %r68;
fma.rn.f32 %r27, %r66, %r77, %r69;
fma.rn.f32 %r28, %r66, %r78, %r70;
fma.rn.f32 %r29, %r66, %r79, %r71;
fma.rn.f32 %r30, %r66, %r80, %r72;
fma.rn.f32 %r31, %r66, %r81, %r73;
fma.rn.f32 %r32, %r66, %r82, %r74;
$L__tmp3:
sub.rn.f32 %r83, %r54, %r14;
sub.rn.f32 %r84, %r55, %r15;
sub.rn.f32 %r85, %r56, %r16;
sub.rn.f32 %r86, %r57, %r17;
sub.rn.f32 %r87, %r58, %r18;
sub.rn.f32 %r88, %r59, %r19;
sub.rn.f32 %r89, %r60, %r20;
sub.rn.f32 %r90, %r61, %r21;
fma.rn.f32 %r33, %r83, 0f3CA3D70A, %r14;
fma.rn.f32 %r34, %r84, 0f3CA3D70A, %r15;
fma.rn.f32 %r35, %r85, 0f3CA3D70A, %r16;
fma.rn.f32 %r36, %r86, 0f3CA3D70A, %r17;
fma.rn.f32 %r37, %r87, 0f3CA3D70A, %r18;
fma.rn.f32 %r38, %r88, 0f3CA3D70A, %r19;
fma.rn.f32 %r39, %r89, 0f3CA3D70A, %r20;
fma.rn.f32 %r40, %r90, 0f3CA3D70A, %r21;
sub.rn.f32 %r91, %r62, %r24;
mul.rn.f32 %r92, %r33, %r91;
mul.rn.f32 %r93, %r34, %r91;
mul.rn.f32 %r94, %r35, %r91;
mul.rn.f32 %r95, %r36, %r91;
mul.rn.f32 %r96, %r37, %r91;
mul.rn.f32 %r97, %r38, %r91;
mul.rn.f32 %r98, %r39, %r91;
mul.rn.f32 %r99, %r40, %r91;
fma.rn.f32 %r100, %r24, %r25, %r92;
fma.rn.f32 %r101, %r24, %r26, %r93;
fma.rn.f32 %r102, %r24, %r27, %r94;
fma.rn.f32 %r103, %r24, %r28, %r95;
fma.rn.f32 %r104, %r24, %r29, %r96;
fma.rn.f32 %r105, %r24, %r30, %r97;
fma.rn.f32 %r106, %r24, %r31, %r98;
fma.rn.f32 %r107, %r24, %r32, %r99;
$L__tmp4:
abs.f32 %r108, %r22;
setp.ge.f32 %p5, %r108, 0f3F000000;
add.rn.f32 %r109, %r22, 0fBF800000;
selp.f32 %r110, %r109, %r22, %p5;
selp.f32 %r111, %r100, %r54, %p5;
selp.f32 %r112, %r101, %r55, %p5;
selp.f32 %r113, %r102, %r56, %p5;
selp.f32 %r114, %r103, %r57, %p5;
selp.f32 %r115, %r104, %r58, %p5;
selp.f32 %r116, %r105, %r59, %p5;
selp.f32 %r117, %r106, %r60, %p5;
selp.f32 %r118, %r107, %r61, %p5;
sub.rn.f32 %r119, %r100, %r54;
sub.rn.f32 %r120, %r101, %r55;
sub.rn.f32 %r121, %r102, %r56;
sub.rn.f32 %r122, %r103, %r57;
sub.rn.f32 %r123, %r104, %r58;
sub.rn.f32 %r124, %r105, %r59;
sub.rn.f32 %r125, %r106, %r60;
sub.rn.f32 %r126, %r107, %r61;
fma.rn.f32 %r127, %r110, %r119, %r111;
fma.rn.f32 %r128, %r110, %r120, %r112;
fma.rn.f32 %r129, %r110, %r121, %r113;
fma.rn.f32 %r130, %r110, %r122, %r114;
fma.rn.f32 %r131, %r110, %r123, %r115;
fma.rn.f32 %r132, %r110, %r124, %r116;
fma.rn.f32 %r133, %r110, %r125, %r117;
fma.rn.f32 %r134, %r110, %r126, %r118;
$L__tmp5:
st.global.v4.b32 [ %rd3 + 0 ], { %r25, %r26, %r27, %r28 };
st.global.v4.b32 [ %rd4 + 0 ], { %r29, %r30, %r31, %r32 };
st.global.v4.b32 [ %rd5 + 0 ], { %r33, %r34, %r35, %r36 };
st.global.v4.b32 [ %rd6 + 0 ], { %r37, %r38, %r39, %r40 };
add.s64 %rd10, %rd14, %rd15;
cvt.rn.bf16x2.f32 %r41, %r128, %r127;
cvt.rn.bf16x2.f32 %r42, %r130, %r129;
cvt.rn.bf16x2.f32 %r43, %r132, %r131;
cvt.rn.bf16x2.f32 %r44, %r134, %r133;
st.global.v4.b32 [ %rd10 + 0 ], { %r41, %r42, %r43, %r44 };
$L__tmp6:
ret;
$L__tmp7:
$L__func_end0:
}
"#
  | .vo => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _prepare_rails_and_seed(
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_0,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_1,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_2,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_3,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_4,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_5,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_6,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_7,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_8
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b16 %rs<9>;
.reg .b32 %r<135>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd11, [_prepare_rails_and_seed_param_0];
ld.param.b64 %rd12, [_prepare_rails_and_seed_param_1];
$L__tmp0:
mov.u32 %r45, %tid.x;
and.b32 %r46, %r45, 63;
ld.param.b64 %rd7, [_prepare_rails_and_seed_param_2];
ld.param.b64 %rd8, [_prepare_rails_and_seed_param_3];
ld.param.b64 %rd13, [_prepare_rails_and_seed_param_6];
mad.wide.u32 %rd1, %r46, 4, %rd13;
ld.param.b64 %rd9, [_prepare_rails_and_seed_param_4];
ld.param.b64 %rd14, [_prepare_rails_and_seed_param_5];
mov.u32 %r47, %ctaid.x;
setp.lt.u32 %p2, %r46, 12;
and.b32 %r48, %r45, 64;
or.b32 %r49, %r48, %r47;
setp.eq.b32 %p3, %r49, 0;
and.pred %p1, %p2, %p3;
mov.b32 %r1, 0;
@%p1 st.global.b32 [ %rd1 + 0 ], { %r1 };
$L__tmp1:
shl.b32 %r50, %r47, 10;
shl.b32 %r51, %r45, 3;
and.b32 %r52, %r51, 1016;
or.b32 %r53, %r52, %r50;
mul.wide.s32 %rd15, %r53, 2;
add.s64 %rd2, %rd11, %rd15;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
cvt.f32.bf16 %r54, %rs1;
cvt.f32.bf16 %r55, %rs2;
cvt.f32.bf16 %r56, %rs3;
cvt.f32.bf16 %r57, %rs4;
cvt.f32.bf16 %r58, %rs5;
cvt.f32.bf16 %r59, %rs6;
cvt.f32.bf16 %r60, %rs7;
cvt.f32.bf16 %r61, %rs8;
mad.wide.s32 %rd3, %r53, 4, %rd12;
add.s64 %rd4, %rd3, 16;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd4 + 0 ];
add.s64 %rd5, %rd3, 4718592;
add.s64 %rd6, %rd3, 4718608;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd5 + 0 ];
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
mov.u32 %r21, 0x0;
ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd6 + 0 ];
mov.u32 %r22, 0x0;
ld.global.b32 { %r22 }, [ %rd7 + 0 ];
mov.u32 %r23, 0x0;
ld.global.b32 { %r23 }, [ %rd8 + 0 ];
mov.u32 %r24, 0x0;
ld.global.b32 { %r24 }, [ %rd9 + 0 ];
mov.b32 %r62, 0f3F800000;
sub.rn.f32 %r63, %r62, %r23;
$L__tmp2:
abs.f32 %r64, %r63;
setp.ge.f32 %p4, %r64, 0f3F000000;
add.rn.f32 %r65, %r63, 0fBF800000;
selp.f32 %r66, %r65, %r63, %p4;
selp.f32 %r67, %r54, %r6, %p4;
selp.f32 %r68, %r55, %r7, %p4;
selp.f32 %r69, %r56, %r8, %p4;
selp.f32 %r70, %r57, %r9, %p4;
selp.f32 %r71, %r58, %r10, %p4;
selp.f32 %r72, %r59, %r11, %p4;
selp.f32 %r73, %r60, %r12, %p4;
selp.f32 %r74, %r61, %r13, %p4;
sub.rn.f32 %r75, %r54, %r6;
sub.rn.f32 %r76, %r55, %r7;
sub.rn.f32 %r77, %r56, %r8;
sub.rn.f32 %r78, %r57, %r9;
sub.rn.f32 %r79, %r58, %r10;
sub.rn.f32 %r80, %r59, %r11;
sub.rn.f32 %r81, %r60, %r12;
sub.rn.f32 %r82, %r61, %r13;
fma.rn.f32 %r25, %r66, %r75, %r67;
fma.rn.f32 %r26, %r66, %r76, %r68;
fma.rn.f32 %r27, %r66, %r77, %r69;
fma.rn.f32 %r28, %r66, %r78, %r70;
fma.rn.f32 %r29, %r66, %r79, %r71;
fma.rn.f32 %r30, %r66, %r80, %r72;
fma.rn.f32 %r31, %r66, %r81, %r73;
fma.rn.f32 %r32, %r66, %r82, %r74;
$L__tmp3:
sub.rn.f32 %r83, %r54, %r14;
sub.rn.f32 %r84, %r55, %r15;
sub.rn.f32 %r85, %r56, %r16;
sub.rn.f32 %r86, %r57, %r17;
sub.rn.f32 %r87, %r58, %r18;
sub.rn.f32 %r88, %r59, %r19;
sub.rn.f32 %r89, %r60, %r20;
sub.rn.f32 %r90, %r61, %r21;
fma.rn.f32 %r33, %r83, 0f3CA3D70A, %r14;
fma.rn.f32 %r34, %r84, 0f3CA3D70A, %r15;
fma.rn.f32 %r35, %r85, 0f3CA3D70A, %r16;
fma.rn.f32 %r36, %r86, 0f3CA3D70A, %r17;
fma.rn.f32 %r37, %r87, 0f3CA3D70A, %r18;
fma.rn.f32 %r38, %r88, 0f3CA3D70A, %r19;
fma.rn.f32 %r39, %r89, 0f3CA3D70A, %r20;
fma.rn.f32 %r40, %r90, 0f3CA3D70A, %r21;
sub.rn.f32 %r91, %r62, %r24;
mul.rn.f32 %r92, %r33, %r91;
mul.rn.f32 %r93, %r34, %r91;
mul.rn.f32 %r94, %r35, %r91;
mul.rn.f32 %r95, %r36, %r91;
mul.rn.f32 %r96, %r37, %r91;
mul.rn.f32 %r97, %r38, %r91;
mul.rn.f32 %r98, %r39, %r91;
mul.rn.f32 %r99, %r40, %r91;
fma.rn.f32 %r100, %r24, %r25, %r92;
fma.rn.f32 %r101, %r24, %r26, %r93;
fma.rn.f32 %r102, %r24, %r27, %r94;
fma.rn.f32 %r103, %r24, %r28, %r95;
fma.rn.f32 %r104, %r24, %r29, %r96;
fma.rn.f32 %r105, %r24, %r30, %r97;
fma.rn.f32 %r106, %r24, %r31, %r98;
fma.rn.f32 %r107, %r24, %r32, %r99;
$L__tmp4:
abs.f32 %r108, %r22;
setp.ge.f32 %p5, %r108, 0f3F000000;
add.rn.f32 %r109, %r22, 0fBF800000;
selp.f32 %r110, %r109, %r22, %p5;
selp.f32 %r111, %r100, %r54, %p5;
selp.f32 %r112, %r101, %r55, %p5;
selp.f32 %r113, %r102, %r56, %p5;
selp.f32 %r114, %r103, %r57, %p5;
selp.f32 %r115, %r104, %r58, %p5;
selp.f32 %r116, %r105, %r59, %p5;
selp.f32 %r117, %r106, %r60, %p5;
selp.f32 %r118, %r107, %r61, %p5;
sub.rn.f32 %r119, %r100, %r54;
sub.rn.f32 %r120, %r101, %r55;
sub.rn.f32 %r121, %r102, %r56;
sub.rn.f32 %r122, %r103, %r57;
sub.rn.f32 %r123, %r104, %r58;
sub.rn.f32 %r124, %r105, %r59;
sub.rn.f32 %r125, %r106, %r60;
sub.rn.f32 %r126, %r107, %r61;
fma.rn.f32 %r127, %r110, %r119, %r111;
fma.rn.f32 %r128, %r110, %r120, %r112;
fma.rn.f32 %r129, %r110, %r121, %r113;
fma.rn.f32 %r130, %r110, %r122, %r114;
fma.rn.f32 %r131, %r110, %r123, %r115;
fma.rn.f32 %r132, %r110, %r124, %r116;
fma.rn.f32 %r133, %r110, %r125, %r117;
fma.rn.f32 %r134, %r110, %r126, %r118;
$L__tmp5:
st.global.v4.b32 [ %rd3 + 0 ], { %r25, %r26, %r27, %r28 };
st.global.v4.b32 [ %rd4 + 0 ], { %r29, %r30, %r31, %r32 };
st.global.v4.b32 [ %rd5 + 0 ], { %r33, %r34, %r35, %r36 };
st.global.v4.b32 [ %rd6 + 0 ], { %r37, %r38, %r39, %r40 };
add.s64 %rd10, %rd14, %rd15;
cvt.rn.bf16x2.f32 %r41, %r128, %r127;
cvt.rn.bf16x2.f32 %r42, %r130, %r129;
cvt.rn.bf16x2.f32 %r43, %r132, %r131;
cvt.rn.bf16x2.f32 %r44, %r134, %r133;
st.global.v4.b32 [ %rd10 + 0 ], { %r41, %r42, %r43, %r44 };
$L__tmp6:
ret;
$L__tmp7:
$L__func_end0:
}
"#
  | .mlp => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _prepare_rails_and_seed(
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_0,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_1,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_2,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_3,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_4,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_5,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_6,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_7,
.param .u64 .ptr .global .align 1 _prepare_rails_and_seed_param_8
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b16 %rs<9>;
.reg .b32 %r<135>;
.reg .b64 %rd<16>;
$L__func_begin0:
ld.param.b64 %rd11, [_prepare_rails_and_seed_param_0];
ld.param.b64 %rd12, [_prepare_rails_and_seed_param_1];
$L__tmp0:
mov.u32 %r45, %tid.x;
and.b32 %r46, %r45, 63;
ld.param.b64 %rd7, [_prepare_rails_and_seed_param_2];
ld.param.b64 %rd8, [_prepare_rails_and_seed_param_3];
ld.param.b64 %rd13, [_prepare_rails_and_seed_param_6];
mad.wide.u32 %rd1, %r46, 4, %rd13;
ld.param.b64 %rd9, [_prepare_rails_and_seed_param_4];
ld.param.b64 %rd14, [_prepare_rails_and_seed_param_5];
mov.u32 %r47, %ctaid.x;
setp.lt.u32 %p2, %r46, 18;
and.b32 %r48, %r45, 64;
or.b32 %r49, %r48, %r47;
setp.eq.b32 %p3, %r49, 0;
and.pred %p1, %p2, %p3;
mov.b32 %r1, 0;
@%p1 st.global.b32 [ %rd1 + 0 ], { %r1 };
$L__tmp1:
shl.b32 %r50, %r47, 10;
shl.b32 %r51, %r45, 3;
and.b32 %r52, %r51, 1016;
or.b32 %r53, %r52, %r50;
mul.wide.s32 %rd15, %r53, 2;
add.s64 %rd2, %rd11, %rd15;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs1, %rs2}, %r2;
mov.b32 {%rs3, %rs4}, %r3;
mov.b32 {%rs5, %rs6}, %r4;
mov.b32 {%rs7, %rs8}, %r5;
cvt.f32.bf16 %r54, %rs1;
cvt.f32.bf16 %r55, %rs2;
cvt.f32.bf16 %r56, %rs3;
cvt.f32.bf16 %r57, %rs4;
cvt.f32.bf16 %r58, %rs5;
cvt.f32.bf16 %r59, %rs6;
cvt.f32.bf16 %r60, %rs7;
cvt.f32.bf16 %r61, %rs8;
mad.wide.s32 %rd3, %r53, 4, %rd12;
add.s64 %rd4, %rd3, 16;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd3 + 0 ];
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd4 + 0 ];
add.s64 %rd5, %rd3, 25952256;
add.s64 %rd6, %rd3, 25952272;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd5 + 0 ];
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
mov.u32 %r21, 0x0;
ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd6 + 0 ];
mov.u32 %r22, 0x0;
ld.global.b32 { %r22 }, [ %rd7 + 0 ];
mov.u32 %r23, 0x0;
ld.global.b32 { %r23 }, [ %rd8 + 0 ];
mov.u32 %r24, 0x0;
ld.global.b32 { %r24 }, [ %rd9 + 0 ];
mov.b32 %r62, 0f3F800000;
sub.rn.f32 %r63, %r62, %r23;
$L__tmp2:
abs.f32 %r64, %r63;
setp.ge.f32 %p4, %r64, 0f3F000000;
add.rn.f32 %r65, %r63, 0fBF800000;
selp.f32 %r66, %r65, %r63, %p4;
selp.f32 %r67, %r54, %r6, %p4;
selp.f32 %r68, %r55, %r7, %p4;
selp.f32 %r69, %r56, %r8, %p4;
selp.f32 %r70, %r57, %r9, %p4;
selp.f32 %r71, %r58, %r10, %p4;
selp.f32 %r72, %r59, %r11, %p4;
selp.f32 %r73, %r60, %r12, %p4;
selp.f32 %r74, %r61, %r13, %p4;
sub.rn.f32 %r75, %r54, %r6;
sub.rn.f32 %r76, %r55, %r7;
sub.rn.f32 %r77, %r56, %r8;
sub.rn.f32 %r78, %r57, %r9;
sub.rn.f32 %r79, %r58, %r10;
sub.rn.f32 %r80, %r59, %r11;
sub.rn.f32 %r81, %r60, %r12;
sub.rn.f32 %r82, %r61, %r13;
fma.rn.f32 %r25, %r66, %r75, %r67;
fma.rn.f32 %r26, %r66, %r76, %r68;
fma.rn.f32 %r27, %r66, %r77, %r69;
fma.rn.f32 %r28, %r66, %r78, %r70;
fma.rn.f32 %r29, %r66, %r79, %r71;
fma.rn.f32 %r30, %r66, %r80, %r72;
fma.rn.f32 %r31, %r66, %r81, %r73;
fma.rn.f32 %r32, %r66, %r82, %r74;
$L__tmp3:
sub.rn.f32 %r83, %r54, %r14;
sub.rn.f32 %r84, %r55, %r15;
sub.rn.f32 %r85, %r56, %r16;
sub.rn.f32 %r86, %r57, %r17;
sub.rn.f32 %r87, %r58, %r18;
sub.rn.f32 %r88, %r59, %r19;
sub.rn.f32 %r89, %r60, %r20;
sub.rn.f32 %r90, %r61, %r21;
fma.rn.f32 %r33, %r83, 0f3CA3D70A, %r14;
fma.rn.f32 %r34, %r84, 0f3CA3D70A, %r15;
fma.rn.f32 %r35, %r85, 0f3CA3D70A, %r16;
fma.rn.f32 %r36, %r86, 0f3CA3D70A, %r17;
fma.rn.f32 %r37, %r87, 0f3CA3D70A, %r18;
fma.rn.f32 %r38, %r88, 0f3CA3D70A, %r19;
fma.rn.f32 %r39, %r89, 0f3CA3D70A, %r20;
fma.rn.f32 %r40, %r90, 0f3CA3D70A, %r21;
sub.rn.f32 %r91, %r62, %r24;
mul.rn.f32 %r92, %r33, %r91;
mul.rn.f32 %r93, %r34, %r91;
mul.rn.f32 %r94, %r35, %r91;
mul.rn.f32 %r95, %r36, %r91;
mul.rn.f32 %r96, %r37, %r91;
mul.rn.f32 %r97, %r38, %r91;
mul.rn.f32 %r98, %r39, %r91;
mul.rn.f32 %r99, %r40, %r91;
fma.rn.f32 %r100, %r24, %r25, %r92;
fma.rn.f32 %r101, %r24, %r26, %r93;
fma.rn.f32 %r102, %r24, %r27, %r94;
fma.rn.f32 %r103, %r24, %r28, %r95;
fma.rn.f32 %r104, %r24, %r29, %r96;
fma.rn.f32 %r105, %r24, %r30, %r97;
fma.rn.f32 %r106, %r24, %r31, %r98;
fma.rn.f32 %r107, %r24, %r32, %r99;
$L__tmp4:
abs.f32 %r108, %r22;
setp.ge.f32 %p5, %r108, 0f3F000000;
add.rn.f32 %r109, %r22, 0fBF800000;
selp.f32 %r110, %r109, %r22, %p5;
selp.f32 %r111, %r100, %r54, %p5;
selp.f32 %r112, %r101, %r55, %p5;
selp.f32 %r113, %r102, %r56, %p5;
selp.f32 %r114, %r103, %r57, %p5;
selp.f32 %r115, %r104, %r58, %p5;
selp.f32 %r116, %r105, %r59, %p5;
selp.f32 %r117, %r106, %r60, %p5;
selp.f32 %r118, %r107, %r61, %p5;
sub.rn.f32 %r119, %r100, %r54;
sub.rn.f32 %r120, %r101, %r55;
sub.rn.f32 %r121, %r102, %r56;
sub.rn.f32 %r122, %r103, %r57;
sub.rn.f32 %r123, %r104, %r58;
sub.rn.f32 %r124, %r105, %r59;
sub.rn.f32 %r125, %r106, %r60;
sub.rn.f32 %r126, %r107, %r61;
fma.rn.f32 %r127, %r110, %r119, %r111;
fma.rn.f32 %r128, %r110, %r120, %r112;
fma.rn.f32 %r129, %r110, %r121, %r113;
fma.rn.f32 %r130, %r110, %r122, %r114;
fma.rn.f32 %r131, %r110, %r123, %r115;
fma.rn.f32 %r132, %r110, %r124, %r116;
fma.rn.f32 %r133, %r110, %r125, %r117;
fma.rn.f32 %r134, %r110, %r126, %r118;
$L__tmp5:
st.global.v4.b32 [ %rd3 + 0 ], { %r25, %r26, %r27, %r28 };
st.global.v4.b32 [ %rd4 + 0 ], { %r29, %r30, %r31, %r32 };
st.global.v4.b32 [ %rd5 + 0 ], { %r33, %r34, %r35, %r36 };
st.global.v4.b32 [ %rd6 + 0 ], { %r37, %r38, %r39, %r40 };
add.s64 %rd10, %rd14, %rd15;
cvt.rn.bf16x2.f32 %r41, %r128, %r127;
cvt.rn.bf16x2.f32 %r42, %r130, %r129;
cvt.rn.bf16x2.f32 %r43, %r132, %r131;
cvt.rn.bf16x2.f32 %r44, %r134, %r133;
st.global.v4.b32 [ %rd10 + 0 ], { %r41, %r42, %r43, %r44 };
$L__tmp6:
ret;
$L__tmp7:
$L__func_end0:
}
"#

private def initialGraph : Bank → String
  | .qk => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0]], warp = [[1], [0], [0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_initial(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_8 = arith.constant dense<128> : tensor<2x1xi32, #blocked>
%cst_9 = arith.constant dense<257> : tensor<2x32xi32, #blocked>
%cst_10 = arith.constant dense<2> : tensor<2x1xi32, #blocked>
%cst_11 = arith.constant dense<0.000000e+00> : tensor<2x32xbf16, #blocked>
%cst_12 = arith.constant dense<8224> : tensor<2x32xi32, #blocked>
%cst_13 = arith.constant dense<16448> : tensor<2x32xi32, #blocked>
%cst_14 = arith.constant dense<24672> : tensor<2x32xi32, #blocked>
%cst_15 = arith.constant dense<8> : tensor<2x32xi32, #blocked>
%cst_16 = arith.constant dense<4> : tensor<2x32xi32, #blocked>
%cst_17 = arith.constant dense<2> : tensor<2x32xi32, #blocked>
%cst_18 = arith.constant dense<1> : tensor<2x32xi32, #blocked>
%cst_19 = arith.constant dense<16> : tensor<2x32xi32, #blocked>
%cst_20 = arith.constant dense<0> : tensor<2x1xi32, #blocked>
%cst_21 = arith.constant 1.050000e+00 : f32
%cst_22 = arith.constant 9.99999997E-7 : f32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_23 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_23 : i32 to i64
%row_24 = arith.shrui %c72_i64, %row : i64
%row_25 = arith.andi %row_24, %c7_i64 : i64
%row_26 = arith.trunci %row_25 : i64 to i32
%row_27 = arith.muli %row_26, 128 : i32
%col = arith.shrui %c64_i64, %row : i64
%col_28 = arith.andi %col, %c7_i64 : i64
%col_29 = arith.trunci %col_28 : i64 to i32
%col_30 = arith.muli %col_29, 128 : i32
%batch = arith.divsi %pid, 3 : i32
%offdiag = arith.cmpi ne, %row_27, %col_30 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%offdiag_31 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_32 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_33 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 256 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_27 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_40 = arith.addi %acc, %col_30 : i32
ttng.async_tma_copy_global_to_local %X[%coords_40, %c0_i32] %acc_39, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_41 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_42 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_43 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_44 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_44, %phase_43, %true deps %base, %acc_39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_39 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_45 = ttng.warp_group_dot %base, %right, %acc_42, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_46 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_47 = ttg.memdesc_index %barriers[%nxt_46] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_48 = ttg.memdesc_index %tiles[%nxt_46] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_47, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_48, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_49 = ttng.warp_group_dot_wait %acc_45 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords_40, %k] %acc_39, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_49 : tensor<128x128xf32, #mma>
}
scf.yield %acc_41 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 256 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_27 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_41 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_42 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_42, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %base, %right, %acc_41, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_45 = ttng.warp_group_dot_wait %acc_44 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_46 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_47 = ttg.memdesc_index %barriers[%nxt_46] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_48 = ttg.memdesc_index %tiles[%nxt_46] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_47, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_48, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_45 : tensor<128x128xf32, #mma>
}
%acc_40 = ttng.warp_group_dot_wait %acc_39 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_40 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %offdiag_31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %offdiag_32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %offdiag_33 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_34 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%offdiag_38 = arith.muli %batch, 256 : i32
%offdiag_39 = arith.addi %offdiag_38, %row_27 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_39, %col_30] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%offdiag_35 = arith.muli %batch, 256 : i32
%offdiag_36 = arith.addi %offdiag_35, %col_30 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_36, %row_27] %mirror_34 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%offdiag_37 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
scf.if %offdiag {
} else {
%0 = tt.elementwise_inline_asm "bar.sync 0; fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%ordinal = tt.addptr %Counter, %batch : !tt.ptr<i32>, i32
%ordinal_38 = tt.atomic_rmw add, acq_rel, gpu, %ordinal, 1, %true : (!tt.ptr<i32>, i32, i1) -> i32
%1 = arith.cmpi eq, %ordinal_38, 1 : i32
scf.if %1 {
%2 = tt.elementwise_inline_asm "fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%chunk = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%diagonal = tt.expand_dims %chunk {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<2x1xi32, #blocked>
%diagonal_39 = arith.muli %diagonal, %cst_8 : tensor<2x1xi32, #blocked>
%diagonal_40 = tt.expand_dims %lane {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
%diagonal_41 = tt.broadcast %diagonal_39 : tensor<2x1xi32, #blocked> -> tensor<2x32xi32, #blocked>
%diagonal_42 = tt.broadcast %diagonal_40 : tensor<1x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
%diagonal_43 = arith.addi %diagonal_41, %diagonal_42 : tensor<2x32xi32, #blocked>
%ptr = arith.muli %batch, 65536 : i32
%ptr_44 = tt.addptr %Gram_ptr, %ptr : !tt.ptr<bf16>, i32
%ptr_45 = arith.muli %diagonal_43, %cst_9 : tensor<2x32xi32, #blocked>
%ptr_46 = tt.splat %ptr_44 : !tt.ptr<bf16> -> tensor<2x32x!tt.ptr<bf16>, #blocked>
%ptr_47 = tt.addptr %ptr_46, %ptr_45 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%active = arith.cmpi slt, %diagonal, %cst_10 : tensor<2x1xi32, #blocked>
%x0 = tt.broadcast %active : tensor<2x1xi1, #blocked> -> tensor<2x32xi1, #blocked>
%x0_48 = tt.load %ptr_47, %x0, %cst_11 cacheModifier = cg : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x0_49 = arith.extf %x0_48 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x32 = tt.addptr %ptr_47, %cst_12 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x32_50 = tt.load %x32, %x0, %cst_11 cacheModifier = cg : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x32_51 = arith.extf %x32_50 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x64 = tt.addptr %ptr_47, %cst_13 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x64_52 = tt.load %x64, %x0, %cst_11 cacheModifier = cg : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x64_53 = arith.extf %x64_52 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x96 = tt.addptr %ptr_47, %cst_14 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x96_54 = tt.load %x96, %x0, %cst_11 cacheModifier = cg : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x96_55 = arith.extf %x96_54 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%partial = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x0_49 : tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_56 = arith.addf %partial, %x64_53 : tensor<2x32xf32, #blocked>
%partial_57 = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x32_51 : tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_58 = arith.addf %partial_57, %x96_55 : tensor<2x32xf32, #blocked>
%partial_59 = arith.addf %partial_56, %partial_58 : tensor<2x32xf32, #blocked>
%peer = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_59, %cst_15 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_60 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_59, %peer : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_61 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_60, %cst_16 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_62 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_60, %peer_61 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_63 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_62, %cst_17 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_64 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_62, %peer_63 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_65 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_64, %cst_18 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_66 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_64, %peer_65 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_67 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_66, %cst_19 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_68 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_66, %peer_67 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%sums = tt.gather %partial_68[%cst_20] {axis = 1 : i32} : (tensor<2x32xf32, #blocked>, tensor<2x1xi32, #blocked>) -> tensor<2x1xf32, #blocked>
%sums_69 = tt.reshape %sums : tensor<2x1xf32, #blocked> -> tensor<2xf32, #linear1>
%sums_70 = ttg.convert_layout %sums_69 : tensor<2xf32, #linear1> -> tensor<2xf32, #blocked1>
%trace = "tt.reduce"(%sums_70) <{axis = 0 : i32}> ({
^bb0(%trace_72: f32, %trace_73: f32):
%trace_74 = arith.addf %trace_72, %trace_73 : f32
tt.reduce.return %trace_74 : f32
}) : (tensor<2xf32, #blocked1>) -> f32
%denominator = tt.precise_sqrt %trace : f32
%denominator_71 = math.fma %denominator, %cst_21, %cst_22 : f32
%3 = arith.muli %batch, 2 : i32
%4 = tt.addptr %Divisors, %3 : !tt.ptr<f32>, i32
tt.store %4, %denominator_71 : !tt.ptr<f32>
%5 = tt.addptr %4, 1 : !tt.ptr<f32>, i32
%6 = arith.mulf %denominator_71, %denominator_71 : f32
tt.store %5, %6 : !tt.ptr<f32>
} else {
}
}
tt.return
}
}
"#
  | .vo => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0]], warp = [[1], [2], [4]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_initial(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_8 = arith.constant dense<128> : tensor<8x1xi32, #blocked>
%cst_9 = arith.constant dense<769> : tensor<8x32xi32, #blocked>
%cst_10 = arith.constant dense<6> : tensor<8x1xi32, #blocked>
%cst_11 = arith.constant dense<0.000000e+00> : tensor<8x32xbf16, #blocked>
%cst_12 = arith.constant dense<24608> : tensor<8x32xi32, #blocked>
%cst_13 = arith.constant dense<49216> : tensor<8x32xi32, #blocked>
%cst_14 = arith.constant dense<73824> : tensor<8x32xi32, #blocked>
%cst_15 = arith.constant dense<8> : tensor<8x32xi32, #blocked>
%cst_16 = arith.constant dense<4> : tensor<8x32xi32, #blocked>
%cst_17 = arith.constant dense<2> : tensor<8x32xi32, #blocked>
%cst_18 = arith.constant dense<1> : tensor<8x32xi32, #blocked>
%cst_19 = arith.constant dense<16> : tensor<8x32xi32, #blocked>
%cst_20 = arith.constant dense<0> : tensor<8x1xi32, #blocked>
%cst_21 = arith.constant 1.050000e+00 : f32
%cst_22 = arith.constant 9.99999997E-7 : f32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_23 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_23 : i32 to i64
%row_24 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_25 = arith.andi %row_24, %c7_i64 : i64
%row_26 = arith.trunci %row_25 : i64 to i32
%row_27 = arith.muli %row_26, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_28 = arith.andi %col, %c7_i64 : i64
%col_29 = arith.trunci %col_28 : i64 to i32
%col_30 = arith.muli %col_29, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_27, %col_30 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%offdiag_31 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_32 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_33 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 768 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_27 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_40 = arith.addi %acc, %col_30 : i32
ttng.async_tma_copy_global_to_local %X[%coords_40, %c0_i32] %acc_39, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_41 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_42 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_43 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_44 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_44, %phase_43, %true deps %base, %acc_39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_39 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_45 = ttng.warp_group_dot %base, %right, %acc_42, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_46 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_47 = ttg.memdesc_index %barriers[%nxt_46] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_48 = ttg.memdesc_index %tiles[%nxt_46] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_47, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_48, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_49 = ttng.warp_group_dot_wait %acc_45 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords_40, %k] %acc_39, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_49 : tensor<128x128xf32, #mma>
}
scf.yield %acc_41 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 768 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_27 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_41 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_42 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_42, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %base, %right, %acc_41, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_45 = ttng.warp_group_dot_wait %acc_44 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_46 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_47 = ttg.memdesc_index %barriers[%nxt_46] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_48 = ttg.memdesc_index %tiles[%nxt_46] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_47, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_48, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_45 : tensor<128x128xf32, #mma>
}
%acc_40 = ttng.warp_group_dot_wait %acc_39 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_40 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %offdiag_31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %offdiag_32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %offdiag_33 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_34 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%offdiag_38 = arith.muli %batch, 768 : i32
%offdiag_39 = arith.addi %offdiag_38, %row_27 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_39, %col_30] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%offdiag_35 = arith.muli %batch, 768 : i32
%offdiag_36 = arith.addi %offdiag_35, %col_30 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_36, %row_27] %mirror_34 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%offdiag_37 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
scf.if %offdiag {
} else {
%0 = tt.elementwise_inline_asm "bar.sync 0; fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%ordinal = tt.addptr %Counter, %batch : !tt.ptr<i32>, i32
%ordinal_38 = tt.atomic_rmw add, acq_rel, gpu, %ordinal, 1, %true : (!tt.ptr<i32>, i32, i1) -> i32
%1 = arith.cmpi eq, %ordinal_38, 5 : i32
scf.if %1 {
%2 = tt.elementwise_inline_asm "fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%chunk = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%diagonal = tt.expand_dims %chunk {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
%diagonal_39 = arith.muli %diagonal, %cst_8 : tensor<8x1xi32, #blocked>
%diagonal_40 = tt.expand_dims %lane {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
%diagonal_41 = tt.broadcast %diagonal_39 : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_42 = tt.broadcast %diagonal_40 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_43 = arith.addi %diagonal_41, %diagonal_42 : tensor<8x32xi32, #blocked>
%ptr = arith.muli %batch, 589824 : i32
%ptr_44 = tt.addptr %Gram_ptr, %ptr : !tt.ptr<bf16>, i32
%ptr_45 = arith.muli %diagonal_43, %cst_9 : tensor<8x32xi32, #blocked>
%ptr_46 = tt.splat %ptr_44 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked>
%ptr_47 = tt.addptr %ptr_46, %ptr_45 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%active = arith.cmpi slt, %diagonal, %cst_10 : tensor<8x1xi32, #blocked>
%x0 = tt.broadcast %active : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
%x0_48 = tt.load %ptr_47, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x0_49 = arith.extf %x0_48 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x32 = tt.addptr %ptr_47, %cst_12 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x32_50 = tt.load %x32, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x32_51 = arith.extf %x32_50 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x64 = tt.addptr %ptr_47, %cst_13 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x64_52 = tt.load %x64, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x64_53 = arith.extf %x64_52 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x96 = tt.addptr %ptr_47, %cst_14 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x96_54 = tt.load %x96, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x96_55 = arith.extf %x96_54 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%partial = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x0_49 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_56 = arith.addf %partial, %x64_53 : tensor<8x32xf32, #blocked>
%partial_57 = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x32_51 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_58 = arith.addf %partial_57, %x96_55 : tensor<8x32xf32, #blocked>
%partial_59 = arith.addf %partial_56, %partial_58 : tensor<8x32xf32, #blocked>
%peer = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_59, %cst_15 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_60 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_59, %peer : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_61 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_60, %cst_16 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_62 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_60, %peer_61 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_63 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_62, %cst_17 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_64 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_62, %peer_63 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_65 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_64, %cst_18 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_66 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_64, %peer_65 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_67 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_66, %cst_19 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_68 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_66, %peer_67 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%sums = tt.gather %partial_68[%cst_20] {axis = 1 : i32} : (tensor<8x32xf32, #blocked>, tensor<8x1xi32, #blocked>) -> tensor<8x1xf32, #blocked>
%sums_69 = tt.reshape %sums : tensor<8x1xf32, #blocked> -> tensor<8xf32, #linear1>
%sums_70 = ttg.convert_layout %sums_69 : tensor<8xf32, #linear1> -> tensor<8xf32, #blocked1>
%trace = "tt.reduce"(%sums_70) <{axis = 0 : i32}> ({
^bb0(%trace_72: f32, %trace_73: f32):
%trace_74 = arith.addf %trace_72, %trace_73 : f32
tt.reduce.return %trace_74 : f32
}) : (tensor<8xf32, #blocked1>) -> f32
%denominator = tt.precise_sqrt %trace : f32
%denominator_71 = math.fma %denominator, %cst_21, %cst_22 : f32
%3 = arith.muli %batch, 2 : i32
%4 = tt.addptr %Divisors, %3 : !tt.ptr<f32>, i32
tt.store %4, %denominator_71 : !tt.ptr<f32>
%5 = tt.addptr %4, 1 : !tt.ptr<f32>, i32
%6 = arith.mulf %denominator_71, %denominator_71 : f32
tt.store %5, %6 : !tt.ptr<f32>
} else {
}
}
tt.return
}
}
"#
  | .mlp => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#linear1 = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0]], warp = [[1], [2], [4]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_initial(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_8 = arith.constant dense<128> : tensor<8x1xi32, #blocked>
%cst_9 = arith.constant dense<769> : tensor<8x32xi32, #blocked>
%cst_10 = arith.constant dense<6> : tensor<8x1xi32, #blocked>
%cst_11 = arith.constant dense<0.000000e+00> : tensor<8x32xbf16, #blocked>
%cst_12 = arith.constant dense<24608> : tensor<8x32xi32, #blocked>
%cst_13 = arith.constant dense<49216> : tensor<8x32xi32, #blocked>
%cst_14 = arith.constant dense<73824> : tensor<8x32xi32, #blocked>
%cst_15 = arith.constant dense<8> : tensor<8x32xi32, #blocked>
%cst_16 = arith.constant dense<4> : tensor<8x32xi32, #blocked>
%cst_17 = arith.constant dense<2> : tensor<8x32xi32, #blocked>
%cst_18 = arith.constant dense<1> : tensor<8x32xi32, #blocked>
%cst_19 = arith.constant dense<16> : tensor<8x32xi32, #blocked>
%cst_20 = arith.constant dense<0> : tensor<8x1xi32, #blocked>
%cst_21 = arith.constant 1.050000e+00 : f32
%cst_22 = arith.constant 9.99999997E-7 : f32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_23 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_23 : i32 to i64
%row_24 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_25 = arith.andi %row_24, %c7_i64 : i64
%row_26 = arith.trunci %row_25 : i64 to i32
%row_27 = arith.muli %row_26, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_28 = arith.andi %col, %c7_i64 : i64
%col_29 = arith.trunci %col_28 : i64 to i32
%col_30 = arith.muli %col_29, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_27, %col_30 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%offdiag_31 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_32 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %offdiag_32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%offdiag_33 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 2816 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %row_27] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %col_30] %acc_39, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_40 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_41 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_42 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_42, %true deps %base, %acc_39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %left, %acc_39, %acc_41, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_45 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_46 = ttg.memdesc_index %barriers[%nxt_45] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_47 = ttg.memdesc_index %tiles[%nxt_45] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_46, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %X[%coords, %row_27] %acc_47, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_48 = ttng.warp_group_dot_wait %acc_44 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords, %col_30] %acc_39, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_48 : tensor<128x128xf32, #mma>
}
scf.yield %acc_40 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 2816 : i32
%acc_38 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %offdiag_31, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %row_27] %acc_38, %offdiag_31, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_41 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_42 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_42, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %left, %base, %acc_41, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_45 = ttng.warp_group_dot_wait %acc_44 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_46 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_47 = ttg.memdesc_index %barriers[%nxt_46] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_48 = ttg.memdesc_index %tiles[%nxt_46] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_47, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %X[%coords, %row_27] %acc_48, %acc_47, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_45 : tensor<128x128xf32, #mma>
}
%acc_40 = ttng.warp_group_dot_wait %acc_39 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_40 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %offdiag_31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %offdiag_32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %offdiag_33 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_34 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%offdiag_38 = arith.muli %batch, 768 : i32
%offdiag_39 = arith.addi %offdiag_38, %row_27 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_39, %col_30] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%offdiag_35 = arith.muli %batch, 768 : i32
%offdiag_36 = arith.addi %offdiag_35, %col_30 : i32
ttng.async_tma_copy_local_to_global %Gram[%offdiag_36, %row_27] %mirror_34 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%offdiag_37 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
scf.if %offdiag {
} else {
%0 = tt.elementwise_inline_asm "bar.sync 0; fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%ordinal = tt.addptr %Counter, %batch : !tt.ptr<i32>, i32
%ordinal_38 = tt.atomic_rmw add, acq_rel, gpu, %ordinal, 1, %true : (!tt.ptr<i32>, i32, i1) -> i32
%1 = arith.cmpi eq, %ordinal_38, 5 : i32
scf.if %1 {
%2 = tt.elementwise_inline_asm "fence.proxy.async.global; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%chunk = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%diagonal = tt.expand_dims %chunk {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
%diagonal_39 = arith.muli %diagonal, %cst_8 : tensor<8x1xi32, #blocked>
%diagonal_40 = tt.expand_dims %lane {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
%diagonal_41 = tt.broadcast %diagonal_39 : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_42 = tt.broadcast %diagonal_40 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_43 = arith.addi %diagonal_41, %diagonal_42 : tensor<8x32xi32, #blocked>
%ptr = arith.muli %batch, 589824 : i32
%ptr_44 = tt.addptr %Gram_ptr, %ptr : !tt.ptr<bf16>, i32
%ptr_45 = arith.muli %diagonal_43, %cst_9 : tensor<8x32xi32, #blocked>
%ptr_46 = tt.splat %ptr_44 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked>
%ptr_47 = tt.addptr %ptr_46, %ptr_45 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%active = arith.cmpi slt, %diagonal, %cst_10 : tensor<8x1xi32, #blocked>
%x0 = tt.broadcast %active : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
%x0_48 = tt.load %ptr_47, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x0_49 = arith.extf %x0_48 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x32 = tt.addptr %ptr_47, %cst_12 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x32_50 = tt.load %x32, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x32_51 = arith.extf %x32_50 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x64 = tt.addptr %ptr_47, %cst_13 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x64_52 = tt.load %x64, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x64_53 = arith.extf %x64_52 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x96 = tt.addptr %ptr_47, %cst_14 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x96_54 = tt.load %x96, %x0, %cst_11 cacheModifier = cg : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x96_55 = arith.extf %x96_54 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%partial = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x0_49 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_56 = arith.addf %partial, %x64_53 : tensor<8x32xf32, #blocked>
%partial_57 = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x32_51 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_58 = arith.addf %partial_57, %x96_55 : tensor<8x32xf32, #blocked>
%partial_59 = arith.addf %partial_56, %partial_58 : tensor<8x32xf32, #blocked>
%peer = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_59, %cst_15 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_60 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_59, %peer : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_61 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_60, %cst_16 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_62 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_60, %peer_61 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_63 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_62, %cst_17 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_64 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_62, %peer_63 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_65 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_64, %cst_18 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_66 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_64, %peer_65 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_67 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_66, %cst_19 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_68 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_66, %peer_67 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%sums = tt.gather %partial_68[%cst_20] {axis = 1 : i32} : (tensor<8x32xf32, #blocked>, tensor<8x1xi32, #blocked>) -> tensor<8x1xf32, #blocked>
%sums_69 = tt.reshape %sums : tensor<8x1xf32, #blocked> -> tensor<8xf32, #linear1>
%sums_70 = ttg.convert_layout %sums_69 : tensor<8xf32, #linear1> -> tensor<8xf32, #blocked1>
%trace = "tt.reduce"(%sums_70) <{axis = 0 : i32}> ({
^bb0(%trace_72: f32, %trace_73: f32):
%trace_74 = arith.addf %trace_72, %trace_73 : f32
tt.reduce.return %trace_74 : f32
}) : (tensor<8xf32, #blocked1>) -> f32
%denominator = tt.precise_sqrt %trace : f32
%denominator_71 = math.fma %denominator, %cst_21, %cst_22 : f32
%3 = arith.muli %batch, 2 : i32
%4 = tt.addptr %Divisors, %3 : !tt.ptr<f32>, i32
tt.store %4, %denominator_71 : !tt.ptr<f32>
%5 = tt.addptr %4, 1 : !tt.ptr<f32>, i32
%6 = arith.mulf %denominator_71, %denominator_71 : f32
tt.store %5, %6 : !tt.ptr<f32>
} else {
}
}
tt.return
}
}
"#

private def initialAssembly : Bank → String
  | .qk => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_initial(
.param .align 64 .b8 cooperative_initial_param_0[128],
.param .u32 cooperative_initial_param_1,
.param .u32 cooperative_initial_param_2,
.param .u64 cooperative_initial_param_3,
.param .u64 cooperative_initial_param_4,
.param .align 64 .b8 cooperative_initial_param_5[128],
.param .u32 cooperative_initial_param_6,
.param .u32 cooperative_initial_param_7,
.param .u64 cooperative_initial_param_8,
.param .u64 cooperative_initial_param_9,
.param .u64 .ptr .global .align 1 cooperative_initial_param_10,
.param .u64 .ptr .global .align 1 cooperative_initial_param_11,
.param .u64 .ptr .global .align 1 cooperative_initial_param_12,
.param .u64 .ptr .global .align 1 cooperative_initial_param_13,
.param .u64 .ptr .global .align 1 cooperative_initial_param_14
)
.reqntid 256
{
.reg .pred %p<37>;
.reg .b16 %rs<8>;
.reg .b32 %r<265>;
.reg .b64 %rd<52>;
$L__func_begin0:
mov.b64 %rd4, cooperative_initial_param_0;
mov.b64 %rd5, cooperative_initial_param_5;
$L__tmp0:
cvta.param.u64 %rd43, %rd5;
cvta.param.u64 %rd24, %rd4;
mov.u32 %r47, %ctaid.x;
mul.hi.u32 %r1, %r47, 1431655766;
$L__tmp1:
mul.lo.s32 %r48, %r1, 3;
sub.s32 %r49, %r47, %r48;
mul.lo.s32 %r50, %r49, 3;
mov.b32 %r51, 9216;
shr.u32 %r52, %r51, %r50;
and.b32 %r2, %r52, 896;
mov.b32 %r53, 8192;
shr.u32 %r54, %r53, %r50;
and.b32 %r3, %r54, 896;
$L__tmp2:
setp.eq.b32 %p2, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p1, %r5, 0;
mov.b32 %r55, global_smem;
add.s32 %r85, %r55, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r85], 1;
bar.sync 0;
add.s32 %r46, %r55, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r46], 1;
shl.b32 %r6, %r1, 8;
$L__tmp3:
add.s32 %r136, %r2, %r6;
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p36, %r5, 64;
add.s32 %r71, %r3, %r6;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r85], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r88, %r7, 0, 31, -1;
elect.sync %r89|%p18, -1;
and.pred %p17, %p36, %p18;
and.b32 %r90, %r88, 1;
shl.b32 %r91, %r90, 14;
add.s32 %r86, %r55, %r91;
shl.b32 %r87, %r90, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r86], [%rd24, {%r87, %r136}], [%r85];
shl.b32 %r92, %r88, 7;
and.b32 %r10, %r92, 512;
$L__tmp6:
or.b32 %r261, %r87, 128;
mov.b32 %r262, 0;
mov.b32 %r196, 0f00000000;
mov.b32 %r197, %r196;
mov.b32 %r198, %r196;
mov.b32 %r199, %r196;
mov.b32 %r200, %r196;
mov.b32 %r201, %r196;
mov.b32 %r202, %r196;
mov.b32 %r203, %r196;
mov.b32 %r204, %r196;
mov.b32 %r205, %r196;
mov.b32 %r206, %r196;
mov.b32 %r207, %r196;
mov.b32 %r208, %r196;
mov.b32 %r209, %r196;
mov.b32 %r210, %r196;
mov.b32 %r211, %r196;
mov.b32 %r212, %r196;
mov.b32 %r213, %r196;
mov.b32 %r214, %r196;
mov.b32 %r215, %r196;
mov.b32 %r216, %r196;
mov.b32 %r217, %r196;
mov.b32 %r218, %r196;
mov.b32 %r219, %r196;
mov.b32 %r220, %r196;
mov.b32 %r221, %r196;
mov.b32 %r222, %r196;
mov.b32 %r223, %r196;
mov.b32 %r224, %r196;
mov.b32 %r225, %r196;
mov.b32 %r226, %r196;
mov.b32 %r227, %r196;
mov.b32 %r228, %r196;
mov.b32 %r229, %r196;
mov.b32 %r230, %r196;
mov.b32 %r231, %r196;
mov.b32 %r232, %r196;
mov.b32 %r233, %r196;
mov.b32 %r234, %r196;
mov.b32 %r235, %r196;
mov.b32 %r236, %r196;
mov.b32 %r237, %r196;
mov.b32 %r238, %r196;
mov.b32 %r239, %r196;
mov.b32 %r240, %r196;
mov.b32 %r241, %r196;
mov.b32 %r242, %r196;
mov.b32 %r243, %r196;
mov.b32 %r244, %r196;
mov.b32 %r245, %r196;
mov.b32 %r246, %r196;
mov.b32 %r247, %r196;
mov.b32 %r248, %r196;
mov.b32 %r249, %r196;
mov.b32 %r250, %r196;
mov.b32 %r251, %r196;
mov.b32 %r252, %r196;
mov.b32 %r253, %r196;
mov.b32 %r254, %r196;
mov.b32 %r255, %r196;
mov.b32 %r256, %r196;
mov.b32 %r257, %r196;
mov.b32 %r258, %r196;
mov.b32 %r259, %r196;
$L__BB0_4:
and.b32 %r97, %r262, 1;
bfe.u32 %r94, %r262, 1, 1;
shl.b32 %r98, %r97, 15;
add.s32 %r99, %r55, %r98;
mad.lo.s32 %r100, %r97, -32760, %r99;
add.s32 %r93, %r100, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r93], %r94;
@!complete bra.uni waitLoop;
}
shr.u32 %r101, %r99, 4;
add.s32 %r102, %r101, %r10;
and.b32 %r103, %r102, 16383;
cvt.u64.u32 %rd41, %r103;
bfe.u32 %r104, %r99, 4, 14;
cvt.u64.u32 %rd42, %r104;
wgmma.fence.sync.aligned;
or.b64 %rd25, %rd41, 4611686293305294848;
or.b64 %rd26, %rd42, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd25, %rd26, %p19, 1, 1, 0, 0;
add.s64 %rd27, %rd41, 4611686293305294850;
add.s64 %rd28, %rd42, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd27, %rd28, %p19, 1, 1, 0, 0;
add.s64 %rd29, %rd41, 4611686293305294852;
add.s64 %rd30, %rd42, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd29, %rd30, %p19, 1, 1, 0, 0;
add.s64 %rd31, %rd41, 4611686293305294854;
add.s64 %rd32, %rd42, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd31, %rd32, %p19, 1, 1, 0, 0;
add.s64 %rd33, %rd41, 4611686293305295872;
add.s64 %rd34, %rd42, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd33, %rd34, %p19, 1, 1, 0, 0;
add.s64 %rd35, %rd41, 4611686293305295874;
add.s64 %rd36, %rd42, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd35, %rd36, %p19, 1, 1, 0, 0;
add.s64 %rd37, %rd41, 4611686293305295876;
add.s64 %rd38, %rd42, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd37, %rd38, %p19, 1, 1, 0, 0;
add.s64 %rd39, %rd41, 4611686293305295878;
add.s64 %rd40, %rd42, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd39, %rd40, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r11, %r262, 1;
and.b32 %r105, %r11, 1;
setp.ne.b32 %p22, %r262, 5;
shl.b32 %r106, %r105, 3;
add.s32 %r107, %r55, %r106;
add.s32 %r95, %r107, 98304;
shl.b32 %r108, %r105, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r95], 32768;
bar.sync 0;
elect.sync %r109|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p36, %p24;
add.s32 %r96, %r86, %r108;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r96], [%rd24, {%r261, %r136}], [%r95];
$L__tmp8:
add.s32 %r261, %r261, 128;
setp.ne.b32 %p25, %r11, 6;
mov.b32 %r262, %r11;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r85], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r58, %r7, 0, 31, -1;
elect.sync %r59|%p5, -1;
and.pred %p3, %p36, %p5;
and.b32 %r60, %r58, 1;
shl.b32 %r61, %r60, 14;
add.s32 %r56, %r55, %r61;
shl.b32 %r57, %r60, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r56], [%rd24, {%r57, %r136}], [%r85];
$L__tmp10:
bar.sync 0;
elect.sync %r62|%p6, -1;
and.pred %p4, %p36, %p6;
add.s32 %r70, %r56, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r70], [%rd24, {%r57, %r71}], [%r85];
shl.b32 %r63, %r58, 7;
and.b32 %r8, %r63, 512;
add.s32 %r64, %r55, 65536;
bfe.u32 %r65, %r64, 4, 14;
cvt.u64.u32 %rd6, %r65;
or.b64 %rd8, %rd6, 4611686293305294848;
add.s64 %rd10, %rd6, 4611686293305294850;
add.s64 %rd12, %rd6, 4611686293305294852;
add.s64 %rd14, %rd6, 4611686293305294854;
add.s64 %rd16, %rd6, 4611686293305295872;
add.s64 %rd18, %rd6, 4611686293305295874;
add.s64 %rd20, %rd6, 4611686293305295876;
add.s64 %rd22, %rd6, 4611686293305295878;
$L__tmp11:
or.b32 %r195, %r57, 128;
mov.b32 %r260, 0;
mov.b32 %r196, 0f00000000;
mov.b32 %r197, %r196;
mov.b32 %r198, %r196;
mov.b32 %r199, %r196;
mov.b32 %r200, %r196;
mov.b32 %r201, %r196;
mov.b32 %r202, %r196;
mov.b32 %r203, %r196;
mov.b32 %r204, %r196;
mov.b32 %r205, %r196;
mov.b32 %r206, %r196;
mov.b32 %r207, %r196;
mov.b32 %r208, %r196;
mov.b32 %r209, %r196;
mov.b32 %r210, %r196;
mov.b32 %r211, %r196;
mov.b32 %r212, %r196;
mov.b32 %r213, %r196;
mov.b32 %r214, %r196;
mov.b32 %r215, %r196;
mov.b32 %r216, %r196;
mov.b32 %r217, %r196;
mov.b32 %r218, %r196;
mov.b32 %r219, %r196;
mov.b32 %r220, %r196;
mov.b32 %r221, %r196;
mov.b32 %r222, %r196;
mov.b32 %r223, %r196;
mov.b32 %r224, %r196;
mov.b32 %r225, %r196;
mov.b32 %r226, %r196;
mov.b32 %r227, %r196;
mov.b32 %r228, %r196;
mov.b32 %r229, %r196;
mov.b32 %r230, %r196;
mov.b32 %r231, %r196;
mov.b32 %r232, %r196;
mov.b32 %r233, %r196;
mov.b32 %r234, %r196;
mov.b32 %r235, %r196;
mov.b32 %r236, %r196;
mov.b32 %r237, %r196;
mov.b32 %r238, %r196;
mov.b32 %r239, %r196;
mov.b32 %r240, %r196;
mov.b32 %r241, %r196;
mov.b32 %r242, %r196;
mov.b32 %r243, %r196;
mov.b32 %r244, %r196;
mov.b32 %r245, %r196;
mov.b32 %r246, %r196;
mov.b32 %r247, %r196;
mov.b32 %r248, %r196;
mov.b32 %r249, %r196;
mov.b32 %r250, %r196;
mov.b32 %r251, %r196;
mov.b32 %r252, %r196;
mov.b32 %r253, %r196;
mov.b32 %r254, %r196;
mov.b32 %r255, %r196;
mov.b32 %r256, %r196;
mov.b32 %r257, %r196;
mov.b32 %r258, %r196;
mov.b32 %r259, %r196;
$L__BB0_2:
and.b32 %r72, %r260, 1;
bfe.u32 %r67, %r260, 1, 1;
shl.b32 %r73, %r72, 15;
add.s32 %r74, %r55, %r73;
mad.lo.s32 %r75, %r72, -32760, %r74;
add.s32 %r66, %r75, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r66], %r67;
@!complete bra.uni waitLoop;
}
shr.u32 %r76, %r74, 4;
add.s32 %r77, %r76, %r8;
and.b32 %r78, %r77, 16383;
cvt.u64.u32 %rd23, %r78;
wgmma.fence.sync.aligned;
or.b64 %rd7, %rd23, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd7, %rd8, %p7, 1, 1, 0, 0;
add.s64 %rd9, %rd23, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd9, %rd10, %p7, 1, 1, 0, 0;
add.s64 %rd11, %rd23, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd11, %rd12, %p7, 1, 1, 0, 0;
add.s64 %rd13, %rd23, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd13, %rd14, %p7, 1, 1, 0, 0;
add.s64 %rd15, %rd23, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd15, %rd16, %p7, 1, 1, 0, 0;
add.s64 %rd17, %rd23, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd17, %rd18, %p7, 1, 1, 0, 0;
add.s64 %rd19, %rd23, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd19, %rd20, %p7, 1, 1, 0, 0;
add.s64 %rd21, %rd23, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259}, %rd21, %rd22, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r9, %r260, 1;
and.b32 %r79, %r9, 1;
setp.ne.b32 %p11, %r260, 5;
shl.b32 %r80, %r79, 3;
add.s32 %r81, %r55, %r80;
add.s32 %r68, %r81, 98304;
shl.b32 %r82, %r79, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r68], 65536;
bar.sync 0;
elect.sync %r83|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p36, %p13;
add.s32 %r69, %r56, %r82;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r69], [%rd24, {%r195, %r136}], [%r68];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r84|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p36, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r70], [%rd24, {%r195, %r71}], [%r68];
$L__tmp15:
add.s32 %r195, %r195, 128;
setp.eq.b32 %p16, %r9, 6;
mov.b32 %r260, %r9;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r2, %r3;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r85];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r46];
cvt.rn.bf16x2.f32 %r12, %r197, %r196;
cvt.rn.bf16x2.f32 %r13, %r199, %r198;
cvt.rn.bf16x2.f32 %r14, %r201, %r200;
cvt.rn.bf16x2.f32 %r15, %r203, %r202;
cvt.rn.bf16x2.f32 %r16, %r205, %r204;
cvt.rn.bf16x2.f32 %r17, %r207, %r206;
cvt.rn.bf16x2.f32 %r18, %r209, %r208;
cvt.rn.bf16x2.f32 %r19, %r211, %r210;
cvt.rn.bf16x2.f32 %r20, %r213, %r212;
cvt.rn.bf16x2.f32 %r21, %r215, %r214;
cvt.rn.bf16x2.f32 %r22, %r217, %r216;
cvt.rn.bf16x2.f32 %r23, %r219, %r218;
cvt.rn.bf16x2.f32 %r24, %r221, %r220;
cvt.rn.bf16x2.f32 %r25, %r223, %r222;
cvt.rn.bf16x2.f32 %r26, %r225, %r224;
cvt.rn.bf16x2.f32 %r27, %r227, %r226;
cvt.rn.bf16x2.f32 %r28, %r229, %r228;
cvt.rn.bf16x2.f32 %r29, %r231, %r230;
cvt.rn.bf16x2.f32 %r30, %r233, %r232;
cvt.rn.bf16x2.f32 %r31, %r235, %r234;
cvt.rn.bf16x2.f32 %r32, %r237, %r236;
cvt.rn.bf16x2.f32 %r33, %r239, %r238;
cvt.rn.bf16x2.f32 %r34, %r241, %r240;
cvt.rn.bf16x2.f32 %r35, %r243, %r242;
cvt.rn.bf16x2.f32 %r36, %r245, %r244;
cvt.rn.bf16x2.f32 %r37, %r247, %r246;
cvt.rn.bf16x2.f32 %r38, %r249, %r248;
cvt.rn.bf16x2.f32 %r39, %r251, %r250;
cvt.rn.bf16x2.f32 %r40, %r253, %r252;
cvt.rn.bf16x2.f32 %r41, %r255, %r254;
cvt.rn.bf16x2.f32 %r42, %r257, %r256;
cvt.rn.bf16x2.f32 %r43, %r259, %r258;
shl.b32 %r44, %r4, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r124, %r4, 4;
and.b32 %r264, %r124, 112;
shl.b32 %r263, %r4, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r110, %r44, 1920;
shl.b32 %r263, %r4, 6;
and.b32 %r111, %r263, 14336;
shl.b32 %r112, %r4, 4;
and.b32 %r264, %r112, 112;
and.b32 %r113, %r4, 16;
or.b32 %r114, %r110, %r264;
xor.b32 %r115, %r114, %r113;
or.b32 %r116, %r115, %r111;
add.s32 %r117, %r55, %r116;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r117], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r117+16384], {%r28, %r29, %r30, %r31};
xor.b32 %r118, %r116, 32;
add.s32 %r119, %r55, %r118;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r119], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r119+16384], {%r32, %r33, %r34, %r35};
xor.b32 %r120, %r116, 64;
add.s32 %r121, %r55, %r120;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r121], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r121+16384], {%r36, %r37, %r38, %r39};
xor.b32 %r122, %r116, 96;
add.s32 %r123, %r55, %r122;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123+16384], {%r40, %r41, %r42, %r43};
$L__BB0_9:
and.b32 %r125, %r44, 17280;
and.b32 %r126, %r4, 96;
shl.b32 %r127, %r4, 1;
and.b32 %r128, %r127, 16;
and.b32 %r129, %r263, 1024;
or.b32 %r130, %r125, %r264;
or.b32 %r131, %r128, %r126;
xor.b32 %r132, %r130, %r131;
add.s32 %r133, %r55, %r129;
add.s32 %r134, %r133, %r132;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+32768], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+34816], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+36864], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+38912], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+40960], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+43008], {%r32, %r33, %r34, %r35};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+45056], {%r36, %r37, %r38, %r39};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+47104], {%r40, %r41, %r42, %r43};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_11;
bar.sync 0;
elect.sync %r138|%p28, -1;
shfl.sync.idx.b32 %r139, %r7, 0, 31, -1;
and.pred %p27, %p36, %p28;
and.b32 %r140, %r139, 1;
shl.b32 %r141, %r140, 14;
add.s32 %r137, %r55, %r141;
shl.b32 %r142, %r140, 6;
or.b32 %r135, %r142, %r3;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd43, {%r135, %r136}], [%r137];
cp.async.bulk.commit_group;
$L__BB0_11:
bar.sync 0;
elect.sync %r146|%p30, -1;
shfl.sync.idx.b32 %r147, %r7, 0, 31, -1;
and.pred %p29, %p36, %p30;
and.b32 %r148, %r147, 1;
shl.b32 %r149, %r148, 14;
add.s32 %r150, %r55, %r149;
add.s32 %r144, %r150, 32768;
shl.b32 %r151, %r148, 6;
or.b32 %r143, %r151, %r2;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd43, {%r143, %r71}], [%r144];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r145, 0;
$L__tmp17:
@%p26 bra $L__BB0_14;
ld.param.b64 %rd3, [cooperative_initial_param_12];
and.b32 %r45, %r4, 31;
bar.sync 0; fence.proxy.async.global; mov.u32 %r152, 0;
mad.wide.u32 %rd44, %r1, 4, %rd3;
setp.eq.b32 %p31, %r45, 0;
setp.lt.u32 %p32, %r5, 32;
and.pred %p35, %p31, %p32;
mov.b32 %r154, 1;
mov.u32 %r153, 0x0;
@%p35 atom.global.gpu.acq_rel.add.u32 %r153, [ %rd44 + 0 ], %r154;
@%p35 st.shared.b32 [ %r55 + 0 ], %r153;
bar.sync 0;
ld.shared.b32 %r155, [global_smem];
setp.ne.b32 %p33, %r155, 1;
@%p33 bra $L__BB0_14;
ld.param.b64 %rd2, [cooperative_initial_param_11];
ld.param.b64 %rd1, [cooperative_initial_param_10];
fence.proxy.async.global; mov.u32 %r156, 0;
$L__tmp18:
and.b32 %r174, %r4, 32;
shl.b32 %r175, %r174, 2;
or.b32 %r176, %r175, %r45;
shl.b32 %r177, %r1, 16;
mad.wide.s32 %rd51, %r177, 2, %rd1;
mul.lo.s32 %r178, %r176, 257;
mad.wide.u32 %rd45, %r178, 2, %rd51;
mov.b16 %rs2, 0;
mov.pred %p34, -1;
mov.u16 %rs1, %rs2;
@%p34 ld.global.cg.b16 { %rs1 }, [ %rd45 + 0 ];
cvt.f32.bf16 %r158, %rs1;
add.s64 %rd46, %rd45, 16448;
mov.u16 %rs3, %rs2;
@%p34 ld.global.cg.b16 { %rs3 }, [ %rd46 + 0 ];
cvt.f32.bf16 %r160, %rs3;
add.s64 %rd47, %rd45, 32896;
mov.u16 %rs4, %rs2;
@%p34 ld.global.cg.b16 { %rs4 }, [ %rd47 + 0 ];
add.s64 %rd48, %rd45, 49344;
mov.u16 %rs5, %rs2;
@%p34 ld.global.cg.b16 { %rs5 }, [ %rd48 + 0 ];
$L__tmp19:
add.f32 %r157, %r158, 0f00000000;
$L__tmp20:
add.f32 %r159, %r160, 0f00000000;
$L__tmp21:
mov.b32 %r179, {%rs4, %rs5};
mov.b32 {%rs6, %rs7}, %r179;
cvt.f32.bf16 %r180, %rs6;
cvt.f32.bf16 %r181, %rs7;
add.f32 %r182, %r159, %r181;
add.f32 %r183, %r157, %r180;
add.f32 %r161, %r183, %r182;
$L__tmp22:
shfl.sync.bfly.b32 %r162, %r161, 8, 31, -1;
add.f32 %r163, %r161, %r162;
shfl.sync.bfly.b32 %r164, %r163, 4, 31, -1;
add.f32 %r165, %r163, %r164;
shfl.sync.bfly.b32 %r166, %r165, 2, 31, -1;
add.f32 %r167, %r165, %r166;
shfl.sync.bfly.b32 %r168, %r167, 1, 31, -1;
add.f32 %r169, %r167, %r168;
shfl.sync.bfly.b32 %r171, %r169, 16, 31, -1;
add.f32 %r170, %r169, %r171;
$L__tmp23:
shfl.sync.idx.b32 %r184, %r170, 0, 31, -1;
bar.sync 0;
shr.u32 %r185, %r174, 3;
add.s32 %r186, %r55, %r185;
st.shared.b32 [%r186], %r184;
bar.sync 0;
shl.b32 %r187, %r4, 2;
and.b32 %r188, %r187, 4;
add.s32 %r189, %r55, %r188;
ld.shared.b32 %r190, [%r189];
$L__tmp24:
shfl.sync.bfly.b32 %r191, %r190, 1, 31, -1;
$L__tmp25:
add.f32 %r192, %r190, %r191;
$L__tmp26:
sqrt.rn.f32 %r193, %r192;
fma.rn.f32 %r172, %r193, 0f3F866666, 0f358637BD;
shl.b32 %r194, %r1, 1;
mad.wide.u32 %rd49, %r194, 4, %rd2;
@%p35 st.global.b32 [ %rd49 + 0 ], { %r172 };
add.s64 %rd50, %rd49, 4;
mul.f32 %r173, %r172, %r172;
@%p35 st.global.b32 [ %rd50 + 0 ], { %r173 };
$L__tmp27:
$L__BB0_14:
ret;
$L__tmp28:
$L__func_end0:
}
"#
  | .vo => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_initial(
.param .align 64 .b8 cooperative_initial_param_0[128],
.param .u32 cooperative_initial_param_1,
.param .u32 cooperative_initial_param_2,
.param .u64 cooperative_initial_param_3,
.param .u64 cooperative_initial_param_4,
.param .align 64 .b8 cooperative_initial_param_5[128],
.param .u32 cooperative_initial_param_6,
.param .u32 cooperative_initial_param_7,
.param .u64 cooperative_initial_param_8,
.param .u64 cooperative_initial_param_9,
.param .u64 .ptr .global .align 1 cooperative_initial_param_10,
.param .u64 .ptr .global .align 1 cooperative_initial_param_11,
.param .u64 .ptr .global .align 1 cooperative_initial_param_12,
.param .u64 .ptr .global .align 1 cooperative_initial_param_13,
.param .u64 .ptr .global .align 1 cooperative_initial_param_14
)
.reqntid 256
{
.reg .pred %p<37>;
.reg .b16 %rs<8>;
.reg .b32 %r<270>;
.reg .b64 %rd<56>;
$L__func_begin0:
mov.b64 %rd4, cooperative_initial_param_0;
mov.b64 %rd5, cooperative_initial_param_5;
$L__tmp0:
cvta.param.u64 %rd47, %rd5;
cvta.param.u64 %rd28, %rd4;
mov.u32 %r48, %ctaid.x;
mul.hi.u32 %r49, %r48, 818089009;
shr.u32 %r1, %r49, 2;
$L__tmp1:
mul.lo.s32 %r50, %r1, 21;
sub.s32 %r51, %r48, %r50;
mul.lo.s32 %r52, %r51, 3;
mov.b64 %rd6, 6569755928097638024;
shr.u64 %rd7, %rd6, %r52;
cvt.u32.u64 %r53, %rd7;
shl.b32 %r54, %r53, 7;
and.b32 %r2, %r54, 896;
mov.b64 %rd8, 6420841298313609216;
shr.u64 %rd9, %rd8, %r52;
cvt.u32.u64 %r55, %rd9;
shl.b32 %r56, %r55, 7;
and.b32 %r3, %r56, 896;
$L__tmp2:
setp.eq.b32 %p2, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p1, %r5, 0;
mov.b32 %r57, global_smem;
add.s32 %r87, %r57, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r87], 1;
bar.sync 0;
add.s32 %r47, %r57, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r47], 1;
mul.lo.s32 %r6, %r1, 768;
$L__tmp3:
add.s32 %r139, %r2, %r6;
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p36, %r5, 64;
add.s32 %r73, %r3, %r6;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r87], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r90, %r7, 0, 31, -1;
elect.sync %r91|%p18, -1;
and.pred %p17, %p36, %p18;
and.b32 %r92, %r90, 1;
shl.b32 %r93, %r92, 14;
add.s32 %r88, %r57, %r93;
shl.b32 %r89, %r92, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r88], [%rd28, {%r89, %r139}], [%r87];
shl.b32 %r94, %r90, 7;
and.b32 %r10, %r94, 512;
$L__tmp6:
or.b32 %r267, %r89, 128;
mov.b32 %r268, 0;
mov.b32 %r202, 0f00000000;
mov.b32 %r203, %r202;
mov.b32 %r204, %r202;
mov.b32 %r205, %r202;
mov.b32 %r206, %r202;
mov.b32 %r207, %r202;
mov.b32 %r208, %r202;
mov.b32 %r209, %r202;
mov.b32 %r210, %r202;
mov.b32 %r211, %r202;
mov.b32 %r212, %r202;
mov.b32 %r213, %r202;
mov.b32 %r214, %r202;
mov.b32 %r215, %r202;
mov.b32 %r216, %r202;
mov.b32 %r217, %r202;
mov.b32 %r218, %r202;
mov.b32 %r219, %r202;
mov.b32 %r220, %r202;
mov.b32 %r221, %r202;
mov.b32 %r222, %r202;
mov.b32 %r223, %r202;
mov.b32 %r224, %r202;
mov.b32 %r225, %r202;
mov.b32 %r226, %r202;
mov.b32 %r227, %r202;
mov.b32 %r228, %r202;
mov.b32 %r229, %r202;
mov.b32 %r230, %r202;
mov.b32 %r231, %r202;
mov.b32 %r232, %r202;
mov.b32 %r233, %r202;
mov.b32 %r234, %r202;
mov.b32 %r235, %r202;
mov.b32 %r236, %r202;
mov.b32 %r237, %r202;
mov.b32 %r238, %r202;
mov.b32 %r239, %r202;
mov.b32 %r240, %r202;
mov.b32 %r241, %r202;
mov.b32 %r242, %r202;
mov.b32 %r243, %r202;
mov.b32 %r244, %r202;
mov.b32 %r245, %r202;
mov.b32 %r246, %r202;
mov.b32 %r247, %r202;
mov.b32 %r248, %r202;
mov.b32 %r249, %r202;
mov.b32 %r250, %r202;
mov.b32 %r251, %r202;
mov.b32 %r252, %r202;
mov.b32 %r253, %r202;
mov.b32 %r254, %r202;
mov.b32 %r255, %r202;
mov.b32 %r256, %r202;
mov.b32 %r257, %r202;
mov.b32 %r258, %r202;
mov.b32 %r259, %r202;
mov.b32 %r260, %r202;
mov.b32 %r261, %r202;
mov.b32 %r262, %r202;
mov.b32 %r263, %r202;
mov.b32 %r264, %r202;
mov.b32 %r265, %r202;
$L__BB0_4:
and.b32 %r99, %r268, 1;
bfe.u32 %r96, %r268, 1, 1;
shl.b32 %r100, %r99, 15;
add.s32 %r101, %r57, %r100;
mad.lo.s32 %r102, %r99, -32760, %r101;
add.s32 %r95, %r102, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r95], %r96;
@!complete bra.uni waitLoop;
}
shr.u32 %r103, %r101, 4;
add.s32 %r104, %r103, %r10;
and.b32 %r105, %r104, 16383;
cvt.u64.u32 %rd45, %r105;
bfe.u32 %r106, %r101, 4, 14;
cvt.u64.u32 %rd46, %r106;
wgmma.fence.sync.aligned;
or.b64 %rd29, %rd45, 4611686293305294848;
or.b64 %rd30, %rd46, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd29, %rd30, %p19, 1, 1, 0, 0;
add.s64 %rd31, %rd45, 4611686293305294850;
add.s64 %rd32, %rd46, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd31, %rd32, %p19, 1, 1, 0, 0;
add.s64 %rd33, %rd45, 4611686293305294852;
add.s64 %rd34, %rd46, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd33, %rd34, %p19, 1, 1, 0, 0;
add.s64 %rd35, %rd45, 4611686293305294854;
add.s64 %rd36, %rd46, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd35, %rd36, %p19, 1, 1, 0, 0;
add.s64 %rd37, %rd45, 4611686293305295872;
add.s64 %rd38, %rd46, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd37, %rd38, %p19, 1, 1, 0, 0;
add.s64 %rd39, %rd45, 4611686293305295874;
add.s64 %rd40, %rd46, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd39, %rd40, %p19, 1, 1, 0, 0;
add.s64 %rd41, %rd45, 4611686293305295876;
add.s64 %rd42, %rd46, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd41, %rd42, %p19, 1, 1, 0, 0;
add.s64 %rd43, %rd45, 4611686293305295878;
add.s64 %rd44, %rd46, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd43, %rd44, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r11, %r268, 1;
and.b32 %r107, %r11, 1;
setp.ne.b32 %p22, %r268, 5;
shl.b32 %r108, %r107, 3;
add.s32 %r109, %r57, %r108;
add.s32 %r97, %r109, 98304;
shl.b32 %r110, %r107, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r97], 32768;
bar.sync 0;
elect.sync %r111|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p36, %p24;
add.s32 %r98, %r88, %r110;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r98], [%rd28, {%r267, %r139}], [%r97];
$L__tmp8:
add.s32 %r267, %r267, 128;
setp.ne.b32 %p25, %r11, 6;
mov.b32 %r268, %r11;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r87], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r60, %r7, 0, 31, -1;
elect.sync %r61|%p5, -1;
and.pred %p3, %p36, %p5;
and.b32 %r62, %r60, 1;
shl.b32 %r63, %r62, 14;
add.s32 %r58, %r57, %r63;
shl.b32 %r59, %r62, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r58], [%rd28, {%r59, %r139}], [%r87];
$L__tmp10:
bar.sync 0;
elect.sync %r64|%p6, -1;
and.pred %p4, %p36, %p6;
add.s32 %r72, %r58, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r72], [%rd28, {%r59, %r73}], [%r87];
shl.b32 %r65, %r60, 7;
and.b32 %r8, %r65, 512;
add.s32 %r66, %r57, 65536;
bfe.u32 %r67, %r66, 4, 14;
cvt.u64.u32 %rd10, %r67;
or.b64 %rd12, %rd10, 4611686293305294848;
add.s64 %rd14, %rd10, 4611686293305294850;
add.s64 %rd16, %rd10, 4611686293305294852;
add.s64 %rd18, %rd10, 4611686293305294854;
add.s64 %rd20, %rd10, 4611686293305295872;
add.s64 %rd22, %rd10, 4611686293305295874;
add.s64 %rd24, %rd10, 4611686293305295876;
add.s64 %rd26, %rd10, 4611686293305295878;
$L__tmp11:
or.b32 %r201, %r59, 128;
mov.b32 %r266, 0;
mov.b32 %r202, 0f00000000;
mov.b32 %r203, %r202;
mov.b32 %r204, %r202;
mov.b32 %r205, %r202;
mov.b32 %r206, %r202;
mov.b32 %r207, %r202;
mov.b32 %r208, %r202;
mov.b32 %r209, %r202;
mov.b32 %r210, %r202;
mov.b32 %r211, %r202;
mov.b32 %r212, %r202;
mov.b32 %r213, %r202;
mov.b32 %r214, %r202;
mov.b32 %r215, %r202;
mov.b32 %r216, %r202;
mov.b32 %r217, %r202;
mov.b32 %r218, %r202;
mov.b32 %r219, %r202;
mov.b32 %r220, %r202;
mov.b32 %r221, %r202;
mov.b32 %r222, %r202;
mov.b32 %r223, %r202;
mov.b32 %r224, %r202;
mov.b32 %r225, %r202;
mov.b32 %r226, %r202;
mov.b32 %r227, %r202;
mov.b32 %r228, %r202;
mov.b32 %r229, %r202;
mov.b32 %r230, %r202;
mov.b32 %r231, %r202;
mov.b32 %r232, %r202;
mov.b32 %r233, %r202;
mov.b32 %r234, %r202;
mov.b32 %r235, %r202;
mov.b32 %r236, %r202;
mov.b32 %r237, %r202;
mov.b32 %r238, %r202;
mov.b32 %r239, %r202;
mov.b32 %r240, %r202;
mov.b32 %r241, %r202;
mov.b32 %r242, %r202;
mov.b32 %r243, %r202;
mov.b32 %r244, %r202;
mov.b32 %r245, %r202;
mov.b32 %r246, %r202;
mov.b32 %r247, %r202;
mov.b32 %r248, %r202;
mov.b32 %r249, %r202;
mov.b32 %r250, %r202;
mov.b32 %r251, %r202;
mov.b32 %r252, %r202;
mov.b32 %r253, %r202;
mov.b32 %r254, %r202;
mov.b32 %r255, %r202;
mov.b32 %r256, %r202;
mov.b32 %r257, %r202;
mov.b32 %r258, %r202;
mov.b32 %r259, %r202;
mov.b32 %r260, %r202;
mov.b32 %r261, %r202;
mov.b32 %r262, %r202;
mov.b32 %r263, %r202;
mov.b32 %r264, %r202;
mov.b32 %r265, %r202;
$L__BB0_2:
and.b32 %r74, %r266, 1;
bfe.u32 %r69, %r266, 1, 1;
shl.b32 %r75, %r74, 15;
add.s32 %r76, %r57, %r75;
mad.lo.s32 %r77, %r74, -32760, %r76;
add.s32 %r68, %r77, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r68], %r69;
@!complete bra.uni waitLoop;
}
shr.u32 %r78, %r76, 4;
add.s32 %r79, %r78, %r8;
and.b32 %r80, %r79, 16383;
cvt.u64.u32 %rd27, %r80;
wgmma.fence.sync.aligned;
or.b64 %rd11, %rd27, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd11, %rd12, %p7, 1, 1, 0, 0;
add.s64 %rd13, %rd27, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd13, %rd14, %p7, 1, 1, 0, 0;
add.s64 %rd15, %rd27, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd15, %rd16, %p7, 1, 1, 0, 0;
add.s64 %rd17, %rd27, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd17, %rd18, %p7, 1, 1, 0, 0;
add.s64 %rd19, %rd27, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd19, %rd20, %p7, 1, 1, 0, 0;
add.s64 %rd21, %rd27, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd21, %rd22, %p7, 1, 1, 0, 0;
add.s64 %rd23, %rd27, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd23, %rd24, %p7, 1, 1, 0, 0;
add.s64 %rd25, %rd27, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265}, %rd25, %rd26, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r9, %r266, 1;
and.b32 %r81, %r9, 1;
setp.ne.b32 %p11, %r266, 5;
shl.b32 %r82, %r81, 3;
add.s32 %r83, %r57, %r82;
add.s32 %r70, %r83, 98304;
shl.b32 %r84, %r81, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r70], 65536;
bar.sync 0;
elect.sync %r85|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p36, %p13;
add.s32 %r71, %r58, %r84;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r71], [%rd28, {%r201, %r139}], [%r70];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r86|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p36, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r72], [%rd28, {%r201, %r73}], [%r70];
$L__tmp15:
add.s32 %r201, %r201, 128;
setp.eq.b32 %p16, %r9, 6;
mov.b32 %r266, %r9;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r2, %r3;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r87];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r47];
cvt.rn.bf16x2.f32 %r12, %r203, %r202;
cvt.rn.bf16x2.f32 %r13, %r205, %r204;
cvt.rn.bf16x2.f32 %r14, %r207, %r206;
cvt.rn.bf16x2.f32 %r15, %r209, %r208;
cvt.rn.bf16x2.f32 %r16, %r211, %r210;
cvt.rn.bf16x2.f32 %r17, %r213, %r212;
cvt.rn.bf16x2.f32 %r18, %r215, %r214;
cvt.rn.bf16x2.f32 %r19, %r217, %r216;
cvt.rn.bf16x2.f32 %r20, %r219, %r218;
cvt.rn.bf16x2.f32 %r21, %r221, %r220;
cvt.rn.bf16x2.f32 %r22, %r223, %r222;
cvt.rn.bf16x2.f32 %r23, %r225, %r224;
cvt.rn.bf16x2.f32 %r24, %r227, %r226;
cvt.rn.bf16x2.f32 %r25, %r229, %r228;
cvt.rn.bf16x2.f32 %r26, %r231, %r230;
cvt.rn.bf16x2.f32 %r27, %r233, %r232;
cvt.rn.bf16x2.f32 %r28, %r235, %r234;
cvt.rn.bf16x2.f32 %r29, %r237, %r236;
cvt.rn.bf16x2.f32 %r30, %r239, %r238;
cvt.rn.bf16x2.f32 %r31, %r241, %r240;
cvt.rn.bf16x2.f32 %r32, %r243, %r242;
cvt.rn.bf16x2.f32 %r33, %r245, %r244;
cvt.rn.bf16x2.f32 %r34, %r247, %r246;
cvt.rn.bf16x2.f32 %r35, %r249, %r248;
cvt.rn.bf16x2.f32 %r36, %r251, %r250;
cvt.rn.bf16x2.f32 %r37, %r253, %r252;
cvt.rn.bf16x2.f32 %r38, %r255, %r254;
cvt.rn.bf16x2.f32 %r39, %r257, %r256;
cvt.rn.bf16x2.f32 %r40, %r259, %r258;
cvt.rn.bf16x2.f32 %r41, %r261, %r260;
cvt.rn.bf16x2.f32 %r42, %r263, %r262;
cvt.rn.bf16x2.f32 %r43, %r265, %r264;
shl.b32 %r44, %r4, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r269, %r4, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r112, %r44, 1920;
shl.b32 %r269, %r4, 6;
and.b32 %r113, %r269, 14336;
shl.b32 %r114, %r4, 4;
and.b32 %r115, %r114, 112;
and.b32 %r116, %r4, 16;
or.b32 %r117, %r112, %r115;
xor.b32 %r118, %r117, %r116;
or.b32 %r119, %r118, %r113;
add.s32 %r120, %r57, %r119;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120+16384], {%r28, %r29, %r30, %r31};
xor.b32 %r121, %r119, 32;
add.s32 %r122, %r57, %r121;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122+16384], {%r32, %r33, %r34, %r35};
xor.b32 %r123, %r119, 64;
add.s32 %r124, %r57, %r123;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r124], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r124+16384], {%r36, %r37, %r38, %r39};
xor.b32 %r125, %r119, 96;
add.s32 %r126, %r57, %r125;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r126], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r126+16384], {%r40, %r41, %r42, %r43};
$L__BB0_9:
and.b32 %r127, %r44, 17280;
and.b32 %r46, %r4, 7;
shl.b32 %r128, %r46, 4;
and.b32 %r129, %r4, 96;
shl.b32 %r130, %r4, 1;
and.b32 %r131, %r130, 16;
and.b32 %r132, %r269, 1024;
or.b32 %r133, %r127, %r128;
or.b32 %r134, %r131, %r129;
xor.b32 %r135, %r133, %r134;
add.s32 %r136, %r57, %r132;
add.s32 %r137, %r136, %r135;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+32768], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+34816], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+36864], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+38912], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+40960], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+43008], {%r32, %r33, %r34, %r35};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+45056], {%r36, %r37, %r38, %r39};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+47104], {%r40, %r41, %r42, %r43};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_11;
bar.sync 0;
elect.sync %r141|%p28, -1;
shfl.sync.idx.b32 %r142, %r7, 0, 31, -1;
and.pred %p27, %p36, %p28;
and.b32 %r143, %r142, 1;
shl.b32 %r144, %r143, 14;
add.s32 %r140, %r57, %r144;
shl.b32 %r145, %r143, 6;
or.b32 %r138, %r145, %r3;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd47, {%r138, %r139}], [%r140];
cp.async.bulk.commit_group;
$L__BB0_11:
bar.sync 0;
elect.sync %r149|%p30, -1;
shfl.sync.idx.b32 %r150, %r7, 0, 31, -1;
and.pred %p29, %p36, %p30;
and.b32 %r151, %r150, 1;
shl.b32 %r152, %r151, 14;
add.s32 %r153, %r57, %r152;
add.s32 %r147, %r153, 32768;
shl.b32 %r154, %r151, 6;
or.b32 %r146, %r154, %r2;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd47, {%r146, %r73}], [%r147];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r148, 0;
$L__tmp17:
@%p26 bra $L__BB0_14;
ld.param.b64 %rd3, [cooperative_initial_param_12];
and.b32 %r45, %r4, 31;
bar.sync 0; fence.proxy.async.global; mov.u32 %r155, 0;
mad.wide.u32 %rd48, %r1, 4, %rd3;
setp.eq.b32 %p31, %r45, 0;
setp.lt.u32 %p32, %r5, 32;
and.pred %p35, %p31, %p32;
mov.b32 %r157, 1;
mov.u32 %r156, 0x0;
@%p35 atom.global.gpu.acq_rel.add.u32 %r156, [ %rd48 + 0 ], %r157;
@%p35 st.shared.b32 [ %r57 + 0 ], %r156;
bar.sync 0;
ld.shared.b32 %r158, [global_smem];
setp.ne.b32 %p33, %r158, 5;
@%p33 bra $L__BB0_14;
ld.param.b64 %rd2, [cooperative_initial_param_11];
ld.param.b64 %rd1, [cooperative_initial_param_10];
fence.proxy.async.global; mov.u32 %r159, 0;
$L__tmp18:
and.b32 %r177, %r4, 224;
shl.b32 %r178, %r177, 2;
or.b32 %r179, %r178, %r45;
mul.lo.s32 %r180, %r1, 589824;
mad.wide.s32 %rd55, %r180, 2, %rd1;
mul.lo.s32 %r181, %r179, 769;
mad.wide.u32 %rd49, %r181, 2, %rd55;
setp.lt.u32 %p34, %r177, 192;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p34 ld.global.cg.b16 { %rs1 }, [ %rd49 + 0 ];
cvt.f32.bf16 %r161, %rs1;
add.s64 %rd50, %rd49, 49216;
mov.u16 %rs3, %rs2;
@%p34 ld.global.cg.b16 { %rs3 }, [ %rd50 + 0 ];
cvt.f32.bf16 %r163, %rs3;
add.s64 %rd51, %rd49, 98432;
mov.u16 %rs4, %rs2;
@%p34 ld.global.cg.b16 { %rs4 }, [ %rd51 + 0 ];
add.s64 %rd52, %rd49, 147648;
mov.u16 %rs5, %rs2;
@%p34 ld.global.cg.b16 { %rs5 }, [ %rd52 + 0 ];
$L__tmp19:
add.f32 %r160, %r161, 0f00000000;
$L__tmp20:
add.f32 %r162, %r163, 0f00000000;
$L__tmp21:
mov.b32 %r182, {%rs4, %rs5};
mov.b32 {%rs6, %rs7}, %r182;
cvt.f32.bf16 %r183, %rs6;
cvt.f32.bf16 %r184, %rs7;
add.f32 %r185, %r162, %r184;
add.f32 %r186, %r160, %r183;
add.f32 %r164, %r186, %r185;
$L__tmp22:
shfl.sync.bfly.b32 %r165, %r164, 8, 31, -1;
add.f32 %r166, %r164, %r165;
shfl.sync.bfly.b32 %r167, %r166, 4, 31, -1;
add.f32 %r168, %r166, %r167;
shfl.sync.bfly.b32 %r169, %r168, 2, 31, -1;
add.f32 %r170, %r168, %r169;
shfl.sync.bfly.b32 %r171, %r170, 1, 31, -1;
add.f32 %r172, %r170, %r171;
shfl.sync.bfly.b32 %r174, %r172, 16, 31, -1;
add.f32 %r173, %r172, %r174;
$L__tmp23:
shfl.sync.idx.b32 %r187, %r173, 0, 31, -1;
bar.sync 0;
shr.u32 %r188, %r177, 3;
add.s32 %r189, %r57, %r188;
st.shared.b32 [%r189], %r187;
bar.sync 0;
shl.b32 %r190, %r46, 2;
add.s32 %r191, %r57, %r190;
ld.shared.b32 %r192, [%r191];
$L__tmp24:
shfl.sync.bfly.b32 %r193, %r192, 4, 31, -1;
$L__tmp25:
add.f32 %r194, %r192, %r193;
$L__tmp26:
shfl.sync.bfly.b32 %r195, %r194, 2, 31, -1;
$L__tmp27:
add.f32 %r196, %r194, %r195;
$L__tmp28:
shfl.sync.bfly.b32 %r197, %r196, 1, 31, -1;
$L__tmp29:
add.f32 %r198, %r196, %r197;
$L__tmp30:
sqrt.rn.f32 %r199, %r198;
fma.rn.f32 %r175, %r199, 0f3F866666, 0f358637BD;
shl.b32 %r200, %r1, 1;
mad.wide.u32 %rd53, %r200, 4, %rd2;
@%p35 st.global.b32 [ %rd53 + 0 ], { %r175 };
add.s64 %rd54, %rd53, 4;
mul.f32 %r176, %r175, %r175;
@%p35 st.global.b32 [ %rd54 + 0 ], { %r176 };
$L__tmp31:
$L__BB0_14:
ret;
$L__tmp32:
$L__func_end0:
}
"#
  | .mlp => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_initial(
.param .align 64 .b8 cooperative_initial_param_0[128],
.param .u32 cooperative_initial_param_1,
.param .u32 cooperative_initial_param_2,
.param .u64 cooperative_initial_param_3,
.param .u64 cooperative_initial_param_4,
.param .align 64 .b8 cooperative_initial_param_5[128],
.param .u32 cooperative_initial_param_6,
.param .u32 cooperative_initial_param_7,
.param .u64 cooperative_initial_param_8,
.param .u64 cooperative_initial_param_9,
.param .u64 .ptr .global .align 1 cooperative_initial_param_10,
.param .u64 .ptr .global .align 1 cooperative_initial_param_11,
.param .u64 .ptr .global .align 1 cooperative_initial_param_12,
.param .u64 .ptr .global .align 1 cooperative_initial_param_13,
.param .u64 .ptr .global .align 1 cooperative_initial_param_14
)
.reqntid 256
{
.reg .pred %p<37>;
.reg .b16 %rs<8>;
.reg .b32 %r<274>;
.reg .b64 %rd<56>;
$L__func_begin0:
mov.b64 %rd4, cooperative_initial_param_0;
mov.b64 %rd5, cooperative_initial_param_5;
$L__tmp0:
cvta.param.u64 %rd47, %rd5;
cvta.param.u64 %rd28, %rd4;
mov.u32 %r49, %ctaid.x;
mul.hi.u32 %r50, %r49, 818089009;
shr.u32 %r1, %r50, 2;
$L__tmp1:
mul.lo.s32 %r51, %r1, 21;
sub.s32 %r52, %r49, %r51;
mul.lo.s32 %r53, %r52, 3;
mov.b64 %rd6, 6569755928097638024;
shr.u64 %rd7, %rd6, %r53;
cvt.u32.u64 %r54, %rd7;
shl.b32 %r55, %r54, 7;
and.b32 %r2, %r55, 896;
mov.b64 %rd8, 6420841298313609216;
shr.u64 %rd9, %rd8, %r53;
cvt.u32.u64 %r56, %rd9;
shl.b32 %r57, %r56, 7;
and.b32 %r3, %r57, 896;
$L__tmp2:
setp.eq.b32 %p2, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p1, %r5, 0;
mov.b32 %r58, global_smem;
add.s32 %r89, %r58, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r89], 1;
bar.sync 0;
add.s32 %r48, %r58, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r48], 1;
mul.lo.s32 %r6, %r1, 2816;
$L__tmp3:
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p36, %r5, 64;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r89], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r91, %r7, 0, 31, -1;
elect.sync %r92|%p18, -1;
and.pred %p17, %p36, %p18;
and.b32 %r93, %r91, 1;
shl.b32 %r94, %r93, 14;
add.s32 %r90, %r58, %r94;
shl.b32 %r95, %r93, 6;
or.b32 %r101, %r95, %r2;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r90], [%rd28, {%r101, %r6}], [%r89];
shl.b32 %r96, %r91, 8;
and.b32 %r10, %r96, 1024;
$L__tmp6:
add.s32 %r271, %r6, 128;
mov.b32 %r272, 0;
mov.b32 %r206, 0f00000000;
mov.b32 %r207, %r206;
mov.b32 %r208, %r206;
mov.b32 %r209, %r206;
mov.b32 %r210, %r206;
mov.b32 %r211, %r206;
mov.b32 %r212, %r206;
mov.b32 %r213, %r206;
mov.b32 %r214, %r206;
mov.b32 %r215, %r206;
mov.b32 %r216, %r206;
mov.b32 %r217, %r206;
mov.b32 %r218, %r206;
mov.b32 %r219, %r206;
mov.b32 %r220, %r206;
mov.b32 %r221, %r206;
mov.b32 %r222, %r206;
mov.b32 %r223, %r206;
mov.b32 %r224, %r206;
mov.b32 %r225, %r206;
mov.b32 %r226, %r206;
mov.b32 %r227, %r206;
mov.b32 %r228, %r206;
mov.b32 %r229, %r206;
mov.b32 %r230, %r206;
mov.b32 %r231, %r206;
mov.b32 %r232, %r206;
mov.b32 %r233, %r206;
mov.b32 %r234, %r206;
mov.b32 %r235, %r206;
mov.b32 %r236, %r206;
mov.b32 %r237, %r206;
mov.b32 %r238, %r206;
mov.b32 %r239, %r206;
mov.b32 %r240, %r206;
mov.b32 %r241, %r206;
mov.b32 %r242, %r206;
mov.b32 %r243, %r206;
mov.b32 %r244, %r206;
mov.b32 %r245, %r206;
mov.b32 %r246, %r206;
mov.b32 %r247, %r206;
mov.b32 %r248, %r206;
mov.b32 %r249, %r206;
mov.b32 %r250, %r206;
mov.b32 %r251, %r206;
mov.b32 %r252, %r206;
mov.b32 %r253, %r206;
mov.b32 %r254, %r206;
mov.b32 %r255, %r206;
mov.b32 %r256, %r206;
mov.b32 %r257, %r206;
mov.b32 %r258, %r206;
mov.b32 %r259, %r206;
mov.b32 %r260, %r206;
mov.b32 %r261, %r206;
mov.b32 %r262, %r206;
mov.b32 %r263, %r206;
mov.b32 %r264, %r206;
mov.b32 %r265, %r206;
mov.b32 %r266, %r206;
mov.b32 %r267, %r206;
mov.b32 %r268, %r206;
mov.b32 %r269, %r206;
$L__BB0_4:
and.b32 %r102, %r272, 1;
bfe.u32 %r98, %r272, 1, 1;
shl.b32 %r103, %r102, 15;
add.s32 %r104, %r58, %r103;
mad.lo.s32 %r105, %r102, -32760, %r104;
add.s32 %r97, %r105, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r97], %r98;
@!complete bra.uni waitLoop;
}
shr.u32 %r106, %r104, 4;
add.s32 %r107, %r106, %r10;
and.b32 %r108, %r107, 16383;
cvt.u64.u32 %rd45, %r108;
bfe.u32 %r109, %r104, 4, 14;
cvt.u64.u32 %rd46, %r109;
wgmma.fence.sync.aligned;
or.b64 %rd29, %rd45, 4611686293305294848;
or.b64 %rd30, %rd46, 4611686293372403712;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd29, %rd30, %p19, 1, 1, 1, 1;
add.s64 %rd31, %rd45, 4611686293305294976;
add.s64 %rd32, %rd46, 4611686293372403840;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd31, %rd32, %p19, 1, 1, 1, 1;
add.s64 %rd33, %rd45, 4611686293305295104;
add.s64 %rd34, %rd46, 4611686293372403968;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd33, %rd34, %p19, 1, 1, 1, 1;
add.s64 %rd35, %rd45, 4611686293305295232;
add.s64 %rd36, %rd46, 4611686293372404096;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd35, %rd36, %p19, 1, 1, 1, 1;
add.s64 %rd37, %rd45, 4611686293305295360;
add.s64 %rd38, %rd46, 4611686293372404224;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd37, %rd38, %p19, 1, 1, 1, 1;
add.s64 %rd39, %rd45, 4611686293305295488;
add.s64 %rd40, %rd46, 4611686293372404352;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd39, %rd40, %p19, 1, 1, 1, 1;
add.s64 %rd41, %rd45, 4611686293305295616;
add.s64 %rd42, %rd46, 4611686293372404480;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd41, %rd42, %p19, 1, 1, 1, 1;
add.s64 %rd43, %rd45, 4611686293305295744;
add.s64 %rd44, %rd46, 4611686293372404608;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd43, %rd44, %p19, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r11, %r272, 1;
and.b32 %r110, %r11, 1;
setp.ne.b32 %p22, %r272, 21;
shl.b32 %r111, %r110, 3;
add.s32 %r112, %r58, %r111;
add.s32 %r99, %r112, 98304;
shl.b32 %r113, %r110, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r99], 32768;
bar.sync 0;
elect.sync %r114|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p36, %p24;
add.s32 %r100, %r90, %r113;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r100], [%rd28, {%r101, %r271}], [%r99];
$L__tmp8:
add.s32 %r271, %r271, 128;
setp.ne.b32 %p25, %r11, 22;
mov.b32 %r272, %r11;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r89], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r60, %r7, 0, 31, -1;
elect.sync %r61|%p5, -1;
and.pred %p3, %p36, %p5;
and.b32 %r62, %r60, 1;
shl.b32 %r63, %r62, 14;
add.s32 %r59, %r58, %r63;
shl.b32 %r64, %r62, 6;
or.b32 %r73, %r64, %r2;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r59], [%rd28, {%r73, %r6}], [%r89];
$L__tmp10:
bar.sync 0;
elect.sync %r65|%p6, -1;
and.pred %p4, %p36, %p6;
add.s32 %r74, %r59, 65536;
or.b32 %r75, %r64, %r3;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r74], [%rd28, {%r75, %r6}], [%r89];
shl.b32 %r66, %r60, 8;
and.b32 %r8, %r66, 1024;
add.s32 %r67, %r58, 65536;
bfe.u32 %r68, %r67, 4, 14;
cvt.u64.u32 %rd10, %r68;
or.b64 %rd12, %rd10, 4611686293372403712;
add.s64 %rd14, %rd10, 4611686293372403840;
add.s64 %rd16, %rd10, 4611686293372403968;
add.s64 %rd18, %rd10, 4611686293372404096;
add.s64 %rd20, %rd10, 4611686293372404224;
add.s64 %rd22, %rd10, 4611686293372404352;
add.s64 %rd24, %rd10, 4611686293372404480;
add.s64 %rd26, %rd10, 4611686293372404608;
$L__tmp11:
add.s32 %r205, %r6, 128;
mov.b32 %r270, 0;
mov.b32 %r206, 0f00000000;
mov.b32 %r207, %r206;
mov.b32 %r208, %r206;
mov.b32 %r209, %r206;
mov.b32 %r210, %r206;
mov.b32 %r211, %r206;
mov.b32 %r212, %r206;
mov.b32 %r213, %r206;
mov.b32 %r214, %r206;
mov.b32 %r215, %r206;
mov.b32 %r216, %r206;
mov.b32 %r217, %r206;
mov.b32 %r218, %r206;
mov.b32 %r219, %r206;
mov.b32 %r220, %r206;
mov.b32 %r221, %r206;
mov.b32 %r222, %r206;
mov.b32 %r223, %r206;
mov.b32 %r224, %r206;
mov.b32 %r225, %r206;
mov.b32 %r226, %r206;
mov.b32 %r227, %r206;
mov.b32 %r228, %r206;
mov.b32 %r229, %r206;
mov.b32 %r230, %r206;
mov.b32 %r231, %r206;
mov.b32 %r232, %r206;
mov.b32 %r233, %r206;
mov.b32 %r234, %r206;
mov.b32 %r235, %r206;
mov.b32 %r236, %r206;
mov.b32 %r237, %r206;
mov.b32 %r238, %r206;
mov.b32 %r239, %r206;
mov.b32 %r240, %r206;
mov.b32 %r241, %r206;
mov.b32 %r242, %r206;
mov.b32 %r243, %r206;
mov.b32 %r244, %r206;
mov.b32 %r245, %r206;
mov.b32 %r246, %r206;
mov.b32 %r247, %r206;
mov.b32 %r248, %r206;
mov.b32 %r249, %r206;
mov.b32 %r250, %r206;
mov.b32 %r251, %r206;
mov.b32 %r252, %r206;
mov.b32 %r253, %r206;
mov.b32 %r254, %r206;
mov.b32 %r255, %r206;
mov.b32 %r256, %r206;
mov.b32 %r257, %r206;
mov.b32 %r258, %r206;
mov.b32 %r259, %r206;
mov.b32 %r260, %r206;
mov.b32 %r261, %r206;
mov.b32 %r262, %r206;
mov.b32 %r263, %r206;
mov.b32 %r264, %r206;
mov.b32 %r265, %r206;
mov.b32 %r266, %r206;
mov.b32 %r267, %r206;
mov.b32 %r268, %r206;
mov.b32 %r269, %r206;
$L__BB0_2:
and.b32 %r76, %r270, 1;
bfe.u32 %r70, %r270, 1, 1;
shl.b32 %r77, %r76, 15;
add.s32 %r78, %r58, %r77;
mad.lo.s32 %r79, %r76, -32760, %r78;
add.s32 %r69, %r79, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r69], %r70;
@!complete bra.uni waitLoop;
}
shr.u32 %r80, %r78, 4;
add.s32 %r81, %r80, %r8;
and.b32 %r82, %r81, 16383;
cvt.u64.u32 %rd27, %r82;
wgmma.fence.sync.aligned;
or.b64 %rd11, %rd27, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd11, %rd12, %p7, 1, 1, 1, 1;
add.s64 %rd13, %rd27, 4611686293305294976;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd13, %rd14, %p7, 1, 1, 1, 1;
add.s64 %rd15, %rd27, 4611686293305295104;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd15, %rd16, %p7, 1, 1, 1, 1;
add.s64 %rd17, %rd27, 4611686293305295232;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd17, %rd18, %p7, 1, 1, 1, 1;
add.s64 %rd19, %rd27, 4611686293305295360;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd19, %rd20, %p7, 1, 1, 1, 1;
add.s64 %rd21, %rd27, 4611686293305295488;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd21, %rd22, %p7, 1, 1, 1, 1;
add.s64 %rd23, %rd27, 4611686293305295616;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd23, %rd24, %p7, 1, 1, 1, 1;
add.s64 %rd25, %rd27, 4611686293305295744;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220,%r221,%r222,%r223,%r224,%r225,%r226,%r227,%r228,%r229,%r230,%r231,%r232,%r233,%r234,%r235,%r236,%r237,%r238,%r239,%r240,%r241,%r242,%r243,%r244,%r245,%r246,%r247,%r248,%r249,%r250,%r251,%r252,%r253,%r254,%r255,%r256,%r257,%r258,%r259,%r260,%r261,%r262,%r263,%r264,%r265,%r266,%r267,%r268,%r269}, %rd25, %rd26, %p7, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
add.s32 %r9, %r270, 1;
and.b32 %r83, %r9, 1;
setp.ne.b32 %p11, %r270, 21;
shl.b32 %r84, %r83, 3;
add.s32 %r85, %r58, %r84;
add.s32 %r71, %r85, 98304;
shl.b32 %r86, %r83, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r71], 65536;
bar.sync 0;
elect.sync %r87|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p36, %p13;
add.s32 %r72, %r59, %r86;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r72], [%rd28, {%r73, %r205}], [%r71];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r88|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p36, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r74], [%rd28, {%r75, %r205}], [%r71];
$L__tmp15:
add.s32 %r205, %r205, 128;
setp.eq.b32 %p16, %r9, 22;
mov.b32 %r270, %r9;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r2, %r3;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r89];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r48];
cvt.rn.bf16x2.f32 %r12, %r207, %r206;
cvt.rn.bf16x2.f32 %r13, %r209, %r208;
cvt.rn.bf16x2.f32 %r14, %r211, %r210;
cvt.rn.bf16x2.f32 %r15, %r213, %r212;
cvt.rn.bf16x2.f32 %r16, %r215, %r214;
cvt.rn.bf16x2.f32 %r17, %r217, %r216;
cvt.rn.bf16x2.f32 %r18, %r219, %r218;
cvt.rn.bf16x2.f32 %r19, %r221, %r220;
cvt.rn.bf16x2.f32 %r20, %r223, %r222;
cvt.rn.bf16x2.f32 %r21, %r225, %r224;
cvt.rn.bf16x2.f32 %r22, %r227, %r226;
cvt.rn.bf16x2.f32 %r23, %r229, %r228;
cvt.rn.bf16x2.f32 %r24, %r231, %r230;
cvt.rn.bf16x2.f32 %r25, %r233, %r232;
cvt.rn.bf16x2.f32 %r26, %r235, %r234;
cvt.rn.bf16x2.f32 %r27, %r237, %r236;
cvt.rn.bf16x2.f32 %r28, %r239, %r238;
cvt.rn.bf16x2.f32 %r29, %r241, %r240;
cvt.rn.bf16x2.f32 %r30, %r243, %r242;
cvt.rn.bf16x2.f32 %r31, %r245, %r244;
cvt.rn.bf16x2.f32 %r32, %r247, %r246;
cvt.rn.bf16x2.f32 %r33, %r249, %r248;
cvt.rn.bf16x2.f32 %r34, %r251, %r250;
cvt.rn.bf16x2.f32 %r35, %r253, %r252;
cvt.rn.bf16x2.f32 %r36, %r255, %r254;
cvt.rn.bf16x2.f32 %r37, %r257, %r256;
cvt.rn.bf16x2.f32 %r38, %r259, %r258;
cvt.rn.bf16x2.f32 %r39, %r261, %r260;
cvt.rn.bf16x2.f32 %r40, %r263, %r262;
cvt.rn.bf16x2.f32 %r41, %r265, %r264;
cvt.rn.bf16x2.f32 %r42, %r267, %r266;
cvt.rn.bf16x2.f32 %r43, %r269, %r268;
shl.b32 %r44, %r4, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r273, %r4, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r115, %r44, 1920;
shl.b32 %r273, %r4, 6;
and.b32 %r116, %r273, 14336;
shl.b32 %r117, %r4, 4;
and.b32 %r118, %r117, 112;
and.b32 %r119, %r4, 16;
or.b32 %r120, %r115, %r118;
xor.b32 %r121, %r120, %r119;
or.b32 %r122, %r121, %r116;
add.s32 %r123, %r58, %r122;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123+16384], {%r28, %r29, %r30, %r31};
xor.b32 %r124, %r122, 32;
add.s32 %r125, %r58, %r124;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r125], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r125+16384], {%r32, %r33, %r34, %r35};
xor.b32 %r126, %r122, 64;
add.s32 %r127, %r58, %r126;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r127], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r127+16384], {%r36, %r37, %r38, %r39};
xor.b32 %r128, %r122, 96;
add.s32 %r129, %r58, %r128;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r129], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r129+16384], {%r40, %r41, %r42, %r43};
$L__BB0_9:
and.b32 %r130, %r44, 17280;
and.b32 %r46, %r4, 7;
shl.b32 %r131, %r46, 4;
and.b32 %r132, %r4, 96;
shl.b32 %r133, %r4, 1;
and.b32 %r134, %r133, 16;
and.b32 %r135, %r273, 1024;
or.b32 %r136, %r130, %r131;
or.b32 %r137, %r134, %r132;
xor.b32 %r138, %r136, %r137;
add.s32 %r139, %r58, %r135;
add.s32 %r140, %r139, %r138;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+32768], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+34816], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+36864], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+38912], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+40960], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+43008], {%r32, %r33, %r34, %r35};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+45056], {%r36, %r37, %r38, %r39};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r140+47104], {%r40, %r41, %r42, %r43};
fence.proxy.async.shared::cta;
mul.lo.s32 %r47, %r1, 768;
@%p2 bra $L__BB0_11;
add.s32 %r142, %r2, %r47;
bar.sync 0;
elect.sync %r144|%p28, -1;
shfl.sync.idx.b32 %r145, %r7, 0, 31, -1;
and.pred %p27, %p36, %p28;
and.b32 %r146, %r145, 1;
shl.b32 %r147, %r146, 14;
add.s32 %r143, %r58, %r147;
shl.b32 %r148, %r146, 6;
or.b32 %r141, %r148, %r3;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd47, {%r141, %r142}], [%r143];
cp.async.bulk.commit_group;
$L__BB0_11:
add.s32 %r150, %r3, %r47;
bar.sync 0;
elect.sync %r153|%p30, -1;
shfl.sync.idx.b32 %r154, %r7, 0, 31, -1;
and.pred %p29, %p36, %p30;
and.b32 %r155, %r154, 1;
shl.b32 %r156, %r155, 14;
add.s32 %r157, %r58, %r156;
add.s32 %r151, %r157, 32768;
shl.b32 %r158, %r155, 6;
or.b32 %r149, %r158, %r2;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd47, {%r149, %r150}], [%r151];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r152, 0;
$L__tmp17:
@%p26 bra $L__BB0_14;
ld.param.b64 %rd3, [cooperative_initial_param_12];
and.b32 %r45, %r4, 31;
bar.sync 0; fence.proxy.async.global; mov.u32 %r159, 0;
mad.wide.u32 %rd48, %r1, 4, %rd3;
setp.eq.b32 %p31, %r45, 0;
setp.lt.u32 %p32, %r5, 32;
and.pred %p35, %p31, %p32;
mov.b32 %r161, 1;
mov.u32 %r160, 0x0;
@%p35 atom.global.gpu.acq_rel.add.u32 %r160, [ %rd48 + 0 ], %r161;
@%p35 st.shared.b32 [ %r58 + 0 ], %r160;
bar.sync 0;
ld.shared.b32 %r162, [global_smem];
setp.ne.b32 %p33, %r162, 5;
@%p33 bra $L__BB0_14;
ld.param.b64 %rd2, [cooperative_initial_param_11];
ld.param.b64 %rd1, [cooperative_initial_param_10];
fence.proxy.async.global; mov.u32 %r163, 0;
$L__tmp18:
and.b32 %r181, %r4, 224;
shl.b32 %r182, %r181, 2;
or.b32 %r183, %r182, %r45;
mul.lo.s32 %r184, %r1, 589824;
mad.wide.s32 %rd55, %r184, 2, %rd1;
mul.lo.s32 %r185, %r183, 769;
mad.wide.u32 %rd49, %r185, 2, %rd55;
setp.lt.u32 %p34, %r181, 192;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p34 ld.global.cg.b16 { %rs1 }, [ %rd49 + 0 ];
cvt.f32.bf16 %r165, %rs1;
add.s64 %rd50, %rd49, 49216;
mov.u16 %rs3, %rs2;
@%p34 ld.global.cg.b16 { %rs3 }, [ %rd50 + 0 ];
cvt.f32.bf16 %r167, %rs3;
add.s64 %rd51, %rd49, 98432;
mov.u16 %rs4, %rs2;
@%p34 ld.global.cg.b16 { %rs4 }, [ %rd51 + 0 ];
add.s64 %rd52, %rd49, 147648;
mov.u16 %rs5, %rs2;
@%p34 ld.global.cg.b16 { %rs5 }, [ %rd52 + 0 ];
$L__tmp19:
add.f32 %r164, %r165, 0f00000000;
$L__tmp20:
add.f32 %r166, %r167, 0f00000000;
$L__tmp21:
mov.b32 %r186, {%rs4, %rs5};
mov.b32 {%rs6, %rs7}, %r186;
cvt.f32.bf16 %r187, %rs6;
cvt.f32.bf16 %r188, %rs7;
add.f32 %r189, %r166, %r188;
add.f32 %r190, %r164, %r187;
add.f32 %r168, %r190, %r189;
$L__tmp22:
shfl.sync.bfly.b32 %r169, %r168, 8, 31, -1;
add.f32 %r170, %r168, %r169;
shfl.sync.bfly.b32 %r171, %r170, 4, 31, -1;
add.f32 %r172, %r170, %r171;
shfl.sync.bfly.b32 %r173, %r172, 2, 31, -1;
add.f32 %r174, %r172, %r173;
shfl.sync.bfly.b32 %r175, %r174, 1, 31, -1;
add.f32 %r176, %r174, %r175;
shfl.sync.bfly.b32 %r178, %r176, 16, 31, -1;
add.f32 %r177, %r176, %r178;
$L__tmp23:
shfl.sync.idx.b32 %r191, %r177, 0, 31, -1;
bar.sync 0;
shr.u32 %r192, %r181, 3;
add.s32 %r193, %r58, %r192;
st.shared.b32 [%r193], %r191;
bar.sync 0;
shl.b32 %r194, %r46, 2;
add.s32 %r195, %r58, %r194;
ld.shared.b32 %r196, [%r195];
$L__tmp24:
shfl.sync.bfly.b32 %r197, %r196, 4, 31, -1;
$L__tmp25:
add.f32 %r198, %r196, %r197;
$L__tmp26:
shfl.sync.bfly.b32 %r199, %r198, 2, 31, -1;
$L__tmp27:
add.f32 %r200, %r198, %r199;
$L__tmp28:
shfl.sync.bfly.b32 %r201, %r200, 1, 31, -1;
$L__tmp29:
add.f32 %r202, %r200, %r201;
$L__tmp30:
sqrt.rn.f32 %r203, %r202;
fma.rn.f32 %r179, %r203, 0f3F866666, 0f358637BD;
shl.b32 %r204, %r1, 1;
mad.wide.u32 %rd53, %r204, 4, %rd2;
@%p35 st.global.b32 [ %rd53 + 0 ], { %r179 };
add.s64 %rd54, %rd53, 4;
mul.f32 %r180, %r179, %r179;
@%p35 st.global.b32 [ %rd54 + 0 ], { %r180 };
$L__tmp31:
$L__BB0_14:
ret;
$L__tmp32:
$L__func_end0:
}
"#

private def roundGraph : Bank → String
  | .qk => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_round(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Output: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Output_8: i32, %Output_9: i32, %Output_10: i64, %Output_11: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %round_offset: i32, %alpha: f32, %beta: f32) attributes {noinline = false} {
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_12 = arith.constant dense<256> : tensor<128x1xi32, #blocked>
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_13 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_13 : i32 to i64
%row_14 = arith.shrui %c72_i64, %row : i64
%row_15 = arith.andi %row_14, %c7_i64 : i64
%row_16 = arith.trunci %row_15 : i64 to i32
%row_17 = arith.muli %row_16, 128 : i32
%col = arith.shrui %c64_i64, %row : i64
%col_18 = arith.andi %col, %c7_i64 : i64
%col_19 = arith.trunci %col_18 : i64 to i32
%col_20 = arith.muli %col_19, 128 : i32
%batch = arith.divsi %pid, 3 : i32
%offdiag = arith.cmpi ne, %row_17, %col_20 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 256 : i32
%acc_50 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_17 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_50, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_51 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_52 = arith.addi %acc, %col_20 : i32
ttng.async_tma_copy_global_to_local %X[%coords_52, %c0_i32] %acc_51, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_53 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_54 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_55 = arith.remsi %phase, 2 : i32
%base_56 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_57 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_57, %phase_55, %true deps %base_56, %acc_51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_51 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_58 = ttng.warp_group_dot %base_56, %right, %acc_54, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_59 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_60 = ttg.memdesc_index %barriers[%nxt_59] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_61 = ttg.memdesc_index %tiles[%nxt_59] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_60, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_61, %acc_60, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_62 = ttng.warp_group_dot_wait %acc_58 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords_52, %k] %acc_51, %acc_60, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_62 : tensor<128x128xf32, #mma>
}
scf.yield %acc_53 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 256 : i32
%acc_50 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_17 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_50, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_51 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_53 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_54 = arith.remsi %phase, 2 : i32
%base_55 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_56 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_56, %phase_54, %true deps %base_55, %base_55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_55 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_57 = ttng.warp_group_dot %base_55, %right, %acc_53, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_58 = ttng.warp_group_dot_wait %acc_57 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_59 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_60 = ttg.memdesc_index %barriers[%nxt_59] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_61 = ttg.memdesc_index %tiles[%nxt_59] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_60, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_61, %acc_60, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_58 : tensor<128x128xf32, #mma>
}
%acc_52 = ttng.warp_group_dot_wait %acc_51 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_52 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_21 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.muli %batch, 256 : i32
%14 = arith.addi %13, %row_17 : i32
ttng.async_tma_copy_local_to_global %Gram[%14, %col_20] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 256 : i32
%4 = arith.addi %3, %col_20 : i32
ttng.async_tma_copy_local_to_global %Gram[%4, %row_17] %mirror_21 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%bit = arith.shli 1, %shift : i32
%needed = arith.divsi %row_17, 128 : i32
%needed_22 = arith.shli 3, %needed : i32
%needed_23 = arith.divsi %col_20, 128 : i32
%needed_24 = arith.shli 3, %needed_23 : i32
%needed_25 = arith.ori %needed_22, %needed_24 : i32
%6 = tt.addptr %Counter, %round_offset : !tt.ptr<i32>, i32
%7 = tt.addptr %6, %batch : !tt.ptr<i32>, i32
%8 = tt.elementwise_inline_asm "{\0A .reg .u32 tid, bits, covered;\0A .reg .pred observer, ready;\0A bar.sync 0;\0A mov.u32 tid, %tid.x;\0A setp.eq.u32 observer, tid, 0;\0A @!observer bra gram_ready;\0A fence.proxy.async.global;\0A atom.acq_rel.gpu.global.or.b32 bits, [$1], $2;\0A or.b32 bits, bits, $2;\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @ready bra gram_ready;\0A observe_gram:\0A ld.acquire.gpu.global.u32 bits, [$1];\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @!ready bra observe_gram;\0A gram_ready:\0A bar.sync 0;\0A fence.proxy.async.global;\0A mov.u32 $0, 0;\0A }" {constraints = "=r,l,r,r,~{memory}", packed_element = 1 : i32, pure = false} %7, %bit, %needed_25 : !tt.ptr<i32>, i32, i32 -> i32
%tiles_26 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers_27 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%9 = ttg.memdesc_index %barriers_27[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%10 = ttg.memdesc_index %barriers_27[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%11 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = ttg.memdesc_index %tiles_26[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_50 = ttg.memdesc_index %tiles_26[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%4, %c0_i32] %acc_50, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_51 = scf.for %chunk = 0 to 2 step 1 iter_args(%acc_52 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_53 = arith.remsi %phase, 2 : i32
%base_54 = ttg.memdesc_index %tiles_26[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_55 = ttg.memdesc_index %barriers_27[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_55, %phase_53, %true deps %base_54, %acc_50 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_50 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_56 = ttng.warp_group_dot %base_54, %right, %acc_52, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_57 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 2 : i32
%k = arith.muli %nxt, 128 : i32
%acc_58 = ttg.memdesc_index %barriers_27[%nxt_57] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_59 = ttg.memdesc_index %tiles_26[%nxt_57] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_58, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_59, %acc_58, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_60 = ttng.warp_group_dot_wait %acc_56 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %Gram[%4, %k] %acc_50, %acc_58, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_60 : tensor<128x128xf32, #mma>
}
scf.yield %acc_51 : tensor<128x128xf32, #mma>
} else {
%acc = ttg.memdesc_index %tiles_26[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_50 = scf.for %chunk = 0 to 2 step 1 iter_args(%acc_52 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_53 = arith.remsi %phase, 2 : i32
%base_54 = ttg.memdesc_index %tiles_26[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_55 = ttg.memdesc_index %barriers_27[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_55, %phase_53, %true deps %base_54, %base_54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_54 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_56 = ttng.warp_group_dot %base_54, %right, %acc_52, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_57 = ttng.warp_group_dot_wait %acc_56 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_58 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 2 : i32
%k = arith.muli %nxt, 128 : i32
%acc_59 = ttg.memdesc_index %barriers_27[%nxt_58] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_60 = ttg.memdesc_index %tiles_26[%nxt_58] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_59, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_60, %acc_59, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_57 : tensor<128x128xf32, #mma>
}
%acc_51 = ttng.warp_group_dot_wait %acc_50 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_51 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%i = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%j = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %batch, 65536 : i32
%dst = tt.expand_dims %i {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%dst_28 = tt.splat %row_17 : i32 -> tensor<128x1xi32, #blocked>
%dst_29 = arith.addi %dst_28, %dst : tensor<128x1xi32, #blocked>
%dst_30 = arith.muli %dst_29, %cst_12 : tensor<128x1xi32, #blocked>
%dst_31 = tt.expand_dims %j {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%dst_32 = tt.splat %col_20 : i32 -> tensor<1x128xi32, #blocked>
%dst_33 = arith.addi %dst_32, %dst_31 : tensor<1x128xi32, #blocked>
%dst_34 = tt.broadcast %dst_30 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_35 = tt.broadcast %dst_33 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_36 = arith.addi %dst_34, %dst_35 : tensor<128x128xi32, #blocked>
%bias_words = tt.addptr %Gram_ptr, %base : !tt.ptr<bf16>, i32
%bias_words_37 = tt.splat %bias_words : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias_words_38 = tt.addptr %bias_words_37, %dst_36 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%bias_words_39 = tt.load %bias_words_38 cacheModifier = cg : tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias = ttg.convert_layout %bias_words_39 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%bias_40 = arith.extf %bias : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%words_41 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%words_42 = arith.mulf %11, %words_41 : tensor<128x128xf32, #mma>
%words_43 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%words_44 = arith.mulf %bias_40, %words_43 : tensor<128x128xf32, #mma>
%words_45 = arith.addf %words_42, %words_44 : tensor<128x128xf32, #mma>
%words_46 = arith.truncf %words_45 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct_47 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words_46, %direct_47 : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror_48 = tt.trans %words_46 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_49 = ttg.local_alloc %mirror_48 : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_local_to_global %Output[%13, %col_20] %direct_47 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
ttng.async_tma_copy_local_to_global %Output[%4, %row_17] %mirror_49 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%12 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .vo => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_round(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Output: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Output_8: i32, %Output_9: i32, %Output_10: i64, %Output_11: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %round_offset: i32, %alpha: f32, %beta: f32) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_12 = arith.constant dense<768> : tensor<128x1xi32, #blocked>
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_13 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_13 : i32 to i64
%row_14 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_15 = arith.andi %row_14, %c7_i64 : i64
%row_16 = arith.trunci %row_15 : i64 to i32
%row_17 = arith.muli %row_16, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_18 = arith.andi %col, %c7_i64 : i64
%col_19 = arith.trunci %col_18 : i64 to i32
%col_20 = arith.muli %col_19, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_17, %col_20 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 768 : i32
%acc_65 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_17 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_65, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_67 = arith.addi %acc, %col_20 : i32
ttng.async_tma_copy_global_to_local %X[%coords_67, %c0_i32] %acc_66, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_68 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_69 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_70 = arith.remsi %phase, 2 : i32
%base_71 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_72 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_72, %phase_70, %true deps %base_71, %acc_66 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_66 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_73 = ttng.warp_group_dot %base_71, %right, %acc_69, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_74 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_75 = ttg.memdesc_index %barriers[%nxt_74] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_76 = ttg.memdesc_index %tiles[%nxt_74] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_75, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_76, %acc_75, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_77 = ttng.warp_group_dot_wait %acc_73 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords_67, %k] %acc_66, %acc_75, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_77 : tensor<128x128xf32, #mma>
}
scf.yield %acc_68 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 768 : i32
%acc_65 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_17 : i32
ttng.async_tma_copy_global_to_local %X[%coords, %c0_i32] %acc_65, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_68 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_69 = arith.remsi %phase, 2 : i32
%base_70 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_71 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_71, %phase_69, %true deps %base_70, %base_70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_70 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_72 = ttng.warp_group_dot %base_70, %right, %acc_68, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_73 = ttng.warp_group_dot_wait %acc_72 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_74 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_75 = ttg.memdesc_index %barriers[%nxt_74] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_76 = ttg.memdesc_index %tiles[%nxt_74] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_75, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%coords, %k] %acc_76, %acc_75, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_73 : tensor<128x128xf32, #mma>
}
%acc_67 = ttng.warp_group_dot_wait %acc_66 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_67 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_21 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.muli %batch, 768 : i32
%14 = arith.addi %13, %row_17 : i32
ttng.async_tma_copy_local_to_global %Gram[%14, %col_20] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 768 : i32
%4 = arith.addi %3, %col_20 : i32
ttng.async_tma_copy_local_to_global %Gram[%4, %row_17] %mirror_21 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%bit = arith.shli 1, %shift : i32
%needed = arith.divsi %row_17, 128 : i32
%start = arith.subi 13, %needed : i32
%start_22 = arith.muli %needed, %start : i32
%start_23 = arith.divsi %start_22, 2 : i32
%prefix = arith.subi %start_23, %needed : i32
%prefix_24 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, 0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %prefix : i32 -> i32
%tail = arith.subi 6, %needed : i32
%tail_25 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, $1, $2;" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %start_23, %tail : i32, i32 -> i32
%needed_26 = arith.andi %prefix_24, 53793 : i32
%needed_27 = arith.shli %needed_26, %needed : i32
%needed_28 = arith.ori %needed_27, %tail_25 : i32
%needed_29 = arith.divsi %col_20, 128 : i32
%start_30 = arith.subi 13, %needed_29 : i32
%start_31 = arith.muli %needed_29, %start_30 : i32
%start_32 = arith.divsi %start_31, 2 : i32
%prefix_33 = arith.subi %start_32, %needed_29 : i32
%prefix_34 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, 0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %prefix_33 : i32 -> i32
%tail_35 = arith.subi 6, %needed_29 : i32
%tail_36 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, $1, $2;" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %start_32, %tail_35 : i32, i32 -> i32
%needed_37 = arith.andi %prefix_34, 53793 : i32
%needed_38 = arith.shli %needed_37, %needed_29 : i32
%needed_39 = arith.ori %needed_38, %tail_36 : i32
%needed_40 = arith.ori %needed_28, %needed_39 : i32
%6 = tt.addptr %Counter, %round_offset : !tt.ptr<i32>, i32
%7 = tt.addptr %6, %batch : !tt.ptr<i32>, i32
%8 = tt.elementwise_inline_asm "{\0A .reg .u32 tid, bits, covered;\0A .reg .pred observer, ready;\0A bar.sync 0;\0A mov.u32 tid, %tid.x;\0A setp.eq.u32 observer, tid, 0;\0A @!observer bra gram_ready;\0A fence.proxy.async.global;\0A atom.acq_rel.gpu.global.or.b32 bits, [$1], $2;\0A or.b32 bits, bits, $2;\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @ready bra gram_ready;\0A observe_gram:\0A ld.acquire.gpu.global.u32 bits, [$1];\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @!ready bra observe_gram;\0A gram_ready:\0A bar.sync 0;\0A fence.proxy.async.global;\0A mov.u32 $0, 0;\0A }" {constraints = "=r,l,r,r,~{memory}", packed_element = 1 : i32, pure = false} %7, %bit, %needed_40 : !tt.ptr<i32>, i32, i32 -> i32
%tiles_41 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers_42 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%9 = ttg.memdesc_index %barriers_42[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%10 = ttg.memdesc_index %barriers_42[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%11 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = ttg.memdesc_index %tiles_41[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_65 = ttg.memdesc_index %tiles_41[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%4, %c0_i32] %acc_65, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_67 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_68 = arith.remsi %phase, 2 : i32
%base_69 = ttg.memdesc_index %tiles_41[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_70 = ttg.memdesc_index %barriers_42[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_70, %phase_68, %true deps %base_69, %acc_65 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_65 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_71 = ttng.warp_group_dot %base_69, %right, %acc_67, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_72 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_73 = ttg.memdesc_index %barriers_42[%nxt_72] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_74 = ttg.memdesc_index %tiles_41[%nxt_72] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_73, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_74, %acc_73, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_75 = ttng.warp_group_dot_wait %acc_71 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %Gram[%4, %k] %acc_65, %acc_73, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_75 : tensor<128x128xf32, #mma>
}
scf.yield %acc_66 : tensor<128x128xf32, #mma>
} else {
%acc = ttg.memdesc_index %tiles_41[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_65 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_67 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_68 = arith.remsi %phase, 2 : i32
%base_69 = ttg.memdesc_index %tiles_41[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_70 = ttg.memdesc_index %barriers_42[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_70, %phase_68, %true deps %base_69, %base_69 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_69 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_71 = ttng.warp_group_dot %base_69, %right, %acc_67, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_72 = ttng.warp_group_dot_wait %acc_71 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_73 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_74 = ttg.memdesc_index %barriers_42[%nxt_73] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_75 = ttg.memdesc_index %tiles_41[%nxt_73] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_74, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_75, %acc_74, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_72 : tensor<128x128xf32, #mma>
}
%acc_66 = ttng.warp_group_dot_wait %acc_65 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_66 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%i = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%j = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %batch, 589824 : i32
%dst = tt.expand_dims %i {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%dst_43 = tt.splat %row_17 : i32 -> tensor<128x1xi32, #blocked>
%dst_44 = arith.addi %dst_43, %dst : tensor<128x1xi32, #blocked>
%dst_45 = arith.muli %dst_44, %cst_12 : tensor<128x1xi32, #blocked>
%dst_46 = tt.expand_dims %j {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%dst_47 = tt.splat %col_20 : i32 -> tensor<1x128xi32, #blocked>
%dst_48 = arith.addi %dst_47, %dst_46 : tensor<1x128xi32, #blocked>
%dst_49 = tt.broadcast %dst_45 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_50 = tt.broadcast %dst_48 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_51 = arith.addi %dst_49, %dst_50 : tensor<128x128xi32, #blocked>
%bias_words = tt.addptr %Gram_ptr, %base : !tt.ptr<bf16>, i32
%bias_words_52 = tt.splat %bias_words : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias_words_53 = tt.addptr %bias_words_52, %dst_51 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%bias_words_54 = tt.load %bias_words_53 cacheModifier = cg : tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias = ttg.convert_layout %bias_words_54 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%bias_55 = arith.extf %bias : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%words_56 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%words_57 = arith.mulf %11, %words_56 : tensor<128x128xf32, #mma>
%words_58 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%words_59 = arith.mulf %bias_55, %words_58 : tensor<128x128xf32, #mma>
%words_60 = arith.addf %words_57, %words_59 : tensor<128x128xf32, #mma>
%words_61 = arith.truncf %words_60 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct_62 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words_61, %direct_62 : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror_63 = tt.trans %words_61 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_64 = ttg.local_alloc %mirror_63 : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_local_to_global %Output[%13, %col_20] %direct_62 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
ttng.async_tma_copy_local_to_global %Output[%4, %row_17] %mirror_64 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%12 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .mlp => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @cooperative_round(%X: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %X_0: i32, %X_1: i32, %X_2: i64, %X_3: i64, %Gram: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Gram_4: i32, %Gram_5: i32, %Gram_6: i64, %Gram_7: i64, %Output: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %Output_8: i32, %Output_9: i32, %Output_10: i64, %Output_11: i64, %Gram_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Counter: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %round_offset: i32, %alpha: f32, %beta: f32) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_12 = arith.constant dense<768> : tensor<128x1xi32, #blocked>
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_13 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_13 : i32 to i64
%row_14 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_15 = arith.andi %row_14, %c7_i64 : i64
%row_16 = arith.trunci %row_15 : i64 to i32
%row_17 = arith.muli %row_16, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_18 = arith.andi %col, %c7_i64 : i64
%col_19 = arith.trunci %col_18 : i64 to i32
%col_20 = arith.muli %col_19, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_17, %col_20 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 2816 : i32
%acc_65 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %row_17] %acc_65, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %col_20] %acc_66, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_67 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_68 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_69 = arith.remsi %phase, 2 : i32
%base_70 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_71 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_71, %phase_69, %true deps %base_70, %acc_66 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base_70 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_72 = ttng.warp_group_dot %left, %acc_66, %acc_68, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_73 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_74 = ttg.memdesc_index %barriers[%nxt_73] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_75 = ttg.memdesc_index %tiles[%nxt_73] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_74, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %X[%coords, %row_17] %acc_75, %acc_74, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_76 = ttng.warp_group_dot_wait %acc_72 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %X[%coords, %col_20] %acc_66, %acc_74, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_76 : tensor<128x128xf32, #mma>
}
scf.yield %acc_67 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 2816 : i32
%acc_65 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %X[%acc, %row_17] %acc_65, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_68 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_69 = arith.remsi %phase, 2 : i32
%base_70 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_71 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_71, %phase_69, %true deps %base_70, %base_70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base_70 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_72 = ttng.warp_group_dot %left, %base_70, %acc_68, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_73 = ttng.warp_group_dot_wait %acc_72 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_74 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_75 = ttg.memdesc_index %barriers[%nxt_74] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_76 = ttg.memdesc_index %tiles[%nxt_74] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_75, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %X[%coords, %row_17] %acc_76, %acc_75, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_73 : tensor<128x128xf32, #mma>
}
%acc_67 = ttng.warp_group_dot_wait %acc_66 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_67 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_21 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.muli %batch, 768 : i32
%14 = arith.addi %13, %row_17 : i32
ttng.async_tma_copy_local_to_global %Gram[%14, %col_20] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 768 : i32
%4 = arith.addi %3, %col_20 : i32
ttng.async_tma_copy_local_to_global %Gram[%4, %row_17] %mirror_21 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
%bit = arith.shli 1, %shift : i32
%needed = arith.divsi %row_17, 128 : i32
%start = arith.subi 13, %needed : i32
%start_22 = arith.muli %needed, %start : i32
%start_23 = arith.divsi %start_22, 2 : i32
%prefix = arith.subi %start_23, %needed : i32
%prefix_24 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, 0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %prefix : i32 -> i32
%tail = arith.subi 6, %needed : i32
%tail_25 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, $1, $2;" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %start_23, %tail : i32, i32 -> i32
%needed_26 = arith.andi %prefix_24, 53793 : i32
%needed_27 = arith.shli %needed_26, %needed : i32
%needed_28 = arith.ori %needed_27, %tail_25 : i32
%needed_29 = arith.divsi %col_20, 128 : i32
%start_30 = arith.subi 13, %needed_29 : i32
%start_31 = arith.muli %needed_29, %start_30 : i32
%start_32 = arith.divsi %start_31, 2 : i32
%prefix_33 = arith.subi %start_32, %needed_29 : i32
%prefix_34 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, 0, $1;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %prefix_33 : i32 -> i32
%tail_35 = arith.subi 6, %needed_29 : i32
%tail_36 = tt.elementwise_inline_asm "bmsk.clamp.b32 $0, $1, $2;" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %start_32, %tail_35 : i32, i32 -> i32
%needed_37 = arith.andi %prefix_34, 53793 : i32
%needed_38 = arith.shli %needed_37, %needed_29 : i32
%needed_39 = arith.ori %needed_38, %tail_36 : i32
%needed_40 = arith.ori %needed_28, %needed_39 : i32
%6 = tt.addptr %Counter, %round_offset : !tt.ptr<i32>, i32
%7 = tt.addptr %6, %batch : !tt.ptr<i32>, i32
%8 = tt.elementwise_inline_asm "{\0A .reg .u32 tid, bits, covered;\0A .reg .pred observer, ready;\0A bar.sync 0;\0A mov.u32 tid, %tid.x;\0A setp.eq.u32 observer, tid, 0;\0A @!observer bra gram_ready;\0A fence.proxy.async.global;\0A atom.acq_rel.gpu.global.or.b32 bits, [$1], $2;\0A or.b32 bits, bits, $2;\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @ready bra gram_ready;\0A observe_gram:\0A ld.acquire.gpu.global.u32 bits, [$1];\0A and.b32 covered, bits, $3;\0A setp.eq.u32 ready, covered, $3;\0A @!ready bra observe_gram;\0A gram_ready:\0A bar.sync 0;\0A fence.proxy.async.global;\0A mov.u32 $0, 0;\0A }" {constraints = "=r,l,r,r,~{memory}", packed_element = 1 : i32, pure = false} %7, %bit, %needed_40 : !tt.ptr<i32>, i32, i32 -> i32
%tiles_41 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers_42 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%9 = ttg.memdesc_index %barriers_42[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%10 = ttg.memdesc_index %barriers_42[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %10, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%11 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = ttg.memdesc_index %tiles_41[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_65 = ttg.memdesc_index %tiles_41[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%4, %c0_i32] %acc_65, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_66 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_67 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_68 = arith.remsi %phase, 2 : i32
%base_69 = ttg.memdesc_index %tiles_41[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_70 = ttg.memdesc_index %barriers_42[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_70, %phase_68, %true deps %base_69, %acc_65 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_65 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_71 = ttng.warp_group_dot %base_69, %right, %acc_67, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_72 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_73 = ttg.memdesc_index %barriers_42[%nxt_72] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_74 = ttg.memdesc_index %tiles_41[%nxt_72] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_73, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_74, %acc_73, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_75 = ttng.warp_group_dot_wait %acc_71 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %Gram[%4, %k] %acc_65, %acc_73, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_75 : tensor<128x128xf32, #mma>
}
scf.yield %acc_66 : tensor<128x128xf32, #mma>
} else {
%acc = ttg.memdesc_index %tiles_41[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %9, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_global_to_local %Gram[%coords, %c0_i32] %acc, %9, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_65 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_67 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_68 = arith.remsi %phase, 2 : i32
%base_69 = ttg.memdesc_index %tiles_41[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_70 = ttg.memdesc_index %barriers_42[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_70, %phase_68, %true deps %base_69, %base_69 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_69 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_71 = ttng.warp_group_dot %base_69, %right, %acc_67, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_72 = ttng.warp_group_dot_wait %acc_71 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_73 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_74 = ttg.memdesc_index %barriers_42[%nxt_73] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_75 = ttg.memdesc_index %tiles_41[%nxt_73] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_74, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %Gram[%coords, %k] %acc_75, %acc_74, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_72 : tensor<128x128xf32, #mma>
}
%acc_66 = ttng.warp_group_dot_wait %acc_65 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_66 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %9 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %10 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%i = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%j = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %batch, 589824 : i32
%dst = tt.expand_dims %i {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%dst_43 = tt.splat %row_17 : i32 -> tensor<128x1xi32, #blocked>
%dst_44 = arith.addi %dst_43, %dst : tensor<128x1xi32, #blocked>
%dst_45 = arith.muli %dst_44, %cst_12 : tensor<128x1xi32, #blocked>
%dst_46 = tt.expand_dims %j {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%dst_47 = tt.splat %col_20 : i32 -> tensor<1x128xi32, #blocked>
%dst_48 = arith.addi %dst_47, %dst_46 : tensor<1x128xi32, #blocked>
%dst_49 = tt.broadcast %dst_45 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_50 = tt.broadcast %dst_48 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_51 = arith.addi %dst_49, %dst_50 : tensor<128x128xi32, #blocked>
%bias_words = tt.addptr %Gram_ptr, %base : !tt.ptr<bf16>, i32
%bias_words_52 = tt.splat %bias_words : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias_words_53 = tt.addptr %bias_words_52, %dst_51 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%bias_words_54 = tt.load %bias_words_53 cacheModifier = cg : tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias = ttg.convert_layout %bias_words_54 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%bias_55 = arith.extf %bias : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%words_56 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%words_57 = arith.mulf %11, %words_56 : tensor<128x128xf32, #mma>
%words_58 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%words_59 = arith.mulf %bias_55, %words_58 : tensor<128x128xf32, #mma>
%words_60 = arith.addf %words_57, %words_59 : tensor<128x128xf32, #mma>
%words_61 = arith.truncf %words_60 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct_62 = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words_61, %direct_62 : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror_63 = tt.trans %words_61 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_64 = ttg.local_alloc %mirror_63 : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%13 = arith.addi %3, %row_17 : i32
ttng.async_tma_copy_local_to_global %Output[%13, %col_20] %direct_62 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
ttng.async_tma_copy_local_to_global %Output[%4, %row_17] %mirror_64 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%12 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#

private def roundAssembly : Bank → String
  | .qk => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_round(
.param .align 64 .b8 cooperative_round_param_0[128],
.param .u32 cooperative_round_param_1,
.param .u32 cooperative_round_param_2,
.param .u64 cooperative_round_param_3,
.param .u64 cooperative_round_param_4,
.param .align 64 .b8 cooperative_round_param_5[128],
.param .u32 cooperative_round_param_6,
.param .u32 cooperative_round_param_7,
.param .u64 cooperative_round_param_8,
.param .u64 cooperative_round_param_9,
.param .align 64 .b8 cooperative_round_param_10[128],
.param .u32 cooperative_round_param_11,
.param .u32 cooperative_round_param_12,
.param .u64 cooperative_round_param_13,
.param .u64 cooperative_round_param_14,
.param .u64 .ptr .global .align 1 cooperative_round_param_15,
.param .u64 .ptr .global .align 1 cooperative_round_param_16,
.param .u32 cooperative_round_param_17,
.param .f32 cooperative_round_param_18,
.param .f32 cooperative_round_param_19,
.param .u64 .ptr .global .align 1 cooperative_round_param_20,
.param .u64 .ptr .global .align 1 cooperative_round_param_21
)
.reqntid 256
{
.reg .pred %p<59>;
.reg .b16 %rs<65>;
.reg .b32 %r<704>;
.reg .b64 %rd<92>;
$L__func_begin0:
mov.b64 %rd3, cooperative_round_param_0;
mov.b64 %rd4, cooperative_round_param_10;
$L__tmp0:
cvta.param.u64 %rd91, %rd4;
mov.b64 %rd5, cooperative_round_param_5;
cvta.param.u64 %rd63, %rd5;
cvta.param.u64 %rd24, %rd3;
mov.u32 %r97, %ctaid.x;
mul.hi.u32 %r1, %r97, 1431655766;
$L__tmp1:
mul.lo.s32 %r98, %r1, 3;
sub.s32 %r2, %r97, %r98;
mul.lo.s32 %r99, %r2, 3;
mov.b32 %r100, 9216;
shr.u32 %r101, %r100, %r99;
and.b32 %r3, %r101, 896;
mov.b32 %r102, 8192;
shr.u32 %r103, %r102, %r99;
and.b32 %r4, %r103, 896;
$L__tmp2:
setp.eq.b32 %p4, %r3, %r4;
mov.u32 %r5, %tid.x;
and.b32 %r6, %r5, 255;
setp.eq.b32 %p3, %r6, 0;
mov.b32 %r104, global_smem;
add.s32 %r228, %r104, 98304;
@%p3 mbarrier.init.shared::cta.b64 [%r228], 1;
bar.sync 0;
add.s32 %r96, %r104, 98312;
@%p3 mbarrier.init.shared::cta.b64 [%r96], 1;
shl.b32 %r7, %r1, 8;
$L__tmp3:
add.s32 %r561, %r3, %r7;
shr.u32 %r8, %r5, 5;
setp.lt.u32 %p56, %r6, 64;
add.s32 %r565, %r4, %r7;
$L__tmp4:
@%p4 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r228], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r135, %r8, 0, 31, -1;
elect.sync %r136|%p20, -1;
and.pred %p19, %p56, %p20;
and.b32 %r137, %r135, 1;
shl.b32 %r138, %r137, 14;
add.s32 %r133, %r104, %r138;
shl.b32 %r134, %r137, 6;
@%p19 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r133], [%rd24, {%r134, %r561}], [%r228];
shl.b32 %r139, %r135, 7;
and.b32 %r11, %r139, 512;
$L__tmp6:
or.b32 %r635, %r134, 128;
mov.b32 %r636, 0;
mov.b32 %r570, 0f00000000;
mov.b32 %r571, %r570;
mov.b32 %r572, %r570;
mov.b32 %r573, %r570;
mov.b32 %r574, %r570;
mov.b32 %r575, %r570;
mov.b32 %r576, %r570;
mov.b32 %r577, %r570;
mov.b32 %r578, %r570;
mov.b32 %r579, %r570;
mov.b32 %r580, %r570;
mov.b32 %r581, %r570;
mov.b32 %r582, %r570;
mov.b32 %r583, %r570;
mov.b32 %r584, %r570;
mov.b32 %r585, %r570;
mov.b32 %r586, %r570;
mov.b32 %r587, %r570;
mov.b32 %r588, %r570;
mov.b32 %r589, %r570;
mov.b32 %r590, %r570;
mov.b32 %r591, %r570;
mov.b32 %r592, %r570;
mov.b32 %r593, %r570;
mov.b32 %r594, %r570;
mov.b32 %r595, %r570;
mov.b32 %r596, %r570;
mov.b32 %r597, %r570;
mov.b32 %r598, %r570;
mov.b32 %r599, %r570;
mov.b32 %r600, %r570;
mov.b32 %r601, %r570;
mov.b32 %r602, %r570;
mov.b32 %r603, %r570;
mov.b32 %r604, %r570;
mov.b32 %r605, %r570;
mov.b32 %r606, %r570;
mov.b32 %r607, %r570;
mov.b32 %r608, %r570;
mov.b32 %r609, %r570;
mov.b32 %r610, %r570;
mov.b32 %r611, %r570;
mov.b32 %r612, %r570;
mov.b32 %r613, %r570;
mov.b32 %r614, %r570;
mov.b32 %r615, %r570;
mov.b32 %r616, %r570;
mov.b32 %r617, %r570;
mov.b32 %r618, %r570;
mov.b32 %r619, %r570;
mov.b32 %r620, %r570;
mov.b32 %r621, %r570;
mov.b32 %r622, %r570;
mov.b32 %r623, %r570;
mov.b32 %r624, %r570;
mov.b32 %r625, %r570;
mov.b32 %r626, %r570;
mov.b32 %r627, %r570;
mov.b32 %r628, %r570;
mov.b32 %r629, %r570;
mov.b32 %r630, %r570;
mov.b32 %r631, %r570;
mov.b32 %r632, %r570;
mov.b32 %r633, %r570;
$L__BB0_4:
and.b32 %r144, %r636, 1;
bfe.u32 %r141, %r636, 1, 1;
shl.b32 %r145, %r144, 15;
add.s32 %r146, %r104, %r145;
mad.lo.s32 %r147, %r144, -32760, %r146;
add.s32 %r140, %r147, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r140], %r141;
@!complete bra.uni waitLoop;
}
shr.u32 %r148, %r146, 4;
add.s32 %r149, %r148, %r11;
and.b32 %r150, %r149, 16383;
cvt.u64.u32 %rd41, %r150;
bfe.u32 %r151, %r146, 4, 14;
cvt.u64.u32 %rd42, %r151;
wgmma.fence.sync.aligned;
or.b64 %rd25, %rd41, 4611686293305294848;
or.b64 %rd26, %rd42, 4611686293305294848;
mov.pred %p21, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd25, %rd26, %p21, 1, 1, 0, 0;
add.s64 %rd27, %rd41, 4611686293305294850;
add.s64 %rd28, %rd42, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd27, %rd28, %p21, 1, 1, 0, 0;
add.s64 %rd29, %rd41, 4611686293305294852;
add.s64 %rd30, %rd42, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd29, %rd30, %p21, 1, 1, 0, 0;
add.s64 %rd31, %rd41, 4611686293305294854;
add.s64 %rd32, %rd42, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd31, %rd32, %p21, 1, 1, 0, 0;
add.s64 %rd33, %rd41, 4611686293305295872;
add.s64 %rd34, %rd42, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd33, %rd34, %p21, 1, 1, 0, 0;
add.s64 %rd35, %rd41, 4611686293305295874;
add.s64 %rd36, %rd42, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd35, %rd36, %p21, 1, 1, 0, 0;
add.s64 %rd37, %rd41, 4611686293305295876;
add.s64 %rd38, %rd42, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd37, %rd38, %p21, 1, 1, 0, 0;
add.s64 %rd39, %rd41, 4611686293305295878;
add.s64 %rd40, %rd42, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd39, %rd40, %p21, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r12, %r636, 1;
and.b32 %r152, %r12, 1;
setp.ne.b32 %p24, %r636, 5;
shl.b32 %r153, %r152, 3;
add.s32 %r154, %r104, %r153;
add.s32 %r142, %r154, 98304;
shl.b32 %r155, %r152, 15;
$L__tmp7:
bar.sync 0;
and.pred %p22, %p3, %p24;
@%p22 mbarrier.arrive.expect_tx.shared.b64 _, [%r142], 32768;
bar.sync 0;
elect.sync %r156|%p25, -1;
and.pred %p26, %p24, %p25;
and.pred %p23, %p56, %p26;
add.s32 %r143, %r133, %r155;
@%p23 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r143], [%rd24, {%r635, %r561}], [%r142];
$L__tmp8:
add.s32 %r635, %r635, 128;
setp.ne.b32 %p27, %r12, 6;
mov.b32 %r636, %r12;
@%p27 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r228], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r107, %r8, 0, 31, -1;
elect.sync %r108|%p7, -1;
and.pred %p5, %p56, %p7;
and.b32 %r109, %r107, 1;
shl.b32 %r110, %r109, 14;
add.s32 %r105, %r104, %r110;
shl.b32 %r106, %r109, 6;
@%p5 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r105], [%rd24, {%r106, %r561}], [%r228];
$L__tmp10:
bar.sync 0;
elect.sync %r111|%p8, -1;
and.pred %p6, %p56, %p8;
add.s32 %r119, %r105, 65536;
@%p6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r119], [%rd24, {%r106, %r565}], [%r228];
shl.b32 %r112, %r107, 7;
and.b32 %r9, %r112, 512;
add.s32 %r113, %r104, 65536;
bfe.u32 %r114, %r113, 4, 14;
cvt.u64.u32 %rd6, %r114;
or.b64 %rd8, %rd6, 4611686293305294848;
add.s64 %rd10, %rd6, 4611686293305294850;
add.s64 %rd12, %rd6, 4611686293305294852;
add.s64 %rd14, %rd6, 4611686293305294854;
add.s64 %rd16, %rd6, 4611686293305295872;
add.s64 %rd18, %rd6, 4611686293305295874;
add.s64 %rd20, %rd6, 4611686293305295876;
add.s64 %rd22, %rd6, 4611686293305295878;
$L__tmp11:
or.b32 %r569, %r106, 128;
mov.b32 %r634, 0;
mov.b32 %r570, 0f00000000;
mov.b32 %r571, %r570;
mov.b32 %r572, %r570;
mov.b32 %r573, %r570;
mov.b32 %r574, %r570;
mov.b32 %r575, %r570;
mov.b32 %r576, %r570;
mov.b32 %r577, %r570;
mov.b32 %r578, %r570;
mov.b32 %r579, %r570;
mov.b32 %r580, %r570;
mov.b32 %r581, %r570;
mov.b32 %r582, %r570;
mov.b32 %r583, %r570;
mov.b32 %r584, %r570;
mov.b32 %r585, %r570;
mov.b32 %r586, %r570;
mov.b32 %r587, %r570;
mov.b32 %r588, %r570;
mov.b32 %r589, %r570;
mov.b32 %r590, %r570;
mov.b32 %r591, %r570;
mov.b32 %r592, %r570;
mov.b32 %r593, %r570;
mov.b32 %r594, %r570;
mov.b32 %r595, %r570;
mov.b32 %r596, %r570;
mov.b32 %r597, %r570;
mov.b32 %r598, %r570;
mov.b32 %r599, %r570;
mov.b32 %r600, %r570;
mov.b32 %r601, %r570;
mov.b32 %r602, %r570;
mov.b32 %r603, %r570;
mov.b32 %r604, %r570;
mov.b32 %r605, %r570;
mov.b32 %r606, %r570;
mov.b32 %r607, %r570;
mov.b32 %r608, %r570;
mov.b32 %r609, %r570;
mov.b32 %r610, %r570;
mov.b32 %r611, %r570;
mov.b32 %r612, %r570;
mov.b32 %r613, %r570;
mov.b32 %r614, %r570;
mov.b32 %r615, %r570;
mov.b32 %r616, %r570;
mov.b32 %r617, %r570;
mov.b32 %r618, %r570;
mov.b32 %r619, %r570;
mov.b32 %r620, %r570;
mov.b32 %r621, %r570;
mov.b32 %r622, %r570;
mov.b32 %r623, %r570;
mov.b32 %r624, %r570;
mov.b32 %r625, %r570;
mov.b32 %r626, %r570;
mov.b32 %r627, %r570;
mov.b32 %r628, %r570;
mov.b32 %r629, %r570;
mov.b32 %r630, %r570;
mov.b32 %r631, %r570;
mov.b32 %r632, %r570;
mov.b32 %r633, %r570;
$L__BB0_2:
and.b32 %r120, %r634, 1;
bfe.u32 %r116, %r634, 1, 1;
shl.b32 %r121, %r120, 15;
add.s32 %r122, %r104, %r121;
mad.lo.s32 %r123, %r120, -32760, %r122;
add.s32 %r115, %r123, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r115], %r116;
@!complete bra.uni waitLoop;
}
shr.u32 %r124, %r122, 4;
add.s32 %r125, %r124, %r9;
and.b32 %r126, %r125, 16383;
cvt.u64.u32 %rd23, %r126;
wgmma.fence.sync.aligned;
or.b64 %rd7, %rd23, 4611686293305294848;
mov.pred %p9, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd7, %rd8, %p9, 1, 1, 0, 0;
add.s64 %rd9, %rd23, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd9, %rd10, %p9, 1, 1, 0, 0;
add.s64 %rd11, %rd23, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd11, %rd12, %p9, 1, 1, 0, 0;
add.s64 %rd13, %rd23, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd13, %rd14, %p9, 1, 1, 0, 0;
add.s64 %rd15, %rd23, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd15, %rd16, %p9, 1, 1, 0, 0;
add.s64 %rd17, %rd23, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd17, %rd18, %p9, 1, 1, 0, 0;
add.s64 %rd19, %rd23, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd19, %rd20, %p9, 1, 1, 0, 0;
add.s64 %rd21, %rd23, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r570,%r571,%r572,%r573,%r574,%r575,%r576,%r577,%r578,%r579,%r580,%r581,%r582,%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633}, %rd21, %rd22, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r10, %r634, 1;
and.b32 %r127, %r10, 1;
setp.ne.b32 %p13, %r634, 5;
shl.b32 %r128, %r127, 3;
add.s32 %r129, %r104, %r128;
add.s32 %r117, %r129, 98304;
shl.b32 %r130, %r127, 15;
$L__tmp12:
bar.sync 0;
and.pred %p10, %p3, %p13;
@%p10 mbarrier.arrive.expect_tx.shared.b64 _, [%r117], 65536;
bar.sync 0;
elect.sync %r131|%p14, -1;
and.pred %p15, %p13, %p14;
and.pred %p11, %p56, %p15;
add.s32 %r118, %r105, %r130;
@%p11 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r118], [%rd24, {%r569, %r561}], [%r117];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r132|%p16, -1;
and.pred %p17, %p13, %p16;
and.pred %p12, %p56, %p17;
@%p12 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r119], [%rd24, {%r569, %r565}], [%r117];
$L__tmp15:
add.s32 %r569, %r569, 128;
setp.eq.b32 %p18, %r10, 6;
mov.b32 %r634, %r10;
@%p18 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p28, %r3, %r4;
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r228];
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r96];
cvt.rn.bf16x2.f32 %r13, %r571, %r570;
cvt.rn.bf16x2.f32 %r14, %r573, %r572;
cvt.rn.bf16x2.f32 %r15, %r575, %r574;
cvt.rn.bf16x2.f32 %r16, %r577, %r576;
cvt.rn.bf16x2.f32 %r17, %r579, %r578;
cvt.rn.bf16x2.f32 %r18, %r581, %r580;
cvt.rn.bf16x2.f32 %r19, %r583, %r582;
cvt.rn.bf16x2.f32 %r20, %r585, %r584;
cvt.rn.bf16x2.f32 %r21, %r587, %r586;
cvt.rn.bf16x2.f32 %r22, %r589, %r588;
cvt.rn.bf16x2.f32 %r23, %r591, %r590;
cvt.rn.bf16x2.f32 %r24, %r593, %r592;
cvt.rn.bf16x2.f32 %r25, %r595, %r594;
cvt.rn.bf16x2.f32 %r26, %r597, %r596;
cvt.rn.bf16x2.f32 %r27, %r599, %r598;
cvt.rn.bf16x2.f32 %r28, %r601, %r600;
cvt.rn.bf16x2.f32 %r29, %r603, %r602;
cvt.rn.bf16x2.f32 %r30, %r605, %r604;
cvt.rn.bf16x2.f32 %r31, %r607, %r606;
cvt.rn.bf16x2.f32 %r32, %r609, %r608;
cvt.rn.bf16x2.f32 %r33, %r611, %r610;
cvt.rn.bf16x2.f32 %r34, %r613, %r612;
cvt.rn.bf16x2.f32 %r35, %r615, %r614;
cvt.rn.bf16x2.f32 %r36, %r617, %r616;
cvt.rn.bf16x2.f32 %r37, %r619, %r618;
cvt.rn.bf16x2.f32 %r38, %r621, %r620;
cvt.rn.bf16x2.f32 %r39, %r623, %r622;
cvt.rn.bf16x2.f32 %r40, %r625, %r624;
cvt.rn.bf16x2.f32 %r41, %r627, %r626;
cvt.rn.bf16x2.f32 %r42, %r629, %r628;
cvt.rn.bf16x2.f32 %r43, %r631, %r630;
cvt.rn.bf16x2.f32 %r44, %r633, %r632;
shl.b32 %r45, %r5, 7;
@%p28 bra $L__BB0_8;
and.b32 %r637, %r5, 16;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r157, %r45, 1920;
shl.b32 %r158, %r5, 6;
and.b32 %r159, %r158, 14336;
shl.b32 %r160, %r5, 4;
and.b32 %r161, %r160, 112;
and.b32 %r637, %r5, 16;
or.b32 %r162, %r157, %r161;
xor.b32 %r163, %r162, %r637;
or.b32 %r164, %r163, %r159;
add.s32 %r165, %r104, %r164;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r165], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r165+16384], {%r29, %r30, %r31, %r32};
xor.b32 %r166, %r164, 32;
add.s32 %r167, %r104, %r166;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r167], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r167+16384], {%r33, %r34, %r35, %r36};
xor.b32 %r168, %r164, 64;
add.s32 %r169, %r104, %r168;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r169], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r169+16384], {%r37, %r38, %r39, %r40};
xor.b32 %r170, %r164, 96;
add.s32 %r171, %r104, %r170;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r171], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r171+16384], {%r41, %r42, %r43, %r44};
$L__BB0_9:
ld.param.b32 %r93, [cooperative_round_param_17];
ld.param.b64 %rd2, [cooperative_round_param_16];
and.b32 %r172, %r45, 17280;
and.b32 %r46, %r5, 7;
shl.b32 %r47, %r46, 4;
and.b32 %r173, %r5, 96;
shl.b32 %r174, %r5, 1;
and.b32 %r175, %r174, 16;
shl.b32 %r176, %r637, 6;
or.b32 %r177, %r172, %r47;
or.b32 %r178, %r175, %r173;
xor.b32 %r179, %r177, %r178;
add.s32 %r180, %r104, %r176;
add.s32 %r181, %r180, %r179;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+32768], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+34816], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+36864], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+38912], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+40960], {%r29, %r30, %r31, %r32};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+43008], {%r33, %r34, %r35, %r36};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+45056], {%r37, %r38, %r39, %r40};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r181+47104], {%r41, %r42, %r43, %r44};
fence.proxy.async.shared::cta;
@%p4 bra $L__BB0_11;
bar.sync 0;
elect.sync %r184|%p30, -1;
shfl.sync.idx.b32 %r185, %r8, 0, 31, -1;
and.pred %p29, %p56, %p30;
and.b32 %r186, %r185, 1;
shl.b32 %r187, %r186, 14;
add.s32 %r183, %r104, %r187;
shl.b32 %r188, %r186, 6;
or.b32 %r182, %r188, %r4;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd63, {%r182, %r561}], [%r183];
cp.async.bulk.commit_group;
$L__BB0_11:
ld.param.b32 %r95, [cooperative_round_param_19];
ld.param.b32 %r94, [cooperative_round_param_18];
ld.param.b64 %rd1, [cooperative_round_param_15];
bar.sync 0;
elect.sync %r193|%p32, -1;
shfl.sync.idx.b32 %r56, %r8, 0, 31, -1;
and.pred %p31, %p56, %p32;
and.b32 %r194, %r56, 1;
shl.b32 %r195, %r194, 14;
add.s32 %r562, %r104, %r195;
add.s32 %r566, %r562, 32768;
shl.b32 %r229, %r194, 6;
or.b32 %r564, %r229, %r3;
@%p31 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd63, {%r564, %r565}], [%r566];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r189, 0;
mov.b32 %r196, 1;
$L__tmp17:
shl.b32 %r191, %r196, %r2;
shr.u32 %r197, %r3, 7;
mov.b32 %r198, 3;
$L__tmp18:
shl.b32 %r199, %r198, %r197;
$L__tmp19:
shr.u32 %r200, %r4, 7;
$L__tmp20:
shl.b32 %r201, %r198, %r200;
$L__tmp21:
or.b32 %r192, %r199, %r201;
mad.wide.s32 %rd44, %r93, 4, %rd2;
mad.wide.u32 %rd43, %r1, 4, %rd44;
$L__tmp22:
{
.reg .u32 tid, bits, covered;
.reg .pred observer, ready;
bar.sync 0;
mov.u32 tid, %tid.x;
setp.eq.u32 observer, tid, 0;
@!observer bra gram_ready;
fence.proxy.async.global;
atom.acq_rel.gpu.global.or.b32 bits, [%rd43], %r191;
or.b32 bits, bits, %r191;
and.b32 covered, bits, %r192;
setp.eq.u32 ready, covered, %r192;
@ready bra gram_ready;
observe_gram:
ld.acquire.gpu.global.u32 bits, [%rd43];
and.b32 covered, bits, %r192;
setp.eq.u32 ready, covered, %r192;
@!ready bra observe_gram;
gram_ready:
bar.sync 0;
fence.proxy.async.global;
mov.u32 %r190, 0;
}
$L__tmp23:
@%p3 mbarrier.init.shared::cta.b64 [%r228], 1;
bar.sync 0;
@%p3 mbarrier.init.shared::cta.b64 [%r96], 1;
@%p4 bra $L__BB0_14;
bra.uni $L__BB0_12;
$L__BB0_14:
$L__tmp24:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r228], 32768;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r231|%p46, -1;
and.pred %p45, %p56, %p46;
@%p45 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r562], [%rd63, {%r229, %r561}], [%r228];
shl.b32 %r232, %r56, 7;
and.b32 %r58, %r232, 512;
mov.b32 %r230, 0;
mov.pred %p47, -1;
mov.b32 %r638, 0f00000000;
mov.b32 %r639, %r638;
mov.b32 %r640, %r638;
mov.b32 %r641, %r638;
mov.b32 %r642, %r638;
mov.b32 %r643, %r638;
mov.b32 %r644, %r638;
mov.b32 %r645, %r638;
mov.b32 %r646, %r638;
mov.b32 %r647, %r638;
mov.b32 %r648, %r638;
mov.b32 %r649, %r638;
mov.b32 %r650, %r638;
mov.b32 %r651, %r638;
mov.b32 %r652, %r638;
mov.b32 %r653, %r638;
mov.b32 %r654, %r638;
mov.b32 %r655, %r638;
mov.b32 %r656, %r638;
mov.b32 %r657, %r638;
mov.b32 %r658, %r638;
mov.b32 %r659, %r638;
mov.b32 %r660, %r638;
mov.b32 %r661, %r638;
mov.b32 %r662, %r638;
mov.b32 %r663, %r638;
mov.b32 %r664, %r638;
mov.b32 %r665, %r638;
mov.b32 %r666, %r638;
mov.b32 %r667, %r638;
mov.b32 %r668, %r638;
mov.b32 %r669, %r638;
mov.b32 %r670, %r638;
mov.b32 %r671, %r638;
mov.b32 %r672, %r638;
mov.b32 %r673, %r638;
mov.b32 %r674, %r638;
mov.b32 %r675, %r638;
mov.b32 %r676, %r638;
mov.b32 %r677, %r638;
mov.b32 %r678, %r638;
mov.b32 %r679, %r638;
mov.b32 %r680, %r638;
mov.b32 %r681, %r638;
mov.b32 %r682, %r638;
mov.b32 %r683, %r638;
mov.b32 %r684, %r638;
mov.b32 %r685, %r638;
mov.b32 %r686, %r638;
mov.b32 %r687, %r638;
mov.b32 %r688, %r638;
mov.b32 %r689, %r638;
mov.b32 %r690, %r638;
mov.b32 %r691, %r638;
mov.b32 %r692, %r638;
mov.b32 %r693, %r638;
mov.b32 %r694, %r638;
mov.b32 %r695, %r638;
mov.b32 %r696, %r638;
mov.b32 %r697, %r638;
mov.b32 %r698, %r638;
mov.b32 %r699, %r638;
mov.b32 %r700, %r638;
mov.b32 %r701, %r638;
mov.pred %p58, %p47;
mov.b32 %r703, %r230;
$L__tmp25:
$L__BB0_15:
mov.pred %p2, %p58;
shl.b32 %r237, %r703, 15;
add.s32 %r238, %r104, %r237;
mad.lo.s32 %r239, %r703, -32760, %r238;
add.s32 %r233, %r239, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r233], %r230;
@!complete bra.uni waitLoop;
}
shr.u32 %r240, %r238, 4;
add.s32 %r241, %r240, %r58;
and.b32 %r242, %r241, 16383;
cvt.u64.u32 %rd80, %r242;
bfe.u32 %r243, %r238, 4, 14;
cvt.u64.u32 %rd81, %r243;
wgmma.fence.sync.aligned;
or.b64 %rd64, %rd80, 4611686293305294848;
or.b64 %rd65, %rd81, 4611686293305294848;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd64, %rd65, %p47, 1, 1, 0, 0;
add.s64 %rd66, %rd80, 4611686293305294850;
add.s64 %rd67, %rd81, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd66, %rd67, %p47, 1, 1, 0, 0;
add.s64 %rd68, %rd80, 4611686293305294852;
add.s64 %rd69, %rd81, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd68, %rd69, %p47, 1, 1, 0, 0;
add.s64 %rd70, %rd80, 4611686293305294854;
add.s64 %rd71, %rd81, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd70, %rd71, %p47, 1, 1, 0, 0;
add.s64 %rd72, %rd80, 4611686293305295872;
add.s64 %rd73, %rd81, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd72, %rd73, %p47, 1, 1, 0, 0;
add.s64 %rd74, %rd80, 4611686293305295874;
add.s64 %rd75, %rd81, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd74, %rd75, %p47, 1, 1, 0, 0;
add.s64 %rd76, %rd80, 4611686293305295876;
add.s64 %rd77, %rd81, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd76, %rd77, %p47, 1, 1, 0, 0;
add.s64 %rd78, %rd80, 4611686293305295878;
add.s64 %rd79, %rd81, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd78, %rd79, %p47, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r244, %r703, 1;
and.b32 %r245, %r244, 1;
shl.b32 %r246, %r244, 7;
shl.b32 %r247, %r245, 3;
add.s32 %r248, %r104, %r247;
add.s32 %r234, %r248, 98304;
shl.b32 %r249, %r245, 15;
$L__tmp26:
bar.sync 0;
or.b32 %r250, %r703, %r6;
setp.eq.b32 %p48, %r250, 0;
@%p48 mbarrier.arrive.expect_tx.shared.b64 _, [%r234], 32768;
bar.sync 0;
elect.sync %r251|%p50, -1;
and.pred %p51, %p2, %p50;
and.pred %p49, %p56, %p51;
add.s32 %r235, %r562, %r249;
or.b32 %r236, %r246, %r229;
@%p49 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r235], [%rd63, {%r236, %r561}], [%r234];
mov.b32 %r703, 1;
mov.pred %p58, 0;
$L__tmp27:
@%p2 bra $L__BB0_15;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_17;
$L__tmp28:
$L__BB0_12:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r228], 65536;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r203|%p35, -1;
and.pred %p33, %p56, %p35;
@%p33 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r562], [%rd63, {%r229, %r561}], [%r228];
$L__tmp29:
bar.sync 0;
elect.sync %r204|%p36, -1;
and.pred %p34, %p56, %p36;
add.s32 %r212, %r562, 65536;
@%p34 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r212], [%rd63, {%r229, %r565}], [%r228];
shl.b32 %r205, %r56, 7;
and.b32 %r57, %r205, 512;
add.s32 %r206, %r104, 65536;
bfe.u32 %r207, %r206, 4, 14;
cvt.u64.u32 %rd45, %r207;
or.b64 %rd47, %rd45, 4611686293305294848;
add.s64 %rd49, %rd45, 4611686293305294850;
add.s64 %rd51, %rd45, 4611686293305294852;
add.s64 %rd53, %rd45, 4611686293305294854;
add.s64 %rd55, %rd45, 4611686293305295872;
add.s64 %rd57, %rd45, 4611686293305295874;
add.s64 %rd59, %rd45, 4611686293305295876;
add.s64 %rd61, %rd45, 4611686293305295878;
mov.b32 %r202, 0;
mov.pred %p37, -1;
mov.b32 %r638, 0f00000000;
mov.b32 %r639, %r638;
mov.b32 %r640, %r638;
mov.b32 %r641, %r638;
mov.b32 %r642, %r638;
mov.b32 %r643, %r638;
mov.b32 %r644, %r638;
mov.b32 %r645, %r638;
mov.b32 %r646, %r638;
mov.b32 %r647, %r638;
mov.b32 %r648, %r638;
mov.b32 %r649, %r638;
mov.b32 %r650, %r638;
mov.b32 %r651, %r638;
mov.b32 %r652, %r638;
mov.b32 %r653, %r638;
mov.b32 %r654, %r638;
mov.b32 %r655, %r638;
mov.b32 %r656, %r638;
mov.b32 %r657, %r638;
mov.b32 %r658, %r638;
mov.b32 %r659, %r638;
mov.b32 %r660, %r638;
mov.b32 %r661, %r638;
mov.b32 %r662, %r638;
mov.b32 %r663, %r638;
mov.b32 %r664, %r638;
mov.b32 %r665, %r638;
mov.b32 %r666, %r638;
mov.b32 %r667, %r638;
mov.b32 %r668, %r638;
mov.b32 %r669, %r638;
mov.b32 %r670, %r638;
mov.b32 %r671, %r638;
mov.b32 %r672, %r638;
mov.b32 %r673, %r638;
mov.b32 %r674, %r638;
mov.b32 %r675, %r638;
mov.b32 %r676, %r638;
mov.b32 %r677, %r638;
mov.b32 %r678, %r638;
mov.b32 %r679, %r638;
mov.b32 %r680, %r638;
mov.b32 %r681, %r638;
mov.b32 %r682, %r638;
mov.b32 %r683, %r638;
mov.b32 %r684, %r638;
mov.b32 %r685, %r638;
mov.b32 %r686, %r638;
mov.b32 %r687, %r638;
mov.b32 %r688, %r638;
mov.b32 %r689, %r638;
mov.b32 %r690, %r638;
mov.b32 %r691, %r638;
mov.b32 %r692, %r638;
mov.b32 %r693, %r638;
mov.b32 %r694, %r638;
mov.b32 %r695, %r638;
mov.b32 %r696, %r638;
mov.b32 %r697, %r638;
mov.b32 %r698, %r638;
mov.b32 %r699, %r638;
mov.b32 %r700, %r638;
mov.b32 %r701, %r638;
mov.pred %p57, %p37;
mov.b32 %r702, %r202;
$L__tmp30:
$L__BB0_13:
mov.pred %p1, %p57;
shl.b32 %r213, %r702, 15;
add.s32 %r214, %r104, %r213;
mad.lo.s32 %r215, %r702, -32760, %r214;
add.s32 %r208, %r215, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r208], %r202;
@!complete bra.uni waitLoop;
}
shr.u32 %r216, %r214, 4;
add.s32 %r217, %r216, %r57;
and.b32 %r218, %r217, 16383;
cvt.u64.u32 %rd62, %r218;
wgmma.fence.sync.aligned;
or.b64 %rd46, %rd62, 4611686293305294848;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd46, %rd47, %p37, 1, 1, 0, 0;
add.s64 %rd48, %rd62, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd48, %rd49, %p37, 1, 1, 0, 0;
add.s64 %rd50, %rd62, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd50, %rd51, %p37, 1, 1, 0, 0;
add.s64 %rd52, %rd62, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd52, %rd53, %p37, 1, 1, 0, 0;
add.s64 %rd54, %rd62, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd54, %rd55, %p37, 1, 1, 0, 0;
add.s64 %rd56, %rd62, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd56, %rd57, %p37, 1, 1, 0, 0;
add.s64 %rd58, %rd62, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd58, %rd59, %p37, 1, 1, 0, 0;
add.s64 %rd60, %rd62, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701}, %rd60, %rd61, %p37, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r219, %r702, 1;
and.b32 %r220, %r219, 1;
shl.b32 %r221, %r219, 7;
shl.b32 %r222, %r220, 3;
add.s32 %r223, %r104, %r222;
add.s32 %r209, %r223, 98304;
shl.b32 %r224, %r220, 15;
$L__tmp31:
bar.sync 0;
or.b32 %r225, %r702, %r6;
setp.eq.b32 %p38, %r225, 0;
@%p38 mbarrier.arrive.expect_tx.shared.b64 _, [%r209], 65536;
bar.sync 0;
elect.sync %r226|%p41, -1;
and.pred %p42, %p1, %p41;
and.pred %p39, %p56, %p42;
add.s32 %r210, %r562, %r224;
or.b32 %r211, %r221, %r229;
@%p39 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r210], [%rd63, {%r211, %r561}], [%r209];
$L__tmp32:
wgmma.wait_group.sync.aligned 0;
$L__tmp33:
bar.sync 0;
elect.sync %r227|%p43, -1;
and.pred %p44, %p1, %p43;
and.pred %p40, %p56, %p44;
@%p40 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r212], [%rd63, {%r211, %r565}], [%r209];
mov.b32 %r702, 1;
mov.pred %p57, 0;
$L__tmp34:
@%p1 bra $L__BB0_13;
$L__tmp35:
$L__BB0_17:
add.s32 %r48, %r181, 32768;
add.s32 %r49, %r181, 34816;
add.s32 %r50, %r181, 36864;
add.s32 %r51, %r181, 38912;
add.s32 %r52, %r181, 40960;
add.s32 %r53, %r181, 43008;
add.s32 %r54, %r181, 45056;
add.s32 %r55, %r181, 47104;
$L__tmp36:
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r228];
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r96];
and.b32 %r284, %r5, 240;
bfe.u32 %r285, %r5, 4, 4;
and.b32 %r59, %r5, 15;
shl.b32 %r286, %r59, 3;
shl.b32 %r287, %r1, 16;
or.b32 %r288, %r3, %r285;
shl.b32 %r289, %r288, 8;
or.b32 %r290, %r289, 4096;
or.b32 %r291, %r289, 8192;
or.b32 %r292, %r289, 12288;
or.b32 %r293, %r289, 16384;
or.b32 %r294, %r289, 20480;
or.b32 %r295, %r289, 24576;
or.b32 %r296, %r289, 28672;
or.b32 %r297, %r4, %r286;
add.s32 %r298, %r289, %r297;
add.s32 %r299, %r290, %r297;
add.s32 %r300, %r291, %r297;
add.s32 %r301, %r292, %r297;
add.s32 %r302, %r293, %r297;
add.s32 %r303, %r294, %r297;
add.s32 %r304, %r295, %r297;
add.s32 %r305, %r296, %r297;
mad.wide.s32 %rd90, %r287, 2, %rd1;
mad.wide.u32 %rd82, %r298, 2, %rd90;
mad.wide.u32 %rd83, %r299, 2, %rd90;
mad.wide.u32 %rd84, %r300, 2, %rd90;
mad.wide.u32 %rd85, %r301, 2, %rd90;
mad.wide.u32 %rd86, %r302, 2, %rd90;
mad.wide.u32 %rd87, %r303, 2, %rd90;
mad.wide.u32 %rd88, %r304, 2, %rd90;
mad.wide.u32 %rd89, %r305, 2, %rd90;
mov.u32 %r252, 0x0;
mov.u32 %r253, 0x0;
mov.u32 %r254, 0x0;
mov.u32 %r255, 0x0;
ld.global.cg.v4.b32 { %r252, %r253, %r254, %r255 }, [ %rd82 + 0 ];
mov.u32 %r256, 0x0;
mov.u32 %r257, 0x0;
mov.u32 %r258, 0x0;
mov.u32 %r259, 0x0;
ld.global.cg.v4.b32 { %r256, %r257, %r258, %r259 }, [ %rd83 + 0 ];
mov.u32 %r260, 0x0;
mov.u32 %r261, 0x0;
mov.u32 %r262, 0x0;
mov.u32 %r263, 0x0;
ld.global.cg.v4.b32 { %r260, %r261, %r262, %r263 }, [ %rd84 + 0 ];
mov.u32 %r264, 0x0;
mov.u32 %r265, 0x0;
mov.u32 %r266, 0x0;
mov.u32 %r267, 0x0;
ld.global.cg.v4.b32 { %r264, %r265, %r266, %r267 }, [ %rd85 + 0 ];
mov.u32 %r268, 0x0;
mov.u32 %r269, 0x0;
mov.u32 %r270, 0x0;
mov.u32 %r271, 0x0;
ld.global.cg.v4.b32 { %r268, %r269, %r270, %r271 }, [ %rd86 + 0 ];
mov.u32 %r272, 0x0;
mov.u32 %r273, 0x0;
mov.u32 %r274, 0x0;
mov.u32 %r275, 0x0;
ld.global.cg.v4.b32 { %r272, %r273, %r274, %r275 }, [ %rd87 + 0 ];
mov.u32 %r276, 0x0;
mov.u32 %r277, 0x0;
mov.u32 %r278, 0x0;
mov.u32 %r279, 0x0;
ld.global.cg.v4.b32 { %r276, %r277, %r278, %r279 }, [ %rd88 + 0 ];
mov.u32 %r280, 0x0;
mov.u32 %r281, 0x0;
mov.u32 %r282, 0x0;
mov.u32 %r283, 0x0;
ld.global.cg.v4.b32 { %r280, %r281, %r282, %r283 }, [ %rd89 + 0 ];
shl.b32 %r306, %r5, 8;
and.b32 %r307, %r306, 30720;
or.b32 %r308, %r307, %r47;
xor.b32 %r309, %r308, %r284;
add.s32 %r310, %r104, %r309;
st.shared.v4.b32 [%r310], {%r252, %r253, %r254, %r255};
st.shared.v4.b32 [%r310+256], {%r256, %r257, %r258, %r259};
st.shared.v4.b32 [%r310+512], {%r260, %r261, %r262, %r263};
st.shared.v4.b32 [%r310+768], {%r264, %r265, %r266, %r267};
st.shared.v4.b32 [%r310+1024], {%r268, %r269, %r270, %r271};
st.shared.v4.b32 [%r310+1280], {%r272, %r273, %r274, %r275};
st.shared.v4.b32 [%r310+1536], {%r276, %r277, %r278, %r279};
st.shared.v4.b32 [%r310+1792], {%r280, %r281, %r282, %r283};
bar.sync 0;
shl.b32 %r311, %r46, 12;
shl.b32 %r312, %r59, 4;
and.b32 %r60, %r5, 224;
shl.b32 %r313, %r60, 3;
or.b32 %r314, %r312, %r313;
xor.b32 %r315, %r314, %r637;
or.b32 %r316, %r315, %r311;
add.s32 %r317, %r104, %r316;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r318, %r319, %r320, %r321}, [%r317];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r322, %r323, %r324, %r325}, [%r317+2048];
xor.b32 %r326, %r316, 32;
add.s32 %r327, %r104, %r326;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r328, %r329, %r330, %r331}, [%r327];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r332, %r333, %r334, %r335}, [%r327+2048];
xor.b32 %r336, %r316, 64;
add.s32 %r337, %r104, %r336;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r338, %r339, %r340, %r341}, [%r337];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r342, %r343, %r344, %r345}, [%r337+2048];
xor.b32 %r346, %r316, 96;
add.s32 %r347, %r104, %r346;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r348, %r349, %r350, %r351}, [%r347];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r352, %r353, %r354, %r355}, [%r347+2048];
mov.b32 {%rs1, %rs2}, %r318;
cvt.f32.bf16 %r356, %rs1;
cvt.f32.bf16 %r357, %rs2;
mul.f32 %r358, %r95, %r357;
mul.f32 %r359, %r95, %r356;
fma.rn.f32 %r360, %r94, %r638, %r359;
fma.rn.f32 %r361, %r94, %r639, %r358;
mov.b32 {%rs3, %rs4}, %r319;
cvt.f32.bf16 %r362, %rs3;
cvt.f32.bf16 %r363, %rs4;
mul.f32 %r364, %r95, %r363;
mul.f32 %r365, %r95, %r362;
fma.rn.f32 %r366, %r94, %r640, %r365;
fma.rn.f32 %r367, %r94, %r641, %r364;
mov.b32 {%rs5, %rs6}, %r320;
cvt.f32.bf16 %r368, %rs5;
cvt.f32.bf16 %r369, %rs6;
mul.f32 %r370, %r95, %r369;
mul.f32 %r371, %r95, %r368;
fma.rn.f32 %r372, %r94, %r642, %r371;
fma.rn.f32 %r373, %r94, %r643, %r370;
mov.b32 {%rs7, %rs8}, %r321;
cvt.f32.bf16 %r374, %rs7;
cvt.f32.bf16 %r375, %rs8;
mul.f32 %r376, %r95, %r375;
mul.f32 %r377, %r95, %r374;
fma.rn.f32 %r378, %r94, %r644, %r377;
fma.rn.f32 %r379, %r94, %r645, %r376;
mov.b32 {%rs9, %rs10}, %r328;
cvt.f32.bf16 %r380, %rs9;
cvt.f32.bf16 %r381, %rs10;
mul.f32 %r382, %r95, %r381;
mul.f32 %r383, %r95, %r380;
fma.rn.f32 %r384, %r94, %r646, %r383;
fma.rn.f32 %r385, %r94, %r647, %r382;
mov.b32 {%rs11, %rs12}, %r329;
cvt.f32.bf16 %r386, %rs11;
cvt.f32.bf16 %r387, %rs12;
mul.f32 %r388, %r95, %r387;
mul.f32 %r389, %r95, %r386;
fma.rn.f32 %r390, %r94, %r648, %r389;
fma.rn.f32 %r391, %r94, %r649, %r388;
mov.b32 {%rs13, %rs14}, %r330;
cvt.f32.bf16 %r392, %rs13;
cvt.f32.bf16 %r393, %rs14;
mul.f32 %r394, %r95, %r393;
mul.f32 %r395, %r95, %r392;
fma.rn.f32 %r396, %r94, %r650, %r395;
fma.rn.f32 %r397, %r94, %r651, %r394;
mov.b32 {%rs15, %rs16}, %r331;
cvt.f32.bf16 %r398, %rs15;
cvt.f32.bf16 %r399, %rs16;
mul.f32 %r400, %r95, %r399;
mul.f32 %r401, %r95, %r398;
fma.rn.f32 %r402, %r94, %r652, %r401;
fma.rn.f32 %r403, %r94, %r653, %r400;
mov.b32 {%rs17, %rs18}, %r338;
cvt.f32.bf16 %r404, %rs17;
cvt.f32.bf16 %r405, %rs18;
mul.f32 %r406, %r95, %r405;
mul.f32 %r407, %r95, %r404;
fma.rn.f32 %r408, %r94, %r654, %r407;
fma.rn.f32 %r409, %r94, %r655, %r406;
mov.b32 {%rs19, %rs20}, %r339;
cvt.f32.bf16 %r410, %rs19;
cvt.f32.bf16 %r411, %rs20;
mul.f32 %r412, %r95, %r411;
mul.f32 %r413, %r95, %r410;
fma.rn.f32 %r414, %r94, %r656, %r413;
fma.rn.f32 %r415, %r94, %r657, %r412;
mov.b32 {%rs21, %rs22}, %r340;
cvt.f32.bf16 %r416, %rs21;
cvt.f32.bf16 %r417, %rs22;
mul.f32 %r418, %r95, %r417;
mul.f32 %r419, %r95, %r416;
fma.rn.f32 %r420, %r94, %r658, %r419;
fma.rn.f32 %r421, %r94, %r659, %r418;
mov.b32 {%rs23, %rs24}, %r341;
cvt.f32.bf16 %r422, %rs23;
cvt.f32.bf16 %r423, %rs24;
mul.f32 %r424, %r95, %r423;
mul.f32 %r425, %r95, %r422;
fma.rn.f32 %r426, %r94, %r660, %r425;
fma.rn.f32 %r427, %r94, %r661, %r424;
mov.b32 {%rs25, %rs26}, %r348;
cvt.f32.bf16 %r428, %rs25;
cvt.f32.bf16 %r429, %rs26;
mul.f32 %r430, %r95, %r429;
mul.f32 %r431, %r95, %r428;
fma.rn.f32 %r432, %r94, %r662, %r431;
fma.rn.f32 %r433, %r94, %r663, %r430;
mov.b32 {%rs27, %rs28}, %r349;
cvt.f32.bf16 %r434, %rs27;
cvt.f32.bf16 %r435, %rs28;
mul.f32 %r436, %r95, %r435;
mul.f32 %r437, %r95, %r434;
fma.rn.f32 %r438, %r94, %r664, %r437;
fma.rn.f32 %r439, %r94, %r665, %r436;
mov.b32 {%rs29, %rs30}, %r350;
cvt.f32.bf16 %r440, %rs29;
cvt.f32.bf16 %r441, %rs30;
mul.f32 %r442, %r95, %r441;
mul.f32 %r443, %r95, %r440;
fma.rn.f32 %r444, %r94, %r666, %r443;
fma.rn.f32 %r445, %r94, %r667, %r442;
mov.b32 {%rs31, %rs32}, %r351;
cvt.f32.bf16 %r446, %rs31;
cvt.f32.bf16 %r447, %rs32;
mul.f32 %r448, %r95, %r447;
mul.f32 %r449, %r95, %r446;
fma.rn.f32 %r450, %r94, %r668, %r449;
fma.rn.f32 %r451, %r94, %r669, %r448;
mov.b32 {%rs33, %rs34}, %r322;
cvt.f32.bf16 %r452, %rs33;
cvt.f32.bf16 %r453, %rs34;
mul.f32 %r454, %r95, %r453;
mul.f32 %r455, %r95, %r452;
fma.rn.f32 %r456, %r94, %r670, %r455;
fma.rn.f32 %r457, %r94, %r671, %r454;
mov.b32 {%rs35, %rs36}, %r323;
cvt.f32.bf16 %r458, %rs35;
cvt.f32.bf16 %r459, %rs36;
mul.f32 %r460, %r95, %r459;
mul.f32 %r461, %r95, %r458;
fma.rn.f32 %r462, %r94, %r672, %r461;
fma.rn.f32 %r463, %r94, %r673, %r460;
mov.b32 {%rs37, %rs38}, %r324;
cvt.f32.bf16 %r464, %rs37;
cvt.f32.bf16 %r465, %rs38;
mul.f32 %r466, %r95, %r465;
mul.f32 %r467, %r95, %r464;
fma.rn.f32 %r468, %r94, %r674, %r467;
fma.rn.f32 %r469, %r94, %r675, %r466;
mov.b32 {%rs39, %rs40}, %r325;
cvt.f32.bf16 %r470, %rs39;
cvt.f32.bf16 %r471, %rs40;
mul.f32 %r472, %r95, %r471;
mul.f32 %r473, %r95, %r470;
fma.rn.f32 %r474, %r94, %r676, %r473;
fma.rn.f32 %r475, %r94, %r677, %r472;
mov.b32 {%rs41, %rs42}, %r332;
cvt.f32.bf16 %r476, %rs41;
cvt.f32.bf16 %r477, %rs42;
mul.f32 %r478, %r95, %r477;
mul.f32 %r479, %r95, %r476;
fma.rn.f32 %r480, %r94, %r678, %r479;
fma.rn.f32 %r481, %r94, %r679, %r478;
mov.b32 {%rs43, %rs44}, %r333;
cvt.f32.bf16 %r482, %rs43;
cvt.f32.bf16 %r483, %rs44;
mul.f32 %r484, %r95, %r483;
mul.f32 %r485, %r95, %r482;
fma.rn.f32 %r486, %r94, %r680, %r485;
fma.rn.f32 %r487, %r94, %r681, %r484;
mov.b32 {%rs45, %rs46}, %r334;
cvt.f32.bf16 %r488, %rs45;
cvt.f32.bf16 %r489, %rs46;
mul.f32 %r490, %r95, %r489;
mul.f32 %r491, %r95, %r488;
fma.rn.f32 %r492, %r94, %r682, %r491;
fma.rn.f32 %r493, %r94, %r683, %r490;
mov.b32 {%rs47, %rs48}, %r335;
cvt.f32.bf16 %r494, %rs47;
cvt.f32.bf16 %r495, %rs48;
mul.f32 %r496, %r95, %r495;
mul.f32 %r497, %r95, %r494;
fma.rn.f32 %r498, %r94, %r684, %r497;
fma.rn.f32 %r499, %r94, %r685, %r496;
mov.b32 {%rs49, %rs50}, %r342;
cvt.f32.bf16 %r500, %rs49;
cvt.f32.bf16 %r501, %rs50;
mul.f32 %r502, %r95, %r501;
mul.f32 %r503, %r95, %r500;
fma.rn.f32 %r504, %r94, %r686, %r503;
fma.rn.f32 %r505, %r94, %r687, %r502;
mov.b32 {%rs51, %rs52}, %r343;
cvt.f32.bf16 %r506, %rs51;
cvt.f32.bf16 %r507, %rs52;
mul.f32 %r508, %r95, %r507;
mul.f32 %r509, %r95, %r506;
fma.rn.f32 %r510, %r94, %r688, %r509;
fma.rn.f32 %r511, %r94, %r689, %r508;
mov.b32 {%rs53, %rs54}, %r344;
cvt.f32.bf16 %r512, %rs53;
cvt.f32.bf16 %r513, %rs54;
mul.f32 %r514, %r95, %r513;
mul.f32 %r515, %r95, %r512;
fma.rn.f32 %r516, %r94, %r690, %r515;
fma.rn.f32 %r517, %r94, %r691, %r514;
mov.b32 {%rs55, %rs56}, %r345;
cvt.f32.bf16 %r518, %rs55;
cvt.f32.bf16 %r519, %rs56;
mul.f32 %r520, %r95, %r519;
mul.f32 %r521, %r95, %r518;
fma.rn.f32 %r522, %r94, %r692, %r521;
fma.rn.f32 %r523, %r94, %r693, %r520;
mov.b32 {%rs57, %rs58}, %r352;
cvt.f32.bf16 %r524, %rs57;
cvt.f32.bf16 %r525, %rs58;
mul.f32 %r526, %r95, %r525;
mul.f32 %r527, %r95, %r524;
fma.rn.f32 %r528, %r94, %r694, %r527;
fma.rn.f32 %r529, %r94, %r695, %r526;
mov.b32 {%rs59, %rs60}, %r353;
cvt.f32.bf16 %r530, %rs59;
cvt.f32.bf16 %r531, %rs60;
mul.f32 %r532, %r95, %r531;
mul.f32 %r533, %r95, %r530;
fma.rn.f32 %r534, %r94, %r696, %r533;
fma.rn.f32 %r535, %r94, %r697, %r532;
mov.b32 {%rs61, %rs62}, %r354;
cvt.f32.bf16 %r536, %rs61;
cvt.f32.bf16 %r537, %rs62;
mul.f32 %r538, %r95, %r537;
mul.f32 %r539, %r95, %r536;
fma.rn.f32 %r540, %r94, %r698, %r539;
fma.rn.f32 %r541, %r94, %r699, %r538;
mov.b32 {%rs63, %rs64}, %r355;
cvt.f32.bf16 %r542, %rs63;
cvt.f32.bf16 %r543, %rs64;
mul.f32 %r544, %r95, %r543;
mul.f32 %r545, %r95, %r542;
fma.rn.f32 %r546, %r94, %r700, %r545;
fma.rn.f32 %r547, %r94, %r701, %r544;
cvt.rn.bf16x2.f32 %r61, %r361, %r360;
cvt.rn.bf16x2.f32 %r62, %r367, %r366;
cvt.rn.bf16x2.f32 %r63, %r373, %r372;
cvt.rn.bf16x2.f32 %r64, %r379, %r378;
cvt.rn.bf16x2.f32 %r65, %r385, %r384;
cvt.rn.bf16x2.f32 %r66, %r391, %r390;
cvt.rn.bf16x2.f32 %r67, %r397, %r396;
cvt.rn.bf16x2.f32 %r68, %r403, %r402;
cvt.rn.bf16x2.f32 %r69, %r409, %r408;
cvt.rn.bf16x2.f32 %r70, %r415, %r414;
cvt.rn.bf16x2.f32 %r71, %r421, %r420;
cvt.rn.bf16x2.f32 %r72, %r427, %r426;
cvt.rn.bf16x2.f32 %r73, %r433, %r432;
cvt.rn.bf16x2.f32 %r74, %r439, %r438;
cvt.rn.bf16x2.f32 %r75, %r445, %r444;
cvt.rn.bf16x2.f32 %r76, %r451, %r450;
cvt.rn.bf16x2.f32 %r77, %r457, %r456;
cvt.rn.bf16x2.f32 %r78, %r463, %r462;
cvt.rn.bf16x2.f32 %r79, %r469, %r468;
cvt.rn.bf16x2.f32 %r80, %r475, %r474;
cvt.rn.bf16x2.f32 %r81, %r481, %r480;
cvt.rn.bf16x2.f32 %r82, %r487, %r486;
cvt.rn.bf16x2.f32 %r83, %r493, %r492;
cvt.rn.bf16x2.f32 %r84, %r499, %r498;
cvt.rn.bf16x2.f32 %r85, %r505, %r504;
cvt.rn.bf16x2.f32 %r86, %r511, %r510;
cvt.rn.bf16x2.f32 %r87, %r517, %r516;
cvt.rn.bf16x2.f32 %r88, %r523, %r522;
cvt.rn.bf16x2.f32 %r89, %r529, %r528;
cvt.rn.bf16x2.f32 %r90, %r535, %r534;
cvt.rn.bf16x2.f32 %r91, %r541, %r540;
cvt.rn.bf16x2.f32 %r92, %r547, %r546;
@%p28 bra $L__BB0_18;
bra.uni $L__BB0_19;
$L__BB0_18:
bar.sync 0;
shl.b32 %r548, %r59, 7;
shl.b32 %r549, %r60, 6;
or.b32 %r550, %r548, %r47;
xor.b32 %r551, %r550, %r637;
or.b32 %r552, %r551, %r549;
add.s32 %r553, %r104, %r552;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r553], {%r61, %r62, %r63, %r64};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r553+16384], {%r77, %r78, %r79, %r80};
xor.b32 %r554, %r552, 32;
add.s32 %r555, %r104, %r554;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r555], {%r65, %r66, %r67, %r68};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r555+16384], {%r81, %r82, %r83, %r84};
xor.b32 %r556, %r552, 64;
add.s32 %r557, %r104, %r556;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r557], {%r69, %r70, %r71, %r72};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r557+16384], {%r85, %r86, %r87, %r88};
xor.b32 %r558, %r552, 96;
add.s32 %r559, %r104, %r558;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r559], {%r73, %r74, %r75, %r76};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r559+16384], {%r89, %r90, %r91, %r92};
$L__BB0_19:
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r48], {%r61, %r62, %r63, %r64};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r49], {%r65, %r66, %r67, %r68};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r50], {%r69, %r70, %r71, %r72};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r51], {%r73, %r74, %r75, %r76};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r52], {%r77, %r78, %r79, %r80};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r53], {%r81, %r82, %r83, %r84};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r85, %r86, %r87, %r88};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r55], {%r89, %r90, %r91, %r92};
fence.proxy.async.shared::cta;
@%p4 bra $L__BB0_21;
bar.sync 0;
elect.sync %r563|%p53, -1;
and.pred %p52, %p56, %p53;
or.b32 %r560, %r229, %r4;
@%p52 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd91, {%r560, %r561}], [%r562];
cp.async.bulk.commit_group;
$L__BB0_21:
bar.sync 0;
elect.sync %r568|%p55, -1;
and.pred %p54, %p56, %p55;
@%p54 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd91, {%r564, %r565}], [%r566];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r567, 0;
$L__tmp37:
ret;
$L__tmp38:
$L__func_end0:
}
"#
  | .vo => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_round(
.param .align 64 .b8 cooperative_round_param_0[128],
.param .u32 cooperative_round_param_1,
.param .u32 cooperative_round_param_2,
.param .u64 cooperative_round_param_3,
.param .u64 cooperative_round_param_4,
.param .align 64 .b8 cooperative_round_param_5[128],
.param .u32 cooperative_round_param_6,
.param .u32 cooperative_round_param_7,
.param .u64 cooperative_round_param_8,
.param .u64 cooperative_round_param_9,
.param .align 64 .b8 cooperative_round_param_10[128],
.param .u32 cooperative_round_param_11,
.param .u32 cooperative_round_param_12,
.param .u64 cooperative_round_param_13,
.param .u64 cooperative_round_param_14,
.param .u64 .ptr .global .align 1 cooperative_round_param_15,
.param .u64 .ptr .global .align 1 cooperative_round_param_16,
.param .u32 cooperative_round_param_17,
.param .f32 cooperative_round_param_18,
.param .f32 cooperative_round_param_19,
.param .u64 .ptr .global .align 1 cooperative_round_param_20,
.param .u64 .ptr .global .align 1 cooperative_round_param_21
)
.reqntid 256
{
.reg .pred %p<59>;
.reg .b16 %rs<65>;
.reg .b32 %r<719>;
.reg .b64 %rd<96>;
$L__func_begin0:
mov.b64 %rd3, cooperative_round_param_0;
mov.b64 %rd4, cooperative_round_param_10;
$L__tmp0:
cvta.param.u64 %rd95, %rd4;
mov.b64 %rd5, cooperative_round_param_5;
cvta.param.u64 %rd67, %rd5;
cvta.param.u64 %rd28, %rd3;
mov.u32 %r99, %ctaid.x;
mul.hi.u32 %r100, %r99, 818089009;
shr.u32 %r1, %r100, 2;
$L__tmp1:
mul.lo.s32 %r101, %r1, 21;
sub.s32 %r2, %r99, %r101;
mul.lo.s32 %r102, %r2, 3;
mov.b64 %rd6, 6569755928097638024;
shr.u64 %rd7, %rd6, %r102;
cvt.u32.u64 %r103, %rd7;
shl.b32 %r104, %r103, 7;
and.b32 %r3, %r104, 896;
mov.b64 %rd8, 6420841298313609216;
shr.u64 %rd9, %rd8, %r102;
cvt.u32.u64 %r105, %rd9;
shl.b32 %r106, %r105, 7;
and.b32 %r4, %r106, 896;
$L__tmp2:
setp.eq.b32 %p2, %r3, %r4;
mov.u32 %r5, %tid.x;
and.b32 %r6, %r5, 255;
setp.eq.b32 %p1, %r6, 0;
mov.b32 %r107, global_smem;
add.s32 %r245, %r107, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r245], 1;
bar.sync 0;
add.s32 %r98, %r107, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r98], 1;
mul.lo.s32 %r7, %r1, 768;
$L__tmp3:
add.s32 %r574, %r3, %r7;
shr.u32 %r8, %r5, 5;
setp.lt.u32 %p58, %r6, 64;
add.s32 %r578, %r4, %r7;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r245], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r138, %r8, 0, 31, -1;
elect.sync %r139|%p18, -1;
and.pred %p17, %p58, %p18;
and.b32 %r140, %r138, 1;
shl.b32 %r141, %r140, 14;
add.s32 %r136, %r107, %r141;
shl.b32 %r137, %r140, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r136], [%rd28, {%r137, %r574}], [%r245];
shl.b32 %r142, %r138, 7;
and.b32 %r11, %r142, 512;
$L__tmp6:
or.b32 %r648, %r137, 128;
mov.b32 %r649, 0;
mov.b32 %r583, 0f00000000;
mov.b32 %r584, %r583;
mov.b32 %r585, %r583;
mov.b32 %r586, %r583;
mov.b32 %r587, %r583;
mov.b32 %r588, %r583;
mov.b32 %r589, %r583;
mov.b32 %r590, %r583;
mov.b32 %r591, %r583;
mov.b32 %r592, %r583;
mov.b32 %r593, %r583;
mov.b32 %r594, %r583;
mov.b32 %r595, %r583;
mov.b32 %r596, %r583;
mov.b32 %r597, %r583;
mov.b32 %r598, %r583;
mov.b32 %r599, %r583;
mov.b32 %r600, %r583;
mov.b32 %r601, %r583;
mov.b32 %r602, %r583;
mov.b32 %r603, %r583;
mov.b32 %r604, %r583;
mov.b32 %r605, %r583;
mov.b32 %r606, %r583;
mov.b32 %r607, %r583;
mov.b32 %r608, %r583;
mov.b32 %r609, %r583;
mov.b32 %r610, %r583;
mov.b32 %r611, %r583;
mov.b32 %r612, %r583;
mov.b32 %r613, %r583;
mov.b32 %r614, %r583;
mov.b32 %r615, %r583;
mov.b32 %r616, %r583;
mov.b32 %r617, %r583;
mov.b32 %r618, %r583;
mov.b32 %r619, %r583;
mov.b32 %r620, %r583;
mov.b32 %r621, %r583;
mov.b32 %r622, %r583;
mov.b32 %r623, %r583;
mov.b32 %r624, %r583;
mov.b32 %r625, %r583;
mov.b32 %r626, %r583;
mov.b32 %r627, %r583;
mov.b32 %r628, %r583;
mov.b32 %r629, %r583;
mov.b32 %r630, %r583;
mov.b32 %r631, %r583;
mov.b32 %r632, %r583;
mov.b32 %r633, %r583;
mov.b32 %r634, %r583;
mov.b32 %r635, %r583;
mov.b32 %r636, %r583;
mov.b32 %r637, %r583;
mov.b32 %r638, %r583;
mov.b32 %r639, %r583;
mov.b32 %r640, %r583;
mov.b32 %r641, %r583;
mov.b32 %r642, %r583;
mov.b32 %r643, %r583;
mov.b32 %r644, %r583;
mov.b32 %r645, %r583;
mov.b32 %r646, %r583;
$L__BB0_4:
and.b32 %r147, %r649, 1;
bfe.u32 %r144, %r649, 1, 1;
shl.b32 %r148, %r147, 15;
add.s32 %r149, %r107, %r148;
mad.lo.s32 %r150, %r147, -32760, %r149;
add.s32 %r143, %r150, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r143], %r144;
@!complete bra.uni waitLoop;
}
shr.u32 %r151, %r149, 4;
add.s32 %r152, %r151, %r11;
and.b32 %r153, %r152, 16383;
cvt.u64.u32 %rd45, %r153;
bfe.u32 %r154, %r149, 4, 14;
cvt.u64.u32 %rd46, %r154;
wgmma.fence.sync.aligned;
or.b64 %rd29, %rd45, 4611686293305294848;
or.b64 %rd30, %rd46, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd29, %rd30, %p19, 1, 1, 0, 0;
add.s64 %rd31, %rd45, 4611686293305294850;
add.s64 %rd32, %rd46, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd31, %rd32, %p19, 1, 1, 0, 0;
add.s64 %rd33, %rd45, 4611686293305294852;
add.s64 %rd34, %rd46, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd33, %rd34, %p19, 1, 1, 0, 0;
add.s64 %rd35, %rd45, 4611686293305294854;
add.s64 %rd36, %rd46, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd35, %rd36, %p19, 1, 1, 0, 0;
add.s64 %rd37, %rd45, 4611686293305295872;
add.s64 %rd38, %rd46, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd37, %rd38, %p19, 1, 1, 0, 0;
add.s64 %rd39, %rd45, 4611686293305295874;
add.s64 %rd40, %rd46, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd39, %rd40, %p19, 1, 1, 0, 0;
add.s64 %rd41, %rd45, 4611686293305295876;
add.s64 %rd42, %rd46, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd41, %rd42, %p19, 1, 1, 0, 0;
add.s64 %rd43, %rd45, 4611686293305295878;
add.s64 %rd44, %rd46, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd43, %rd44, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r12, %r649, 1;
and.b32 %r155, %r12, 1;
setp.ne.b32 %p22, %r649, 5;
shl.b32 %r156, %r155, 3;
add.s32 %r157, %r107, %r156;
add.s32 %r145, %r157, 98304;
shl.b32 %r158, %r155, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r145], 32768;
bar.sync 0;
elect.sync %r159|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p58, %p24;
add.s32 %r146, %r136, %r158;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r146], [%rd28, {%r648, %r574}], [%r145];
$L__tmp8:
add.s32 %r648, %r648, 128;
setp.ne.b32 %p25, %r12, 6;
mov.b32 %r649, %r12;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r245], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r110, %r8, 0, 31, -1;
elect.sync %r111|%p5, -1;
and.pred %p3, %p58, %p5;
and.b32 %r112, %r110, 1;
shl.b32 %r113, %r112, 14;
add.s32 %r108, %r107, %r113;
shl.b32 %r109, %r112, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r108], [%rd28, {%r109, %r574}], [%r245];
$L__tmp10:
bar.sync 0;
elect.sync %r114|%p6, -1;
and.pred %p4, %p58, %p6;
add.s32 %r122, %r108, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r122], [%rd28, {%r109, %r578}], [%r245];
shl.b32 %r115, %r110, 7;
and.b32 %r9, %r115, 512;
add.s32 %r116, %r107, 65536;
bfe.u32 %r117, %r116, 4, 14;
cvt.u64.u32 %rd10, %r117;
or.b64 %rd12, %rd10, 4611686293305294848;
add.s64 %rd14, %rd10, 4611686293305294850;
add.s64 %rd16, %rd10, 4611686293305294852;
add.s64 %rd18, %rd10, 4611686293305294854;
add.s64 %rd20, %rd10, 4611686293305295872;
add.s64 %rd22, %rd10, 4611686293305295874;
add.s64 %rd24, %rd10, 4611686293305295876;
add.s64 %rd26, %rd10, 4611686293305295878;
$L__tmp11:
or.b32 %r582, %r109, 128;
mov.b32 %r647, 0;
mov.b32 %r583, 0f00000000;
mov.b32 %r584, %r583;
mov.b32 %r585, %r583;
mov.b32 %r586, %r583;
mov.b32 %r587, %r583;
mov.b32 %r588, %r583;
mov.b32 %r589, %r583;
mov.b32 %r590, %r583;
mov.b32 %r591, %r583;
mov.b32 %r592, %r583;
mov.b32 %r593, %r583;
mov.b32 %r594, %r583;
mov.b32 %r595, %r583;
mov.b32 %r596, %r583;
mov.b32 %r597, %r583;
mov.b32 %r598, %r583;
mov.b32 %r599, %r583;
mov.b32 %r600, %r583;
mov.b32 %r601, %r583;
mov.b32 %r602, %r583;
mov.b32 %r603, %r583;
mov.b32 %r604, %r583;
mov.b32 %r605, %r583;
mov.b32 %r606, %r583;
mov.b32 %r607, %r583;
mov.b32 %r608, %r583;
mov.b32 %r609, %r583;
mov.b32 %r610, %r583;
mov.b32 %r611, %r583;
mov.b32 %r612, %r583;
mov.b32 %r613, %r583;
mov.b32 %r614, %r583;
mov.b32 %r615, %r583;
mov.b32 %r616, %r583;
mov.b32 %r617, %r583;
mov.b32 %r618, %r583;
mov.b32 %r619, %r583;
mov.b32 %r620, %r583;
mov.b32 %r621, %r583;
mov.b32 %r622, %r583;
mov.b32 %r623, %r583;
mov.b32 %r624, %r583;
mov.b32 %r625, %r583;
mov.b32 %r626, %r583;
mov.b32 %r627, %r583;
mov.b32 %r628, %r583;
mov.b32 %r629, %r583;
mov.b32 %r630, %r583;
mov.b32 %r631, %r583;
mov.b32 %r632, %r583;
mov.b32 %r633, %r583;
mov.b32 %r634, %r583;
mov.b32 %r635, %r583;
mov.b32 %r636, %r583;
mov.b32 %r637, %r583;
mov.b32 %r638, %r583;
mov.b32 %r639, %r583;
mov.b32 %r640, %r583;
mov.b32 %r641, %r583;
mov.b32 %r642, %r583;
mov.b32 %r643, %r583;
mov.b32 %r644, %r583;
mov.b32 %r645, %r583;
mov.b32 %r646, %r583;
$L__BB0_2:
and.b32 %r123, %r647, 1;
bfe.u32 %r119, %r647, 1, 1;
shl.b32 %r124, %r123, 15;
add.s32 %r125, %r107, %r124;
mad.lo.s32 %r126, %r123, -32760, %r125;
add.s32 %r118, %r126, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r118], %r119;
@!complete bra.uni waitLoop;
}
shr.u32 %r127, %r125, 4;
add.s32 %r128, %r127, %r9;
and.b32 %r129, %r128, 16383;
cvt.u64.u32 %rd27, %r129;
wgmma.fence.sync.aligned;
or.b64 %rd11, %rd27, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd11, %rd12, %p7, 1, 1, 0, 0;
add.s64 %rd13, %rd27, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd13, %rd14, %p7, 1, 1, 0, 0;
add.s64 %rd15, %rd27, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd15, %rd16, %p7, 1, 1, 0, 0;
add.s64 %rd17, %rd27, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd17, %rd18, %p7, 1, 1, 0, 0;
add.s64 %rd19, %rd27, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd19, %rd20, %p7, 1, 1, 0, 0;
add.s64 %rd21, %rd27, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd21, %rd22, %p7, 1, 1, 0, 0;
add.s64 %rd23, %rd27, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd23, %rd24, %p7, 1, 1, 0, 0;
add.s64 %rd25, %rd27, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r583,%r584,%r585,%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646}, %rd25, %rd26, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r10, %r647, 1;
and.b32 %r130, %r10, 1;
setp.ne.b32 %p11, %r647, 5;
shl.b32 %r131, %r130, 3;
add.s32 %r132, %r107, %r131;
add.s32 %r120, %r132, 98304;
shl.b32 %r133, %r130, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r120], 65536;
bar.sync 0;
elect.sync %r134|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p58, %p13;
add.s32 %r121, %r108, %r133;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r121], [%rd28, {%r582, %r574}], [%r120];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r135|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p58, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r122], [%rd28, {%r582, %r578}], [%r120];
$L__tmp15:
add.s32 %r582, %r582, 128;
setp.eq.b32 %p16, %r10, 6;
mov.b32 %r647, %r10;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r3, %r4;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r245];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r98];
cvt.rn.bf16x2.f32 %r13, %r584, %r583;
cvt.rn.bf16x2.f32 %r14, %r586, %r585;
cvt.rn.bf16x2.f32 %r15, %r588, %r587;
cvt.rn.bf16x2.f32 %r16, %r590, %r589;
cvt.rn.bf16x2.f32 %r17, %r592, %r591;
cvt.rn.bf16x2.f32 %r18, %r594, %r593;
cvt.rn.bf16x2.f32 %r19, %r596, %r595;
cvt.rn.bf16x2.f32 %r20, %r598, %r597;
cvt.rn.bf16x2.f32 %r21, %r600, %r599;
cvt.rn.bf16x2.f32 %r22, %r602, %r601;
cvt.rn.bf16x2.f32 %r23, %r604, %r603;
cvt.rn.bf16x2.f32 %r24, %r606, %r605;
cvt.rn.bf16x2.f32 %r25, %r608, %r607;
cvt.rn.bf16x2.f32 %r26, %r610, %r609;
cvt.rn.bf16x2.f32 %r27, %r612, %r611;
cvt.rn.bf16x2.f32 %r28, %r614, %r613;
cvt.rn.bf16x2.f32 %r29, %r616, %r615;
cvt.rn.bf16x2.f32 %r30, %r618, %r617;
cvt.rn.bf16x2.f32 %r31, %r620, %r619;
cvt.rn.bf16x2.f32 %r32, %r622, %r621;
cvt.rn.bf16x2.f32 %r33, %r624, %r623;
cvt.rn.bf16x2.f32 %r34, %r626, %r625;
cvt.rn.bf16x2.f32 %r35, %r628, %r627;
cvt.rn.bf16x2.f32 %r36, %r630, %r629;
cvt.rn.bf16x2.f32 %r37, %r632, %r631;
cvt.rn.bf16x2.f32 %r38, %r634, %r633;
cvt.rn.bf16x2.f32 %r39, %r636, %r635;
cvt.rn.bf16x2.f32 %r40, %r638, %r637;
cvt.rn.bf16x2.f32 %r41, %r640, %r639;
cvt.rn.bf16x2.f32 %r42, %r642, %r641;
cvt.rn.bf16x2.f32 %r43, %r644, %r643;
cvt.rn.bf16x2.f32 %r44, %r646, %r645;
shl.b32 %r45, %r5, 7;
@%p26 bra $L__BB0_8;
and.b32 %r650, %r5, 16;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r160, %r45, 1920;
shl.b32 %r161, %r5, 6;
and.b32 %r162, %r161, 14336;
shl.b32 %r163, %r5, 4;
and.b32 %r164, %r163, 112;
and.b32 %r650, %r5, 16;
or.b32 %r165, %r160, %r164;
xor.b32 %r166, %r165, %r650;
or.b32 %r167, %r166, %r162;
add.s32 %r168, %r107, %r167;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r168], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r168+16384], {%r29, %r30, %r31, %r32};
xor.b32 %r169, %r167, 32;
add.s32 %r170, %r107, %r169;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r170], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r170+16384], {%r33, %r34, %r35, %r36};
xor.b32 %r171, %r167, 64;
add.s32 %r172, %r107, %r171;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r172], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r172+16384], {%r37, %r38, %r39, %r40};
xor.b32 %r173, %r167, 96;
add.s32 %r174, %r107, %r173;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r174], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r174+16384], {%r41, %r42, %r43, %r44};
$L__BB0_9:
ld.param.b32 %r95, [cooperative_round_param_17];
ld.param.b64 %rd2, [cooperative_round_param_16];
and.b32 %r175, %r45, 17280;
and.b32 %r46, %r5, 7;
shl.b32 %r47, %r46, 4;
and.b32 %r176, %r5, 96;
shl.b32 %r177, %r5, 1;
and.b32 %r178, %r177, 16;
shl.b32 %r179, %r650, 6;
or.b32 %r180, %r175, %r47;
or.b32 %r181, %r178, %r176;
xor.b32 %r182, %r180, %r181;
add.s32 %r183, %r107, %r179;
add.s32 %r184, %r183, %r182;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+32768], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+34816], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+36864], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+38912], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+40960], {%r29, %r30, %r31, %r32};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+43008], {%r33, %r34, %r35, %r36};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+45056], {%r37, %r38, %r39, %r40};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r184+47104], {%r41, %r42, %r43, %r44};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_11;
bar.sync 0;
elect.sync %r187|%p28, -1;
shfl.sync.idx.b32 %r188, %r8, 0, 31, -1;
and.pred %p27, %p58, %p28;
and.b32 %r189, %r188, 1;
shl.b32 %r190, %r189, 14;
add.s32 %r186, %r107, %r190;
shl.b32 %r191, %r189, 6;
or.b32 %r185, %r191, %r4;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd67, {%r185, %r574}], [%r186];
cp.async.bulk.commit_group;
$L__BB0_11:
ld.param.b32 %r97, [cooperative_round_param_19];
ld.param.b32 %r96, [cooperative_round_param_18];
ld.param.b64 %rd1, [cooperative_round_param_15];
bar.sync 0;
elect.sync %r206|%p30, -1;
shfl.sync.idx.b32 %r56, %r8, 0, 31, -1;
and.pred %p29, %p58, %p30;
and.b32 %r207, %r56, 1;
shl.b32 %r208, %r207, 14;
add.s32 %r575, %r107, %r208;
add.s32 %r579, %r575, 32768;
shl.b32 %r246, %r207, 6;
or.b32 %r577, %r246, %r3;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd67, {%r577, %r578}], [%r579];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r192, 0;
mov.b32 %r209, 1;
$L__tmp17:
shl.b32 %r204, %r209, %r2;
shr.u32 %r210, %r3, 7;
$L__tmp18:
sub.s32 %r211, 13, %r210;
mul.lo.s32 %r212, %r211, %r210;
shr.u32 %r196, %r212, 1;
sub.s32 %r194, %r196, %r210;
bmsk.clamp.b32 %r193, 0, %r194;
sub.s32 %r197, 6, %r210;
bmsk.clamp.b32 %r195, %r196, %r197;
and.b32 %r213, %r193, 53793;
shl.b32 %r214, %r213, %r210;
$L__tmp19:
shr.u32 %r215, %r4, 7;
$L__tmp20:
sub.s32 %r216, 13, %r215;
mul.lo.s32 %r217, %r216, %r215;
shr.u32 %r201, %r217, 1;
sub.s32 %r199, %r201, %r215;
bmsk.clamp.b32 %r198, 0, %r199;
sub.s32 %r202, 6, %r215;
bmsk.clamp.b32 %r200, %r201, %r202;
and.b32 %r218, %r198, 53793;
shl.b32 %r219, %r218, %r215;
or.b32 %r220, %r214, %r195;
$L__tmp21:
or.b32 %r221, %r220, %r200;
$L__tmp22:
or.b32 %r205, %r221, %r219;
mad.wide.s32 %rd48, %r95, 4, %rd2;
mad.wide.u32 %rd47, %r1, 4, %rd48;
$L__tmp23:
{
.reg .u32 tid, bits, covered;
.reg .pred observer, ready;
bar.sync 0;
mov.u32 tid, %tid.x;
setp.eq.u32 observer, tid, 0;
@!observer bra gram_ready;
fence.proxy.async.global;
atom.acq_rel.gpu.global.or.b32 bits, [%rd47], %r204;
or.b32 bits, bits, %r204;
and.b32 covered, bits, %r205;
setp.eq.u32 ready, covered, %r205;
@ready bra gram_ready;
observe_gram:
ld.acquire.gpu.global.u32 bits, [%rd47];
and.b32 covered, bits, %r205;
setp.eq.u32 ready, covered, %r205;
@!ready bra observe_gram;
gram_ready:
bar.sync 0;
fence.proxy.async.global;
mov.u32 %r203, 0;
}
$L__tmp24:
@%p1 mbarrier.init.shared::cta.b64 [%r245], 1;
bar.sync 0;
@%p1 mbarrier.init.shared::cta.b64 [%r98], 1;
@%p2 bra $L__BB0_14;
bra.uni $L__BB0_12;
$L__BB0_14:
$L__tmp25:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r245], 32768;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r247|%p46, -1;
and.pred %p45, %p58, %p46;
@%p45 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r575], [%rd67, {%r246, %r574}], [%r245];
shl.b32 %r248, %r56, 7;
and.b32 %r59, %r248, 512;
$L__tmp26:
or.b32 %r717, %r246, 128;
mov.b32 %r718, 0;
mov.b32 %r652, 0f00000000;
mov.b32 %r653, %r652;
mov.b32 %r654, %r652;
mov.b32 %r655, %r652;
mov.b32 %r656, %r652;
mov.b32 %r657, %r652;
mov.b32 %r658, %r652;
mov.b32 %r659, %r652;
mov.b32 %r660, %r652;
mov.b32 %r661, %r652;
mov.b32 %r662, %r652;
mov.b32 %r663, %r652;
mov.b32 %r664, %r652;
mov.b32 %r665, %r652;
mov.b32 %r666, %r652;
mov.b32 %r667, %r652;
mov.b32 %r668, %r652;
mov.b32 %r669, %r652;
mov.b32 %r670, %r652;
mov.b32 %r671, %r652;
mov.b32 %r672, %r652;
mov.b32 %r673, %r652;
mov.b32 %r674, %r652;
mov.b32 %r675, %r652;
mov.b32 %r676, %r652;
mov.b32 %r677, %r652;
mov.b32 %r678, %r652;
mov.b32 %r679, %r652;
mov.b32 %r680, %r652;
mov.b32 %r681, %r652;
mov.b32 %r682, %r652;
mov.b32 %r683, %r652;
mov.b32 %r684, %r652;
mov.b32 %r685, %r652;
mov.b32 %r686, %r652;
mov.b32 %r687, %r652;
mov.b32 %r688, %r652;
mov.b32 %r689, %r652;
mov.b32 %r690, %r652;
mov.b32 %r691, %r652;
mov.b32 %r692, %r652;
mov.b32 %r693, %r652;
mov.b32 %r694, %r652;
mov.b32 %r695, %r652;
mov.b32 %r696, %r652;
mov.b32 %r697, %r652;
mov.b32 %r698, %r652;
mov.b32 %r699, %r652;
mov.b32 %r700, %r652;
mov.b32 %r701, %r652;
mov.b32 %r702, %r652;
mov.b32 %r703, %r652;
mov.b32 %r704, %r652;
mov.b32 %r705, %r652;
mov.b32 %r706, %r652;
mov.b32 %r707, %r652;
mov.b32 %r708, %r652;
mov.b32 %r709, %r652;
mov.b32 %r710, %r652;
mov.b32 %r711, %r652;
mov.b32 %r712, %r652;
mov.b32 %r713, %r652;
mov.b32 %r714, %r652;
mov.b32 %r715, %r652;
$L__BB0_15:
and.b32 %r253, %r718, 1;
bfe.u32 %r250, %r718, 1, 1;
shl.b32 %r254, %r253, 15;
add.s32 %r255, %r107, %r254;
mad.lo.s32 %r256, %r253, -32760, %r255;
add.s32 %r249, %r256, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r249], %r250;
@!complete bra.uni waitLoop;
}
shr.u32 %r257, %r255, 4;
add.s32 %r258, %r257, %r59;
and.b32 %r259, %r258, 16383;
cvt.u64.u32 %rd84, %r259;
bfe.u32 %r260, %r255, 4, 14;
cvt.u64.u32 %rd85, %r260;
wgmma.fence.sync.aligned;
or.b64 %rd68, %rd84, 4611686293305294848;
or.b64 %rd69, %rd85, 4611686293305294848;
mov.pred %p47, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd68, %rd69, %p47, 1, 1, 0, 0;
add.s64 %rd70, %rd84, 4611686293305294850;
add.s64 %rd71, %rd85, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd70, %rd71, %p47, 1, 1, 0, 0;
add.s64 %rd72, %rd84, 4611686293305294852;
add.s64 %rd73, %rd85, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd72, %rd73, %p47, 1, 1, 0, 0;
add.s64 %rd74, %rd84, 4611686293305294854;
add.s64 %rd75, %rd85, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd74, %rd75, %p47, 1, 1, 0, 0;
add.s64 %rd76, %rd84, 4611686293305295872;
add.s64 %rd77, %rd85, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd76, %rd77, %p47, 1, 1, 0, 0;
add.s64 %rd78, %rd84, 4611686293305295874;
add.s64 %rd79, %rd85, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd78, %rd79, %p47, 1, 1, 0, 0;
add.s64 %rd80, %rd84, 4611686293305295876;
add.s64 %rd81, %rd85, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd80, %rd81, %p47, 1, 1, 0, 0;
add.s64 %rd82, %rd84, 4611686293305295878;
add.s64 %rd83, %rd85, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd82, %rd83, %p47, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r60, %r718, 1;
and.b32 %r261, %r60, 1;
setp.ne.b32 %p50, %r718, 5;
shl.b32 %r262, %r261, 3;
add.s32 %r263, %r107, %r262;
add.s32 %r251, %r263, 98304;
shl.b32 %r264, %r261, 15;
$L__tmp27:
bar.sync 0;
and.pred %p48, %p1, %p50;
@%p48 mbarrier.arrive.expect_tx.shared.b64 _, [%r251], 32768;
bar.sync 0;
elect.sync %r265|%p51, -1;
and.pred %p52, %p50, %p51;
and.pred %p49, %p58, %p52;
add.s32 %r252, %r575, %r264;
@%p49 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r252], [%rd67, {%r717, %r574}], [%r251];
$L__tmp28:
add.s32 %r717, %r717, 128;
setp.ne.b32 %p53, %r60, 6;
mov.b32 %r718, %r60;
@%p53 bra $L__BB0_15;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_17;
$L__tmp29:
$L__BB0_12:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r245], 65536;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r222|%p33, -1;
and.pred %p31, %p58, %p33;
@%p31 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r575], [%rd67, {%r246, %r574}], [%r245];
$L__tmp30:
bar.sync 0;
elect.sync %r223|%p34, -1;
and.pred %p32, %p58, %p34;
add.s32 %r231, %r575, 65536;
@%p32 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r231], [%rd67, {%r246, %r578}], [%r245];
shl.b32 %r224, %r56, 7;
and.b32 %r57, %r224, 512;
add.s32 %r225, %r107, 65536;
bfe.u32 %r226, %r225, 4, 14;
cvt.u64.u32 %rd49, %r226;
or.b64 %rd51, %rd49, 4611686293305294848;
add.s64 %rd53, %rd49, 4611686293305294850;
add.s64 %rd55, %rd49, 4611686293305294852;
add.s64 %rd57, %rd49, 4611686293305294854;
add.s64 %rd59, %rd49, 4611686293305295872;
add.s64 %rd61, %rd49, 4611686293305295874;
add.s64 %rd63, %rd49, 4611686293305295876;
add.s64 %rd65, %rd49, 4611686293305295878;
$L__tmp31:
or.b32 %r651, %r246, 128;
mov.b32 %r716, 0;
mov.b32 %r652, 0f00000000;
mov.b32 %r653, %r652;
mov.b32 %r654, %r652;
mov.b32 %r655, %r652;
mov.b32 %r656, %r652;
mov.b32 %r657, %r652;
mov.b32 %r658, %r652;
mov.b32 %r659, %r652;
mov.b32 %r660, %r652;
mov.b32 %r661, %r652;
mov.b32 %r662, %r652;
mov.b32 %r663, %r652;
mov.b32 %r664, %r652;
mov.b32 %r665, %r652;
mov.b32 %r666, %r652;
mov.b32 %r667, %r652;
mov.b32 %r668, %r652;
mov.b32 %r669, %r652;
mov.b32 %r670, %r652;
mov.b32 %r671, %r652;
mov.b32 %r672, %r652;
mov.b32 %r673, %r652;
mov.b32 %r674, %r652;
mov.b32 %r675, %r652;
mov.b32 %r676, %r652;
mov.b32 %r677, %r652;
mov.b32 %r678, %r652;
mov.b32 %r679, %r652;
mov.b32 %r680, %r652;
mov.b32 %r681, %r652;
mov.b32 %r682, %r652;
mov.b32 %r683, %r652;
mov.b32 %r684, %r652;
mov.b32 %r685, %r652;
mov.b32 %r686, %r652;
mov.b32 %r687, %r652;
mov.b32 %r688, %r652;
mov.b32 %r689, %r652;
mov.b32 %r690, %r652;
mov.b32 %r691, %r652;
mov.b32 %r692, %r652;
mov.b32 %r693, %r652;
mov.b32 %r694, %r652;
mov.b32 %r695, %r652;
mov.b32 %r696, %r652;
mov.b32 %r697, %r652;
mov.b32 %r698, %r652;
mov.b32 %r699, %r652;
mov.b32 %r700, %r652;
mov.b32 %r701, %r652;
mov.b32 %r702, %r652;
mov.b32 %r703, %r652;
mov.b32 %r704, %r652;
mov.b32 %r705, %r652;
mov.b32 %r706, %r652;
mov.b32 %r707, %r652;
mov.b32 %r708, %r652;
mov.b32 %r709, %r652;
mov.b32 %r710, %r652;
mov.b32 %r711, %r652;
mov.b32 %r712, %r652;
mov.b32 %r713, %r652;
mov.b32 %r714, %r652;
mov.b32 %r715, %r652;
$L__BB0_13:
and.b32 %r232, %r716, 1;
bfe.u32 %r228, %r716, 1, 1;
shl.b32 %r233, %r232, 15;
add.s32 %r234, %r107, %r233;
mad.lo.s32 %r235, %r232, -32760, %r234;
add.s32 %r227, %r235, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r227], %r228;
@!complete bra.uni waitLoop;
}
shr.u32 %r236, %r234, 4;
add.s32 %r237, %r236, %r57;
and.b32 %r238, %r237, 16383;
cvt.u64.u32 %rd66, %r238;
wgmma.fence.sync.aligned;
or.b64 %rd50, %rd66, 4611686293305294848;
mov.pred %p35, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd50, %rd51, %p35, 1, 1, 0, 0;
add.s64 %rd52, %rd66, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd52, %rd53, %p35, 1, 1, 0, 0;
add.s64 %rd54, %rd66, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd54, %rd55, %p35, 1, 1, 0, 0;
add.s64 %rd56, %rd66, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd56, %rd57, %p35, 1, 1, 0, 0;
add.s64 %rd58, %rd66, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd58, %rd59, %p35, 1, 1, 0, 0;
add.s64 %rd60, %rd66, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd60, %rd61, %p35, 1, 1, 0, 0;
add.s64 %rd62, %rd66, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd62, %rd63, %p35, 1, 1, 0, 0;
add.s64 %rd64, %rd66, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715}, %rd64, %rd65, %p35, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r58, %r716, 1;
and.b32 %r239, %r58, 1;
setp.ne.b32 %p39, %r716, 5;
shl.b32 %r240, %r239, 3;
add.s32 %r241, %r107, %r240;
add.s32 %r229, %r241, 98304;
shl.b32 %r242, %r239, 15;
$L__tmp32:
bar.sync 0;
and.pred %p36, %p1, %p39;
@%p36 mbarrier.arrive.expect_tx.shared.b64 _, [%r229], 65536;
bar.sync 0;
elect.sync %r243|%p40, -1;
and.pred %p41, %p39, %p40;
and.pred %p37, %p58, %p41;
add.s32 %r230, %r575, %r242;
@%p37 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r230], [%rd67, {%r651, %r574}], [%r229];
$L__tmp33:
wgmma.wait_group.sync.aligned 0;
$L__tmp34:
bar.sync 0;
elect.sync %r244|%p42, -1;
and.pred %p43, %p39, %p42;
and.pred %p38, %p58, %p43;
@%p38 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r231], [%rd67, {%r651, %r578}], [%r229];
$L__tmp35:
add.s32 %r651, %r651, 128;
setp.eq.b32 %p44, %r58, 6;
mov.b32 %r716, %r58;
@%p44 bra $L__BB0_17;
bra.uni $L__BB0_13;
$L__tmp36:
$L__BB0_17:
add.s32 %r48, %r184, 32768;
add.s32 %r49, %r184, 34816;
add.s32 %r50, %r184, 36864;
add.s32 %r51, %r184, 38912;
add.s32 %r52, %r184, 40960;
add.s32 %r53, %r184, 43008;
add.s32 %r54, %r184, 45056;
add.s32 %r55, %r184, 47104;
$L__tmp37:
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r245];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r98];
and.b32 %r298, %r5, 240;
bfe.u32 %r299, %r5, 4, 4;
and.b32 %r61, %r5, 15;
shl.b32 %r300, %r61, 3;
mul.lo.s32 %r301, %r1, 589824;
or.b32 %r302, %r3, %r299;
or.b32 %r303, %r302, 16;
or.b32 %r304, %r302, 32;
or.b32 %r305, %r302, 48;
or.b32 %r306, %r302, 64;
or.b32 %r307, %r302, 80;
or.b32 %r308, %r302, 96;
or.b32 %r309, %r302, 112;
or.b32 %r310, %r4, %r300;
mad.lo.s32 %r311, %r302, 768, %r310;
mad.lo.s32 %r312, %r303, 768, %r310;
mad.lo.s32 %r313, %r304, 768, %r310;
mad.lo.s32 %r314, %r305, 768, %r310;
mad.lo.s32 %r315, %r306, 768, %r310;
mad.lo.s32 %r316, %r307, 768, %r310;
mad.lo.s32 %r317, %r308, 768, %r310;
mad.lo.s32 %r318, %r309, 768, %r310;
mad.wide.s32 %rd94, %r301, 2, %rd1;
mad.wide.u32 %rd86, %r311, 2, %rd94;
mad.wide.u32 %rd87, %r312, 2, %rd94;
mad.wide.u32 %rd88, %r313, 2, %rd94;
mad.wide.u32 %rd89, %r314, 2, %rd94;
mad.wide.u32 %rd90, %r315, 2, %rd94;
mad.wide.u32 %rd91, %r316, 2, %rd94;
mad.wide.u32 %rd92, %r317, 2, %rd94;
mad.wide.u32 %rd93, %r318, 2, %rd94;
mov.u32 %r266, 0x0;
mov.u32 %r267, 0x0;
mov.u32 %r268, 0x0;
mov.u32 %r269, 0x0;
ld.global.cg.v4.b32 { %r266, %r267, %r268, %r269 }, [ %rd86 + 0 ];
mov.u32 %r270, 0x0;
mov.u32 %r271, 0x0;
mov.u32 %r272, 0x0;
mov.u32 %r273, 0x0;
ld.global.cg.v4.b32 { %r270, %r271, %r272, %r273 }, [ %rd87 + 0 ];
mov.u32 %r274, 0x0;
mov.u32 %r275, 0x0;
mov.u32 %r276, 0x0;
mov.u32 %r277, 0x0;
ld.global.cg.v4.b32 { %r274, %r275, %r276, %r277 }, [ %rd88 + 0 ];
mov.u32 %r278, 0x0;
mov.u32 %r279, 0x0;
mov.u32 %r280, 0x0;
mov.u32 %r281, 0x0;
ld.global.cg.v4.b32 { %r278, %r279, %r280, %r281 }, [ %rd89 + 0 ];
mov.u32 %r282, 0x0;
mov.u32 %r283, 0x0;
mov.u32 %r284, 0x0;
mov.u32 %r285, 0x0;
ld.global.cg.v4.b32 { %r282, %r283, %r284, %r285 }, [ %rd90 + 0 ];
mov.u32 %r286, 0x0;
mov.u32 %r287, 0x0;
mov.u32 %r288, 0x0;
mov.u32 %r289, 0x0;
ld.global.cg.v4.b32 { %r286, %r287, %r288, %r289 }, [ %rd91 + 0 ];
mov.u32 %r290, 0x0;
mov.u32 %r291, 0x0;
mov.u32 %r292, 0x0;
mov.u32 %r293, 0x0;
ld.global.cg.v4.b32 { %r290, %r291, %r292, %r293 }, [ %rd92 + 0 ];
mov.u32 %r294, 0x0;
mov.u32 %r295, 0x0;
mov.u32 %r296, 0x0;
mov.u32 %r297, 0x0;
ld.global.cg.v4.b32 { %r294, %r295, %r296, %r297 }, [ %rd93 + 0 ];
shl.b32 %r319, %r5, 8;
and.b32 %r320, %r319, 30720;
or.b32 %r321, %r320, %r47;
xor.b32 %r322, %r321, %r298;
add.s32 %r323, %r107, %r322;
st.shared.v4.b32 [%r323], {%r266, %r267, %r268, %r269};
st.shared.v4.b32 [%r323+256], {%r270, %r271, %r272, %r273};
st.shared.v4.b32 [%r323+512], {%r274, %r275, %r276, %r277};
st.shared.v4.b32 [%r323+768], {%r278, %r279, %r280, %r281};
st.shared.v4.b32 [%r323+1024], {%r282, %r283, %r284, %r285};
st.shared.v4.b32 [%r323+1280], {%r286, %r287, %r288, %r289};
st.shared.v4.b32 [%r323+1536], {%r290, %r291, %r292, %r293};
st.shared.v4.b32 [%r323+1792], {%r294, %r295, %r296, %r297};
bar.sync 0;
shl.b32 %r324, %r46, 12;
shl.b32 %r325, %r61, 4;
and.b32 %r62, %r5, 224;
shl.b32 %r326, %r62, 3;
or.b32 %r327, %r325, %r326;
xor.b32 %r328, %r327, %r650;
or.b32 %r329, %r328, %r324;
add.s32 %r330, %r107, %r329;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r331, %r332, %r333, %r334}, [%r330];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r335, %r336, %r337, %r338}, [%r330+2048];
xor.b32 %r339, %r329, 32;
add.s32 %r340, %r107, %r339;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r341, %r342, %r343, %r344}, [%r340];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r345, %r346, %r347, %r348}, [%r340+2048];
xor.b32 %r349, %r329, 64;
add.s32 %r350, %r107, %r349;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r351, %r352, %r353, %r354}, [%r350];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r355, %r356, %r357, %r358}, [%r350+2048];
xor.b32 %r359, %r329, 96;
add.s32 %r360, %r107, %r359;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r361, %r362, %r363, %r364}, [%r360];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r365, %r366, %r367, %r368}, [%r360+2048];
mov.b32 {%rs1, %rs2}, %r331;
cvt.f32.bf16 %r369, %rs1;
cvt.f32.bf16 %r370, %rs2;
mul.f32 %r371, %r97, %r370;
mul.f32 %r372, %r97, %r369;
fma.rn.f32 %r373, %r96, %r652, %r372;
fma.rn.f32 %r374, %r96, %r653, %r371;
mov.b32 {%rs3, %rs4}, %r332;
cvt.f32.bf16 %r375, %rs3;
cvt.f32.bf16 %r376, %rs4;
mul.f32 %r377, %r97, %r376;
mul.f32 %r378, %r97, %r375;
fma.rn.f32 %r379, %r96, %r654, %r378;
fma.rn.f32 %r380, %r96, %r655, %r377;
mov.b32 {%rs5, %rs6}, %r333;
cvt.f32.bf16 %r381, %rs5;
cvt.f32.bf16 %r382, %rs6;
mul.f32 %r383, %r97, %r382;
mul.f32 %r384, %r97, %r381;
fma.rn.f32 %r385, %r96, %r656, %r384;
fma.rn.f32 %r386, %r96, %r657, %r383;
mov.b32 {%rs7, %rs8}, %r334;
cvt.f32.bf16 %r387, %rs7;
cvt.f32.bf16 %r388, %rs8;
mul.f32 %r389, %r97, %r388;
mul.f32 %r390, %r97, %r387;
fma.rn.f32 %r391, %r96, %r658, %r390;
fma.rn.f32 %r392, %r96, %r659, %r389;
mov.b32 {%rs9, %rs10}, %r341;
cvt.f32.bf16 %r393, %rs9;
cvt.f32.bf16 %r394, %rs10;
mul.f32 %r395, %r97, %r394;
mul.f32 %r396, %r97, %r393;
fma.rn.f32 %r397, %r96, %r660, %r396;
fma.rn.f32 %r398, %r96, %r661, %r395;
mov.b32 {%rs11, %rs12}, %r342;
cvt.f32.bf16 %r399, %rs11;
cvt.f32.bf16 %r400, %rs12;
mul.f32 %r401, %r97, %r400;
mul.f32 %r402, %r97, %r399;
fma.rn.f32 %r403, %r96, %r662, %r402;
fma.rn.f32 %r404, %r96, %r663, %r401;
mov.b32 {%rs13, %rs14}, %r343;
cvt.f32.bf16 %r405, %rs13;
cvt.f32.bf16 %r406, %rs14;
mul.f32 %r407, %r97, %r406;
mul.f32 %r408, %r97, %r405;
fma.rn.f32 %r409, %r96, %r664, %r408;
fma.rn.f32 %r410, %r96, %r665, %r407;
mov.b32 {%rs15, %rs16}, %r344;
cvt.f32.bf16 %r411, %rs15;
cvt.f32.bf16 %r412, %rs16;
mul.f32 %r413, %r97, %r412;
mul.f32 %r414, %r97, %r411;
fma.rn.f32 %r415, %r96, %r666, %r414;
fma.rn.f32 %r416, %r96, %r667, %r413;
mov.b32 {%rs17, %rs18}, %r351;
cvt.f32.bf16 %r417, %rs17;
cvt.f32.bf16 %r418, %rs18;
mul.f32 %r419, %r97, %r418;
mul.f32 %r420, %r97, %r417;
fma.rn.f32 %r421, %r96, %r668, %r420;
fma.rn.f32 %r422, %r96, %r669, %r419;
mov.b32 {%rs19, %rs20}, %r352;
cvt.f32.bf16 %r423, %rs19;
cvt.f32.bf16 %r424, %rs20;
mul.f32 %r425, %r97, %r424;
mul.f32 %r426, %r97, %r423;
fma.rn.f32 %r427, %r96, %r670, %r426;
fma.rn.f32 %r428, %r96, %r671, %r425;
mov.b32 {%rs21, %rs22}, %r353;
cvt.f32.bf16 %r429, %rs21;
cvt.f32.bf16 %r430, %rs22;
mul.f32 %r431, %r97, %r430;
mul.f32 %r432, %r97, %r429;
fma.rn.f32 %r433, %r96, %r672, %r432;
fma.rn.f32 %r434, %r96, %r673, %r431;
mov.b32 {%rs23, %rs24}, %r354;
cvt.f32.bf16 %r435, %rs23;
cvt.f32.bf16 %r436, %rs24;
mul.f32 %r437, %r97, %r436;
mul.f32 %r438, %r97, %r435;
fma.rn.f32 %r439, %r96, %r674, %r438;
fma.rn.f32 %r440, %r96, %r675, %r437;
mov.b32 {%rs25, %rs26}, %r361;
cvt.f32.bf16 %r441, %rs25;
cvt.f32.bf16 %r442, %rs26;
mul.f32 %r443, %r97, %r442;
mul.f32 %r444, %r97, %r441;
fma.rn.f32 %r445, %r96, %r676, %r444;
fma.rn.f32 %r446, %r96, %r677, %r443;
mov.b32 {%rs27, %rs28}, %r362;
cvt.f32.bf16 %r447, %rs27;
cvt.f32.bf16 %r448, %rs28;
mul.f32 %r449, %r97, %r448;
mul.f32 %r450, %r97, %r447;
fma.rn.f32 %r451, %r96, %r678, %r450;
fma.rn.f32 %r452, %r96, %r679, %r449;
mov.b32 {%rs29, %rs30}, %r363;
cvt.f32.bf16 %r453, %rs29;
cvt.f32.bf16 %r454, %rs30;
mul.f32 %r455, %r97, %r454;
mul.f32 %r456, %r97, %r453;
fma.rn.f32 %r457, %r96, %r680, %r456;
fma.rn.f32 %r458, %r96, %r681, %r455;
mov.b32 {%rs31, %rs32}, %r364;
cvt.f32.bf16 %r459, %rs31;
cvt.f32.bf16 %r460, %rs32;
mul.f32 %r461, %r97, %r460;
mul.f32 %r462, %r97, %r459;
fma.rn.f32 %r463, %r96, %r682, %r462;
fma.rn.f32 %r464, %r96, %r683, %r461;
mov.b32 {%rs33, %rs34}, %r335;
cvt.f32.bf16 %r465, %rs33;
cvt.f32.bf16 %r466, %rs34;
mul.f32 %r467, %r97, %r466;
mul.f32 %r468, %r97, %r465;
fma.rn.f32 %r469, %r96, %r684, %r468;
fma.rn.f32 %r470, %r96, %r685, %r467;
mov.b32 {%rs35, %rs36}, %r336;
cvt.f32.bf16 %r471, %rs35;
cvt.f32.bf16 %r472, %rs36;
mul.f32 %r473, %r97, %r472;
mul.f32 %r474, %r97, %r471;
fma.rn.f32 %r475, %r96, %r686, %r474;
fma.rn.f32 %r476, %r96, %r687, %r473;
mov.b32 {%rs37, %rs38}, %r337;
cvt.f32.bf16 %r477, %rs37;
cvt.f32.bf16 %r478, %rs38;
mul.f32 %r479, %r97, %r478;
mul.f32 %r480, %r97, %r477;
fma.rn.f32 %r481, %r96, %r688, %r480;
fma.rn.f32 %r482, %r96, %r689, %r479;
mov.b32 {%rs39, %rs40}, %r338;
cvt.f32.bf16 %r483, %rs39;
cvt.f32.bf16 %r484, %rs40;
mul.f32 %r485, %r97, %r484;
mul.f32 %r486, %r97, %r483;
fma.rn.f32 %r487, %r96, %r690, %r486;
fma.rn.f32 %r488, %r96, %r691, %r485;
mov.b32 {%rs41, %rs42}, %r345;
cvt.f32.bf16 %r489, %rs41;
cvt.f32.bf16 %r490, %rs42;
mul.f32 %r491, %r97, %r490;
mul.f32 %r492, %r97, %r489;
fma.rn.f32 %r493, %r96, %r692, %r492;
fma.rn.f32 %r494, %r96, %r693, %r491;
mov.b32 {%rs43, %rs44}, %r346;
cvt.f32.bf16 %r495, %rs43;
cvt.f32.bf16 %r496, %rs44;
mul.f32 %r497, %r97, %r496;
mul.f32 %r498, %r97, %r495;
fma.rn.f32 %r499, %r96, %r694, %r498;
fma.rn.f32 %r500, %r96, %r695, %r497;
mov.b32 {%rs45, %rs46}, %r347;
cvt.f32.bf16 %r501, %rs45;
cvt.f32.bf16 %r502, %rs46;
mul.f32 %r503, %r97, %r502;
mul.f32 %r504, %r97, %r501;
fma.rn.f32 %r505, %r96, %r696, %r504;
fma.rn.f32 %r506, %r96, %r697, %r503;
mov.b32 {%rs47, %rs48}, %r348;
cvt.f32.bf16 %r507, %rs47;
cvt.f32.bf16 %r508, %rs48;
mul.f32 %r509, %r97, %r508;
mul.f32 %r510, %r97, %r507;
fma.rn.f32 %r511, %r96, %r698, %r510;
fma.rn.f32 %r512, %r96, %r699, %r509;
mov.b32 {%rs49, %rs50}, %r355;
cvt.f32.bf16 %r513, %rs49;
cvt.f32.bf16 %r514, %rs50;
mul.f32 %r515, %r97, %r514;
mul.f32 %r516, %r97, %r513;
fma.rn.f32 %r517, %r96, %r700, %r516;
fma.rn.f32 %r518, %r96, %r701, %r515;
mov.b32 {%rs51, %rs52}, %r356;
cvt.f32.bf16 %r519, %rs51;
cvt.f32.bf16 %r520, %rs52;
mul.f32 %r521, %r97, %r520;
mul.f32 %r522, %r97, %r519;
fma.rn.f32 %r523, %r96, %r702, %r522;
fma.rn.f32 %r524, %r96, %r703, %r521;
mov.b32 {%rs53, %rs54}, %r357;
cvt.f32.bf16 %r525, %rs53;
cvt.f32.bf16 %r526, %rs54;
mul.f32 %r527, %r97, %r526;
mul.f32 %r528, %r97, %r525;
fma.rn.f32 %r529, %r96, %r704, %r528;
fma.rn.f32 %r530, %r96, %r705, %r527;
mov.b32 {%rs55, %rs56}, %r358;
cvt.f32.bf16 %r531, %rs55;
cvt.f32.bf16 %r532, %rs56;
mul.f32 %r533, %r97, %r532;
mul.f32 %r534, %r97, %r531;
fma.rn.f32 %r535, %r96, %r706, %r534;
fma.rn.f32 %r536, %r96, %r707, %r533;
mov.b32 {%rs57, %rs58}, %r365;
cvt.f32.bf16 %r537, %rs57;
cvt.f32.bf16 %r538, %rs58;
mul.f32 %r539, %r97, %r538;
mul.f32 %r540, %r97, %r537;
fma.rn.f32 %r541, %r96, %r708, %r540;
fma.rn.f32 %r542, %r96, %r709, %r539;
mov.b32 {%rs59, %rs60}, %r366;
cvt.f32.bf16 %r543, %rs59;
cvt.f32.bf16 %r544, %rs60;
mul.f32 %r545, %r97, %r544;
mul.f32 %r546, %r97, %r543;
fma.rn.f32 %r547, %r96, %r710, %r546;
fma.rn.f32 %r548, %r96, %r711, %r545;
mov.b32 {%rs61, %rs62}, %r367;
cvt.f32.bf16 %r549, %rs61;
cvt.f32.bf16 %r550, %rs62;
mul.f32 %r551, %r97, %r550;
mul.f32 %r552, %r97, %r549;
fma.rn.f32 %r553, %r96, %r712, %r552;
fma.rn.f32 %r554, %r96, %r713, %r551;
mov.b32 {%rs63, %rs64}, %r368;
cvt.f32.bf16 %r555, %rs63;
cvt.f32.bf16 %r556, %rs64;
mul.f32 %r557, %r97, %r556;
mul.f32 %r558, %r97, %r555;
fma.rn.f32 %r559, %r96, %r714, %r558;
fma.rn.f32 %r560, %r96, %r715, %r557;
cvt.rn.bf16x2.f32 %r63, %r374, %r373;
cvt.rn.bf16x2.f32 %r64, %r380, %r379;
cvt.rn.bf16x2.f32 %r65, %r386, %r385;
cvt.rn.bf16x2.f32 %r66, %r392, %r391;
cvt.rn.bf16x2.f32 %r67, %r398, %r397;
cvt.rn.bf16x2.f32 %r68, %r404, %r403;
cvt.rn.bf16x2.f32 %r69, %r410, %r409;
cvt.rn.bf16x2.f32 %r70, %r416, %r415;
cvt.rn.bf16x2.f32 %r71, %r422, %r421;
cvt.rn.bf16x2.f32 %r72, %r428, %r427;
cvt.rn.bf16x2.f32 %r73, %r434, %r433;
cvt.rn.bf16x2.f32 %r74, %r440, %r439;
cvt.rn.bf16x2.f32 %r75, %r446, %r445;
cvt.rn.bf16x2.f32 %r76, %r452, %r451;
cvt.rn.bf16x2.f32 %r77, %r458, %r457;
cvt.rn.bf16x2.f32 %r78, %r464, %r463;
cvt.rn.bf16x2.f32 %r79, %r470, %r469;
cvt.rn.bf16x2.f32 %r80, %r476, %r475;
cvt.rn.bf16x2.f32 %r81, %r482, %r481;
cvt.rn.bf16x2.f32 %r82, %r488, %r487;
cvt.rn.bf16x2.f32 %r83, %r494, %r493;
cvt.rn.bf16x2.f32 %r84, %r500, %r499;
cvt.rn.bf16x2.f32 %r85, %r506, %r505;
cvt.rn.bf16x2.f32 %r86, %r512, %r511;
cvt.rn.bf16x2.f32 %r87, %r518, %r517;
cvt.rn.bf16x2.f32 %r88, %r524, %r523;
cvt.rn.bf16x2.f32 %r89, %r530, %r529;
cvt.rn.bf16x2.f32 %r90, %r536, %r535;
cvt.rn.bf16x2.f32 %r91, %r542, %r541;
cvt.rn.bf16x2.f32 %r92, %r548, %r547;
cvt.rn.bf16x2.f32 %r93, %r554, %r553;
cvt.rn.bf16x2.f32 %r94, %r560, %r559;
@%p26 bra $L__BB0_18;
bra.uni $L__BB0_19;
$L__BB0_18:
bar.sync 0;
shl.b32 %r561, %r61, 7;
shl.b32 %r562, %r62, 6;
or.b32 %r563, %r561, %r47;
xor.b32 %r564, %r563, %r650;
or.b32 %r565, %r564, %r562;
add.s32 %r566, %r107, %r565;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r566], {%r63, %r64, %r65, %r66};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r566+16384], {%r79, %r80, %r81, %r82};
xor.b32 %r567, %r565, 32;
add.s32 %r568, %r107, %r567;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r568], {%r67, %r68, %r69, %r70};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r568+16384], {%r83, %r84, %r85, %r86};
xor.b32 %r569, %r565, 64;
add.s32 %r570, %r107, %r569;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r570], {%r71, %r72, %r73, %r74};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r570+16384], {%r87, %r88, %r89, %r90};
xor.b32 %r571, %r565, 96;
add.s32 %r572, %r107, %r571;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r572], {%r75, %r76, %r77, %r78};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r572+16384], {%r91, %r92, %r93, %r94};
$L__BB0_19:
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r48], {%r63, %r64, %r65, %r66};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r49], {%r67, %r68, %r69, %r70};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r50], {%r71, %r72, %r73, %r74};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r51], {%r75, %r76, %r77, %r78};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r52], {%r79, %r80, %r81, %r82};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r53], {%r83, %r84, %r85, %r86};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r87, %r88, %r89, %r90};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r55], {%r91, %r92, %r93, %r94};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_21;
bar.sync 0;
elect.sync %r576|%p55, -1;
and.pred %p54, %p58, %p55;
or.b32 %r573, %r246, %r4;
@%p54 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd95, {%r573, %r574}], [%r575];
cp.async.bulk.commit_group;
$L__BB0_21:
bar.sync 0;
elect.sync %r581|%p57, -1;
and.pred %p56, %p58, %p57;
@%p56 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd95, {%r577, %r578}], [%r579];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r580, 0;
$L__tmp38:
ret;
$L__tmp39:
$L__func_end0:
}
"#
  | .mlp => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry cooperative_round(
.param .align 64 .b8 cooperative_round_param_0[128],
.param .u32 cooperative_round_param_1,
.param .u32 cooperative_round_param_2,
.param .u64 cooperative_round_param_3,
.param .u64 cooperative_round_param_4,
.param .align 64 .b8 cooperative_round_param_5[128],
.param .u32 cooperative_round_param_6,
.param .u32 cooperative_round_param_7,
.param .u64 cooperative_round_param_8,
.param .u64 cooperative_round_param_9,
.param .align 64 .b8 cooperative_round_param_10[128],
.param .u32 cooperative_round_param_11,
.param .u32 cooperative_round_param_12,
.param .u64 cooperative_round_param_13,
.param .u64 cooperative_round_param_14,
.param .u64 .ptr .global .align 1 cooperative_round_param_15,
.param .u64 .ptr .global .align 1 cooperative_round_param_16,
.param .u32 cooperative_round_param_17,
.param .f32 cooperative_round_param_18,
.param .f32 cooperative_round_param_19,
.param .u64 .ptr .global .align 1 cooperative_round_param_20,
.param .u64 .ptr .global .align 1 cooperative_round_param_21
)
.reqntid 256
{
.reg .pred %p<59>;
.reg .b16 %rs<65>;
.reg .b32 %r<723>;
.reg .b64 %rd<96>;
$L__func_begin0:
mov.b64 %rd3, cooperative_round_param_0;
mov.b64 %rd4, cooperative_round_param_10;
$L__tmp0:
cvta.param.u64 %rd95, %rd4;
mov.b64 %rd5, cooperative_round_param_5;
cvta.param.u64 %rd67, %rd5;
cvta.param.u64 %rd28, %rd3;
mov.u32 %r100, %ctaid.x;
mul.hi.u32 %r101, %r100, 818089009;
shr.u32 %r1, %r101, 2;
$L__tmp1:
mul.lo.s32 %r102, %r1, 21;
sub.s32 %r2, %r100, %r102;
mul.lo.s32 %r103, %r2, 3;
mov.b64 %rd6, 6569755928097638024;
shr.u64 %rd7, %rd6, %r103;
cvt.u32.u64 %r104, %rd7;
shl.b32 %r105, %r104, 7;
and.b32 %r3, %r105, 896;
mov.b64 %rd8, 6420841298313609216;
shr.u64 %rd9, %rd8, %r103;
cvt.u32.u64 %r106, %rd9;
shl.b32 %r107, %r106, 7;
and.b32 %r4, %r107, 896;
$L__tmp2:
setp.eq.b32 %p2, %r3, %r4;
mov.u32 %r5, %tid.x;
and.b32 %r6, %r5, 255;
setp.eq.b32 %p1, %r6, 0;
mov.b32 %r108, global_smem;
add.s32 %r249, %r108, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r249], 1;
bar.sync 0;
add.s32 %r99, %r108, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r99], 1;
mul.lo.s32 %r7, %r1, 2816;
$L__tmp3:
shr.u32 %r8, %r5, 5;
setp.lt.u32 %p58, %r6, 64;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r249], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r140, %r8, 0, 31, -1;
elect.sync %r141|%p18, -1;
and.pred %p17, %p58, %p18;
and.b32 %r142, %r140, 1;
shl.b32 %r143, %r142, 14;
add.s32 %r139, %r108, %r143;
shl.b32 %r144, %r142, 6;
or.b32 %r150, %r144, %r3;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r139], [%rd28, {%r150, %r7}], [%r249];
shl.b32 %r145, %r140, 8;
and.b32 %r11, %r145, 1024;
$L__tmp6:
add.s32 %r651, %r7, 128;
mov.b32 %r652, 0;
mov.b32 %r586, 0f00000000;
mov.b32 %r587, %r586;
mov.b32 %r588, %r586;
mov.b32 %r589, %r586;
mov.b32 %r590, %r586;
mov.b32 %r591, %r586;
mov.b32 %r592, %r586;
mov.b32 %r593, %r586;
mov.b32 %r594, %r586;
mov.b32 %r595, %r586;
mov.b32 %r596, %r586;
mov.b32 %r597, %r586;
mov.b32 %r598, %r586;
mov.b32 %r599, %r586;
mov.b32 %r600, %r586;
mov.b32 %r601, %r586;
mov.b32 %r602, %r586;
mov.b32 %r603, %r586;
mov.b32 %r604, %r586;
mov.b32 %r605, %r586;
mov.b32 %r606, %r586;
mov.b32 %r607, %r586;
mov.b32 %r608, %r586;
mov.b32 %r609, %r586;
mov.b32 %r610, %r586;
mov.b32 %r611, %r586;
mov.b32 %r612, %r586;
mov.b32 %r613, %r586;
mov.b32 %r614, %r586;
mov.b32 %r615, %r586;
mov.b32 %r616, %r586;
mov.b32 %r617, %r586;
mov.b32 %r618, %r586;
mov.b32 %r619, %r586;
mov.b32 %r620, %r586;
mov.b32 %r621, %r586;
mov.b32 %r622, %r586;
mov.b32 %r623, %r586;
mov.b32 %r624, %r586;
mov.b32 %r625, %r586;
mov.b32 %r626, %r586;
mov.b32 %r627, %r586;
mov.b32 %r628, %r586;
mov.b32 %r629, %r586;
mov.b32 %r630, %r586;
mov.b32 %r631, %r586;
mov.b32 %r632, %r586;
mov.b32 %r633, %r586;
mov.b32 %r634, %r586;
mov.b32 %r635, %r586;
mov.b32 %r636, %r586;
mov.b32 %r637, %r586;
mov.b32 %r638, %r586;
mov.b32 %r639, %r586;
mov.b32 %r640, %r586;
mov.b32 %r641, %r586;
mov.b32 %r642, %r586;
mov.b32 %r643, %r586;
mov.b32 %r644, %r586;
mov.b32 %r645, %r586;
mov.b32 %r646, %r586;
mov.b32 %r647, %r586;
mov.b32 %r648, %r586;
mov.b32 %r649, %r586;
$L__BB0_4:
and.b32 %r151, %r652, 1;
bfe.u32 %r147, %r652, 1, 1;
shl.b32 %r152, %r151, 15;
add.s32 %r153, %r108, %r152;
mad.lo.s32 %r154, %r151, -32760, %r153;
add.s32 %r146, %r154, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r146], %r147;
@!complete bra.uni waitLoop;
}
shr.u32 %r155, %r153, 4;
add.s32 %r156, %r155, %r11;
and.b32 %r157, %r156, 16383;
cvt.u64.u32 %rd45, %r157;
bfe.u32 %r158, %r153, 4, 14;
cvt.u64.u32 %rd46, %r158;
wgmma.fence.sync.aligned;
or.b64 %rd29, %rd45, 4611686293305294848;
or.b64 %rd30, %rd46, 4611686293372403712;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd29, %rd30, %p19, 1, 1, 1, 1;
add.s64 %rd31, %rd45, 4611686293305294976;
add.s64 %rd32, %rd46, 4611686293372403840;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd31, %rd32, %p19, 1, 1, 1, 1;
add.s64 %rd33, %rd45, 4611686293305295104;
add.s64 %rd34, %rd46, 4611686293372403968;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd33, %rd34, %p19, 1, 1, 1, 1;
add.s64 %rd35, %rd45, 4611686293305295232;
add.s64 %rd36, %rd46, 4611686293372404096;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd35, %rd36, %p19, 1, 1, 1, 1;
add.s64 %rd37, %rd45, 4611686293305295360;
add.s64 %rd38, %rd46, 4611686293372404224;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd37, %rd38, %p19, 1, 1, 1, 1;
add.s64 %rd39, %rd45, 4611686293305295488;
add.s64 %rd40, %rd46, 4611686293372404352;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd39, %rd40, %p19, 1, 1, 1, 1;
add.s64 %rd41, %rd45, 4611686293305295616;
add.s64 %rd42, %rd46, 4611686293372404480;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd41, %rd42, %p19, 1, 1, 1, 1;
add.s64 %rd43, %rd45, 4611686293305295744;
add.s64 %rd44, %rd46, 4611686293372404608;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd43, %rd44, %p19, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r12, %r652, 1;
and.b32 %r159, %r12, 1;
setp.ne.b32 %p22, %r652, 21;
shl.b32 %r160, %r159, 3;
add.s32 %r161, %r108, %r160;
add.s32 %r148, %r161, 98304;
shl.b32 %r162, %r159, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r148], 32768;
bar.sync 0;
elect.sync %r163|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p58, %p24;
add.s32 %r149, %r139, %r162;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r149], [%rd28, {%r150, %r651}], [%r148];
$L__tmp8:
add.s32 %r651, %r651, 128;
setp.ne.b32 %p25, %r12, 22;
mov.b32 %r652, %r12;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r249], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r110, %r8, 0, 31, -1;
elect.sync %r111|%p5, -1;
and.pred %p3, %p58, %p5;
and.b32 %r112, %r110, 1;
shl.b32 %r113, %r112, 14;
add.s32 %r109, %r108, %r113;
shl.b32 %r114, %r112, 6;
or.b32 %r123, %r114, %r3;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r109], [%rd28, {%r123, %r7}], [%r249];
$L__tmp10:
bar.sync 0;
elect.sync %r115|%p6, -1;
and.pred %p4, %p58, %p6;
add.s32 %r124, %r109, 65536;
or.b32 %r125, %r114, %r4;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r124], [%rd28, {%r125, %r7}], [%r249];
shl.b32 %r116, %r110, 8;
and.b32 %r9, %r116, 1024;
add.s32 %r117, %r108, 65536;
bfe.u32 %r118, %r117, 4, 14;
cvt.u64.u32 %rd10, %r118;
or.b64 %rd12, %rd10, 4611686293372403712;
add.s64 %rd14, %rd10, 4611686293372403840;
add.s64 %rd16, %rd10, 4611686293372403968;
add.s64 %rd18, %rd10, 4611686293372404096;
add.s64 %rd20, %rd10, 4611686293372404224;
add.s64 %rd22, %rd10, 4611686293372404352;
add.s64 %rd24, %rd10, 4611686293372404480;
add.s64 %rd26, %rd10, 4611686293372404608;
$L__tmp11:
add.s32 %r585, %r7, 128;
mov.b32 %r650, 0;
mov.b32 %r586, 0f00000000;
mov.b32 %r587, %r586;
mov.b32 %r588, %r586;
mov.b32 %r589, %r586;
mov.b32 %r590, %r586;
mov.b32 %r591, %r586;
mov.b32 %r592, %r586;
mov.b32 %r593, %r586;
mov.b32 %r594, %r586;
mov.b32 %r595, %r586;
mov.b32 %r596, %r586;
mov.b32 %r597, %r586;
mov.b32 %r598, %r586;
mov.b32 %r599, %r586;
mov.b32 %r600, %r586;
mov.b32 %r601, %r586;
mov.b32 %r602, %r586;
mov.b32 %r603, %r586;
mov.b32 %r604, %r586;
mov.b32 %r605, %r586;
mov.b32 %r606, %r586;
mov.b32 %r607, %r586;
mov.b32 %r608, %r586;
mov.b32 %r609, %r586;
mov.b32 %r610, %r586;
mov.b32 %r611, %r586;
mov.b32 %r612, %r586;
mov.b32 %r613, %r586;
mov.b32 %r614, %r586;
mov.b32 %r615, %r586;
mov.b32 %r616, %r586;
mov.b32 %r617, %r586;
mov.b32 %r618, %r586;
mov.b32 %r619, %r586;
mov.b32 %r620, %r586;
mov.b32 %r621, %r586;
mov.b32 %r622, %r586;
mov.b32 %r623, %r586;
mov.b32 %r624, %r586;
mov.b32 %r625, %r586;
mov.b32 %r626, %r586;
mov.b32 %r627, %r586;
mov.b32 %r628, %r586;
mov.b32 %r629, %r586;
mov.b32 %r630, %r586;
mov.b32 %r631, %r586;
mov.b32 %r632, %r586;
mov.b32 %r633, %r586;
mov.b32 %r634, %r586;
mov.b32 %r635, %r586;
mov.b32 %r636, %r586;
mov.b32 %r637, %r586;
mov.b32 %r638, %r586;
mov.b32 %r639, %r586;
mov.b32 %r640, %r586;
mov.b32 %r641, %r586;
mov.b32 %r642, %r586;
mov.b32 %r643, %r586;
mov.b32 %r644, %r586;
mov.b32 %r645, %r586;
mov.b32 %r646, %r586;
mov.b32 %r647, %r586;
mov.b32 %r648, %r586;
mov.b32 %r649, %r586;
$L__BB0_2:
and.b32 %r126, %r650, 1;
bfe.u32 %r120, %r650, 1, 1;
shl.b32 %r127, %r126, 15;
add.s32 %r128, %r108, %r127;
mad.lo.s32 %r129, %r126, -32760, %r128;
add.s32 %r119, %r129, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r119], %r120;
@!complete bra.uni waitLoop;
}
shr.u32 %r130, %r128, 4;
add.s32 %r131, %r130, %r9;
and.b32 %r132, %r131, 16383;
cvt.u64.u32 %rd27, %r132;
wgmma.fence.sync.aligned;
or.b64 %rd11, %rd27, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd11, %rd12, %p7, 1, 1, 1, 1;
add.s64 %rd13, %rd27, 4611686293305294976;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd13, %rd14, %p7, 1, 1, 1, 1;
add.s64 %rd15, %rd27, 4611686293305295104;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd15, %rd16, %p7, 1, 1, 1, 1;
add.s64 %rd17, %rd27, 4611686293305295232;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd17, %rd18, %p7, 1, 1, 1, 1;
add.s64 %rd19, %rd27, 4611686293305295360;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd19, %rd20, %p7, 1, 1, 1, 1;
add.s64 %rd21, %rd27, 4611686293305295488;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd21, %rd22, %p7, 1, 1, 1, 1;
add.s64 %rd23, %rd27, 4611686293305295616;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd23, %rd24, %p7, 1, 1, 1, 1;
add.s64 %rd25, %rd27, 4611686293305295744;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r586,%r587,%r588,%r589,%r590,%r591,%r592,%r593,%r594,%r595,%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649}, %rd25, %rd26, %p7, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
add.s32 %r10, %r650, 1;
and.b32 %r133, %r10, 1;
setp.ne.b32 %p11, %r650, 21;
shl.b32 %r134, %r133, 3;
add.s32 %r135, %r108, %r134;
add.s32 %r121, %r135, 98304;
shl.b32 %r136, %r133, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r121], 65536;
bar.sync 0;
elect.sync %r137|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p58, %p13;
add.s32 %r122, %r109, %r136;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r122], [%rd28, {%r123, %r585}], [%r121];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r138|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p58, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r124], [%rd28, {%r125, %r585}], [%r121];
$L__tmp15:
add.s32 %r585, %r585, 128;
setp.eq.b32 %p16, %r10, 22;
mov.b32 %r650, %r10;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r3, %r4;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r249];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r99];
cvt.rn.bf16x2.f32 %r13, %r587, %r586;
cvt.rn.bf16x2.f32 %r14, %r589, %r588;
cvt.rn.bf16x2.f32 %r15, %r591, %r590;
cvt.rn.bf16x2.f32 %r16, %r593, %r592;
cvt.rn.bf16x2.f32 %r17, %r595, %r594;
cvt.rn.bf16x2.f32 %r18, %r597, %r596;
cvt.rn.bf16x2.f32 %r19, %r599, %r598;
cvt.rn.bf16x2.f32 %r20, %r601, %r600;
cvt.rn.bf16x2.f32 %r21, %r603, %r602;
cvt.rn.bf16x2.f32 %r22, %r605, %r604;
cvt.rn.bf16x2.f32 %r23, %r607, %r606;
cvt.rn.bf16x2.f32 %r24, %r609, %r608;
cvt.rn.bf16x2.f32 %r25, %r611, %r610;
cvt.rn.bf16x2.f32 %r26, %r613, %r612;
cvt.rn.bf16x2.f32 %r27, %r615, %r614;
cvt.rn.bf16x2.f32 %r28, %r617, %r616;
cvt.rn.bf16x2.f32 %r29, %r619, %r618;
cvt.rn.bf16x2.f32 %r30, %r621, %r620;
cvt.rn.bf16x2.f32 %r31, %r623, %r622;
cvt.rn.bf16x2.f32 %r32, %r625, %r624;
cvt.rn.bf16x2.f32 %r33, %r627, %r626;
cvt.rn.bf16x2.f32 %r34, %r629, %r628;
cvt.rn.bf16x2.f32 %r35, %r631, %r630;
cvt.rn.bf16x2.f32 %r36, %r633, %r632;
cvt.rn.bf16x2.f32 %r37, %r635, %r634;
cvt.rn.bf16x2.f32 %r38, %r637, %r636;
cvt.rn.bf16x2.f32 %r39, %r639, %r638;
cvt.rn.bf16x2.f32 %r40, %r641, %r640;
cvt.rn.bf16x2.f32 %r41, %r643, %r642;
cvt.rn.bf16x2.f32 %r42, %r645, %r644;
cvt.rn.bf16x2.f32 %r43, %r647, %r646;
cvt.rn.bf16x2.f32 %r44, %r649, %r648;
shl.b32 %r45, %r5, 7;
@%p26 bra $L__BB0_8;
and.b32 %r653, %r5, 16;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r164, %r45, 1920;
shl.b32 %r165, %r5, 6;
and.b32 %r166, %r165, 14336;
shl.b32 %r167, %r5, 4;
and.b32 %r168, %r167, 112;
and.b32 %r653, %r5, 16;
or.b32 %r169, %r164, %r168;
xor.b32 %r170, %r169, %r653;
or.b32 %r171, %r170, %r166;
add.s32 %r172, %r108, %r171;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r172], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r172+16384], {%r29, %r30, %r31, %r32};
xor.b32 %r173, %r171, 32;
add.s32 %r174, %r108, %r173;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r174], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r174+16384], {%r33, %r34, %r35, %r36};
xor.b32 %r175, %r171, 64;
add.s32 %r176, %r108, %r175;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r176], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r176+16384], {%r37, %r38, %r39, %r40};
xor.b32 %r177, %r171, 96;
add.s32 %r178, %r108, %r177;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r178], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r178+16384], {%r41, %r42, %r43, %r44};
$L__BB0_9:
ld.param.b32 %r96, [cooperative_round_param_17];
ld.param.b64 %rd2, [cooperative_round_param_16];
and.b32 %r179, %r45, 17280;
and.b32 %r46, %r5, 7;
shl.b32 %r47, %r46, 4;
and.b32 %r180, %r5, 96;
shl.b32 %r181, %r5, 1;
and.b32 %r182, %r181, 16;
shl.b32 %r183, %r653, 6;
or.b32 %r184, %r179, %r47;
or.b32 %r185, %r182, %r180;
xor.b32 %r186, %r184, %r185;
add.s32 %r187, %r108, %r183;
add.s32 %r188, %r187, %r186;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+32768], {%r13, %r14, %r15, %r16};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+34816], {%r17, %r18, %r19, %r20};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+36864], {%r21, %r22, %r23, %r24};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+38912], {%r25, %r26, %r27, %r28};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+40960], {%r29, %r30, %r31, %r32};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+43008], {%r33, %r34, %r35, %r36};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+45056], {%r37, %r38, %r39, %r40};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r188+47104], {%r41, %r42, %r43, %r44};
fence.proxy.async.shared::cta;
mul.lo.s32 %r56, %r1, 768;
$L__tmp17:
add.s32 %r654, %r3, %r56;
$L__tmp18:
@%p26 bra $L__BB0_10;
bra.uni $L__BB0_11;
$L__BB0_10:
bar.sync 0;
elect.sync %r191|%p28, -1;
shfl.sync.idx.b32 %r192, %r8, 0, 31, -1;
and.pred %p27, %p58, %p28;
and.b32 %r193, %r192, 1;
shl.b32 %r194, %r193, 14;
add.s32 %r190, %r108, %r194;
shl.b32 %r195, %r193, 6;
or.b32 %r189, %r195, %r4;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd67, {%r189, %r654}], [%r190];
cp.async.bulk.commit_group;
$L__BB0_11:
ld.param.b32 %r98, [cooperative_round_param_19];
ld.param.b32 %r97, [cooperative_round_param_18];
ld.param.b64 %rd1, [cooperative_round_param_15];
add.s32 %r581, %r4, %r56;
bar.sync 0;
elect.sync %r210|%p30, -1;
shfl.sync.idx.b32 %r57, %r8, 0, 31, -1;
and.pred %p29, %p58, %p30;
and.b32 %r211, %r57, 1;
shl.b32 %r212, %r211, 14;
add.s32 %r578, %r108, %r212;
add.s32 %r582, %r578, 32768;
shl.b32 %r250, %r211, 6;
or.b32 %r580, %r250, %r3;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd67, {%r580, %r581}], [%r582];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r196, 0;
mov.b32 %r213, 1;
$L__tmp19:
shl.b32 %r208, %r213, %r2;
shr.u32 %r214, %r3, 7;
$L__tmp20:
sub.s32 %r215, 13, %r214;
mul.lo.s32 %r216, %r215, %r214;
shr.u32 %r200, %r216, 1;
sub.s32 %r198, %r200, %r214;
bmsk.clamp.b32 %r197, 0, %r198;
sub.s32 %r201, 6, %r214;
bmsk.clamp.b32 %r199, %r200, %r201;
and.b32 %r217, %r197, 53793;
shl.b32 %r218, %r217, %r214;
$L__tmp21:
shr.u32 %r219, %r4, 7;
$L__tmp22:
sub.s32 %r220, 13, %r219;
mul.lo.s32 %r221, %r220, %r219;
shr.u32 %r205, %r221, 1;
sub.s32 %r203, %r205, %r219;
bmsk.clamp.b32 %r202, 0, %r203;
sub.s32 %r206, 6, %r219;
bmsk.clamp.b32 %r204, %r205, %r206;
and.b32 %r222, %r202, 53793;
shl.b32 %r223, %r222, %r219;
or.b32 %r224, %r218, %r199;
$L__tmp23:
or.b32 %r225, %r224, %r204;
$L__tmp24:
or.b32 %r209, %r225, %r223;
mad.wide.s32 %rd48, %r96, 4, %rd2;
mad.wide.u32 %rd47, %r1, 4, %rd48;
$L__tmp25:
{
.reg .u32 tid, bits, covered;
.reg .pred observer, ready;
bar.sync 0;
mov.u32 tid, %tid.x;
setp.eq.u32 observer, tid, 0;
@!observer bra gram_ready;
fence.proxy.async.global;
atom.acq_rel.gpu.global.or.b32 bits, [%rd47], %r208;
or.b32 bits, bits, %r208;
and.b32 covered, bits, %r209;
setp.eq.u32 ready, covered, %r209;
@ready bra gram_ready;
observe_gram:
ld.acquire.gpu.global.u32 bits, [%rd47];
and.b32 covered, bits, %r209;
setp.eq.u32 ready, covered, %r209;
@!ready bra observe_gram;
gram_ready:
bar.sync 0;
fence.proxy.async.global;
mov.u32 %r207, 0;
}
$L__tmp26:
@%p1 mbarrier.init.shared::cta.b64 [%r249], 1;
bar.sync 0;
@%p1 mbarrier.init.shared::cta.b64 [%r99], 1;
@%p2 bra $L__BB0_14;
bra.uni $L__BB0_12;
$L__BB0_14:
$L__tmp27:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r249], 32768;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r251|%p46, -1;
and.pred %p45, %p58, %p46;
@%p45 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r578], [%rd67, {%r250, %r654}], [%r249];
shl.b32 %r252, %r57, 7;
and.b32 %r60, %r252, 512;
$L__tmp28:
or.b32 %r721, %r250, 128;
mov.b32 %r722, 0;
mov.b32 %r656, 0f00000000;
mov.b32 %r657, %r656;
mov.b32 %r658, %r656;
mov.b32 %r659, %r656;
mov.b32 %r660, %r656;
mov.b32 %r661, %r656;
mov.b32 %r662, %r656;
mov.b32 %r663, %r656;
mov.b32 %r664, %r656;
mov.b32 %r665, %r656;
mov.b32 %r666, %r656;
mov.b32 %r667, %r656;
mov.b32 %r668, %r656;
mov.b32 %r669, %r656;
mov.b32 %r670, %r656;
mov.b32 %r671, %r656;
mov.b32 %r672, %r656;
mov.b32 %r673, %r656;
mov.b32 %r674, %r656;
mov.b32 %r675, %r656;
mov.b32 %r676, %r656;
mov.b32 %r677, %r656;
mov.b32 %r678, %r656;
mov.b32 %r679, %r656;
mov.b32 %r680, %r656;
mov.b32 %r681, %r656;
mov.b32 %r682, %r656;
mov.b32 %r683, %r656;
mov.b32 %r684, %r656;
mov.b32 %r685, %r656;
mov.b32 %r686, %r656;
mov.b32 %r687, %r656;
mov.b32 %r688, %r656;
mov.b32 %r689, %r656;
mov.b32 %r690, %r656;
mov.b32 %r691, %r656;
mov.b32 %r692, %r656;
mov.b32 %r693, %r656;
mov.b32 %r694, %r656;
mov.b32 %r695, %r656;
mov.b32 %r696, %r656;
mov.b32 %r697, %r656;
mov.b32 %r698, %r656;
mov.b32 %r699, %r656;
mov.b32 %r700, %r656;
mov.b32 %r701, %r656;
mov.b32 %r702, %r656;
mov.b32 %r703, %r656;
mov.b32 %r704, %r656;
mov.b32 %r705, %r656;
mov.b32 %r706, %r656;
mov.b32 %r707, %r656;
mov.b32 %r708, %r656;
mov.b32 %r709, %r656;
mov.b32 %r710, %r656;
mov.b32 %r711, %r656;
mov.b32 %r712, %r656;
mov.b32 %r713, %r656;
mov.b32 %r714, %r656;
mov.b32 %r715, %r656;
mov.b32 %r716, %r656;
mov.b32 %r717, %r656;
mov.b32 %r718, %r656;
mov.b32 %r719, %r656;
$L__BB0_15:
and.b32 %r257, %r722, 1;
bfe.u32 %r254, %r722, 1, 1;
shl.b32 %r258, %r257, 15;
add.s32 %r259, %r108, %r258;
mad.lo.s32 %r260, %r257, -32760, %r259;
add.s32 %r253, %r260, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r253], %r254;
@!complete bra.uni waitLoop;
}
shr.u32 %r261, %r259, 4;
add.s32 %r262, %r261, %r60;
and.b32 %r263, %r262, 16383;
cvt.u64.u32 %rd84, %r263;
bfe.u32 %r264, %r259, 4, 14;
cvt.u64.u32 %rd85, %r264;
wgmma.fence.sync.aligned;
or.b64 %rd68, %rd84, 4611686293305294848;
or.b64 %rd69, %rd85, 4611686293305294848;
mov.pred %p47, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd68, %rd69, %p47, 1, 1, 0, 0;
add.s64 %rd70, %rd84, 4611686293305294850;
add.s64 %rd71, %rd85, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd70, %rd71, %p47, 1, 1, 0, 0;
add.s64 %rd72, %rd84, 4611686293305294852;
add.s64 %rd73, %rd85, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd72, %rd73, %p47, 1, 1, 0, 0;
add.s64 %rd74, %rd84, 4611686293305294854;
add.s64 %rd75, %rd85, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd74, %rd75, %p47, 1, 1, 0, 0;
add.s64 %rd76, %rd84, 4611686293305295872;
add.s64 %rd77, %rd85, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd76, %rd77, %p47, 1, 1, 0, 0;
add.s64 %rd78, %rd84, 4611686293305295874;
add.s64 %rd79, %rd85, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd78, %rd79, %p47, 1, 1, 0, 0;
add.s64 %rd80, %rd84, 4611686293305295876;
add.s64 %rd81, %rd85, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd80, %rd81, %p47, 1, 1, 0, 0;
add.s64 %rd82, %rd84, 4611686293305295878;
add.s64 %rd83, %rd85, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd82, %rd83, %p47, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r61, %r722, 1;
and.b32 %r265, %r61, 1;
setp.ne.b32 %p50, %r722, 5;
shl.b32 %r266, %r265, 3;
add.s32 %r267, %r108, %r266;
add.s32 %r255, %r267, 98304;
shl.b32 %r268, %r265, 15;
$L__tmp29:
bar.sync 0;
and.pred %p48, %p1, %p50;
@%p48 mbarrier.arrive.expect_tx.shared.b64 _, [%r255], 32768;
bar.sync 0;
elect.sync %r269|%p51, -1;
and.pred %p52, %p50, %p51;
and.pred %p49, %p58, %p52;
add.s32 %r256, %r578, %r268;
@%p49 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r256], [%rd67, {%r721, %r654}], [%r255];
$L__tmp30:
add.s32 %r721, %r721, 128;
setp.ne.b32 %p53, %r61, 6;
mov.b32 %r722, %r61;
@%p53 bra $L__BB0_15;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_17;
$L__tmp31:
$L__BB0_12:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r249], 65536;
fence.proxy.async.shared::cta;
bar.sync 0;
elect.sync %r226|%p33, -1;
and.pred %p31, %p58, %p33;
@%p31 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r578], [%rd67, {%r250, %r654}], [%r249];
$L__tmp32:
bar.sync 0;
elect.sync %r227|%p34, -1;
and.pred %p32, %p58, %p34;
add.s32 %r235, %r578, 65536;
@%p32 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r235], [%rd67, {%r250, %r581}], [%r249];
shl.b32 %r228, %r57, 7;
and.b32 %r58, %r228, 512;
add.s32 %r229, %r108, 65536;
bfe.u32 %r230, %r229, 4, 14;
cvt.u64.u32 %rd49, %r230;
or.b64 %rd51, %rd49, 4611686293305294848;
add.s64 %rd53, %rd49, 4611686293305294850;
add.s64 %rd55, %rd49, 4611686293305294852;
add.s64 %rd57, %rd49, 4611686293305294854;
add.s64 %rd59, %rd49, 4611686293305295872;
add.s64 %rd61, %rd49, 4611686293305295874;
add.s64 %rd63, %rd49, 4611686293305295876;
add.s64 %rd65, %rd49, 4611686293305295878;
$L__tmp33:
or.b32 %r655, %r250, 128;
mov.b32 %r720, 0;
mov.b32 %r656, 0f00000000;
mov.b32 %r657, %r656;
mov.b32 %r658, %r656;
mov.b32 %r659, %r656;
mov.b32 %r660, %r656;
mov.b32 %r661, %r656;
mov.b32 %r662, %r656;
mov.b32 %r663, %r656;
mov.b32 %r664, %r656;
mov.b32 %r665, %r656;
mov.b32 %r666, %r656;
mov.b32 %r667, %r656;
mov.b32 %r668, %r656;
mov.b32 %r669, %r656;
mov.b32 %r670, %r656;
mov.b32 %r671, %r656;
mov.b32 %r672, %r656;
mov.b32 %r673, %r656;
mov.b32 %r674, %r656;
mov.b32 %r675, %r656;
mov.b32 %r676, %r656;
mov.b32 %r677, %r656;
mov.b32 %r678, %r656;
mov.b32 %r679, %r656;
mov.b32 %r680, %r656;
mov.b32 %r681, %r656;
mov.b32 %r682, %r656;
mov.b32 %r683, %r656;
mov.b32 %r684, %r656;
mov.b32 %r685, %r656;
mov.b32 %r686, %r656;
mov.b32 %r687, %r656;
mov.b32 %r688, %r656;
mov.b32 %r689, %r656;
mov.b32 %r690, %r656;
mov.b32 %r691, %r656;
mov.b32 %r692, %r656;
mov.b32 %r693, %r656;
mov.b32 %r694, %r656;
mov.b32 %r695, %r656;
mov.b32 %r696, %r656;
mov.b32 %r697, %r656;
mov.b32 %r698, %r656;
mov.b32 %r699, %r656;
mov.b32 %r700, %r656;
mov.b32 %r701, %r656;
mov.b32 %r702, %r656;
mov.b32 %r703, %r656;
mov.b32 %r704, %r656;
mov.b32 %r705, %r656;
mov.b32 %r706, %r656;
mov.b32 %r707, %r656;
mov.b32 %r708, %r656;
mov.b32 %r709, %r656;
mov.b32 %r710, %r656;
mov.b32 %r711, %r656;
mov.b32 %r712, %r656;
mov.b32 %r713, %r656;
mov.b32 %r714, %r656;
mov.b32 %r715, %r656;
mov.b32 %r716, %r656;
mov.b32 %r717, %r656;
mov.b32 %r718, %r656;
mov.b32 %r719, %r656;
$L__BB0_13:
and.b32 %r236, %r720, 1;
bfe.u32 %r232, %r720, 1, 1;
shl.b32 %r237, %r236, 15;
add.s32 %r238, %r108, %r237;
mad.lo.s32 %r239, %r236, -32760, %r238;
add.s32 %r231, %r239, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r231], %r232;
@!complete bra.uni waitLoop;
}
shr.u32 %r240, %r238, 4;
add.s32 %r241, %r240, %r58;
and.b32 %r242, %r241, 16383;
cvt.u64.u32 %rd66, %r242;
wgmma.fence.sync.aligned;
or.b64 %rd50, %rd66, 4611686293305294848;
mov.pred %p35, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd50, %rd51, %p35, 1, 1, 0, 0;
add.s64 %rd52, %rd66, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd52, %rd53, %p35, 1, 1, 0, 0;
add.s64 %rd54, %rd66, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd54, %rd55, %p35, 1, 1, 0, 0;
add.s64 %rd56, %rd66, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd56, %rd57, %p35, 1, 1, 0, 0;
add.s64 %rd58, %rd66, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd58, %rd59, %p35, 1, 1, 0, 0;
add.s64 %rd60, %rd66, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd60, %rd61, %p35, 1, 1, 0, 0;
add.s64 %rd62, %rd66, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd62, %rd63, %p35, 1, 1, 0, 0;
add.s64 %rd64, %rd66, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672,%r673,%r674,%r675,%r676,%r677,%r678,%r679,%r680,%r681,%r682,%r683,%r684,%r685,%r686,%r687,%r688,%r689,%r690,%r691,%r692,%r693,%r694,%r695,%r696,%r697,%r698,%r699,%r700,%r701,%r702,%r703,%r704,%r705,%r706,%r707,%r708,%r709,%r710,%r711,%r712,%r713,%r714,%r715,%r716,%r717,%r718,%r719}, %rd64, %rd65, %p35, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r59, %r720, 1;
and.b32 %r243, %r59, 1;
setp.ne.b32 %p39, %r720, 5;
shl.b32 %r244, %r243, 3;
add.s32 %r245, %r108, %r244;
add.s32 %r233, %r245, 98304;
shl.b32 %r246, %r243, 15;
$L__tmp34:
bar.sync 0;
and.pred %p36, %p1, %p39;
@%p36 mbarrier.arrive.expect_tx.shared.b64 _, [%r233], 65536;
bar.sync 0;
elect.sync %r247|%p40, -1;
and.pred %p41, %p39, %p40;
and.pred %p37, %p58, %p41;
add.s32 %r234, %r578, %r246;
@%p37 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r234], [%rd67, {%r655, %r654}], [%r233];
$L__tmp35:
wgmma.wait_group.sync.aligned 0;
$L__tmp36:
bar.sync 0;
elect.sync %r248|%p42, -1;
and.pred %p43, %p39, %p42;
and.pred %p38, %p58, %p43;
@%p38 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r235], [%rd67, {%r655, %r581}], [%r233];
$L__tmp37:
add.s32 %r655, %r655, 128;
setp.eq.b32 %p44, %r59, 6;
mov.b32 %r720, %r59;
@%p44 bra $L__BB0_17;
bra.uni $L__BB0_13;
$L__tmp38:
$L__BB0_17:
add.s32 %r48, %r188, 32768;
add.s32 %r49, %r188, 34816;
add.s32 %r50, %r188, 36864;
add.s32 %r51, %r188, 38912;
add.s32 %r52, %r188, 40960;
add.s32 %r53, %r188, 43008;
add.s32 %r54, %r188, 45056;
add.s32 %r55, %r188, 47104;
$L__tmp39:
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r249];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r99];
and.b32 %r302, %r5, 240;
bfe.u32 %r303, %r5, 4, 4;
and.b32 %r62, %r5, 15;
shl.b32 %r304, %r62, 3;
mul.lo.s32 %r305, %r1, 589824;
or.b32 %r306, %r3, %r303;
or.b32 %r307, %r306, 16;
or.b32 %r308, %r306, 32;
or.b32 %r309, %r306, 48;
or.b32 %r310, %r306, 64;
or.b32 %r311, %r306, 80;
or.b32 %r312, %r306, 96;
or.b32 %r313, %r306, 112;
or.b32 %r314, %r4, %r304;
mad.lo.s32 %r315, %r306, 768, %r314;
mad.lo.s32 %r316, %r307, 768, %r314;
mad.lo.s32 %r317, %r308, 768, %r314;
mad.lo.s32 %r318, %r309, 768, %r314;
mad.lo.s32 %r319, %r310, 768, %r314;
mad.lo.s32 %r320, %r311, 768, %r314;
mad.lo.s32 %r321, %r312, 768, %r314;
mad.lo.s32 %r322, %r313, 768, %r314;
mad.wide.s32 %rd94, %r305, 2, %rd1;
mad.wide.u32 %rd86, %r315, 2, %rd94;
mad.wide.u32 %rd87, %r316, 2, %rd94;
mad.wide.u32 %rd88, %r317, 2, %rd94;
mad.wide.u32 %rd89, %r318, 2, %rd94;
mad.wide.u32 %rd90, %r319, 2, %rd94;
mad.wide.u32 %rd91, %r320, 2, %rd94;
mad.wide.u32 %rd92, %r321, 2, %rd94;
mad.wide.u32 %rd93, %r322, 2, %rd94;
mov.u32 %r270, 0x0;
mov.u32 %r271, 0x0;
mov.u32 %r272, 0x0;
mov.u32 %r273, 0x0;
ld.global.cg.v4.b32 { %r270, %r271, %r272, %r273 }, [ %rd86 + 0 ];
mov.u32 %r274, 0x0;
mov.u32 %r275, 0x0;
mov.u32 %r276, 0x0;
mov.u32 %r277, 0x0;
ld.global.cg.v4.b32 { %r274, %r275, %r276, %r277 }, [ %rd87 + 0 ];
mov.u32 %r278, 0x0;
mov.u32 %r279, 0x0;
mov.u32 %r280, 0x0;
mov.u32 %r281, 0x0;
ld.global.cg.v4.b32 { %r278, %r279, %r280, %r281 }, [ %rd88 + 0 ];
mov.u32 %r282, 0x0;
mov.u32 %r283, 0x0;
mov.u32 %r284, 0x0;
mov.u32 %r285, 0x0;
ld.global.cg.v4.b32 { %r282, %r283, %r284, %r285 }, [ %rd89 + 0 ];
mov.u32 %r286, 0x0;
mov.u32 %r287, 0x0;
mov.u32 %r288, 0x0;
mov.u32 %r289, 0x0;
ld.global.cg.v4.b32 { %r286, %r287, %r288, %r289 }, [ %rd90 + 0 ];
mov.u32 %r290, 0x0;
mov.u32 %r291, 0x0;
mov.u32 %r292, 0x0;
mov.u32 %r293, 0x0;
ld.global.cg.v4.b32 { %r290, %r291, %r292, %r293 }, [ %rd91 + 0 ];
mov.u32 %r294, 0x0;
mov.u32 %r295, 0x0;
mov.u32 %r296, 0x0;
mov.u32 %r297, 0x0;
ld.global.cg.v4.b32 { %r294, %r295, %r296, %r297 }, [ %rd92 + 0 ];
mov.u32 %r298, 0x0;
mov.u32 %r299, 0x0;
mov.u32 %r300, 0x0;
mov.u32 %r301, 0x0;
ld.global.cg.v4.b32 { %r298, %r299, %r300, %r301 }, [ %rd93 + 0 ];
shl.b32 %r323, %r5, 8;
and.b32 %r324, %r323, 30720;
or.b32 %r325, %r324, %r47;
xor.b32 %r326, %r325, %r302;
add.s32 %r327, %r108, %r326;
st.shared.v4.b32 [%r327], {%r270, %r271, %r272, %r273};
st.shared.v4.b32 [%r327+256], {%r274, %r275, %r276, %r277};
st.shared.v4.b32 [%r327+512], {%r278, %r279, %r280, %r281};
st.shared.v4.b32 [%r327+768], {%r282, %r283, %r284, %r285};
st.shared.v4.b32 [%r327+1024], {%r286, %r287, %r288, %r289};
st.shared.v4.b32 [%r327+1280], {%r290, %r291, %r292, %r293};
st.shared.v4.b32 [%r327+1536], {%r294, %r295, %r296, %r297};
st.shared.v4.b32 [%r327+1792], {%r298, %r299, %r300, %r301};
bar.sync 0;
shl.b32 %r328, %r46, 12;
shl.b32 %r329, %r62, 4;
and.b32 %r63, %r5, 224;
shl.b32 %r330, %r63, 3;
or.b32 %r331, %r329, %r330;
xor.b32 %r332, %r331, %r653;
or.b32 %r333, %r332, %r328;
add.s32 %r334, %r108, %r333;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r335, %r336, %r337, %r338}, [%r334];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r339, %r340, %r341, %r342}, [%r334+2048];
xor.b32 %r343, %r333, 32;
add.s32 %r344, %r108, %r343;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r345, %r346, %r347, %r348}, [%r344];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r349, %r350, %r351, %r352}, [%r344+2048];
xor.b32 %r353, %r333, 64;
add.s32 %r354, %r108, %r353;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r355, %r356, %r357, %r358}, [%r354];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r359, %r360, %r361, %r362}, [%r354+2048];
xor.b32 %r363, %r333, 96;
add.s32 %r364, %r108, %r363;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r365, %r366, %r367, %r368}, [%r364];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r369, %r370, %r371, %r372}, [%r364+2048];
mov.b32 {%rs1, %rs2}, %r335;
cvt.f32.bf16 %r373, %rs1;
cvt.f32.bf16 %r374, %rs2;
mul.f32 %r375, %r98, %r374;
mul.f32 %r376, %r98, %r373;
fma.rn.f32 %r377, %r97, %r656, %r376;
fma.rn.f32 %r378, %r97, %r657, %r375;
mov.b32 {%rs3, %rs4}, %r336;
cvt.f32.bf16 %r379, %rs3;
cvt.f32.bf16 %r380, %rs4;
mul.f32 %r381, %r98, %r380;
mul.f32 %r382, %r98, %r379;
fma.rn.f32 %r383, %r97, %r658, %r382;
fma.rn.f32 %r384, %r97, %r659, %r381;
mov.b32 {%rs5, %rs6}, %r337;
cvt.f32.bf16 %r385, %rs5;
cvt.f32.bf16 %r386, %rs6;
mul.f32 %r387, %r98, %r386;
mul.f32 %r388, %r98, %r385;
fma.rn.f32 %r389, %r97, %r660, %r388;
fma.rn.f32 %r390, %r97, %r661, %r387;
mov.b32 {%rs7, %rs8}, %r338;
cvt.f32.bf16 %r391, %rs7;
cvt.f32.bf16 %r392, %rs8;
mul.f32 %r393, %r98, %r392;
mul.f32 %r394, %r98, %r391;
fma.rn.f32 %r395, %r97, %r662, %r394;
fma.rn.f32 %r396, %r97, %r663, %r393;
mov.b32 {%rs9, %rs10}, %r345;
cvt.f32.bf16 %r397, %rs9;
cvt.f32.bf16 %r398, %rs10;
mul.f32 %r399, %r98, %r398;
mul.f32 %r400, %r98, %r397;
fma.rn.f32 %r401, %r97, %r664, %r400;
fma.rn.f32 %r402, %r97, %r665, %r399;
mov.b32 {%rs11, %rs12}, %r346;
cvt.f32.bf16 %r403, %rs11;
cvt.f32.bf16 %r404, %rs12;
mul.f32 %r405, %r98, %r404;
mul.f32 %r406, %r98, %r403;
fma.rn.f32 %r407, %r97, %r666, %r406;
fma.rn.f32 %r408, %r97, %r667, %r405;
mov.b32 {%rs13, %rs14}, %r347;
cvt.f32.bf16 %r409, %rs13;
cvt.f32.bf16 %r410, %rs14;
mul.f32 %r411, %r98, %r410;
mul.f32 %r412, %r98, %r409;
fma.rn.f32 %r413, %r97, %r668, %r412;
fma.rn.f32 %r414, %r97, %r669, %r411;
mov.b32 {%rs15, %rs16}, %r348;
cvt.f32.bf16 %r415, %rs15;
cvt.f32.bf16 %r416, %rs16;
mul.f32 %r417, %r98, %r416;
mul.f32 %r418, %r98, %r415;
fma.rn.f32 %r419, %r97, %r670, %r418;
fma.rn.f32 %r420, %r97, %r671, %r417;
mov.b32 {%rs17, %rs18}, %r355;
cvt.f32.bf16 %r421, %rs17;
cvt.f32.bf16 %r422, %rs18;
mul.f32 %r423, %r98, %r422;
mul.f32 %r424, %r98, %r421;
fma.rn.f32 %r425, %r97, %r672, %r424;
fma.rn.f32 %r426, %r97, %r673, %r423;
mov.b32 {%rs19, %rs20}, %r356;
cvt.f32.bf16 %r427, %rs19;
cvt.f32.bf16 %r428, %rs20;
mul.f32 %r429, %r98, %r428;
mul.f32 %r430, %r98, %r427;
fma.rn.f32 %r431, %r97, %r674, %r430;
fma.rn.f32 %r432, %r97, %r675, %r429;
mov.b32 {%rs21, %rs22}, %r357;
cvt.f32.bf16 %r433, %rs21;
cvt.f32.bf16 %r434, %rs22;
mul.f32 %r435, %r98, %r434;
mul.f32 %r436, %r98, %r433;
fma.rn.f32 %r437, %r97, %r676, %r436;
fma.rn.f32 %r438, %r97, %r677, %r435;
mov.b32 {%rs23, %rs24}, %r358;
cvt.f32.bf16 %r439, %rs23;
cvt.f32.bf16 %r440, %rs24;
mul.f32 %r441, %r98, %r440;
mul.f32 %r442, %r98, %r439;
fma.rn.f32 %r443, %r97, %r678, %r442;
fma.rn.f32 %r444, %r97, %r679, %r441;
mov.b32 {%rs25, %rs26}, %r365;
cvt.f32.bf16 %r445, %rs25;
cvt.f32.bf16 %r446, %rs26;
mul.f32 %r447, %r98, %r446;
mul.f32 %r448, %r98, %r445;
fma.rn.f32 %r449, %r97, %r680, %r448;
fma.rn.f32 %r450, %r97, %r681, %r447;
mov.b32 {%rs27, %rs28}, %r366;
cvt.f32.bf16 %r451, %rs27;
cvt.f32.bf16 %r452, %rs28;
mul.f32 %r453, %r98, %r452;
mul.f32 %r454, %r98, %r451;
fma.rn.f32 %r455, %r97, %r682, %r454;
fma.rn.f32 %r456, %r97, %r683, %r453;
mov.b32 {%rs29, %rs30}, %r367;
cvt.f32.bf16 %r457, %rs29;
cvt.f32.bf16 %r458, %rs30;
mul.f32 %r459, %r98, %r458;
mul.f32 %r460, %r98, %r457;
fma.rn.f32 %r461, %r97, %r684, %r460;
fma.rn.f32 %r462, %r97, %r685, %r459;
mov.b32 {%rs31, %rs32}, %r368;
cvt.f32.bf16 %r463, %rs31;
cvt.f32.bf16 %r464, %rs32;
mul.f32 %r465, %r98, %r464;
mul.f32 %r466, %r98, %r463;
fma.rn.f32 %r467, %r97, %r686, %r466;
fma.rn.f32 %r468, %r97, %r687, %r465;
mov.b32 {%rs33, %rs34}, %r339;
cvt.f32.bf16 %r469, %rs33;
cvt.f32.bf16 %r470, %rs34;
mul.f32 %r471, %r98, %r470;
mul.f32 %r472, %r98, %r469;
fma.rn.f32 %r473, %r97, %r688, %r472;
fma.rn.f32 %r474, %r97, %r689, %r471;
mov.b32 {%rs35, %rs36}, %r340;
cvt.f32.bf16 %r475, %rs35;
cvt.f32.bf16 %r476, %rs36;
mul.f32 %r477, %r98, %r476;
mul.f32 %r478, %r98, %r475;
fma.rn.f32 %r479, %r97, %r690, %r478;
fma.rn.f32 %r480, %r97, %r691, %r477;
mov.b32 {%rs37, %rs38}, %r341;
cvt.f32.bf16 %r481, %rs37;
cvt.f32.bf16 %r482, %rs38;
mul.f32 %r483, %r98, %r482;
mul.f32 %r484, %r98, %r481;
fma.rn.f32 %r485, %r97, %r692, %r484;
fma.rn.f32 %r486, %r97, %r693, %r483;
mov.b32 {%rs39, %rs40}, %r342;
cvt.f32.bf16 %r487, %rs39;
cvt.f32.bf16 %r488, %rs40;
mul.f32 %r489, %r98, %r488;
mul.f32 %r490, %r98, %r487;
fma.rn.f32 %r491, %r97, %r694, %r490;
fma.rn.f32 %r492, %r97, %r695, %r489;
mov.b32 {%rs41, %rs42}, %r349;
cvt.f32.bf16 %r493, %rs41;
cvt.f32.bf16 %r494, %rs42;
mul.f32 %r495, %r98, %r494;
mul.f32 %r496, %r98, %r493;
fma.rn.f32 %r497, %r97, %r696, %r496;
fma.rn.f32 %r498, %r97, %r697, %r495;
mov.b32 {%rs43, %rs44}, %r350;
cvt.f32.bf16 %r499, %rs43;
cvt.f32.bf16 %r500, %rs44;
mul.f32 %r501, %r98, %r500;
mul.f32 %r502, %r98, %r499;
fma.rn.f32 %r503, %r97, %r698, %r502;
fma.rn.f32 %r504, %r97, %r699, %r501;
mov.b32 {%rs45, %rs46}, %r351;
cvt.f32.bf16 %r505, %rs45;
cvt.f32.bf16 %r506, %rs46;
mul.f32 %r507, %r98, %r506;
mul.f32 %r508, %r98, %r505;
fma.rn.f32 %r509, %r97, %r700, %r508;
fma.rn.f32 %r510, %r97, %r701, %r507;
mov.b32 {%rs47, %rs48}, %r352;
cvt.f32.bf16 %r511, %rs47;
cvt.f32.bf16 %r512, %rs48;
mul.f32 %r513, %r98, %r512;
mul.f32 %r514, %r98, %r511;
fma.rn.f32 %r515, %r97, %r702, %r514;
fma.rn.f32 %r516, %r97, %r703, %r513;
mov.b32 {%rs49, %rs50}, %r359;
cvt.f32.bf16 %r517, %rs49;
cvt.f32.bf16 %r518, %rs50;
mul.f32 %r519, %r98, %r518;
mul.f32 %r520, %r98, %r517;
fma.rn.f32 %r521, %r97, %r704, %r520;
fma.rn.f32 %r522, %r97, %r705, %r519;
mov.b32 {%rs51, %rs52}, %r360;
cvt.f32.bf16 %r523, %rs51;
cvt.f32.bf16 %r524, %rs52;
mul.f32 %r525, %r98, %r524;
mul.f32 %r526, %r98, %r523;
fma.rn.f32 %r527, %r97, %r706, %r526;
fma.rn.f32 %r528, %r97, %r707, %r525;
mov.b32 {%rs53, %rs54}, %r361;
cvt.f32.bf16 %r529, %rs53;
cvt.f32.bf16 %r530, %rs54;
mul.f32 %r531, %r98, %r530;
mul.f32 %r532, %r98, %r529;
fma.rn.f32 %r533, %r97, %r708, %r532;
fma.rn.f32 %r534, %r97, %r709, %r531;
mov.b32 {%rs55, %rs56}, %r362;
cvt.f32.bf16 %r535, %rs55;
cvt.f32.bf16 %r536, %rs56;
mul.f32 %r537, %r98, %r536;
mul.f32 %r538, %r98, %r535;
fma.rn.f32 %r539, %r97, %r710, %r538;
fma.rn.f32 %r540, %r97, %r711, %r537;
mov.b32 {%rs57, %rs58}, %r369;
cvt.f32.bf16 %r541, %rs57;
cvt.f32.bf16 %r542, %rs58;
mul.f32 %r543, %r98, %r542;
mul.f32 %r544, %r98, %r541;
fma.rn.f32 %r545, %r97, %r712, %r544;
fma.rn.f32 %r546, %r97, %r713, %r543;
mov.b32 {%rs59, %rs60}, %r370;
cvt.f32.bf16 %r547, %rs59;
cvt.f32.bf16 %r548, %rs60;
mul.f32 %r549, %r98, %r548;
mul.f32 %r550, %r98, %r547;
fma.rn.f32 %r551, %r97, %r714, %r550;
fma.rn.f32 %r552, %r97, %r715, %r549;
mov.b32 {%rs61, %rs62}, %r371;
cvt.f32.bf16 %r553, %rs61;
cvt.f32.bf16 %r554, %rs62;
mul.f32 %r555, %r98, %r554;
mul.f32 %r556, %r98, %r553;
fma.rn.f32 %r557, %r97, %r716, %r556;
fma.rn.f32 %r558, %r97, %r717, %r555;
mov.b32 {%rs63, %rs64}, %r372;
cvt.f32.bf16 %r559, %rs63;
cvt.f32.bf16 %r560, %rs64;
mul.f32 %r561, %r98, %r560;
mul.f32 %r562, %r98, %r559;
fma.rn.f32 %r563, %r97, %r718, %r562;
fma.rn.f32 %r564, %r97, %r719, %r561;
cvt.rn.bf16x2.f32 %r64, %r378, %r377;
cvt.rn.bf16x2.f32 %r65, %r384, %r383;
cvt.rn.bf16x2.f32 %r66, %r390, %r389;
cvt.rn.bf16x2.f32 %r67, %r396, %r395;
cvt.rn.bf16x2.f32 %r68, %r402, %r401;
cvt.rn.bf16x2.f32 %r69, %r408, %r407;
cvt.rn.bf16x2.f32 %r70, %r414, %r413;
cvt.rn.bf16x2.f32 %r71, %r420, %r419;
cvt.rn.bf16x2.f32 %r72, %r426, %r425;
cvt.rn.bf16x2.f32 %r73, %r432, %r431;
cvt.rn.bf16x2.f32 %r74, %r438, %r437;
cvt.rn.bf16x2.f32 %r75, %r444, %r443;
cvt.rn.bf16x2.f32 %r76, %r450, %r449;
cvt.rn.bf16x2.f32 %r77, %r456, %r455;
cvt.rn.bf16x2.f32 %r78, %r462, %r461;
cvt.rn.bf16x2.f32 %r79, %r468, %r467;
cvt.rn.bf16x2.f32 %r80, %r474, %r473;
cvt.rn.bf16x2.f32 %r81, %r480, %r479;
cvt.rn.bf16x2.f32 %r82, %r486, %r485;
cvt.rn.bf16x2.f32 %r83, %r492, %r491;
cvt.rn.bf16x2.f32 %r84, %r498, %r497;
cvt.rn.bf16x2.f32 %r85, %r504, %r503;
cvt.rn.bf16x2.f32 %r86, %r510, %r509;
cvt.rn.bf16x2.f32 %r87, %r516, %r515;
cvt.rn.bf16x2.f32 %r88, %r522, %r521;
cvt.rn.bf16x2.f32 %r89, %r528, %r527;
cvt.rn.bf16x2.f32 %r90, %r534, %r533;
cvt.rn.bf16x2.f32 %r91, %r540, %r539;
cvt.rn.bf16x2.f32 %r92, %r546, %r545;
cvt.rn.bf16x2.f32 %r93, %r552, %r551;
cvt.rn.bf16x2.f32 %r94, %r558, %r557;
cvt.rn.bf16x2.f32 %r95, %r564, %r563;
@%p26 bra $L__BB0_18;
bra.uni $L__BB0_19;
$L__BB0_18:
bar.sync 0;
shl.b32 %r565, %r62, 7;
shl.b32 %r566, %r63, 6;
or.b32 %r567, %r565, %r47;
xor.b32 %r568, %r567, %r653;
or.b32 %r569, %r568, %r566;
add.s32 %r570, %r108, %r569;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r570], {%r64, %r65, %r66, %r67};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r570+16384], {%r80, %r81, %r82, %r83};
xor.b32 %r571, %r569, 32;
add.s32 %r572, %r108, %r571;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r572], {%r68, %r69, %r70, %r71};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r572+16384], {%r84, %r85, %r86, %r87};
xor.b32 %r573, %r569, 64;
add.s32 %r574, %r108, %r573;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r574], {%r72, %r73, %r74, %r75};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r574+16384], {%r88, %r89, %r90, %r91};
xor.b32 %r575, %r569, 96;
add.s32 %r576, %r108, %r575;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r576], {%r76, %r77, %r78, %r79};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r576+16384], {%r92, %r93, %r94, %r95};
$L__BB0_19:
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r48], {%r64, %r65, %r66, %r67};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r49], {%r68, %r69, %r70, %r71};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r50], {%r72, %r73, %r74, %r75};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r51], {%r76, %r77, %r78, %r79};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r52], {%r80, %r81, %r82, %r83};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r53], {%r84, %r85, %r86, %r87};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r88, %r89, %r90, %r91};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r55], {%r92, %r93, %r94, %r95};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_21;
bar.sync 0;
elect.sync %r579|%p55, -1;
and.pred %p54, %p58, %p55;
or.b32 %r577, %r250, %r4;
@%p54 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd95, {%r577, %r654}], [%r578];
cp.async.bulk.commit_group;
$L__BB0_21:
bar.sync 0;
elect.sync %r584|%p57, -1;
and.pred %p56, %p58, %p57;
@%p56 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd95, {%r580, %r581}], [%r582];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r583, 0;
$L__tmp40:
ret;
$L__tmp41:
$L__func_end0:
}
"#

private def lines (code : String) : List String :=
  (code.splitOn "\n").filter (!·.isEmpty)

def matchesCode (bank : Bank) (phase : Phase) (code : Code) : Bool :=
  let (ttgir, ptx) := match phase with
    | .seed => (seedGraph bank, seedAssembly bank)
    | .initial => (initialGraph bank, initialAssembly bank)
    | .round => (roundGraph bank, roundAssembly bank)
  QKVTTIR.normalizeTTIR code.ttgir == some (lines ttgir) &&
    QKVTTIR.normalizePTX code.ptx == lines ptx

structure Program (bank : Bank) where
  seed : Code
  initial : Code
  round : Code
  seedChecked : matchesCode bank .seed seed = true
  initialChecked : matchesCode bank .initial initial = true
  roundChecked : matchesCode bank .round round = true

def Program.check (bank : Bank) (seed initial round : Code) : Option (Program bank) :=
  if hs : matchesCode bank .seed seed = true then
    if hi : matchesCode bank .initial initial = true then
      if hr : matchesCode bank .round round = true then some ⟨seed, initial, round, hs, hi, hr⟩ else none
    else none
  else none

abbrev Programs := (bank : Bank) → Program bank

def Programs.initialize (programs : Programs) : Initial := fun State rt bank normalize state =>
  let program := programs bank
  if matchesCode bank .seed program.seed && matchesCode bank .initial program.initial then
    fusedInitial State rt bank normalize state
  else splitInitial reference State rt bank normalize state

theorem Programs.initialize_eq (programs : Programs) :
    programs.initialize = splitInitial reference := by
  funext State rt bank normalize state
  simp [Programs.initialize, (programs bank).seedChecked, (programs bank).initialChecked, fused_initial_eq]

def Programs.run (programs : Programs) : Round := fun State rt bank c state =>
  let program := programs bank
  if matchesCode bank .seed program.seed && matchesCode bank .round program.round then
    execute State rt bank c state
  else splitRound reference State rt bank c state

theorem Programs.run_eq (programs : Programs) : programs.run = splitRound reference := by
  funext State rt bank c state
  simp [Programs.run, (programs bank).seedChecked, (programs bank).roundChecked, execute_eq]

#axiomsClean Programs.initialize_eq
#axiomsClean Programs.run_eq

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.Cooperative
