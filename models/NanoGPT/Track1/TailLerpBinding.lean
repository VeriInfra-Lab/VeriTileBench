import models.NanoGPT.Track1.TailLerp
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Triton 3.6.0 artifacts for the full-tile Hopper path and the
two-launch reference. Keep FP operands, lane layouts, pointers and stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TailLerp

structure Artifacts where
  referenceIR : String
  candidateIR : String
  referencePTX : String
  candidatePTX : String
  deriving BEq

private def referenceIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_lerp_upcast_kernel(%SRC: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}, %rate: f32) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<4096xf32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xbf16, #blocked>
%pid = tt.get_program_id x : i32
%offs = arith.muli %pid, 4096 : i32
%offs_1 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
%offs_2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked1>
%offs_3 = tt.splat %offs : i32 -> tensor<4096xi32, #blocked>
%offs_4 = tt.splat %offs : i32 -> tensor<4096xi32, #blocked1>
%offs_5 = arith.addi %offs_3, %offs_1 : tensor<4096xi32, #blocked>
%offs_6 = arith.addi %offs_4, %offs_2 : tensor<4096xi32, #blocked1>
%m = tt.splat %n_elements : i32 -> tensor<4096xi32, #blocked>
%m_7 = tt.splat %n_elements : i32 -> tensor<4096xi32, #blocked1>
%m_8 = arith.cmpi slt, %offs_5, %m : tensor<4096xi32, #blocked>
%m_9 = arith.cmpi slt, %offs_6, %m_7 : tensor<4096xi32, #blocked1>
%s = tt.splat %DST : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
%s_10 = tt.splat %DST : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked1>
%s_11 = tt.addptr %s, %offs_5 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
%s_12 = tt.addptr %s_10, %offs_6 : tensor<4096x!tt.ptr<f32>, #blocked1>, tensor<4096xi32, #blocked1>
%s_13 = tt.load %s_11, %m_8, %cst : tensor<4096x!tt.ptr<f32>, #blocked>
%e = tt.splat %SRC : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>, #blocked>
%e_14 = tt.addptr %e, %offs_5 : tensor<4096x!tt.ptr<bf16>, #blocked>, tensor<4096xi32, #blocked>
%e_15 = tt.load %e_14, %m_8, %cst_0 : tensor<4096x!tt.ptr<bf16>, #blocked>
%e_16 = arith.extf %e_15 : tensor<4096xbf16, #blocked> to tensor<4096xf32, #blocked>
%0 = arith.subf %e_16, %s_13 : tensor<4096xf32, #blocked>
%1 = tt.splat %rate : f32 -> tensor<4096xf32, #blocked>
%2 = math.fma %1, %0, %s_13 : tensor<4096xf32, #blocked>
%3 = ttg.convert_layout %2 : tensor<4096xf32, #blocked> -> tensor<4096xf32, #blocked1>
tt.store %s_12, %3, %m_9 : tensor<4096x!tt.ptr<f32>, #blocked1>
tt.return
}
}"#

private def candidateIR : String := r#"#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_tail_lerp_pair(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %first: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %second: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rate_first: f32, %rate_second: f32) attributes {noinline = false} {
%lane = tt.get_program_id x : i32
%lane_0 = arith.muli %lane, 4096 : i32
%lane_1 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
%lane_2 = tt.splat %lane_0 : i32 -> tensor<4096xi32, #blocked>
%lane_3 = arith.addi %lane_2, %lane_1 : tensor<4096xi32, #blocked>
%weight = tt.splat %src : !tt.ptr<bf16> -> tensor<4096x!tt.ptr<bf16>, #blocked>
%weight_4 = tt.addptr %weight, %lane_3 : tensor<4096x!tt.ptr<bf16>, #blocked>, tensor<4096xi32, #blocked>
%weight_5 = tt.load %weight_4 : tensor<4096x!tt.ptr<bf16>, #blocked>
%weight_6 = arith.extf %weight_5 : tensor<4096xbf16, #blocked> to tensor<4096xf32, #blocked>
%old_first = tt.splat %first : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
%old_first_7 = tt.addptr %old_first, %lane_3 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
%old_first_8 = tt.load %old_first_7 : tensor<4096x!tt.ptr<f32>, #blocked>
%0 = arith.subf %weight_6, %old_first_8 : tensor<4096xf32, #blocked>
%1 = tt.splat %rate_first : f32 -> tensor<4096xf32, #blocked>
%2 = math.fma %1, %0, %old_first_8 : tensor<4096xf32, #blocked>
tt.store %old_first_7, %2 : tensor<4096x!tt.ptr<f32>, #blocked>
%old_second = tt.splat %second : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
%old_second_9 = tt.addptr %old_second, %lane_3 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
%old_second_10 = tt.load %old_second_9 : tensor<4096x!tt.ptr<f32>, #blocked>
%3 = arith.subf %weight_6, %old_second_10 : tensor<4096xf32, #blocked>
%4 = tt.splat %rate_second : f32 -> tensor<4096xf32, #blocked>
%5 = math.fma %4, %3, %old_second_10 : tensor<4096xf32, #blocked>
tt.store %old_second_9, %5 : tensor<4096x!tt.ptr<f32>, #blocked>
tt.return
}
}"#

private def referencePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _lerp_upcast_kernel(
.param .u64 .ptr .global .align 1 _lerp_upcast_kernel_param_0,
.param .u64 .ptr .global .align 1 _lerp_upcast_kernel_param_1,
.param .u32 _lerp_upcast_kernel_param_2,
.param .f32 _lerp_upcast_kernel_param_3,
.param .u64 .ptr .global .align 1 _lerp_upcast_kernel_param_4,
.param .u64 .ptr .global .align 1 _lerp_upcast_kernel_param_5
)
.reqntid 256
{
.reg .pred %p<7>;
.reg .b16 %rs<17>;
.reg .b32 %r<116>;
.reg .b64 %rd<13>;
$L__func_begin0:
ld.param.b64 %rd11, [_lerp_upcast_kernel_param_0];
ld.param.b64 %rd12, [_lerp_upcast_kernel_param_1];
$L__tmp0:
mov.u32 %r42, %ctaid.x;
shl.b32 %r43, %r42, 12;
ld.param.b32 %r44, [_lerp_upcast_kernel_param_2];
ld.param.b32 %r45, [_lerp_upcast_kernel_param_3];
mov.u32 %r46, %tid.x;
and.b32 %r47, %r46, 255;
shl.b32 %r48, %r47, 3;
shl.b32 %r49, %r47, 2;
or.b32 %r50, %r48, %r43;
or.b32 %r51, %r50, 2048;
or.b32 %r52, %r49, %r43;
or.b32 %r53, %r52, 1024;
or.b32 %r54, %r52, 2048;
or.b32 %r55, %r52, 3072;
setp.lt.s32 %p1, %r50, %r44;
setp.lt.s32 %p2, %r51, %r44;
setp.lt.s32 %p3, %r52, %r44;
setp.lt.s32 %p4, %r53, %r44;
setp.lt.s32 %p5, %r54, %r44;
setp.lt.s32 %p6, %r55, %r44;
mad.wide.s32 %rd1, %r50, 4, %rd12;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd1, 8192;
add.s64 %rd4, %rd1, 8208;
mad.wide.s32 %rd7, %r52, 4, %rd12;
add.s64 %rd8, %rd7, 4096;
add.s64 %rd9, %rd7, 8192;
add.s64 %rd10, %rd7, 12288;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mad.wide.s32 %rd5, %r50, 2, %rd11;
add.s64 %rd6, %rd5, 4096;
mov.u32 %r18, %r5;
mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
mov.u32 %r22, %r5;
mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
@%p2 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
mov.b32 {%rs1, %rs2}, %r19;
cvt.f32.bf16 %r56, %rs2;
cvt.f32.bf16 %r57, %rs1;
mov.b32 {%rs3, %rs4}, %r18;
cvt.f32.bf16 %r58, %rs4;
cvt.f32.bf16 %r59, %rs3;
sub.f32 %r60, %r59, %r1;
sub.f32 %r61, %r58, %r2;
sub.f32 %r62, %r57, %r3;
sub.f32 %r63, %r56, %r4;
fma.rn.f32 %r64, %r45, %r63, %r4;
fma.rn.f32 %r65, %r45, %r62, %r3;
fma.rn.f32 %r66, %r45, %r61, %r2;
fma.rn.f32 %r67, %r45, %r60, %r1;
mov.b32 {%rs5, %rs6}, %r21;
cvt.f32.bf16 %r68, %rs6;
cvt.f32.bf16 %r69, %rs5;
mov.b32 {%rs7, %rs8}, %r20;
cvt.f32.bf16 %r70, %rs8;
cvt.f32.bf16 %r71, %rs7;
sub.f32 %r72, %r71, %r6;
sub.f32 %r73, %r70, %r7;
sub.f32 %r74, %r69, %r8;
sub.f32 %r75, %r68, %r9;
fma.rn.f32 %r76, %r45, %r75, %r9;
fma.rn.f32 %r77, %r45, %r74, %r8;
fma.rn.f32 %r78, %r45, %r73, %r7;
fma.rn.f32 %r79, %r45, %r72, %r6;
mov.b32 {%rs9, %rs10}, %r23;
cvt.f32.bf16 %r80, %rs10;
cvt.f32.bf16 %r81, %rs9;
mov.b32 {%rs11, %rs12}, %r22;
cvt.f32.bf16 %r82, %rs12;
cvt.f32.bf16 %r83, %rs11;
sub.f32 %r84, %r83, %r10;
sub.f32 %r85, %r82, %r11;
sub.f32 %r86, %r81, %r12;
sub.f32 %r87, %r80, %r13;
fma.rn.f32 %r88, %r45, %r87, %r13;
fma.rn.f32 %r89, %r45, %r86, %r12;
fma.rn.f32 %r90, %r45, %r85, %r11;
fma.rn.f32 %r91, %r45, %r84, %r10;
mov.b32 {%rs13, %rs14}, %r25;
cvt.f32.bf16 %r92, %rs14;
cvt.f32.bf16 %r93, %rs13;
mov.b32 {%rs15, %rs16}, %r24;
cvt.f32.bf16 %r94, %rs16;
cvt.f32.bf16 %r95, %rs15;
sub.f32 %r96, %r95, %r14;
sub.f32 %r97, %r94, %r15;
sub.f32 %r98, %r93, %r16;
sub.f32 %r99, %r92, %r17;
fma.rn.f32 %r100, %r45, %r99, %r17;
fma.rn.f32 %r101, %r45, %r98, %r16;
fma.rn.f32 %r102, %r45, %r97, %r15;
fma.rn.f32 %r103, %r45, %r96, %r14;
shl.b32 %r104, %r47, 4;
mov.b32 %r105, global_smem;
add.s32 %r106, %r105, %r104;
st.shared.v4.b32 [%r106], {%r67, %r66, %r65, %r64};
xor.b32 %r107, %r104, 64;
add.s32 %r108, %r105, %r107;
st.shared.v4.b32 [%r108+4096], {%r79, %r78, %r77, %r76};
bar.sync 0;
shl.b32 %r109, %r46, 3;
and.b32 %r110, %r109, 2032;
and.b32 %r111, %r46, 1;
neg.s32 %r112, %r111;
and.b32 %r113, %r112, 4160;
xor.b32 %r114, %r113, %r110;
add.s32 %r115, %r105, %r114;
ld.shared.v4.b32 {%r26, %r27, %r28, %r29}, [%r115];
ld.shared.v4.b32 {%r30, %r31, %r32, %r33}, [%r115+2048];
bar.sync 0;
st.shared.v4.b32 [%r106], {%r91, %r90, %r89, %r88};
st.shared.v4.b32 [%r108+4096], {%r103, %r102, %r101, %r100};
bar.sync 0;
ld.shared.v4.b32 {%r34, %r35, %r36, %r37}, [%r115];
ld.shared.v4.b32 {%r38, %r39, %r40, %r41}, [%r115+2048];
@%p3 st.global.v4.b32 [ %rd7 + 0 ], { %r26, %r27, %r28, %r29 };
@%p4 st.global.v4.b32 [ %rd8 + 0 ], { %r30, %r31, %r32, %r33 };
@%p5 st.global.v4.b32 [ %rd9 + 0 ], { %r34, %r35, %r36, %r37 };
@%p6 st.global.v4.b32 [ %rd10 + 0 ], { %r38, %r39, %r40, %r41 };
ret;
$L__tmp1:
$L__func_end0:
}"#

private def candidatePTX : String := r#".version 8.7
.target sm_90a
.address_size 64
.visible .entry _tail_lerp_pair(
.param .u64 .ptr .global .align 1 _tail_lerp_pair_param_0,
.param .u64 .ptr .global .align 1 _tail_lerp_pair_param_1,
.param .u64 .ptr .global .align 1 _tail_lerp_pair_param_2,
.param .f32 _tail_lerp_pair_param_3,
.param .f32 _tail_lerp_pair_param_4,
.param .u64 .ptr .global .align 1 _tail_lerp_pair_param_5,
.param .u64 .ptr .global .align 1 _tail_lerp_pair_param_6
)
.reqntid 256
{
.reg .b16 %rs<17>;
.reg .b32 %r<131>;
.reg .b64 %rd<18>;
$L__func_begin0:
ld.param.b64 %rd13, [_tail_lerp_pair_param_0];
ld.param.b64 %rd14, [_tail_lerp_pair_param_1];
$L__tmp0:
mov.u32 %r73, %ctaid.x;
shl.b32 %r74, %r73, 12;
ld.param.b64 %rd15, [_tail_lerp_pair_param_2];
ld.param.b32 %r75, [_tail_lerp_pair_param_3];
mov.u32 %r76, %tid.x;
shl.b32 %r77, %r76, 2;
ld.param.b32 %r78, [_tail_lerp_pair_param_4];
and.b32 %r79, %r77, 1020;
or.b32 %r80, %r79, %r74;
or.b32 %r81, %r77, %r74;
or.b32 %r82, %r81, 3072;
mad.wide.s32 %rd1, %r80, 2, %rd13;
add.s64 %rd2, %rd1, 2048;
add.s64 %rd3, %rd1, 4096;
mad.wide.s32 %rd4, %r82, 2, %rd13;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
mov.b32 {%rs3, %rs4}, %r2;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v2.b32 { %r3, %r4 }, [ %rd2 + 0 ];
mov.b32 {%rs5, %rs6}, %r3;
mov.b32 {%rs7, %rs8}, %r4;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
ld.global.v2.b32 { %r5, %r6 }, [ %rd3 + 0 ];
mov.b32 {%rs9, %rs10}, %r5;
mov.b32 {%rs11, %rs12}, %r6;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v2.b32 { %r7, %r8 }, [ %rd4 + 0 ];
mov.b32 {%rs13, %rs14}, %r7;
mov.b32 {%rs15, %rs16}, %r8;
cvt.f32.bf16 %r83, %rs1;
cvt.f32.bf16 %r84, %rs2;
cvt.f32.bf16 %r85, %rs3;
cvt.f32.bf16 %r86, %rs4;
cvt.f32.bf16 %r87, %rs5;
cvt.f32.bf16 %r88, %rs6;
cvt.f32.bf16 %r89, %rs7;
cvt.f32.bf16 %r90, %rs8;
cvt.f32.bf16 %r91, %rs9;
cvt.f32.bf16 %r92, %rs10;
cvt.f32.bf16 %r93, %rs11;
cvt.f32.bf16 %r94, %rs12;
cvt.f32.bf16 %r95, %rs13;
cvt.f32.bf16 %r96, %rs14;
cvt.f32.bf16 %r97, %rs15;
cvt.f32.bf16 %r98, %rs16;
mul.wide.s32 %rd16, %r80, 4;
add.s64 %rd5, %rd14, %rd16;
add.s64 %rd6, %rd5, 4096;
add.s64 %rd7, %rd5, 8192;
mul.wide.s32 %rd17, %r82, 4;
add.s64 %rd8, %rd14, %rd17;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd5 + 0 ];
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd6 + 0 ];
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
ld.global.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd7 + 0 ];
mov.u32 %r21, 0x0;
mov.u32 %r22, 0x0;
mov.u32 %r23, 0x0;
mov.u32 %r24, 0x0;
ld.global.v4.b32 { %r21, %r22, %r23, %r24 }, [ %rd8 + 0 ];
sub.f32 %r99, %r83, %r9;
sub.f32 %r100, %r84, %r10;
sub.f32 %r101, %r85, %r11;
sub.f32 %r102, %r86, %r12;
sub.f32 %r103, %r87, %r13;
sub.f32 %r104, %r88, %r14;
sub.f32 %r105, %r89, %r15;
sub.f32 %r106, %r90, %r16;
sub.f32 %r107, %r91, %r17;
sub.f32 %r108, %r92, %r18;
sub.f32 %r109, %r93, %r19;
sub.f32 %r110, %r94, %r20;
sub.f32 %r111, %r95, %r21;
sub.f32 %r112, %r96, %r22;
sub.f32 %r113, %r97, %r23;
sub.f32 %r114, %r98, %r24;
fma.rn.f32 %r25, %r75, %r99, %r9;
fma.rn.f32 %r26, %r75, %r100, %r10;
fma.rn.f32 %r27, %r75, %r101, %r11;
fma.rn.f32 %r28, %r75, %r102, %r12;
fma.rn.f32 %r29, %r75, %r103, %r13;
fma.rn.f32 %r30, %r75, %r104, %r14;
fma.rn.f32 %r31, %r75, %r105, %r15;
fma.rn.f32 %r32, %r75, %r106, %r16;
fma.rn.f32 %r33, %r75, %r107, %r17;
fma.rn.f32 %r34, %r75, %r108, %r18;
fma.rn.f32 %r35, %r75, %r109, %r19;
fma.rn.f32 %r36, %r75, %r110, %r20;
fma.rn.f32 %r37, %r75, %r111, %r21;
fma.rn.f32 %r38, %r75, %r112, %r22;
fma.rn.f32 %r39, %r75, %r113, %r23;
fma.rn.f32 %r40, %r75, %r114, %r24;
st.global.v4.b32 [ %rd5 + 0 ], { %r25, %r26, %r27, %r28 };
st.global.v4.b32 [ %rd6 + 0 ], { %r29, %r30, %r31, %r32 };
st.global.v4.b32 [ %rd7 + 0 ], { %r33, %r34, %r35, %r36 };
st.global.v4.b32 [ %rd8 + 0 ], { %r37, %r38, %r39, %r40 };
add.s64 %rd9, %rd15, %rd16;
add.s64 %rd10, %rd9, 4096;
add.s64 %rd11, %rd9, 8192;
add.s64 %rd12, %rd15, %rd17;
mov.u32 %r41, 0x0;
mov.u32 %r42, 0x0;
mov.u32 %r43, 0x0;
mov.u32 %r44, 0x0;
ld.global.v4.b32 { %r41, %r42, %r43, %r44 }, [ %rd9 + 0 ];
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
ld.global.v4.b32 { %r45, %r46, %r47, %r48 }, [ %rd10 + 0 ];
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
ld.global.v4.b32 { %r49, %r50, %r51, %r52 }, [ %rd11 + 0 ];
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
ld.global.v4.b32 { %r53, %r54, %r55, %r56 }, [ %rd12 + 0 ];
sub.f32 %r115, %r83, %r41;
sub.f32 %r116, %r84, %r42;
sub.f32 %r117, %r85, %r43;
sub.f32 %r118, %r86, %r44;
sub.f32 %r119, %r87, %r45;
sub.f32 %r120, %r88, %r46;
sub.f32 %r121, %r89, %r47;
sub.f32 %r122, %r90, %r48;
sub.f32 %r123, %r91, %r49;
sub.f32 %r124, %r92, %r50;
sub.f32 %r125, %r93, %r51;
sub.f32 %r126, %r94, %r52;
sub.f32 %r127, %r95, %r53;
sub.f32 %r128, %r96, %r54;
sub.f32 %r129, %r97, %r55;
sub.f32 %r130, %r98, %r56;
fma.rn.f32 %r57, %r78, %r115, %r41;
fma.rn.f32 %r58, %r78, %r116, %r42;
fma.rn.f32 %r59, %r78, %r117, %r43;
fma.rn.f32 %r60, %r78, %r118, %r44;
fma.rn.f32 %r61, %r78, %r119, %r45;
fma.rn.f32 %r62, %r78, %r120, %r46;
fma.rn.f32 %r63, %r78, %r121, %r47;
fma.rn.f32 %r64, %r78, %r122, %r48;
fma.rn.f32 %r65, %r78, %r123, %r49;
fma.rn.f32 %r66, %r78, %r124, %r50;
fma.rn.f32 %r67, %r78, %r125, %r51;
fma.rn.f32 %r68, %r78, %r126, %r52;
fma.rn.f32 %r69, %r78, %r127, %r53;
fma.rn.f32 %r70, %r78, %r128, %r54;
fma.rn.f32 %r71, %r78, %r129, %r55;
fma.rn.f32 %r72, %r78, %r130, %r56;
st.global.v4.b32 [ %rd9 + 0 ], { %r57, %r58, %r59, %r60 };
st.global.v4.b32 [ %rd10 + 0 ], { %r61, %r62, %r63, %r64 };
st.global.v4.b32 [ %rd11 + 0 ], { %r65, %r66, %r67, %r68 };
st.global.v4.b32 [ %rd12 + 0 ], { %r69, %r70, %r71, %r72 };
ret;
$L__tmp1:
$L__func_end0:
}"#

def recognized (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (referenceIR.splitOn "\n") &&
    QKVTTIR.normalizeTTIR a.candidateIR == some (candidateIR.splitOn "\n") &&
    QKVTTIR.normalizePTX a.referencePTX == referencePTX.splitOn "\n" &&
    QKVTTIR.normalizePTX a.candidatePTX == candidatePTX.splitOn "\n"

structure Certificate (a : Artifacts) : Type where
  matched : recognized a = true
  laneCoverage : ∀ {tiles} (i : Fin (tiles * tileSize)), (owner i).offset = i.val
  uniqueLane : ∀ {tiles} (p : Lane tiles), owner ⟨p.offset, lane_live p⟩ = p
  exactIndex : ∀ {tiles} (p : Lane tiles), tiles * tileSize < 2 ^ 31 →
    (Int32.ofNat p.offset).toInt = p.offset
  orderedStores : ∀ eligible elements inputs first second i,
    candidate eligible elements inputs first second i = reference inputs first second i

def certify (a : Artifacts) : Option (Certificate a) :=
  if matched : recognized a = true then
    some ⟨matched, owner_offset, offset_owner, index_i32, candidate_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.TailLerp
