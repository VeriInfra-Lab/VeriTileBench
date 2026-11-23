import models.NanoGPT.Track1.SampledDensify
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact compiler artifacts for the original gather and the aligned
Hopper copy. Only the three stride constants depend on the candidate count. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Densify
open SampledRows (Count ClassID Channel)

private def referenceIR : String := r#"
module {
tt.func public @_densify_kernel(%SRC: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %POS: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %DST: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %R: i32 {tt.divisibility = 16 : i32}, %V: i32 {tt.divisibility = 16 : i32}, %src_stride_r: i32 {tt.divisibility = 16 : i32}, %dst_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%src = arith.constant dense<0.000000e+00> : tensor<32x256xbf16>
%cst = arith.constant dense<0> : tensor<256xi32>
%pos = arith.constant dense<-1> : tensor<256xi32>
%offs_v = tt.get_program_id x : i32
%offs_v_0 = arith.muli %offs_v, 256 : i32
%offs_v_1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
%offs_v_2 = tt.splat %offs_v_0 : i32 -> tensor<256xi32>
%offs_v_3 = arith.addi %offs_v_2, %offs_v_1 : tensor<256xi32>
%offs_r = tt.get_program_id y : i32
%offs_r_4 = arith.muli %offs_r, 32 : i32
%offs_r_5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%offs_r_6 = tt.splat %offs_r_4 : i32 -> tensor<32xi32>
%offs_r_7 = arith.addi %offs_r_6, %offs_r_5 : tensor<32xi32>
%0 = tt.splat %V : i32 -> tensor<256xi32>
%1 = arith.cmpi slt, %offs_v_3, %0 : tensor<256xi32>
%2 = tt.splat %R : i32 -> tensor<32xi32>
%3 = arith.cmpi slt, %offs_r_7, %2 : tensor<32xi32>
%pos_8 = tt.splat %POS : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
%pos_9 = tt.addptr %pos_8, %offs_v_3 : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
%pos_10 = tt.load %pos_9, %1, %pos : tensor<256x!tt.ptr<i32>>
%keep = arith.cmpi sge, %pos_10, %cst : tensor<256xi32>
%p = arith.select %keep, %pos_10, %cst : tensor<256xi1>, tensor<256xi32>
%p_11 = arith.extsi %p : tensor<256xi32> to tensor<256xi64>
%src_12 = tt.expand_dims %3 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
%src_13 = tt.expand_dims %keep {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
%src_14 = tt.broadcast %src_12 : tensor<32x1xi1> -> tensor<32x256xi1>
%src_15 = tt.broadcast %src_13 : tensor<1x256xi1> -> tensor<32x256xi1>
%src_16 = arith.andi %src_14, %src_15 : tensor<32x256xi1>
%src_17 = tt.expand_dims %offs_r_7 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
%src_18 = arith.extsi %src_17 : tensor<32x1xi32> to tensor<32x1xi64>
%src_19 = arith.extsi %src_stride_r : i32 to i64
%src_20 = tt.splat %src_19 : i64 -> tensor<32x1xi64>
%src_21 = arith.muli %src_18, %src_20 : tensor<32x1xi64>
%src_22 = tt.splat %SRC : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
%src_23 = tt.addptr %src_22, %src_21 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi64>
%src_24 = tt.expand_dims %p_11 {axis = 0 : i32} : tensor<256xi64> -> tensor<1x256xi64>
%src_25 = tt.broadcast %src_23 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x256x!tt.ptr<bf16>>
%src_26 = tt.broadcast %src_24 : tensor<1x256xi64> -> tensor<32x256xi64>
%src_27 = tt.addptr %src_25, %src_26 : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi64>
%src_28 = tt.load %src_27, %src_16, %src : tensor<32x256x!tt.ptr<bf16>>
%4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
%5 = tt.broadcast %4 : tensor<1x256xi1> -> tensor<32x256xi1>
%6 = arith.andi %src_14, %5 : tensor<32x256xi1>
%7 = arith.extsi %dst_stride_r : i32 to i64
%8 = tt.splat %7 : i64 -> tensor<32x1xi64>
%9 = arith.muli %src_18, %8 : tensor<32x1xi64>
%10 = tt.splat %DST : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
%11 = tt.addptr %10, %9 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi64>
%12 = tt.expand_dims %offs_v_3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
%13 = arith.extsi %12 : tensor<1x256xi32> to tensor<1x256xi64>
%14 = tt.broadcast %11 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x256x!tt.ptr<bf16>>
%15 = tt.broadcast %13 : tensor<1x256xi64> -> tensor<32x256xi64>
%16 = tt.addptr %14, %15 : tensor<32x256x!tt.ptr<bf16>>, tensor<32x256xi64>
tt.store %16, %src_28, %6 : tensor<32x256x!tt.ptr<bf16>>
tt.return
}
}
"#

private def referencePTX : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _densify_kernel(
.param .u64 .ptr .global .align 1 _densify_kernel_param_0,
.param .u64 .ptr .global .align 1 _densify_kernel_param_1,
.param .u64 .ptr .global .align 1 _densify_kernel_param_2,
.param .u32 _densify_kernel_param_3,
.param .u32 _densify_kernel_param_4,
.param .u32 _densify_kernel_param_5,
.param .u32 _densify_kernel_param_6,
.param .u64 .ptr .global .align 1 _densify_kernel_param_7,
.param .u64 .ptr .global .align 1 _densify_kernel_param_8
)
.reqntid 128
{
.reg .pred %p<138>;
.reg .b16 %rs<199>;
.reg .b32 %r<306>;
.reg .b64 %rd<116>;
$L__func_begin0:
ld.param.b64 %rd74, [_densify_kernel_param_0];
ld.param.b64 %rd75, [_densify_kernel_param_1];
$L__tmp0:
mov.u32 %r35, %ctaid.x;
shl.b32 %r36, %r35, 8;
ld.param.b64 %rd76, [_densify_kernel_param_2];
ld.param.b32 %r37, [_densify_kernel_param_3];
mov.u32 %r38, %tid.x;
shl.b32 %r39, %r38, 1;
ld.param.b32 %r40, [_densify_kernel_param_4];
and.b32 %r41, %r39, 254;
ld.param.s32 %rd77, [_densify_kernel_param_5];
and.b32 %r42, %r38, 31;
ld.param.s32 %rd78, [_densify_kernel_param_6];
shl.b32 %r43, %r42, 3;
or.b32 %r44, %r41, %r36;
or.b32 %r45, %r43, %r36;
mov.u32 %r46, %ctaid.y;
shl.b32 %r47, %r46, 5;
and.b32 %r48, %r38, 96;
bfe.u32 %r49, %r38, 5, 2;
or.b32 %r50, %r47, %r42;
or.b32 %r51, %r49, %r47;
or.b32 %r52, %r51, 4;
or.b32 %r53, %r51, 8;
or.b32 %r54, %r51, 12;
or.b32 %r55, %r51, 16;
or.b32 %r56, %r51, 20;
or.b32 %r57, %r51, 24;
or.b32 %r58, %r51, 28;
setp.lt.s32 %p1, %r44, %r40;
setp.lt.s32 %p68, %r45, %r40;
setp.lt.s32 %p69, %r50, %r37;
setp.lt.s32 %p70, %r51, %r37;
setp.lt.s32 %p71, %r55, %r37;
mad.wide.s32 %rd1, %r44, 4, %rd75;
mov.u32 %r1, 0xffffffffffffffff;
mov.u32 %r2, 0xffffffffffffffff;
@%p1 ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
setp.gt.s32 %p72, %r1, -1;
setp.gt.s32 %p73, %r2, -1;
selp.b32 %r59, %r1, 0, %p72;
selp.b32 %r60, %r2, 0, %p73;
shl.b32 %r61, %r48, 3;
and.b32 %r62, %r39, 60;
shl.b32 %r63, %r38, 6;
and.b32 %r64, %r63, 64;
mov.b32 %r65, global_smem;
add.s32 %r66, %r65, %r64;
add.s32 %r67, %r66, %r61;
add.s32 %r68, %r67, %r62;
st.shared.b32 [%r68], %r59;
st.shared.b32 [%r68+128], %r60;
bar.sync 0;
and.b32 %r69, %r38, 32;
shl.b32 %r70, %r69, 2;
and.b32 %r71, %r38, 64;
add.s32 %r72, %r65, %r70;
add.s32 %r73, %r72, %r71;
ld.shared.v4.b32 {%r74, %r75, %r76, %r77}, [%r73];
ld.shared.v4.b32 {%r78, %r79, %r80, %r81}, [%r73+16];
ld.shared.v4.b32 {%r82, %r83, %r84, %r85}, [%r73+32];
ld.shared.v4.b32 {%r86, %r87, %r88, %r89}, [%r73+48];
ld.shared.v4.b32 {%r90, %r91, %r92, %r93}, [%r73+256];
ld.shared.v4.b32 {%r94, %r95, %r96, %r97}, [%r73+272];
ld.shared.v4.b32 {%r98, %r99, %r100, %r101}, [%r73+288];
ld.shared.v4.b32 {%r102, %r103, %r104, %r105}, [%r73+304];
ld.shared.v4.b32 {%r106, %r107, %r108, %r109}, [%r73+512];
ld.shared.v4.b32 {%r110, %r111, %r112, %r113}, [%r73+528];
ld.shared.v4.b32 {%r114, %r115, %r116, %r117}, [%r73+544];
ld.shared.v4.b32 {%r118, %r119, %r120, %r121}, [%r73+560];
ld.shared.v4.b32 {%r122, %r123, %r124, %r125}, [%r73+768];
ld.shared.v4.b32 {%r126, %r127, %r128, %r129}, [%r73+784];
ld.shared.v4.b32 {%r130, %r131, %r132, %r133}, [%r73+800];
ld.shared.v4.b32 {%r134, %r135, %r136, %r137}, [%r73+816];
bar.sync 0;
selp.b16 %rs66, 1, 0, %p72;
selp.b16 %rs67, 1, 0, %p73;
add.s32 %r138, %r66, %r49;
add.s32 %r139, %r138, %r62;
st.shared.b8 [%r139], %rs66;
st.shared.b8 [%r139+128], %rs67;
bar.sync 0;
ld.shared.v4.b32 {%r140, %r141, %r142, %r143}, [%r73];
shr.u32 %r144, %r143, 24;
and.b32 %r145, %r144, 1;
setp.ne.b32 %p74, %r145, 0;
shr.u32 %r146, %r142, 24;
and.b32 %r147, %r146, 1;
setp.ne.b32 %p75, %r147, 0;
shr.u32 %r148, %r141, 24;
and.b32 %r149, %r148, 1;
setp.ne.b32 %p76, %r149, 0;
shr.u32 %r150, %r140, 24;
and.b32 %r151, %r150, 1;
setp.ne.b32 %p77, %r151, 0;
and.b32 %r152, %r143, 1;
setp.ne.b32 %p78, %r152, 0;
and.b32 %r153, %r142, 1;
setp.ne.b32 %p79, %r153, 0;
and.b32 %r154, %r141, 1;
setp.ne.b32 %p80, %r154, 0;
and.b32 %r155, %r140, 1;
setp.ne.b32 %p81, %r155, 0;
mov.b32 {_, %rs68}, %r143;
mov.b32 {_, %rs69}, %r141;
ld.shared.v4.b32 {%r156, %r157, %r158, %r159}, [%r73+16];
shr.u32 %r160, %r159, 24;
and.b32 %r161, %r160, 1;
setp.ne.b32 %p82, %r161, 0;
shr.u32 %r162, %r158, 24;
and.b32 %r163, %r162, 1;
setp.ne.b32 %p83, %r163, 0;
shr.u32 %r164, %r157, 24;
and.b32 %r165, %r164, 1;
setp.ne.b32 %p84, %r165, 0;
shr.u32 %r166, %r156, 24;
and.b32 %r167, %r166, 1;
setp.ne.b32 %p85, %r167, 0;
and.b32 %r168, %r159, 1;
setp.ne.b32 %p86, %r168, 0;
and.b32 %r169, %r158, 1;
setp.ne.b32 %p87, %r169, 0;
and.b32 %r170, %r157, 1;
setp.ne.b32 %p88, %r170, 0;
and.b32 %r171, %r156, 1;
setp.ne.b32 %p89, %r171, 0;
mov.b32 {_, %rs70}, %r159;
mov.b32 {_, %rs71}, %r157;
ld.shared.v4.b32 {%r172, %r173, %r174, %r175}, [%r73+32];
shr.u32 %r176, %r175, 24;
and.b32 %r177, %r176, 1;
setp.ne.b32 %p90, %r177, 0;
shr.u32 %r178, %r174, 24;
and.b32 %r179, %r178, 1;
setp.ne.b32 %p91, %r179, 0;
shr.u32 %r180, %r173, 24;
and.b32 %r181, %r180, 1;
setp.ne.b32 %p92, %r181, 0;
shr.u32 %r182, %r172, 24;
and.b32 %r183, %r182, 1;
setp.ne.b32 %p93, %r183, 0;
and.b32 %r184, %r175, 1;
setp.ne.b32 %p94, %r184, 0;
and.b32 %r185, %r174, 1;
setp.ne.b32 %p95, %r185, 0;
and.b32 %r186, %r173, 1;
setp.ne.b32 %p96, %r186, 0;
and.b32 %r187, %r172, 1;
setp.ne.b32 %p97, %r187, 0;
mov.b32 {_, %rs72}, %r175;
mov.b32 {_, %rs73}, %r173;
ld.shared.v4.b32 {%r188, %r189, %r190, %r191}, [%r73+48];
shr.u32 %r192, %r191, 24;
and.b32 %r193, %r192, 1;
setp.ne.b32 %p98, %r193, 0;
shr.u32 %r194, %r190, 24;
and.b32 %r195, %r194, 1;
setp.ne.b32 %p99, %r195, 0;
shr.u32 %r196, %r189, 24;
and.b32 %r197, %r196, 1;
setp.ne.b32 %p100, %r197, 0;
shr.u32 %r198, %r188, 24;
and.b32 %r199, %r198, 1;
setp.ne.b32 %p101, %r199, 0;
and.b32 %r200, %r191, 1;
setp.ne.b32 %p102, %r200, 0;
and.b32 %r201, %r190, 1;
setp.ne.b32 %p103, %r201, 0;
and.b32 %r202, %r189, 1;
setp.ne.b32 %p104, %r202, 0;
and.b32 %r203, %r188, 1;
setp.ne.b32 %p105, %r203, 0;
and.b32 %r204, %r174, 65536;
and.b32 %r205, %r172, 65536;
and.b32 %r206, %r190, 65536;
and.b32 %r207, %r188, 65536;
and.b32 %r208, %r142, 65536;
and.b32 %r209, %r140, 65536;
and.b32 %r210, %r158, 65536;
and.b32 %r211, %r156, 65536;
and.b16 %rs74, %rs71, 1;
shl.b16 %rs75, %rs74, 2;
mov.b32 {_, %rs76}, %r211;
shl.b16 %rs77, %rs76, 3;
or.b16 %rs78, %rs77, %rs75;
and.b16 %rs79, %rs70, 1;
mov.b32 {_, %rs80}, %r210;
shl.b16 %rs81, %rs80, 1;
or.b16 %rs82, %rs79, %rs81;
or.b16 %rs83, %rs82, %rs78;
shl.b16 %rs84, %rs83, 8;
and.b16 %rs85, %rs69, 1;
shl.b16 %rs86, %rs85, 2;
mov.b32 {_, %rs87}, %r209;
shl.b16 %rs88, %rs87, 3;
or.b16 %rs89, %rs88, %rs86;
and.b16 %rs90, %rs68, 1;
mov.b32 {_, %rs91}, %r208;
shl.b16 %rs92, %rs91, 1;
or.b16 %rs93, %rs90, %rs92;
or.b16 %rs94, %rs93, %rs89;
shl.b16 %rs95, %rs94, 12;
or.b16 %rs96, %rs95, %rs84;
mov.b32 {_, %rs97}, %r189;
and.b16 %rs98, %rs97, 1;
shl.b16 %rs99, %rs98, 2;
mov.b32 {_, %rs100}, %r207;
shl.b16 %rs101, %rs100, 3;
or.b16 %rs102, %rs101, %rs99;
mov.b32 {_, %rs103}, %r191;
and.b16 %rs104, %rs103, 1;
mov.b32 {_, %rs105}, %r206;
shl.b16 %rs106, %rs105, 1;
or.b16 %rs107, %rs104, %rs106;
or.b16 %rs108, %rs107, %rs102;
and.b16 %rs109, %rs73, 1;
shl.b16 %rs110, %rs109, 2;
mov.b32 {_, %rs111}, %r205;
shl.b16 %rs112, %rs111, 3;
or.b16 %rs113, %rs112, %rs110;
and.b16 %rs114, %rs72, 1;
mov.b32 {_, %rs115}, %r204;
shl.b16 %rs116, %rs115, 1;
or.b16 %rs117, %rs114, %rs116;
or.b16 %rs118, %rs117, %rs113;
shl.b16 %rs119, %rs118, 4;
or.b16 %rs120, %rs108, %rs119;
and.b16 %rs121, %rs120, 255;
or.b16 %rs122, %rs121, %rs96;
cvt.u32.u16 %r212, %rs122;
cvt.u16.u32 %rs123, %r157;
shr.u16 %rs124, %rs123, 6;
and.b16 %rs125, %rs124, 4;
cvt.u16.u32 %rs126, %r156;
and.b16 %rs127, %rs126, 256;
shr.u16 %rs128, %rs127, 5;
or.b16 %rs129, %rs128, %rs125;
cvt.u16.u32 %rs130, %r158;
and.b16 %rs131, %rs130, 256;
shr.u16 %rs132, %rs131, 7;
cvt.u16.u32 %rs133, %r159;
shr.u16 %rs134, %rs133, 8;
and.b16 %rs135, %rs134, 1;
or.b16 %rs136, %rs135, %rs132;
or.b16 %rs137, %rs136, %rs129;
shl.b16 %rs138, %rs137, 8;
cvt.u16.u32 %rs139, %r141;
shr.u16 %rs140, %rs139, 6;
and.b16 %rs141, %rs140, 4;
cvt.u16.u32 %rs142, %r140;
and.b16 %rs143, %rs142, 256;
shr.u16 %rs144, %rs143, 5;
or.b16 %rs145, %rs144, %rs141;
cvt.u16.u32 %rs146, %r142;
and.b16 %rs147, %rs146, 256;
shr.u16 %rs148, %rs147, 7;
cvt.u16.u32 %rs149, %r143;
shr.u16 %rs150, %rs149, 8;
and.b16 %rs151, %rs150, 1;
or.b16 %rs152, %rs151, %rs148;
or.b16 %rs153, %rs152, %rs145;
shl.b16 %rs154, %rs153, 12;
or.b16 %rs155, %rs154, %rs138;
shr.u16 %rs156, %rs155, 12;
shr.u16 %rs157, %rs155, 13;
shr.u16 %rs158, %rs155, 14;
shr.u16 %rs159, %rs155, 15;
cvt.u16.u32 %rs160, %r173;
shr.u16 %rs161, %rs160, 6;
and.b16 %rs162, %rs161, 4;
bfe.u32 %r213, %r172, 8, 1;
cvt.u16.u32 %rs163, %r213;
shl.b16 %rs164, %rs163, 3;
or.b16 %rs165, %rs164, %rs162;
bfe.u32 %r214, %r174, 8, 1;
cvt.u16.u32 %rs166, %r214;
shl.b16 %rs167, %rs166, 1;
cvt.u16.u32 %rs168, %r175;
shr.u16 %rs169, %rs168, 8;
and.b16 %rs170, %rs169, 1;
or.b16 %rs171, %rs170, %rs167;
or.b16 %rs172, %rs171, %rs165;
shr.u16 %rs173, %rs172, 2;
shr.u16 %rs174, %rs172, 3;
shr.u32 %r215, %r191, 16;
and.b32 %r216, %r215, 1;
setp.ne.b32 %p106, %r216, 0;
cvt.u16.u32 %rs175, %r191;
shr.u16 %rs176, %rs175, 8;
cvt.u16.u32 %rs177, %r190;
shr.u16 %rs178, %rs177, 8;
cvt.u16.u32 %rs179, %r189;
shr.u16 %rs180, %rs179, 8;
cvt.u16.u32 %rs181, %r188;
shr.u16 %rs182, %rs181, 8;
shr.u16 %rs183, %rs130, 8;
shr.u16 %rs184, %rs123, 8;
shr.u16 %rs185, %rs126, 8;
and.pred %p2, %p69, %p81;
and.pred %p3, %p69, %p80;
and.pred %p4, %p69, %p79;
and.pred %p5, %p69, %p78;
and.pred %p6, %p69, %p89;
and.pred %p7, %p69, %p88;
and.pred %p8, %p69, %p87;
and.pred %p9, %p69, %p86;
and.pred %p10, %p69, %p97;
and.pred %p11, %p69, %p96;
and.pred %p12, %p69, %p95;
and.pred %p13, %p69, %p94;
and.pred %p14, %p69, %p105;
and.pred %p15, %p69, %p104;
and.pred %p16, %p69, %p103;
and.pred %p17, %p69, %p102;
and.b16 %rs186, %rs159, 1;
setp.ne.b16 %p107, %rs186, 0;
and.pred %p18, %p69, %p107;
and.b16 %rs187, %rs158, 1;
setp.ne.b16 %p108, %rs187, 0;
and.pred %p19, %p69, %p108;
and.b16 %rs188, %rs157, 1;
setp.ne.b16 %p109, %rs188, 0;
and.pred %p20, %p69, %p109;
and.b16 %rs189, %rs156, 1;
setp.ne.b16 %p110, %rs189, 0;
and.pred %p21, %p69, %p110;
and.b16 %rs190, %rs185, 1;
setp.ne.b16 %p111, %rs190, 0;
and.pred %p22, %p69, %p111;
and.b16 %rs191, %rs184, 1;
setp.ne.b16 %p112, %rs191, 0;
and.pred %p23, %p69, %p112;
and.b16 %rs192, %rs183, 1;
setp.ne.b16 %p113, %rs192, 0;
and.pred %p24, %p69, %p113;
setp.ne.b16 %p114, %rs135, 0;
and.pred %p25, %p69, %p114;
and.b16 %rs193, %rs174, 1;
setp.ne.b16 %p115, %rs193, 0;
and.pred %p26, %p69, %p115;
and.b16 %rs194, %rs173, 1;
setp.ne.b16 %p116, %rs194, 0;
and.pred %p27, %p69, %p116;
and.b32 %r217, %r214, 1;
setp.ne.b32 %p117, %r217, 0;
and.pred %p28, %p69, %p117;
setp.ne.b16 %p118, %rs170, 0;
and.pred %p29, %p69, %p118;
and.b16 %rs195, %rs182, 1;
setp.ne.b16 %p119, %rs195, 0;
and.pred %p30, %p69, %p119;
and.b16 %rs196, %rs180, 1;
setp.ne.b16 %p120, %rs196, 0;
and.pred %p31, %p69, %p120;
and.b16 %rs197, %rs178, 1;
setp.ne.b16 %p121, %rs197, 0;
and.pred %p32, %p69, %p121;
and.b16 %rs198, %rs176, 1;
setp.ne.b16 %p122, %rs198, 0;
and.pred %p33, %p69, %p122;
shr.u32 %r218, %r212, 15;
and.b32 %r219, %r218, 1;
setp.ne.b32 %p123, %r219, 0;
and.pred %p34, %p69, %p123;
shr.u32 %r220, %r212, 14;
and.b32 %r221, %r220, 1;
setp.ne.b32 %p124, %r221, 0;
and.pred %p35, %p69, %p124;
shr.u32 %r222, %r212, 13;
and.b32 %r223, %r222, 1;
setp.ne.b32 %p125, %r223, 0;
and.pred %p36, %p69, %p125;
shr.u32 %r224, %r212, 12;
and.b32 %r225, %r224, 1;
setp.ne.b32 %p126, %r225, 0;
and.pred %p37, %p69, %p126;
shr.u32 %r226, %r212, 11;
and.b32 %r227, %r226, 1;
setp.ne.b32 %p127, %r227, 0;
and.pred %p38, %p69, %p127;
shr.u32 %r228, %r212, 10;
and.b32 %r229, %r228, 1;
setp.ne.b32 %p128, %r229, 0;
and.pred %p39, %p69, %p128;
shr.u32 %r230, %r212, 9;
and.b32 %r231, %r230, 1;
setp.ne.b32 %p129, %r231, 0;
and.pred %p40, %p69, %p129;
shr.u32 %r232, %r212, 8;
and.b32 %r233, %r232, 1;
setp.ne.b32 %p130, %r233, 0;
and.pred %p41, %p69, %p130;
shr.u32 %r234, %r212, 7;
and.b32 %r235, %r234, 1;
setp.ne.b32 %p131, %r235, 0;
and.pred %p42, %p69, %p131;
shr.u32 %r236, %r212, 6;
and.b32 %r237, %r236, 1;
setp.ne.b32 %p132, %r237, 0;
and.pred %p43, %p69, %p132;
shr.u32 %r238, %r212, 5;
and.b32 %r239, %r238, 1;
setp.ne.b32 %p133, %r239, 0;
and.pred %p44, %p69, %p133;
shr.u32 %r240, %r212, 4;
and.b32 %r241, %r240, 1;
setp.ne.b32 %p134, %r241, 0;
and.pred %p45, %p69, %p134;
shr.u32 %r242, %r212, 3;
and.b32 %r243, %r242, 1;
setp.ne.b32 %p135, %r243, 0;
and.pred %p46, %p69, %p135;
shr.u32 %r244, %r212, 2;
and.b32 %r245, %r244, 1;
setp.ne.b32 %p136, %r245, 0;
and.pred %p47, %p69, %p136;
shr.u32 %r246, %r212, 1;
and.b32 %r247, %r246, 1;
setp.ne.b32 %p137, %r247, 0;
and.pred %p48, %p69, %p137;
and.pred %p49, %p69, %p106;
and.pred %p50, %p69, %p77;
and.pred %p51, %p69, %p76;
and.pred %p52, %p69, %p75;
and.pred %p53, %p69, %p74;
and.pred %p54, %p69, %p85;
and.pred %p55, %p69, %p84;
and.pred %p56, %p69, %p83;
and.pred %p57, %p69, %p82;
and.pred %p58, %p69, %p93;
and.pred %p59, %p69, %p92;
and.pred %p60, %p69, %p91;
and.pred %p61, %p69, %p90;
and.pred %p62, %p69, %p101;
and.pred %p63, %p69, %p100;
and.pred %p64, %p69, %p99;
and.pred %p65, %p69, %p98;
cvt.u64.u32 %rd79, %r50;
cvt.u64.u32 %rd80, %r51;
cvt.u64.u32 %rd81, %r52;
cvt.u64.u32 %rd82, %r53;
cvt.u64.u32 %rd83, %r54;
cvt.u64.u32 %rd84, %r55;
cvt.u64.u32 %rd85, %r56;
cvt.u64.u32 %rd86, %r57;
cvt.u64.u32 %rd87, %r58;
mul.lo.s64 %rd88, %rd77, %rd79;
shl.b64 %rd89, %rd88, 1;
add.s64 %rd90, %rd74, %rd89;
mad.wide.s32 %rd2, %r74, 2, %rd90;
mad.wide.s32 %rd3, %r75, 2, %rd90;
mad.wide.s32 %rd4, %r76, 2, %rd90;
mad.wide.s32 %rd5, %r77, 2, %rd90;
mad.wide.s32 %rd6, %r78, 2, %rd90;
mad.wide.s32 %rd7, %r79, 2, %rd90;
mad.wide.s32 %rd8, %r80, 2, %rd90;
mad.wide.s32 %rd9, %r81, 2, %rd90;
mad.wide.s32 %rd10, %r82, 2, %rd90;
mad.wide.s32 %rd11, %r83, 2, %rd90;
mad.wide.s32 %rd12, %r84, 2, %rd90;
mad.wide.s32 %rd13, %r85, 2, %rd90;
mad.wide.s32 %rd14, %r86, 2, %rd90;
mad.wide.s32 %rd15, %r87, 2, %rd90;
mad.wide.s32 %rd16, %r88, 2, %rd90;
mad.wide.s32 %rd17, %r89, 2, %rd90;
mad.wide.s32 %rd18, %r90, 2, %rd90;
mad.wide.s32 %rd19, %r91, 2, %rd90;
mad.wide.s32 %rd20, %r92, 2, %rd90;
mad.wide.s32 %rd21, %r93, 2, %rd90;
mad.wide.s32 %rd22, %r94, 2, %rd90;
mad.wide.s32 %rd23, %r95, 2, %rd90;
mad.wide.s32 %rd24, %r96, 2, %rd90;
mad.wide.s32 %rd25, %r97, 2, %rd90;
mad.wide.s32 %rd26, %r98, 2, %rd90;
mad.wide.s32 %rd27, %r99, 2, %rd90;
mad.wide.s32 %rd28, %r100, 2, %rd90;
mad.wide.s32 %rd29, %r101, 2, %rd90;
mad.wide.s32 %rd30, %r102, 2, %rd90;
mad.wide.s32 %rd31, %r103, 2, %rd90;
mad.wide.s32 %rd32, %r104, 2, %rd90;
mad.wide.s32 %rd33, %r105, 2, %rd90;
mad.wide.s32 %rd34, %r106, 2, %rd90;
mad.wide.s32 %rd35, %r107, 2, %rd90;
mad.wide.s32 %rd36, %r108, 2, %rd90;
mad.wide.s32 %rd37, %r109, 2, %rd90;
mad.wide.s32 %rd38, %r110, 2, %rd90;
mad.wide.s32 %rd39, %r111, 2, %rd90;
mad.wide.s32 %rd40, %r112, 2, %rd90;
mad.wide.s32 %rd41, %r113, 2, %rd90;
mad.wide.s32 %rd42, %r114, 2, %rd90;
mad.wide.s32 %rd43, %r115, 2, %rd90;
mad.wide.s32 %rd44, %r116, 2, %rd90;
mad.wide.s32 %rd45, %r117, 2, %rd90;
mad.wide.s32 %rd46, %r118, 2, %rd90;
mad.wide.s32 %rd47, %r119, 2, %rd90;
mad.wide.s32 %rd48, %r120, 2, %rd90;
mad.wide.s32 %rd49, %r121, 2, %rd90;
mad.wide.s32 %rd50, %r122, 2, %rd90;
mad.wide.s32 %rd51, %r123, 2, %rd90;
mad.wide.s32 %rd52, %r124, 2, %rd90;
mad.wide.s32 %rd53, %r125, 2, %rd90;
mad.wide.s32 %rd54, %r126, 2, %rd90;
mad.wide.s32 %rd55, %r127, 2, %rd90;
mad.wide.s32 %rd56, %r128, 2, %rd90;
mad.wide.s32 %rd57, %r129, 2, %rd90;
mad.wide.s32 %rd58, %r130, 2, %rd90;
mad.wide.s32 %rd59, %r131, 2, %rd90;
mad.wide.s32 %rd60, %r132, 2, %rd90;
mad.wide.s32 %rd61, %r133, 2, %rd90;
mad.wide.s32 %rd62, %r134, 2, %rd90;
mad.wide.s32 %rd63, %r135, 2, %rd90;
mad.wide.s32 %rd64, %r136, 2, %rd90;
mad.wide.s32 %rd65, %r137, 2, %rd90;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p2 ld.global.b16 { %rs1 }, [ %rd2 + 0 ];
mov.u16 %rs3, %rs2;
@%p3 ld.global.b16 { %rs3 }, [ %rd3 + 0 ];
mov.u16 %rs4, %rs2;
@%p4 ld.global.b16 { %rs4 }, [ %rd4 + 0 ];
mov.u16 %rs5, %rs2;
@%p5 ld.global.b16 { %rs5 }, [ %rd5 + 0 ];
mov.u16 %rs6, %rs2;
@%p6 ld.global.b16 { %rs6 }, [ %rd6 + 0 ];
mov.u16 %rs7, %rs2;
@%p7 ld.global.b16 { %rs7 }, [ %rd7 + 0 ];
mov.u16 %rs8, %rs2;
@%p8 ld.global.b16 { %rs8 }, [ %rd8 + 0 ];
mov.u16 %rs9, %rs2;
@%p9 ld.global.b16 { %rs9 }, [ %rd9 + 0 ];
mov.u16 %rs10, %rs2;
@%p10 ld.global.b16 { %rs10 }, [ %rd10 + 0 ];
mov.u16 %rs11, %rs2;
@%p11 ld.global.b16 { %rs11 }, [ %rd11 + 0 ];
mov.u16 %rs12, %rs2;
@%p12 ld.global.b16 { %rs12 }, [ %rd12 + 0 ];
mov.u16 %rs13, %rs2;
@%p13 ld.global.b16 { %rs13 }, [ %rd13 + 0 ];
mov.u16 %rs14, %rs2;
@%p14 ld.global.b16 { %rs14 }, [ %rd14 + 0 ];
mov.u16 %rs15, %rs2;
@%p15 ld.global.b16 { %rs15 }, [ %rd15 + 0 ];
mov.u16 %rs16, %rs2;
@%p16 ld.global.b16 { %rs16 }, [ %rd16 + 0 ];
mov.u16 %rs17, %rs2;
@%p17 ld.global.b16 { %rs17 }, [ %rd17 + 0 ];
mov.u16 %rs18, %rs2;
@%p18 ld.global.b16 { %rs18 }, [ %rd18 + 0 ];
mov.u16 %rs19, %rs2;
@%p19 ld.global.b16 { %rs19 }, [ %rd19 + 0 ];
mov.u16 %rs20, %rs2;
@%p20 ld.global.b16 { %rs20 }, [ %rd20 + 0 ];
mov.u16 %rs21, %rs2;
@%p21 ld.global.b16 { %rs21 }, [ %rd21 + 0 ];
mov.u16 %rs22, %rs2;
@%p22 ld.global.b16 { %rs22 }, [ %rd22 + 0 ];
mov.u16 %rs23, %rs2;
@%p23 ld.global.b16 { %rs23 }, [ %rd23 + 0 ];
mov.u16 %rs24, %rs2;
@%p24 ld.global.b16 { %rs24 }, [ %rd24 + 0 ];
mov.u16 %rs25, %rs2;
@%p25 ld.global.b16 { %rs25 }, [ %rd25 + 0 ];
mov.u16 %rs26, %rs2;
@%p26 ld.global.b16 { %rs26 }, [ %rd26 + 0 ];
mov.u16 %rs27, %rs2;
@%p27 ld.global.b16 { %rs27 }, [ %rd27 + 0 ];
mov.u16 %rs28, %rs2;
@%p28 ld.global.b16 { %rs28 }, [ %rd28 + 0 ];
mov.u16 %rs29, %rs2;
@%p29 ld.global.b16 { %rs29 }, [ %rd29 + 0 ];
mov.u16 %rs30, %rs2;
@%p30 ld.global.b16 { %rs30 }, [ %rd30 + 0 ];
mov.u16 %rs31, %rs2;
@%p31 ld.global.b16 { %rs31 }, [ %rd31 + 0 ];
mov.u16 %rs32, %rs2;
@%p32 ld.global.b16 { %rs32 }, [ %rd32 + 0 ];
mov.u16 %rs33, %rs2;
@%p33 ld.global.b16 { %rs33 }, [ %rd33 + 0 ];
mov.u16 %rs34, %rs2;
@%p34 ld.global.b16 { %rs34 }, [ %rd34 + 0 ];
mov.u16 %rs35, %rs2;
@%p35 ld.global.b16 { %rs35 }, [ %rd35 + 0 ];
mov.u16 %rs36, %rs2;
@%p36 ld.global.b16 { %rs36 }, [ %rd36 + 0 ];
mov.u16 %rs37, %rs2;
@%p37 ld.global.b16 { %rs37 }, [ %rd37 + 0 ];
mov.u16 %rs38, %rs2;
@%p38 ld.global.b16 { %rs38 }, [ %rd38 + 0 ];
mov.u16 %rs39, %rs2;
@%p39 ld.global.b16 { %rs39 }, [ %rd39 + 0 ];
mov.u16 %rs40, %rs2;
@%p40 ld.global.b16 { %rs40 }, [ %rd40 + 0 ];
mov.u16 %rs41, %rs2;
@%p41 ld.global.b16 { %rs41 }, [ %rd41 + 0 ];
mov.u16 %rs42, %rs2;
@%p42 ld.global.b16 { %rs42 }, [ %rd42 + 0 ];
mov.u16 %rs43, %rs2;
@%p43 ld.global.b16 { %rs43 }, [ %rd43 + 0 ];
mov.u16 %rs44, %rs2;
@%p44 ld.global.b16 { %rs44 }, [ %rd44 + 0 ];
mov.u16 %rs45, %rs2;
@%p45 ld.global.b16 { %rs45 }, [ %rd45 + 0 ];
mov.u16 %rs46, %rs2;
@%p46 ld.global.b16 { %rs46 }, [ %rd46 + 0 ];
mov.u16 %rs47, %rs2;
@%p47 ld.global.b16 { %rs47 }, [ %rd47 + 0 ];
mov.u16 %rs48, %rs2;
@%p48 ld.global.b16 { %rs48 }, [ %rd48 + 0 ];
mov.u16 %rs49, %rs2;
@%p49 ld.global.b16 { %rs49 }, [ %rd49 + 0 ];
mov.u16 %rs50, %rs2;
@%p50 ld.global.b16 { %rs50 }, [ %rd50 + 0 ];
mov.u16 %rs51, %rs2;
@%p51 ld.global.b16 { %rs51 }, [ %rd51 + 0 ];
mov.u16 %rs52, %rs2;
@%p52 ld.global.b16 { %rs52 }, [ %rd52 + 0 ];
mov.u16 %rs53, %rs2;
@%p53 ld.global.b16 { %rs53 }, [ %rd53 + 0 ];
mov.u16 %rs54, %rs2;
@%p54 ld.global.b16 { %rs54 }, [ %rd54 + 0 ];
mov.u16 %rs55, %rs2;
@%p55 ld.global.b16 { %rs55 }, [ %rd55 + 0 ];
mov.u16 %rs56, %rs2;
@%p56 ld.global.b16 { %rs56 }, [ %rd56 + 0 ];
mov.u16 %rs57, %rs2;
@%p57 ld.global.b16 { %rs57 }, [ %rd57 + 0 ];
mov.u16 %rs58, %rs2;
@%p58 ld.global.b16 { %rs58 }, [ %rd58 + 0 ];
mov.u16 %rs59, %rs2;
@%p59 ld.global.b16 { %rs59 }, [ %rd59 + 0 ];
mov.u16 %rs60, %rs2;
@%p60 ld.global.b16 { %rs60 }, [ %rd60 + 0 ];
mov.u16 %rs61, %rs2;
@%p61 ld.global.b16 { %rs61 }, [ %rd61 + 0 ];
mov.u16 %rs62, %rs2;
@%p62 ld.global.b16 { %rs62 }, [ %rd62 + 0 ];
mov.u16 %rs63, %rs2;
@%p63 ld.global.b16 { %rs63 }, [ %rd63 + 0 ];
mov.u16 %rs64, %rs2;
@%p64 ld.global.b16 { %rs64 }, [ %rd64 + 0 ];
mov.u16 %rs65, %rs2;
@%p65 ld.global.b16 { %rs65 }, [ %rd65 + 0 ];
and.pred %p66, %p70, %p68;
and.pred %p67, %p71, %p68;
mul.lo.s64 %rd91, %rd78, %rd80;
mul.lo.s64 %rd92, %rd78, %rd81;
mul.lo.s64 %rd93, %rd78, %rd82;
mul.lo.s64 %rd94, %rd78, %rd83;
mul.lo.s64 %rd95, %rd78, %rd84;
mul.lo.s64 %rd96, %rd78, %rd85;
mul.lo.s64 %rd97, %rd78, %rd86;
mul.lo.s64 %rd98, %rd78, %rd87;
shl.b64 %rd99, %rd91, 1;
add.s64 %rd100, %rd76, %rd99;
shl.b64 %rd101, %rd92, 1;
add.s64 %rd102, %rd76, %rd101;
shl.b64 %rd103, %rd93, 1;
add.s64 %rd104, %rd76, %rd103;
shl.b64 %rd105, %rd94, 1;
add.s64 %rd106, %rd76, %rd105;
shl.b64 %rd107, %rd95, 1;
add.s64 %rd108, %rd76, %rd107;
shl.b64 %rd109, %rd96, 1;
add.s64 %rd110, %rd76, %rd109;
shl.b64 %rd111, %rd97, 1;
add.s64 %rd112, %rd76, %rd111;
shl.b64 %rd113, %rd98, 1;
add.s64 %rd114, %rd76, %rd113;
mul.wide.s32 %rd115, %r45, 2;
add.s64 %rd66, %rd100, %rd115;
add.s64 %rd67, %rd102, %rd115;
add.s64 %rd68, %rd104, %rd115;
add.s64 %rd69, %rd106, %rd115;
add.s64 %rd70, %rd108, %rd115;
add.s64 %rd71, %rd110, %rd115;
add.s64 %rd72, %rd112, %rd115;
add.s64 %rd73, %rd114, %rd115;
bar.sync 0;
and.b32 %r248, %r38, 7;
shl.b32 %r249, %r248, 10;
shl.b32 %r250, %r42, 4;
shr.u32 %r251, %r69, 1;
shl.b32 %r252, %r71, 3;
or.b32 %r253, %r249, %r252;
xor.b32 %r254, %r250, %r251;
or.b32 %r255, %r253, %r254;
add.s32 %r256, %r65, %r255;
mov.b32 %r257, {%rs34, %rs42};
mov.b32 %r258, {%rs1, %rs10};
mov.b32 %r259, {%rs18, %rs26};
mov.b32 %r260, {%rs50, %rs58};
st.shared.v4.b32 [%r256], {%r258, %r259, %r257, %r260};
xor.b32 %r261, %r255, 32;
add.s32 %r262, %r65, %r261;
mov.b32 %r263, {%rs36, %rs44};
mov.b32 %r264, {%rs4, %rs12};
mov.b32 %r265, {%rs20, %rs28};
mov.b32 %r266, {%rs52, %rs60};
st.shared.v4.b32 [%r262], {%r264, %r265, %r263, %r266};
xor.b32 %r267, %r255, 64;
add.s32 %r268, %r65, %r267;
mov.b32 %r269, {%rs38, %rs46};
mov.b32 %r270, {%rs6, %rs14};
mov.b32 %r271, {%rs22, %rs30};
mov.b32 %r272, {%rs54, %rs62};
st.shared.v4.b32 [%r268], {%r270, %r271, %r269, %r272};
xor.b32 %r273, %r255, 96;
add.s32 %r274, %r65, %r273;
mov.b32 %r275, {%rs40, %rs48};
mov.b32 %r276, {%rs8, %rs16};
mov.b32 %r277, {%rs24, %rs32};
mov.b32 %r278, {%rs56, %rs64};
st.shared.v4.b32 [%r274], {%r276, %r277, %r275, %r278};
bar.sync 0;
shl.b32 %r279, %r48, 5;
shl.b32 %r280, %r248, 4;
shr.u32 %r281, %r48, 1;
and.b32 %r282, %r63, 512;
bfe.s32 %r283, %r38, 4, 1;
and.b32 %r284, %r283, 4160;
or.b32 %r285, %r279, %r280;
or.b32 %r286, %r284, %r281;
xor.b32 %r287, %r286, %r285;
add.s32 %r288, %r65, %r282;
add.s32 %r289, %r288, %r287;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r3, %r4, %r7, %r8}, [%r289];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r11, %r12, %r15, %r16}, [%r289+128];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r19, %r20, %r23, %r24}, [%r289+256];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r27, %r28, %r31, %r32}, [%r289+384];
bar.sync 0;
mov.b32 %r290, {%rs35, %rs43};
mov.b32 %r291, {%rs3, %rs11};
mov.b32 %r292, {%rs19, %rs27};
mov.b32 %r293, {%rs51, %rs59};
st.shared.v4.b32 [%r256], {%r291, %r292, %r290, %r293};
mov.b32 %r294, {%rs37, %rs45};
mov.b32 %r295, {%rs5, %rs13};
mov.b32 %r296, {%rs21, %rs29};
mov.b32 %r297, {%rs53, %rs61};
st.shared.v4.b32 [%r262], {%r295, %r296, %r294, %r297};
mov.b32 %r298, {%rs39, %rs47};
mov.b32 %r299, {%rs7, %rs15};
mov.b32 %r300, {%rs23, %rs31};
mov.b32 %r301, {%rs55, %rs63};
st.shared.v4.b32 [%r268], {%r299, %r300, %r298, %r301};
mov.b32 %r302, {%rs41, %rs49};
mov.b32 %r303, {%rs9, %rs17};
mov.b32 %r304, {%rs25, %rs33};
mov.b32 %r305, {%rs57, %rs65};
st.shared.v4.b32 [%r274], {%r303, %r304, %r302, %r305};
bar.sync 0;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r5, %r6, %r9, %r10}, [%r289];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r13, %r14, %r17, %r18}, [%r289+128];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r21, %r22, %r25, %r26}, [%r289+256];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r29, %r30, %r33, %r34}, [%r289+384];
@%p66 st.global.v4.b32 [ %rd66 + 0 ], { %r3, %r4, %r5, %r6 };
@%p66 st.global.v4.b32 [ %rd67 + 0 ], { %r7, %r8, %r9, %r10 };
@%p66 st.global.v4.b32 [ %rd68 + 0 ], { %r11, %r12, %r13, %r14 };
@%p66 st.global.v4.b32 [ %rd69 + 0 ], { %r15, %r16, %r17, %r18 };
@%p67 st.global.v4.b32 [ %rd70 + 0 ], { %r19, %r20, %r21, %r22 };
@%p67 st.global.v4.b32 [ %rd71 + 0 ], { %r23, %r24, %r25, %r26 };
@%p67 st.global.v4.b32 [ %rd72 + 0 ], { %r27, %r28, %r29, %r30 };
@%p67 st.global.v4.b32 [ %rd73 + 0 ], { %r31, %r32, %r33, %r34 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def hopperIRTemplate : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_densify_gluon(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %pos: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0> : tensor<1x128xi32, #blocked>
%cst_0 = arith.constant dense<@COUNT@> : tensor<16x1xi32, #blocked>
%cst_1 = arith.constant dense<50304> : tensor<16x1xi32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<16x128xbf16, #blocked>
%cst_3 = arith.constant dense<@PITCH16@> : tensor<16x128xi32, #blocked>
%cst_4 = arith.constant dense<804864> : tensor<16x128xi32, #blocked>
%cst_5 = arith.constant dense<@PITCH32@> : tensor<16x128xi32, #blocked>
%cst_6 = arith.constant dense<1609728> : tensor<16x128xi32, #blocked>
%cst_7 = arith.constant dense<@PITCH48@> : tensor<16x128xi32, #blocked>
%cst_8 = arith.constant dense<2414592> : tensor<16x128xi32, #blocked>
%v = tt.get_program_id x : i32
%v_9 = arith.muli %v, 128 : i32
%v_10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_11 = tt.splat %v_9 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_12 = arith.addi %v_11, %v_10 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%r = tt.get_program_id y : i32
%r_13 = arith.muli %r, 64 : i32
%r_14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%r_15 = tt.splat %r_13 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%r_16 = arith.addi %r_15, %r_14 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%p = tt.splat %pos : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
%p_17 = tt.addptr %p, %v_12 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%p_18 = tt.load %p_17 : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
%keep = tt.expand_dims %p_18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%keep_19 = arith.cmpi sge, %keep, %cst : tensor<1x128xi32, #blocked>
%base = tt.expand_dims %r_16 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
%base_20 = arith.muli %base, %cst_0 : tensor<16x1xi32, #blocked>
%base_21 = tt.splat %src : !tt.ptr<bf16> -> tensor<16x1x!tt.ptr<bf16>, #blocked>
%base_22 = tt.addptr %base_21, %base_20 : tensor<16x1x!tt.ptr<bf16>, #blocked>, tensor<16x1xi32, #blocked>
%base_23 = tt.broadcast %base_22 : tensor<16x1x!tt.ptr<bf16>, #blocked> -> tensor<16x128x!tt.ptr<bf16>, #blocked>
%base_24 = tt.broadcast %keep : tensor<1x128xi32, #blocked> -> tensor<16x128xi32, #blocked>
%base_25 = tt.addptr %base_23, %base_24 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%out = arith.muli %base, %cst_1 : tensor<16x1xi32, #blocked>
%out_26 = tt.expand_dims %v_12 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%out_27 = tt.broadcast %out : tensor<16x1xi32, #blocked> -> tensor<16x128xi32, #blocked>
%out_28 = tt.broadcast %out_26 : tensor<1x128xi32, #blocked> -> tensor<16x128xi32, #blocked>
%out_29 = arith.addi %out_27, %out_28 : tensor<16x128xi32, #blocked>
%out_30 = tt.splat %dst : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked>
%out_31 = tt.addptr %out_30, %out_29 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%values = tt.broadcast %keep_19 : tensor<1x128xi1, #blocked> -> tensor<16x128xi1, #blocked>
%values_32 = tt.load %base_25, %values, %cst_2 : tensor<16x128x!tt.ptr<bf16>, #blocked>
tt.store %out_31, %values_32 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%values_33 = tt.addptr %base_25, %cst_3 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%values_34 = tt.load %values_33, %values, %cst_2 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%0 = tt.addptr %out_31, %cst_4 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
tt.store %0, %values_34 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%values_35 = tt.addptr %base_25, %cst_5 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%values_36 = tt.load %values_35, %values, %cst_2 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %out_31, %cst_6 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
tt.store %1, %values_36 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%values_37 = tt.addptr %base_25, %cst_7 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
%values_38 = tt.load %values_37, %values, %cst_2 : tensor<16x128x!tt.ptr<bf16>, #blocked>
%2 = tt.addptr %out_31, %cst_8 : tensor<16x128x!tt.ptr<bf16>, #blocked>, tensor<16x128xi32, #blocked>
tt.store %2, %values_38 : tensor<16x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
"#

private def hopperPTXTemplate : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _densify_gluon(
.param .u64 .ptr .global .align 1 _densify_gluon_param_0,
.param .u64 .ptr .global .align 1 _densify_gluon_param_1,
.param .u64 .ptr .global .align 1 _densify_gluon_param_2,
.param .u64 .ptr .global .align 1 _densify_gluon_param_3,
.param .u64 .ptr .global .align 1 _densify_gluon_param_4
)
.reqntid 256
{
.reg .pred %p<9>;
.reg .b16 %rs<34>;
.reg .b32 %r<37>;
.reg .b64 %rd<43>;
$L__func_begin0:
ld.param.b64 %rd39, [_densify_gluon_param_0];
ld.param.b64 %rd40, [_densify_gluon_param_1];
$L__tmp0:
mov.u32 %r25, %ctaid.x;
ld.param.b64 %rd41, [_densify_gluon_param_2];
mov.u32 %r26, %tid.x;
shl.b32 %r27, %r26, 3;
mov.u32 %r28, %ctaid.y;
shl.b32 %r29, %r25, 7;
shl.b32 %r30, %r28, 6;
and.b32 %r31, %r27, 120;
bfe.u32 %r32, %r26, 4, 4;
or.b32 %r33, %r32, %r30;
or.b32 %r34, %r31, %r29;
mad.wide.s32 %rd1, %r34, 4, %rd40;
add.s64 %rd2, %rd1, 16;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
setp.gt.s32 %p1, %r1, -1;
setp.gt.s32 %p2, %r2, -1;
setp.gt.s32 %p3, %r3, -1;
setp.gt.s32 %p4, %r4, -1;
setp.gt.s32 %p5, %r5, -1;
setp.gt.s32 %p6, %r6, -1;
setp.gt.s32 %p7, %r7, -1;
setp.gt.s32 %p8, %r8, -1;
mul.lo.s32 %r35, %r33, @COUNT@;
mad.wide.s32 %rd42, %r35, 2, %rd39;
mad.wide.s32 %rd3, %r1, 2, %rd42;
mad.wide.s32 %rd4, %r2, 2, %rd42;
mad.wide.s32 %rd5, %r3, 2, %rd42;
mad.wide.s32 %rd6, %r4, 2, %rd42;
mad.wide.s32 %rd7, %r5, 2, %rd42;
mad.wide.s32 %rd8, %r6, 2, %rd42;
mad.wide.s32 %rd9, %r7, 2, %rd42;
mad.wide.s32 %rd10, %r8, 2, %rd42;
mad.lo.s32 %r36, %r33, 50304, %r34;
mad.wide.s32 %rd11, %r36, 2, %rd41;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd3 + 0 ];
mov.u16 %rs3, %rs2;
@%p2 ld.global.b16 { %rs3 }, [ %rd4 + 0 ];
mov.u16 %rs4, %rs2;
@%p3 ld.global.b16 { %rs4 }, [ %rd5 + 0 ];
mov.u16 %rs5, %rs2;
@%p4 ld.global.b16 { %rs5 }, [ %rd6 + 0 ];
mov.u16 %rs6, %rs2;
@%p5 ld.global.b16 { %rs6 }, [ %rd7 + 0 ];
mov.u16 %rs7, %rs2;
@%p6 ld.global.b16 { %rs7 }, [ %rd8 + 0 ];
mov.u16 %rs8, %rs2;
@%p7 ld.global.b16 { %rs8 }, [ %rd9 + 0 ];
mov.u16 %rs9, %rs2;
@%p8 ld.global.b16 { %rs9 }, [ %rd10 + 0 ];
mov.b32 %r9, {%rs1, %rs3};
mov.b32 %r10, {%rs4, %rs5};
mov.b32 %r11, {%rs6, %rs7};
mov.b32 %r12, {%rs8, %rs9};
st.global.v4.b32 [ %rd11 + 0 ], { %r9, %r10, %r11, %r12 };
add.s64 %rd12, %rd3, @PITCH32@;
add.s64 %rd13, %rd4, @PITCH32@;
add.s64 %rd14, %rd5, @PITCH32@;
add.s64 %rd15, %rd6, @PITCH32@;
add.s64 %rd16, %rd7, @PITCH32@;
add.s64 %rd17, %rd8, @PITCH32@;
add.s64 %rd18, %rd9, @PITCH32@;
add.s64 %rd19, %rd10, @PITCH32@;
mov.u16 %rs10, %rs2;
@%p1 ld.global.b16 { %rs10 }, [ %rd12 + 0 ];
mov.u16 %rs11, %rs2;
@%p2 ld.global.b16 { %rs11 }, [ %rd13 + 0 ];
mov.u16 %rs12, %rs2;
@%p3 ld.global.b16 { %rs12 }, [ %rd14 + 0 ];
mov.u16 %rs13, %rs2;
@%p4 ld.global.b16 { %rs13 }, [ %rd15 + 0 ];
mov.u16 %rs14, %rs2;
@%p5 ld.global.b16 { %rs14 }, [ %rd16 + 0 ];
mov.u16 %rs15, %rs2;
@%p6 ld.global.b16 { %rs15 }, [ %rd17 + 0 ];
mov.u16 %rs16, %rs2;
@%p7 ld.global.b16 { %rs16 }, [ %rd18 + 0 ];
mov.u16 %rs17, %rs2;
@%p8 ld.global.b16 { %rs17 }, [ %rd19 + 0 ];
add.s64 %rd20, %rd11, 1609728;
mov.b32 %r13, {%rs10, %rs11};
mov.b32 %r14, {%rs12, %rs13};
mov.b32 %r15, {%rs14, %rs15};
mov.b32 %r16, {%rs16, %rs17};
st.global.v4.b32 [ %rd20 + 0 ], { %r13, %r14, %r15, %r16 };
add.s64 %rd21, %rd3, @PITCH64@;
add.s64 %rd22, %rd4, @PITCH64@;
add.s64 %rd23, %rd5, @PITCH64@;
add.s64 %rd24, %rd6, @PITCH64@;
add.s64 %rd25, %rd7, @PITCH64@;
add.s64 %rd26, %rd8, @PITCH64@;
add.s64 %rd27, %rd9, @PITCH64@;
add.s64 %rd28, %rd10, @PITCH64@;
mov.u16 %rs18, %rs2;
@%p1 ld.global.b16 { %rs18 }, [ %rd21 + 0 ];
mov.u16 %rs19, %rs2;
@%p2 ld.global.b16 { %rs19 }, [ %rd22 + 0 ];
mov.u16 %rs20, %rs2;
@%p3 ld.global.b16 { %rs20 }, [ %rd23 + 0 ];
mov.u16 %rs21, %rs2;
@%p4 ld.global.b16 { %rs21 }, [ %rd24 + 0 ];
mov.u16 %rs22, %rs2;
@%p5 ld.global.b16 { %rs22 }, [ %rd25 + 0 ];
mov.u16 %rs23, %rs2;
@%p6 ld.global.b16 { %rs23 }, [ %rd26 + 0 ];
mov.u16 %rs24, %rs2;
@%p7 ld.global.b16 { %rs24 }, [ %rd27 + 0 ];
mov.u16 %rs25, %rs2;
@%p8 ld.global.b16 { %rs25 }, [ %rd28 + 0 ];
add.s64 %rd29, %rd11, 3219456;
mov.b32 %r17, {%rs18, %rs19};
mov.b32 %r18, {%rs20, %rs21};
mov.b32 %r19, {%rs22, %rs23};
mov.b32 %r20, {%rs24, %rs25};
st.global.v4.b32 [ %rd29 + 0 ], { %r17, %r18, %r19, %r20 };
add.s64 %rd30, %rd3, @PITCH96@;
add.s64 %rd31, %rd4, @PITCH96@;
add.s64 %rd32, %rd5, @PITCH96@;
add.s64 %rd33, %rd6, @PITCH96@;
add.s64 %rd34, %rd7, @PITCH96@;
add.s64 %rd35, %rd8, @PITCH96@;
add.s64 %rd36, %rd9, @PITCH96@;
add.s64 %rd37, %rd10, @PITCH96@;
mov.u16 %rs26, %rs2;
@%p1 ld.global.b16 { %rs26 }, [ %rd30 + 0 ];
mov.u16 %rs27, %rs2;
@%p2 ld.global.b16 { %rs27 }, [ %rd31 + 0 ];
mov.u16 %rs28, %rs2;
@%p3 ld.global.b16 { %rs28 }, [ %rd32 + 0 ];
mov.u16 %rs29, %rs2;
@%p4 ld.global.b16 { %rs29 }, [ %rd33 + 0 ];
mov.u16 %rs30, %rs2;
@%p5 ld.global.b16 { %rs30 }, [ %rd34 + 0 ];
mov.u16 %rs31, %rs2;
@%p6 ld.global.b16 { %rs31 }, [ %rd35 + 0 ];
mov.u16 %rs32, %rs2;
@%p7 ld.global.b16 { %rs32 }, [ %rd36 + 0 ];
mov.u16 %rs33, %rs2;
@%p8 ld.global.b16 { %rs33 }, [ %rd37 + 0 ];
add.s64 %rd38, %rd11, 4829184;
mov.b32 %r21, {%rs26, %rs27};
mov.b32 %r22, {%rs28, %rs29};
mov.b32 %r23, {%rs30, %rs31};
mov.b32 %r24, {%rs32, %rs33};
st.global.v4.b32 [ %rd38 + 0 ], { %r21, %r22, %r23, %r24 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

structure Artifacts where
  referenceIR : String
  referencePTX : String
  hopperIR : String
  hopperPTX : String
  deriving BEq

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

private def fillConstants (c : Count) (s : String) : String :=
  s.replace "@COUNT@" (toString c.rows)
    |>.replace "@PITCH16@" (toString (16 * c.rows))
    |>.replace "@PITCH32@" (toString (32 * c.rows))
    |>.replace "@PITCH48@" (toString (48 * c.rows))
    |>.replace "@PITCH64@" (toString (64 * c.rows))
    |>.replace "@PITCH96@" (toString (96 * c.rows))

def matchesArtifacts (c : Count) (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines referenceIR) &&
    QKVTTIR.normalizePTX a.referencePTX == lines referencePTX &&
    QKVTTIR.normalizeTTIR a.hopperIR == some (lines (fillConstants c hopperIRTemplate)) &&
    QKVTTIR.normalizePTX a.hopperPTX == lines (fillConstants c hopperPTXTemplate)

structure Certificate (c : Count) (a : Artifacts) : Type where
  matched : matchesArtifacts c a = true
  coverage : ∀ (r : Channel) (v : ClassID), (point r v).row = r ∧ (point r v).col = v
  owners : ∀ t : Thread, owner (cell t) = t
  unique : ∀ (p q : Point), destOffset p = destOffset q → p = q
  bounds : ∀ (p : Point) (i : Fin c.rows),
    sourceOffset c p.row i + 1 < 2 * 768 * c.rows ∧ destOffset p + 1 < 2 * 768 * 50304
  stores : ∀ (src : Nat → UInt16) (pos : PositionMap c) (p : Point),
    emitted src pos p = reference src pos p
  kept : ∀ (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (src : Nat → UInt16) (row : Channel) (i : Fin c.rows),
    let p := point row (stagedClasses c mark h i)
    emitted src (inverseMap (stagedClasses c mark h)) p =
      (destOffset p, src (sourceOffset c row i))
  zeros : ∀ (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (src : Nat → UInt16) (row : Channel) (col : ClassID), mark col = false →
    let p := point row col
    emitted src (inverseMap (stagedClasses c mark h)) p = (destOffset p, 0)

def certify (c : Count) (a : Artifacts) : Option (Certificate c a) :=
  if h : matchesArtifacts c a = true then
    some ⟨h, covers, owner_unique, dest_address_unique, addresses_in_range c,
      emitted_equal, restores_selected c, clears_unselected c⟩
  else none

structure Program (c : Count) where
  artifacts : Artifacts
  checked : Certificate c artifacts

def Program.check (c : Count) (a : Artifacts) : Option (Program c) := do
  return ⟨a, ← certify c a⟩

abbrev Programs := (c : Count) → Program c

def Programs.run (programs : Programs) : Dispatch := fun fast c src positions p =>
  if fast && matchesArtifacts c (programs c).artifacts then emitted src positions p
  else reference src positions p

theorem Programs.run_eq (programs : Programs) :
    programs.run = fun _ _ src positions p => reference src positions p := by
  funext fast c src positions p
  simp only [Programs.run, (programs c).checked.matched, Bool.and_true]
  cases fast
  · rfl
  · exact (programs c).checked.stores src positions p

#axiomsClean certify
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.Densify
