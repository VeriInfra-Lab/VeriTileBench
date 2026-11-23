import models.NanoGPT.Track1.MuddBackward
import models.NanoGPT.Track1.MuddNormBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddBackward.Binding
open VeriTile.Spec
open scoped VeriTile.Spec
open MuddNorm (BackwardLayout Native BF16)

private def gradientAssembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry mudd_backward(
.param .u64 .ptr .global .align 1 mudd_backward_param_0,
.param .u64 .ptr .global .align 1 mudd_backward_param_1,
.param .u64 .ptr .global .align 1 mudd_backward_param_2,
.param .u64 .ptr .global .align 1 mudd_backward_param_3,
.param .u64 .ptr .global .align 1 mudd_backward_param_4
)
.reqntid 32
{
.reg .pred %p<3>;
.reg .b16 %rs<33>;
.reg .b32 %r<242>;
.reg .b64 %rd<19>;
$L__func_begin0:
ld.param.b64 %rd14, [mudd_backward_param_0];
ld.param.b64 %rd15, [mudd_backward_param_1];
$L__tmp0:
mov.u32 %r139, %tid.x;
shl.b32 %r140, %r139, 3;
ld.param.b64 %rd16, [mudd_backward_param_2];
and.b32 %r141, %r140, 248;
mov.u32 %r142, %ctaid.x;
mul.lo.s32 %r143, %r142, 768;
mad.wide.s32 %rd17, %r143, 2, %rd14;
mad.wide.u32 %rd1, %r141, 2, %rd17;
add.s64 %rd2, %rd1, 512;
add.s64 %rd3, %rd1, 1024;
add.s64 %rd4, %rd1, 1536;
mov.b32 %r5, 0;
mov.pred %p1, -1;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
mov.b32 {%rs3, %rs4}, %r2;
mov.b32 {%rs5, %rs6}, %r3;
mov.b32 {%rs7, %rs8}, %r4;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.b32 {%rs9, %rs10}, %r6;
mov.b32 {%rs11, %rs12}, %r7;
mov.b32 {%rs13, %rs14}, %r8;
mov.b32 {%rs15, %rs16}, %r9;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs17, %rs18}, %r10;
mov.b32 {%rs19, %rs20}, %r11;
mov.b32 {%rs21, %rs22}, %r12;
mov.b32 {%rs23, %rs24}, %r13;
mov.pred %p2, 0;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
cvt.f32.bf16 %r144, %rs1;
cvt.f32.bf16 %r145, %rs2;
cvt.f32.bf16 %r146, %rs3;
cvt.f32.bf16 %r147, %rs4;
cvt.f32.bf16 %r148, %rs5;
cvt.f32.bf16 %r149, %rs6;
cvt.f32.bf16 %r150, %rs7;
cvt.f32.bf16 %r151, %rs8;
cvt.f32.bf16 %r152, %rs9;
cvt.f32.bf16 %r153, %rs10;
cvt.f32.bf16 %r154, %rs11;
cvt.f32.bf16 %r155, %rs12;
cvt.f32.bf16 %r156, %rs13;
cvt.f32.bf16 %r157, %rs14;
cvt.f32.bf16 %r158, %rs15;
cvt.f32.bf16 %r159, %rs16;
cvt.f32.bf16 %r160, %rs17;
cvt.f32.bf16 %r161, %rs18;
cvt.f32.bf16 %r162, %rs19;
cvt.f32.bf16 %r163, %rs20;
cvt.f32.bf16 %r164, %rs21;
cvt.f32.bf16 %r165, %rs22;
cvt.f32.bf16 %r166, %rs23;
cvt.f32.bf16 %r167, %rs24;
mad.wide.s32 %rd18, %r143, 4, %rd15;
mad.wide.u32 %rd5, %r141, 4, %rd18;
add.s64 %rd6, %rd5, 16;
add.s64 %rd7, %rd5, 1024;
add.s64 %rd8, %rd5, 1040;
add.s64 %rd9, %rd5, 2048;
add.s64 %rd10, %rd5, 2064;
add.s64 %rd11, %rd5, 3072;
add.s64 %rd12, %rd5, 3088;
mov.u32 %r18, %r5;
mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
mov.u32 %r22, %r5;
mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
@%p1 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
mov.u32 %r26, %r5;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
@%p1 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
mov.u32 %r30, %r5;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
@%p1 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
mov.u32 %r34, %r5;
mov.u32 %r35, %r5;
mov.u32 %r36, %r5;
mov.u32 %r37, %r5;
@%p1 ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd9 + 0 ];
mov.u32 %r38, %r5;
mov.u32 %r39, %r5;
mov.u32 %r40, %r5;
mov.u32 %r41, %r5;
@%p1 ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd10 + 0 ];
mov.u32 %r42, %r5;
mov.u32 %r43, %r5;
mov.u32 %r44, %r5;
mov.u32 %r45, %r5;
@%p2 ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd11 + 0 ];
mov.u32 %r46, %r5;
mov.u32 %r47, %r5;
mov.u32 %r48, %r5;
mov.u32 %r49, %r5;
@%p2 ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd12 + 0 ];
mad.wide.u32 %rd13, %r142, 4, %rd16;
mov.u32 %r50, 0x0;
ld.global.b32 { %r50 }, [ %rd13 + 0 ];
mul.f32 %r168, %r145, %r19;
$L__tmp1:
fma.rn.f32 %r169, %r144, %r18, %r168;
fma.rn.f32 %r170, %r146, %r20, %r169;
fma.rn.f32 %r171, %r147, %r21, %r170;
fma.rn.f32 %r172, %r148, %r22, %r171;
fma.rn.f32 %r173, %r149, %r23, %r172;
fma.rn.f32 %r174, %r150, %r24, %r173;
fma.rn.f32 %r175, %r151, %r25, %r174;
fma.rn.f32 %r176, %r152, %r26, %r175;
fma.rn.f32 %r177, %r153, %r27, %r176;
fma.rn.f32 %r178, %r154, %r28, %r177;
fma.rn.f32 %r179, %r155, %r29, %r178;
fma.rn.f32 %r180, %r156, %r30, %r179;
fma.rn.f32 %r181, %r157, %r31, %r180;
fma.rn.f32 %r182, %r158, %r32, %r181;
fma.rn.f32 %r183, %r159, %r33, %r182;
fma.rn.f32 %r184, %r160, %r34, %r183;
fma.rn.f32 %r185, %r161, %r35, %r184;
fma.rn.f32 %r186, %r162, %r36, %r185;
fma.rn.f32 %r187, %r163, %r37, %r186;
fma.rn.f32 %r188, %r164, %r38, %r187;
fma.rn.f32 %r189, %r165, %r39, %r188;
fma.rn.f32 %r190, %r166, %r40, %r189;
fma.rn.f32 %r191, %r167, %r41, %r190;
add.f32 %r192, %r191, 0f00000000;
$L__tmp2:
shfl.sync.bfly.b32 %r193, %r192, 16, 31, -1;
$L__tmp3:
add.f32 %r194, %r192, %r193;
$L__tmp4:
shfl.sync.bfly.b32 %r195, %r194, 8, 31, -1;
$L__tmp5:
add.f32 %r196, %r194, %r195;
$L__tmp6:
shfl.sync.bfly.b32 %r197, %r196, 4, 31, -1;
$L__tmp7:
add.f32 %r198, %r196, %r197;
$L__tmp8:
shfl.sync.bfly.b32 %r199, %r198, 2, 31, -1;
$L__tmp9:
add.f32 %r200, %r198, %r199;
$L__tmp10:
shfl.sync.bfly.b32 %r201, %r200, 1, 31, -1;
$L__tmp11:
add.f32 %r52, %r200, %r201;
$L__tmp12:
mul.rn.f32 %r51, %r50, %r50;
$L__tmp13:
mul.rn.f32 %r55, %r50, %r51;
mov.b32 %r53, -1090519040;
$L__tmp14:
mul.rn.f32 %r54, %r52, %r53;
$L__tmp15:
mul.rn.f32 %r56, %r54, %r55;
mov.b32 %r57, 984263339;
$L__tmp16:
mul.rn.f32 %r60, %r56, %r57;
$L__tmp17:
add.f32 %r59, %r18, %r18;
add.f32 %r62, %r19, %r19;
add.f32 %r64, %r20, %r20;
add.f32 %r66, %r21, %r21;
add.f32 %r68, %r22, %r22;
add.f32 %r70, %r23, %r23;
add.f32 %r72, %r24, %r24;
add.f32 %r74, %r25, %r25;
add.f32 %r76, %r26, %r26;
add.f32 %r78, %r27, %r27;
add.f32 %r80, %r28, %r28;
add.f32 %r82, %r29, %r29;
add.f32 %r84, %r30, %r30;
add.f32 %r86, %r31, %r31;
add.f32 %r88, %r32, %r32;
add.f32 %r90, %r33, %r33;
add.f32 %r92, %r34, %r34;
add.f32 %r94, %r35, %r35;
add.f32 %r96, %r36, %r36;
add.f32 %r98, %r37, %r37;
add.f32 %r100, %r38, %r38;
add.f32 %r102, %r39, %r39;
add.f32 %r104, %r40, %r40;
add.f32 %r106, %r41, %r41;
add.f32 %r108, %r42, %r42;
add.f32 %r110, %r43, %r43;
add.f32 %r112, %r44, %r44;
add.f32 %r114, %r45, %r45;
add.f32 %r116, %r46, %r46;
add.f32 %r118, %r47, %r47;
add.f32 %r120, %r48, %r48;
add.f32 %r122, %r49, %r49;
$L__tmp18:
mul.rn.f32 %r58, %r59, %r60;
mul.rn.f32 %r61, %r62, %r60;
mul.rn.f32 %r63, %r64, %r60;
mul.rn.f32 %r65, %r66, %r60;
mul.rn.f32 %r67, %r68, %r60;
mul.rn.f32 %r69, %r70, %r60;
mul.rn.f32 %r71, %r72, %r60;
mul.rn.f32 %r73, %r74, %r60;
mul.rn.f32 %r75, %r76, %r60;
mul.rn.f32 %r77, %r78, %r60;
mul.rn.f32 %r79, %r80, %r60;
mul.rn.f32 %r81, %r82, %r60;
mul.rn.f32 %r83, %r84, %r60;
mul.rn.f32 %r85, %r86, %r60;
mul.rn.f32 %r87, %r88, %r60;
mul.rn.f32 %r89, %r90, %r60;
mul.rn.f32 %r91, %r92, %r60;
mul.rn.f32 %r93, %r94, %r60;
mul.rn.f32 %r95, %r96, %r60;
mul.rn.f32 %r97, %r98, %r60;
mul.rn.f32 %r99, %r100, %r60;
mul.rn.f32 %r101, %r102, %r60;
mul.rn.f32 %r103, %r104, %r60;
mul.rn.f32 %r105, %r106, %r60;
mul.rn.f32 %r107, %r108, %r60;
mul.rn.f32 %r109, %r110, %r60;
mul.rn.f32 %r111, %r112, %r60;
mul.rn.f32 %r113, %r114, %r60;
mul.rn.f32 %r115, %r116, %r60;
mul.rn.f32 %r117, %r118, %r60;
mul.rn.f32 %r119, %r120, %r60;
mul.rn.f32 %r121, %r122, %r60;
$L__tmp19:
fma.rn.f32 %r202, %r50, %r144, %r58;
fma.rn.f32 %r203, %r50, %r145, %r61;
fma.rn.f32 %r204, %r50, %r146, %r63;
fma.rn.f32 %r205, %r50, %r147, %r65;
fma.rn.f32 %r206, %r50, %r148, %r67;
fma.rn.f32 %r207, %r50, %r149, %r69;
fma.rn.f32 %r208, %r50, %r150, %r71;
fma.rn.f32 %r209, %r50, %r151, %r73;
fma.rn.f32 %r210, %r50, %r152, %r75;
fma.rn.f32 %r211, %r50, %r153, %r77;
fma.rn.f32 %r212, %r50, %r154, %r79;
fma.rn.f32 %r213, %r50, %r155, %r81;
fma.rn.f32 %r214, %r50, %r156, %r83;
fma.rn.f32 %r215, %r50, %r157, %r85;
fma.rn.f32 %r216, %r50, %r158, %r87;
fma.rn.f32 %r217, %r50, %r159, %r89;
fma.rn.f32 %r218, %r50, %r160, %r91;
fma.rn.f32 %r219, %r50, %r161, %r93;
fma.rn.f32 %r220, %r50, %r162, %r95;
fma.rn.f32 %r221, %r50, %r163, %r97;
fma.rn.f32 %r222, %r50, %r164, %r99;
fma.rn.f32 %r223, %r50, %r165, %r101;
fma.rn.f32 %r224, %r50, %r166, %r103;
fma.rn.f32 %r225, %r50, %r167, %r105;
mov.b32 {%rs25, %rs26}, %r14;
cvt.f32.bf16 %r226, %rs26;
cvt.f32.bf16 %r227, %rs25;
fma.rn.f32 %r228, %r50, %r227, %r107;
fma.rn.f32 %r229, %r50, %r226, %r109;
cvt.rn.bf16x2.f32 %r135, %r229, %r228;
mov.b32 {%rs27, %rs28}, %r15;
cvt.f32.bf16 %r230, %rs28;
cvt.f32.bf16 %r231, %rs27;
fma.rn.f32 %r232, %r50, %r231, %r111;
fma.rn.f32 %r233, %r50, %r230, %r113;
cvt.rn.bf16x2.f32 %r136, %r233, %r232;
mov.b32 {%rs29, %rs30}, %r16;
cvt.f32.bf16 %r234, %rs30;
cvt.f32.bf16 %r235, %rs29;
fma.rn.f32 %r236, %r50, %r235, %r115;
fma.rn.f32 %r237, %r50, %r234, %r117;
cvt.rn.bf16x2.f32 %r137, %r237, %r236;
mov.b32 {%rs31, %rs32}, %r17;
cvt.f32.bf16 %r238, %rs32;
cvt.f32.bf16 %r239, %rs31;
fma.rn.f32 %r240, %r50, %r239, %r119;
fma.rn.f32 %r241, %r50, %r238, %r121;
cvt.rn.bf16x2.f32 %r138, %r241, %r240;
cvt.rn.bf16x2.f32 %r123, %r203, %r202;
cvt.rn.bf16x2.f32 %r124, %r205, %r204;
cvt.rn.bf16x2.f32 %r125, %r207, %r206;
cvt.rn.bf16x2.f32 %r126, %r209, %r208;
@%p1 st.global.v4.b32 [ %rd1 + 0 ], { %r123, %r124, %r125, %r126 };
cvt.rn.bf16x2.f32 %r127, %r211, %r210;
cvt.rn.bf16x2.f32 %r128, %r213, %r212;
cvt.rn.bf16x2.f32 %r129, %r215, %r214;
cvt.rn.bf16x2.f32 %r130, %r217, %r216;
@%p1 st.global.v4.b32 [ %rd2 + 0 ], { %r127, %r128, %r129, %r130 };
cvt.rn.bf16x2.f32 %r131, %r219, %r218;
cvt.rn.bf16x2.f32 %r132, %r221, %r220;
cvt.rn.bf16x2.f32 %r133, %r223, %r222;
cvt.rn.bf16x2.f32 %r134, %r225, %r224;
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r131, %r132, %r133, %r134 };
@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r135, %r136, %r137, %r138 };
ret;
$L__tmp20:
$L__func_end0:
}
"#

private def encodeAssembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry mudd_backward(
.param .u64 .ptr .global .align 1 mudd_backward_param_0,
.param .u64 .ptr .global .align 1 mudd_backward_param_1,
.param .u64 .ptr .global .align 1 mudd_backward_param_2,
.param .u64 .ptr .global .align 1 mudd_backward_param_3,
.param .u64 .ptr .global .align 1 mudd_backward_param_4,
.param .u64 .ptr .global .align 1 mudd_backward_param_5
)
.reqntid 32
{
.reg .pred %p<3>;
.reg .b16 %rs<49>;
.reg .b32 %r<338>;
.reg .b64 %rd<25>;
$L__func_begin0:
ld.param.b64 %rd16, [mudd_backward_param_0];
ld.param.b64 %rd17, [mudd_backward_param_1];
$L__tmp0:
mov.u32 %r243, %tid.x;
shl.b32 %r244, %r243, 4;
ld.param.b64 %rd18, [mudd_backward_param_2];
and.b32 %r245, %r244, 496;
ld.param.b64 %rd19, [mudd_backward_param_3];
or.b32 %r246, %r245, 512;
mov.u32 %r247, %ctaid.x;
mul.lo.s32 %r248, %r247, 768;
setp.lt.u32 %p2, %r246, 768;
cvt.s64.s32 %rd20, %r248;
mad.wide.s32 %rd21, %r248, 2, %rd16;
cvt.u64.u32 %rd22, %r245;
mad.wide.u32 %rd1, %r245, 2, %rd21;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd1, 1024;
add.s64 %rd4, %rd1, 1040;
mov.b32 %r5, 0;
mov.pred %p1, -1;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs17, %rs18}, %r1;
mov.b32 {%rs19, %rs20}, %r2;
mov.b32 {%rs21, %rs22}, %r3;
mov.b32 {%rs23, %rs24}, %r4;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.b32 {%rs25, %rs26}, %r6;
mov.b32 {%rs27, %rs28}, %r7;
mov.b32 {%rs29, %rs30}, %r8;
mov.b32 {%rs31, %rs32}, %r9;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs33, %rs34}, %r10;
mov.b32 {%rs35, %rs36}, %r11;
mov.b32 {%rs37, %rs38}, %r12;
mov.b32 {%rs39, %rs40}, %r13;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs41, %rs42}, %r14;
mov.b32 {%rs43, %rs44}, %r15;
mov.b32 {%rs45, %rs46}, %r16;
mov.b32 {%rs47, %rs48}, %r17;
cvt.f32.bf16 %r249, %rs17;
cvt.f32.bf16 %r250, %rs18;
cvt.f32.bf16 %r251, %rs19;
cvt.f32.bf16 %r252, %rs20;
cvt.f32.bf16 %r253, %rs21;
cvt.f32.bf16 %r254, %rs22;
cvt.f32.bf16 %r255, %rs23;
cvt.f32.bf16 %r256, %rs24;
cvt.f32.bf16 %r257, %rs25;
cvt.f32.bf16 %r258, %rs26;
cvt.f32.bf16 %r259, %rs27;
cvt.f32.bf16 %r260, %rs28;
cvt.f32.bf16 %r261, %rs29;
cvt.f32.bf16 %r262, %rs30;
cvt.f32.bf16 %r263, %rs31;
cvt.f32.bf16 %r264, %rs32;
cvt.f32.bf16 %r19, %rs33;
cvt.f32.bf16 %r22, %rs34;
cvt.f32.bf16 %r25, %rs35;
cvt.f32.bf16 %r28, %rs36;
cvt.f32.bf16 %r31, %rs37;
cvt.f32.bf16 %r34, %rs38;
cvt.f32.bf16 %r37, %rs39;
cvt.f32.bf16 %r40, %rs40;
cvt.f32.bf16 %r43, %rs41;
cvt.f32.bf16 %r46, %rs42;
cvt.f32.bf16 %r49, %rs43;
cvt.f32.bf16 %r52, %rs44;
cvt.f32.bf16 %r55, %rs45;
cvt.f32.bf16 %r58, %rs46;
cvt.f32.bf16 %r61, %rs47;
cvt.f32.bf16 %r64, %rs48;
mad.wide.s32 %rd23, %r248, 4, %rd17;
mad.wide.u32 %rd5, %r245, 4, %rd23;
add.s64 %rd6, %rd5, 16;
add.s64 %rd7, %rd5, 32;
add.s64 %rd8, %rd5, 48;
add.s64 %rd9, %rd5, 2048;
add.s64 %rd10, %rd5, 2064;
add.s64 %rd11, %rd5, 2080;
add.s64 %rd12, %rd5, 2096;
mov.u32 %r156, %r5;
mov.u32 %r158, %r5;
mov.u32 %r160, %r5;
mov.u32 %r162, %r5;
@%p1 ld.global.v4.b32 { %r156, %r158, %r160, %r162 }, [ %rd5 + 0 ];
mov.u32 %r164, %r5;
mov.u32 %r166, %r5;
mov.u32 %r168, %r5;
mov.u32 %r170, %r5;
@%p1 ld.global.v4.b32 { %r164, %r166, %r168, %r170 }, [ %rd6 + 0 ];
mov.u32 %r172, %r5;
mov.u32 %r174, %r5;
mov.u32 %r176, %r5;
mov.u32 %r178, %r5;
@%p1 ld.global.v4.b32 { %r172, %r174, %r176, %r178 }, [ %rd7 + 0 ];
mov.u32 %r180, %r5;
mov.u32 %r182, %r5;
mov.u32 %r184, %r5;
mov.u32 %r186, %r5;
@%p1 ld.global.v4.b32 { %r180, %r182, %r184, %r186 }, [ %rd8 + 0 ];
mov.u32 %r20, %r5;
mov.u32 %r23, %r5;
mov.u32 %r26, %r5;
mov.u32 %r29, %r5;
@%p2 ld.global.v4.b32 { %r20, %r23, %r26, %r29 }, [ %rd9 + 0 ];
mov.u32 %r32, %r5;
mov.u32 %r35, %r5;
mov.u32 %r38, %r5;
mov.u32 %r41, %r5;
@%p2 ld.global.v4.b32 { %r32, %r35, %r38, %r41 }, [ %rd10 + 0 ];
mov.u32 %r44, %r5;
mov.u32 %r47, %r5;
mov.u32 %r50, %r5;
mov.u32 %r53, %r5;
@%p2 ld.global.v4.b32 { %r44, %r47, %r50, %r53 }, [ %rd11 + 0 ];
mov.u32 %r56, %r5;
mov.u32 %r59, %r5;
mov.u32 %r62, %r5;
mov.u32 %r65, %r5;
@%p2 ld.global.v4.b32 { %r56, %r59, %r62, %r65 }, [ %rd12 + 0 ];
mad.wide.u32 %rd13, %r247, 4, %rd18;
mov.u32 %r66, 0x0;
ld.global.b32 { %r66 }, [ %rd13 + 0 ];
mul.f32 %r265, %r250, %r158;
$L__tmp1:
mul.rn.f32 %r18, %r19, %r20;
mul.rn.f32 %r21, %r22, %r23;
mul.rn.f32 %r24, %r25, %r26;
mul.rn.f32 %r27, %r28, %r29;
mul.rn.f32 %r30, %r31, %r32;
mul.rn.f32 %r33, %r34, %r35;
mul.rn.f32 %r36, %r37, %r38;
mul.rn.f32 %r39, %r40, %r41;
mul.rn.f32 %r42, %r43, %r44;
mul.rn.f32 %r45, %r46, %r47;
mul.rn.f32 %r48, %r49, %r50;
mul.rn.f32 %r51, %r52, %r53;
mul.rn.f32 %r54, %r55, %r56;
mul.rn.f32 %r57, %r58, %r59;
mul.rn.f32 %r60, %r61, %r62;
mul.rn.f32 %r63, %r64, %r65;
$L__tmp2:
fma.rn.f32 %r266, %r249, %r156, %r265;
fma.rn.f32 %r267, %r251, %r160, %r266;
fma.rn.f32 %r268, %r252, %r162, %r267;
fma.rn.f32 %r269, %r253, %r164, %r268;
fma.rn.f32 %r270, %r254, %r166, %r269;
fma.rn.f32 %r271, %r255, %r168, %r270;
fma.rn.f32 %r272, %r256, %r170, %r271;
fma.rn.f32 %r273, %r257, %r172, %r272;
fma.rn.f32 %r274, %r258, %r174, %r273;
fma.rn.f32 %r275, %r259, %r176, %r274;
fma.rn.f32 %r276, %r260, %r178, %r275;
fma.rn.f32 %r277, %r261, %r180, %r276;
fma.rn.f32 %r278, %r262, %r182, %r277;
fma.rn.f32 %r279, %r263, %r184, %r278;
fma.rn.f32 %r280, %r264, %r186, %r279;
add.f32 %r281, %r18, %r280;
add.f32 %r282, %r21, %r281;
add.f32 %r283, %r24, %r282;
add.f32 %r284, %r27, %r283;
add.f32 %r285, %r30, %r284;
add.f32 %r286, %r33, %r285;
add.f32 %r287, %r36, %r286;
add.f32 %r288, %r39, %r287;
add.f32 %r289, %r42, %r288;
add.f32 %r290, %r45, %r289;
add.f32 %r291, %r48, %r290;
add.f32 %r292, %r51, %r291;
add.f32 %r293, %r54, %r292;
add.f32 %r294, %r57, %r293;
add.f32 %r295, %r60, %r294;
add.f32 %r296, %r63, %r295;
$L__tmp3:
shfl.sync.bfly.b32 %r297, %r296, 16, 31, -1;
$L__tmp4:
add.f32 %r298, %r296, %r297;
$L__tmp5:
shfl.sync.bfly.b32 %r299, %r298, 8, 31, -1;
$L__tmp6:
add.f32 %r300, %r298, %r299;
$L__tmp7:
shfl.sync.bfly.b32 %r301, %r300, 4, 31, -1;
$L__tmp8:
add.f32 %r302, %r300, %r301;
$L__tmp9:
shfl.sync.bfly.b32 %r303, %r302, 2, 31, -1;
$L__tmp10:
add.f32 %r304, %r302, %r303;
$L__tmp11:
shfl.sync.bfly.b32 %r305, %r304, 1, 31, -1;
$L__tmp12:
add.f32 %r68, %r304, %r305;
$L__tmp13:
mul.rn.f32 %r67, %r66, %r66;
$L__tmp14:
mul.rn.f32 %r71, %r66, %r67;
mov.b32 %r69, -1090519040;
$L__tmp15:
mul.rn.f32 %r70, %r68, %r69;
$L__tmp16:
mul.rn.f32 %r72, %r70, %r71;
mov.b32 %r73, 984263339;
$L__tmp17:
mul.rn.f32 %r76, %r72, %r73;
$L__tmp18:
add.f32 %r75, %r156, %r156;
add.f32 %r78, %r158, %r158;
add.f32 %r80, %r160, %r160;
add.f32 %r82, %r162, %r162;
add.f32 %r84, %r164, %r164;
add.f32 %r86, %r166, %r166;
add.f32 %r88, %r168, %r168;
add.f32 %r90, %r170, %r170;
add.f32 %r92, %r172, %r172;
add.f32 %r94, %r174, %r174;
add.f32 %r96, %r176, %r176;
add.f32 %r98, %r178, %r178;
add.f32 %r100, %r180, %r180;
add.f32 %r102, %r182, %r182;
add.f32 %r104, %r184, %r184;
add.f32 %r106, %r186, %r186;
add.f32 %r108, %r20, %r20;
add.f32 %r110, %r23, %r23;
add.f32 %r112, %r26, %r26;
add.f32 %r114, %r29, %r29;
add.f32 %r116, %r32, %r32;
add.f32 %r118, %r35, %r35;
add.f32 %r120, %r38, %r38;
add.f32 %r122, %r41, %r41;
add.f32 %r124, %r44, %r44;
add.f32 %r126, %r47, %r47;
add.f32 %r128, %r50, %r50;
add.f32 %r130, %r53, %r53;
add.f32 %r132, %r56, %r56;
add.f32 %r134, %r59, %r59;
add.f32 %r136, %r62, %r62;
add.f32 %r138, %r65, %r65;
$L__tmp19:
mul.rn.f32 %r74, %r75, %r76;
mul.rn.f32 %r77, %r78, %r76;
mul.rn.f32 %r79, %r80, %r76;
mul.rn.f32 %r81, %r82, %r76;
mul.rn.f32 %r83, %r84, %r76;
mul.rn.f32 %r85, %r86, %r76;
mul.rn.f32 %r87, %r88, %r76;
mul.rn.f32 %r89, %r90, %r76;
mul.rn.f32 %r91, %r92, %r76;
mul.rn.f32 %r93, %r94, %r76;
mul.rn.f32 %r95, %r96, %r76;
mul.rn.f32 %r97, %r98, %r76;
mul.rn.f32 %r99, %r100, %r76;
mul.rn.f32 %r101, %r102, %r76;
mul.rn.f32 %r103, %r104, %r76;
mul.rn.f32 %r105, %r106, %r76;
mul.rn.f32 %r107, %r108, %r76;
mul.rn.f32 %r109, %r110, %r76;
mul.rn.f32 %r111, %r112, %r76;
mul.rn.f32 %r113, %r114, %r76;
mul.rn.f32 %r115, %r116, %r76;
mul.rn.f32 %r117, %r118, %r76;
mul.rn.f32 %r119, %r120, %r76;
mul.rn.f32 %r121, %r122, %r76;
mul.rn.f32 %r123, %r124, %r76;
mul.rn.f32 %r125, %r126, %r76;
mul.rn.f32 %r127, %r128, %r76;
mul.rn.f32 %r129, %r130, %r76;
mul.rn.f32 %r131, %r132, %r76;
mul.rn.f32 %r133, %r134, %r76;
mul.rn.f32 %r135, %r136, %r76;
mul.rn.f32 %r137, %r138, %r76;
$L__tmp20:
fma.rn.f32 %r306, %r66, %r249, %r74;
fma.rn.f32 %r307, %r66, %r250, %r77;
fma.rn.f32 %r308, %r66, %r251, %r79;
fma.rn.f32 %r309, %r66, %r252, %r81;
fma.rn.f32 %r310, %r66, %r253, %r83;
fma.rn.f32 %r311, %r66, %r254, %r85;
fma.rn.f32 %r312, %r66, %r255, %r87;
fma.rn.f32 %r313, %r66, %r256, %r89;
fma.rn.f32 %r314, %r66, %r257, %r91;
fma.rn.f32 %r315, %r66, %r258, %r93;
fma.rn.f32 %r316, %r66, %r259, %r95;
fma.rn.f32 %r317, %r66, %r260, %r97;
fma.rn.f32 %r318, %r66, %r261, %r99;
fma.rn.f32 %r319, %r66, %r262, %r101;
fma.rn.f32 %r320, %r66, %r263, %r103;
fma.rn.f32 %r321, %r66, %r264, %r105;
fma.rn.f32 %r322, %r66, %r19, %r107;
fma.rn.f32 %r323, %r66, %r22, %r109;
fma.rn.f32 %r324, %r66, %r25, %r111;
fma.rn.f32 %r325, %r66, %r28, %r113;
fma.rn.f32 %r326, %r66, %r31, %r115;
fma.rn.f32 %r327, %r66, %r34, %r117;
fma.rn.f32 %r328, %r66, %r37, %r119;
fma.rn.f32 %r329, %r66, %r40, %r121;
fma.rn.f32 %r330, %r66, %r43, %r123;
fma.rn.f32 %r331, %r66, %r46, %r125;
fma.rn.f32 %r332, %r66, %r49, %r127;
fma.rn.f32 %r333, %r66, %r52, %r129;
fma.rn.f32 %r334, %r66, %r55, %r131;
fma.rn.f32 %r335, %r66, %r58, %r133;
fma.rn.f32 %r336, %r66, %r61, %r135;
fma.rn.f32 %r337, %r66, %r64, %r137;
cvt.rn.bf16x2.f32 %r139, %r307, %r306;
cvt.rn.bf16x2.f32 %r140, %r309, %r308;
cvt.rn.bf16x2.f32 %r141, %r311, %r310;
cvt.rn.bf16x2.f32 %r142, %r313, %r312;
@%p1 st.global.v4.b32 [ %rd1 + 0 ], { %r139, %r140, %r141, %r142 };
cvt.rn.bf16x2.f32 %r143, %r315, %r314;
cvt.rn.bf16x2.f32 %r144, %r317, %r316;
cvt.rn.bf16x2.f32 %r145, %r319, %r318;
cvt.rn.bf16x2.f32 %r146, %r321, %r320;
@%p1 st.global.v4.b32 [ %rd2 + 0 ], { %r143, %r144, %r145, %r146 };
cvt.rn.bf16x2.f32 %r147, %r323, %r322;
cvt.rn.bf16x2.f32 %r148, %r325, %r324;
cvt.rn.bf16x2.f32 %r149, %r327, %r326;
cvt.rn.bf16x2.f32 %r150, %r329, %r328;
@%p2 st.global.v4.b32 [ %rd3 + 0 ], { %r147, %r148, %r149, %r150 };
cvt.rn.bf16x2.f32 %r151, %r331, %r330;
cvt.rn.bf16x2.f32 %r152, %r333, %r332;
cvt.rn.bf16x2.f32 %r153, %r335, %r334;
cvt.rn.bf16x2.f32 %r154, %r337, %r336;
@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r151, %r152, %r153, %r154 };
$L__tmp21:
mul.rn.f32 %r155, %r66, %r156;
mul.rn.f32 %r157, %r66, %r158;
mul.rn.f32 %r159, %r66, %r160;
mul.rn.f32 %r161, %r66, %r162;
mul.rn.f32 %r163, %r66, %r164;
mul.rn.f32 %r165, %r66, %r166;
mul.rn.f32 %r167, %r66, %r168;
mul.rn.f32 %r169, %r66, %r170;
mul.rn.f32 %r171, %r66, %r172;
mul.rn.f32 %r173, %r66, %r174;
mul.rn.f32 %r175, %r66, %r176;
mul.rn.f32 %r177, %r66, %r178;
mul.rn.f32 %r179, %r66, %r180;
mul.rn.f32 %r181, %r66, %r182;
mul.rn.f32 %r183, %r66, %r184;
mul.rn.f32 %r185, %r66, %r186;
mul.rn.f32 %r187, %r66, %r20;
mul.rn.f32 %r188, %r66, %r23;
mul.rn.f32 %r189, %r66, %r26;
mul.rn.f32 %r190, %r66, %r29;
mul.rn.f32 %r191, %r66, %r32;
mul.rn.f32 %r192, %r66, %r35;
mul.rn.f32 %r193, %r66, %r38;
mul.rn.f32 %r194, %r66, %r41;
mul.rn.f32 %r195, %r66, %r44;
mul.rn.f32 %r196, %r66, %r47;
mul.rn.f32 %r197, %r66, %r50;
mul.rn.f32 %r198, %r66, %r53;
mul.rn.f32 %r199, %r66, %r56;
mul.rn.f32 %r200, %r66, %r59;
mul.rn.f32 %r201, %r66, %r62;
mul.rn.f32 %r202, %r66, %r65;
$L__tmp22:
mul.f32 %r203, %r155, 0f408F5C29;
mul.f32 %r204, %r157, 0f408F5C29;
mul.f32 %r205, %r159, 0f408F5C29;
mul.f32 %r206, %r161, 0f408F5C29;
mul.f32 %r207, %r163, 0f408F5C29;
mul.f32 %r208, %r165, 0f408F5C29;
mul.f32 %r209, %r167, 0f408F5C29;
mul.f32 %r210, %r169, 0f408F5C29;
mul.f32 %r211, %r171, 0f408F5C29;
mul.f32 %r212, %r173, 0f408F5C29;
mul.f32 %r213, %r175, 0f408F5C29;
mul.f32 %r214, %r177, 0f408F5C29;
mul.f32 %r215, %r179, 0f408F5C29;
mul.f32 %r216, %r181, 0f408F5C29;
mul.f32 %r217, %r183, 0f408F5C29;
mul.f32 %r218, %r185, 0f408F5C29;
mul.f32 %r219, %r187, 0f408F5C29;
mul.f32 %r220, %r188, 0f408F5C29;
mul.f32 %r221, %r189, 0f408F5C29;
mul.f32 %r222, %r190, 0f408F5C29;
mul.f32 %r223, %r191, 0f408F5C29;
mul.f32 %r224, %r192, 0f408F5C29;
mul.f32 %r225, %r193, 0f408F5C29;
mul.f32 %r226, %r194, 0f408F5C29;
mul.f32 %r227, %r195, 0f408F5C29;
mul.f32 %r228, %r196, 0f408F5C29;
mul.f32 %r229, %r197, 0f408F5C29;
mul.f32 %r230, %r198, 0f408F5C29;
mul.f32 %r231, %r199, 0f408F5C29;
mul.f32 %r232, %r200, 0f408F5C29;
mul.f32 %r233, %r201, 0f408F5C29;
mul.f32 %r234, %r202, 0f408F5C29;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r204, %r203;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r206, %r205;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r208, %r207;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r210, %r209;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r212, %r211;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r214, %r213;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r216, %r215;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r218, %r217;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r220, %r219;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r222, %r221;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r224, %r223;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r226, %r225;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r228, %r227;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r230, %r229;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r232, %r231;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r234, %r233;
add.s64 %rd24, %rd19, %rd20;
add.s64 %rd14, %rd24, %rd22;
add.s64 %rd15, %rd14, 512;
mov.b32 %r235, {%rs1, %rs2};
mov.b32 %r236, {%rs3, %rs4};
mov.b32 %r237, {%rs5, %rs6};
mov.b32 %r238, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd14 + 0 ], { %r235, %r236, %r237, %r238 };
mov.b32 %r239, {%rs9, %rs10};
mov.b32 %r240, {%rs11, %rs12};
mov.b32 %r241, {%rs13, %rs14};
mov.b32 %r242, {%rs15, %rs16};
@%p2 st.global.v4.b32 [ %rd15 + 0 ], { %r239, %r240, %r241, %r242 };
ret;
$L__tmp23:
$L__func_end0:
}
"#

private def retainedAssembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry mudd_backward(
.param .u64 .ptr .global .align 1 mudd_backward_param_0,
.param .u64 .ptr .global .align 1 mudd_backward_param_1,
.param .u64 .ptr .global .align 1 mudd_backward_param_2,
.param .u64 .ptr .global .align 1 mudd_backward_param_3,
.param .u64 .ptr .global .align 1 mudd_backward_param_4
)
.reqntid 32
{
.reg .pred %p<3>;
.reg .b16 %rs<33>;
.reg .b32 %r<266>;
.reg .b64 %rd<19>;
$L__func_begin0:
ld.param.b64 %rd14, [mudd_backward_param_0];
ld.param.b64 %rd15, [mudd_backward_param_1];
$L__tmp0:
mov.u32 %r171, %tid.x;
shl.b32 %r172, %r171, 4;
ld.param.b64 %rd16, [mudd_backward_param_2];
and.b32 %r173, %r172, 496;
or.b32 %r174, %r173, 512;
mov.u32 %r175, %ctaid.x;
mul.lo.s32 %r176, %r175, 768;
setp.lt.u32 %p2, %r174, 768;
mad.wide.s32 %rd17, %r176, 2, %rd14;
mad.wide.u32 %rd1, %r173, 2, %rd17;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd1, 1024;
add.s64 %rd4, %rd1, 1040;
mov.b32 %r5, 0;
mov.pred %p1, -1;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs1, %rs2}, %r1;
mov.b32 {%rs3, %rs4}, %r2;
mov.b32 {%rs5, %rs6}, %r3;
mov.b32 {%rs7, %rs8}, %r4;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.b32 {%rs9, %rs10}, %r6;
mov.b32 {%rs11, %rs12}, %r7;
mov.b32 {%rs13, %rs14}, %r8;
mov.b32 {%rs15, %rs16}, %r9;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs17, %rs18}, %r10;
mov.b32 {%rs19, %rs20}, %r11;
mov.b32 {%rs21, %rs22}, %r12;
mov.b32 {%rs23, %rs24}, %r13;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs25, %rs26}, %r14;
mov.b32 {%rs27, %rs28}, %r15;
mov.b32 {%rs29, %rs30}, %r16;
mov.b32 {%rs31, %rs32}, %r17;
cvt.f32.bf16 %r177, %rs1;
cvt.f32.bf16 %r178, %rs2;
cvt.f32.bf16 %r179, %rs3;
cvt.f32.bf16 %r180, %rs4;
cvt.f32.bf16 %r181, %rs5;
cvt.f32.bf16 %r182, %rs6;
cvt.f32.bf16 %r183, %rs7;
cvt.f32.bf16 %r184, %rs8;
cvt.f32.bf16 %r185, %rs9;
cvt.f32.bf16 %r186, %rs10;
cvt.f32.bf16 %r187, %rs11;
cvt.f32.bf16 %r188, %rs12;
cvt.f32.bf16 %r189, %rs13;
cvt.f32.bf16 %r190, %rs14;
cvt.f32.bf16 %r191, %rs15;
cvt.f32.bf16 %r192, %rs16;
cvt.f32.bf16 %r35, %rs17;
cvt.f32.bf16 %r38, %rs18;
cvt.f32.bf16 %r41, %rs19;
cvt.f32.bf16 %r44, %rs20;
cvt.f32.bf16 %r47, %rs21;
cvt.f32.bf16 %r50, %rs22;
cvt.f32.bf16 %r53, %rs23;
cvt.f32.bf16 %r56, %rs24;
cvt.f32.bf16 %r59, %rs25;
cvt.f32.bf16 %r62, %rs26;
cvt.f32.bf16 %r65, %rs27;
cvt.f32.bf16 %r68, %rs28;
cvt.f32.bf16 %r71, %rs29;
cvt.f32.bf16 %r74, %rs30;
cvt.f32.bf16 %r77, %rs31;
cvt.f32.bf16 %r80, %rs32;
mad.wide.s32 %rd18, %r176, 4, %rd15;
mad.wide.u32 %rd5, %r173, 4, %rd18;
add.s64 %rd6, %rd5, 16;
add.s64 %rd7, %rd5, 32;
add.s64 %rd8, %rd5, 48;
add.s64 %rd9, %rd5, 2048;
add.s64 %rd10, %rd5, 2064;
add.s64 %rd11, %rd5, 2080;
add.s64 %rd12, %rd5, 2096;
mov.u32 %r18, %r5;
mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
mov.u32 %r22, %r5;
mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
@%p1 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
mov.u32 %r26, %r5;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
@%p1 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
mov.u32 %r30, %r5;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
@%p1 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
mov.u32 %r36, %r5;
mov.u32 %r39, %r5;
mov.u32 %r42, %r5;
mov.u32 %r45, %r5;
@%p2 ld.global.v4.b32 { %r36, %r39, %r42, %r45 }, [ %rd9 + 0 ];
mov.u32 %r48, %r5;
mov.u32 %r51, %r5;
mov.u32 %r54, %r5;
mov.u32 %r57, %r5;
@%p2 ld.global.v4.b32 { %r48, %r51, %r54, %r57 }, [ %rd10 + 0 ];
mov.u32 %r60, %r5;
mov.u32 %r63, %r5;
mov.u32 %r66, %r5;
mov.u32 %r69, %r5;
@%p2 ld.global.v4.b32 { %r60, %r63, %r66, %r69 }, [ %rd11 + 0 ];
mov.u32 %r72, %r5;
mov.u32 %r75, %r5;
mov.u32 %r78, %r5;
mov.u32 %r81, %r5;
@%p2 ld.global.v4.b32 { %r72, %r75, %r78, %r81 }, [ %rd12 + 0 ];
mad.wide.u32 %rd13, %r175, 4, %rd16;
mov.u32 %r82, 0x0;
ld.global.b32 { %r82 }, [ %rd13 + 0 ];
mul.f32 %r193, %r178, %r19;
$L__tmp1:
mul.rn.f32 %r34, %r35, %r36;
mul.rn.f32 %r37, %r38, %r39;
mul.rn.f32 %r40, %r41, %r42;
mul.rn.f32 %r43, %r44, %r45;
mul.rn.f32 %r46, %r47, %r48;
mul.rn.f32 %r49, %r50, %r51;
mul.rn.f32 %r52, %r53, %r54;
mul.rn.f32 %r55, %r56, %r57;
mul.rn.f32 %r58, %r59, %r60;
mul.rn.f32 %r61, %r62, %r63;
mul.rn.f32 %r64, %r65, %r66;
mul.rn.f32 %r67, %r68, %r69;
mul.rn.f32 %r70, %r71, %r72;
mul.rn.f32 %r73, %r74, %r75;
mul.rn.f32 %r76, %r77, %r78;
mul.rn.f32 %r79, %r80, %r81;
$L__tmp2:
fma.rn.f32 %r194, %r177, %r18, %r193;
fma.rn.f32 %r195, %r179, %r20, %r194;
fma.rn.f32 %r196, %r180, %r21, %r195;
fma.rn.f32 %r197, %r181, %r22, %r196;
fma.rn.f32 %r198, %r182, %r23, %r197;
fma.rn.f32 %r199, %r183, %r24, %r198;
fma.rn.f32 %r200, %r184, %r25, %r199;
fma.rn.f32 %r201, %r185, %r26, %r200;
fma.rn.f32 %r202, %r186, %r27, %r201;
fma.rn.f32 %r203, %r187, %r28, %r202;
fma.rn.f32 %r204, %r188, %r29, %r203;
fma.rn.f32 %r205, %r189, %r30, %r204;
fma.rn.f32 %r206, %r190, %r31, %r205;
fma.rn.f32 %r207, %r191, %r32, %r206;
fma.rn.f32 %r208, %r192, %r33, %r207;
add.f32 %r209, %r34, %r208;
add.f32 %r210, %r37, %r209;
add.f32 %r211, %r40, %r210;
add.f32 %r212, %r43, %r211;
add.f32 %r213, %r46, %r212;
add.f32 %r214, %r49, %r213;
add.f32 %r215, %r52, %r214;
add.f32 %r216, %r55, %r215;
add.f32 %r217, %r58, %r216;
add.f32 %r218, %r61, %r217;
add.f32 %r219, %r64, %r218;
add.f32 %r220, %r67, %r219;
add.f32 %r221, %r70, %r220;
add.f32 %r222, %r73, %r221;
add.f32 %r223, %r76, %r222;
add.f32 %r224, %r79, %r223;
$L__tmp3:
shfl.sync.bfly.b32 %r225, %r224, 16, 31, -1;
$L__tmp4:
add.f32 %r226, %r224, %r225;
$L__tmp5:
shfl.sync.bfly.b32 %r227, %r226, 8, 31, -1;
$L__tmp6:
add.f32 %r228, %r226, %r227;
$L__tmp7:
shfl.sync.bfly.b32 %r229, %r228, 4, 31, -1;
$L__tmp8:
add.f32 %r230, %r228, %r229;
$L__tmp9:
shfl.sync.bfly.b32 %r231, %r230, 2, 31, -1;
$L__tmp10:
add.f32 %r232, %r230, %r231;
$L__tmp11:
shfl.sync.bfly.b32 %r233, %r232, 1, 31, -1;
$L__tmp12:
add.f32 %r84, %r232, %r233;
$L__tmp13:
mul.rn.f32 %r83, %r82, %r82;
$L__tmp14:
mul.rn.f32 %r87, %r82, %r83;
mov.b32 %r85, -1090519040;
$L__tmp15:
mul.rn.f32 %r86, %r84, %r85;
$L__tmp16:
mul.rn.f32 %r88, %r86, %r87;
mov.b32 %r89, 984263339;
$L__tmp17:
mul.rn.f32 %r92, %r88, %r89;
$L__tmp18:
add.f32 %r91, %r18, %r18;
add.f32 %r94, %r19, %r19;
add.f32 %r96, %r20, %r20;
add.f32 %r98, %r21, %r21;
add.f32 %r100, %r22, %r22;
add.f32 %r102, %r23, %r23;
add.f32 %r104, %r24, %r24;
add.f32 %r106, %r25, %r25;
add.f32 %r108, %r26, %r26;
add.f32 %r110, %r27, %r27;
add.f32 %r112, %r28, %r28;
add.f32 %r114, %r29, %r29;
add.f32 %r116, %r30, %r30;
add.f32 %r118, %r31, %r31;
add.f32 %r120, %r32, %r32;
add.f32 %r122, %r33, %r33;
add.f32 %r124, %r36, %r36;
add.f32 %r126, %r39, %r39;
add.f32 %r128, %r42, %r42;
add.f32 %r130, %r45, %r45;
add.f32 %r132, %r48, %r48;
add.f32 %r134, %r51, %r51;
add.f32 %r136, %r54, %r54;
add.f32 %r138, %r57, %r57;
add.f32 %r140, %r60, %r60;
add.f32 %r142, %r63, %r63;
add.f32 %r144, %r66, %r66;
add.f32 %r146, %r69, %r69;
add.f32 %r148, %r72, %r72;
add.f32 %r150, %r75, %r75;
add.f32 %r152, %r78, %r78;
add.f32 %r154, %r81, %r81;
$L__tmp19:
mul.rn.f32 %r90, %r91, %r92;
mul.rn.f32 %r93, %r94, %r92;
mul.rn.f32 %r95, %r96, %r92;
mul.rn.f32 %r97, %r98, %r92;
mul.rn.f32 %r99, %r100, %r92;
mul.rn.f32 %r101, %r102, %r92;
mul.rn.f32 %r103, %r104, %r92;
mul.rn.f32 %r105, %r106, %r92;
mul.rn.f32 %r107, %r108, %r92;
mul.rn.f32 %r109, %r110, %r92;
mul.rn.f32 %r111, %r112, %r92;
mul.rn.f32 %r113, %r114, %r92;
mul.rn.f32 %r115, %r116, %r92;
mul.rn.f32 %r117, %r118, %r92;
mul.rn.f32 %r119, %r120, %r92;
mul.rn.f32 %r121, %r122, %r92;
mul.rn.f32 %r123, %r124, %r92;
mul.rn.f32 %r125, %r126, %r92;
mul.rn.f32 %r127, %r128, %r92;
mul.rn.f32 %r129, %r130, %r92;
mul.rn.f32 %r131, %r132, %r92;
mul.rn.f32 %r133, %r134, %r92;
mul.rn.f32 %r135, %r136, %r92;
mul.rn.f32 %r137, %r138, %r92;
mul.rn.f32 %r139, %r140, %r92;
mul.rn.f32 %r141, %r142, %r92;
mul.rn.f32 %r143, %r144, %r92;
mul.rn.f32 %r145, %r146, %r92;
mul.rn.f32 %r147, %r148, %r92;
mul.rn.f32 %r149, %r150, %r92;
mul.rn.f32 %r151, %r152, %r92;
mul.rn.f32 %r153, %r154, %r92;
$L__tmp20:
fma.rn.f32 %r234, %r82, %r177, %r90;
fma.rn.f32 %r235, %r82, %r178, %r93;
fma.rn.f32 %r236, %r82, %r179, %r95;
fma.rn.f32 %r237, %r82, %r180, %r97;
fma.rn.f32 %r238, %r82, %r181, %r99;
fma.rn.f32 %r239, %r82, %r182, %r101;
fma.rn.f32 %r240, %r82, %r183, %r103;
fma.rn.f32 %r241, %r82, %r184, %r105;
fma.rn.f32 %r242, %r82, %r185, %r107;
fma.rn.f32 %r243, %r82, %r186, %r109;
fma.rn.f32 %r244, %r82, %r187, %r111;
fma.rn.f32 %r245, %r82, %r188, %r113;
fma.rn.f32 %r246, %r82, %r189, %r115;
fma.rn.f32 %r247, %r82, %r190, %r117;
fma.rn.f32 %r248, %r82, %r191, %r119;
fma.rn.f32 %r249, %r82, %r192, %r121;
fma.rn.f32 %r250, %r82, %r35, %r123;
fma.rn.f32 %r251, %r82, %r38, %r125;
fma.rn.f32 %r252, %r82, %r41, %r127;
fma.rn.f32 %r253, %r82, %r44, %r129;
fma.rn.f32 %r254, %r82, %r47, %r131;
fma.rn.f32 %r255, %r82, %r50, %r133;
fma.rn.f32 %r256, %r82, %r53, %r135;
fma.rn.f32 %r257, %r82, %r56, %r137;
fma.rn.f32 %r258, %r82, %r59, %r139;
fma.rn.f32 %r259, %r82, %r62, %r141;
fma.rn.f32 %r260, %r82, %r65, %r143;
fma.rn.f32 %r261, %r82, %r68, %r145;
fma.rn.f32 %r262, %r82, %r71, %r147;
fma.rn.f32 %r263, %r82, %r74, %r149;
fma.rn.f32 %r264, %r82, %r77, %r151;
fma.rn.f32 %r265, %r82, %r80, %r153;
cvt.rn.bf16x2.f32 %r155, %r235, %r234;
cvt.rn.bf16x2.f32 %r156, %r237, %r236;
cvt.rn.bf16x2.f32 %r157, %r239, %r238;
cvt.rn.bf16x2.f32 %r158, %r241, %r240;
@%p1 st.global.v4.b32 [ %rd1 + 0 ], { %r155, %r156, %r157, %r158 };
cvt.rn.bf16x2.f32 %r159, %r243, %r242;
cvt.rn.bf16x2.f32 %r160, %r245, %r244;
cvt.rn.bf16x2.f32 %r161, %r247, %r246;
cvt.rn.bf16x2.f32 %r162, %r249, %r248;
@%p1 st.global.v4.b32 [ %rd2 + 0 ], { %r159, %r160, %r161, %r162 };
cvt.rn.bf16x2.f32 %r163, %r251, %r250;
cvt.rn.bf16x2.f32 %r164, %r253, %r252;
cvt.rn.bf16x2.f32 %r165, %r255, %r254;
cvt.rn.bf16x2.f32 %r166, %r257, %r256;
@%p2 st.global.v4.b32 [ %rd3 + 0 ], { %r163, %r164, %r165, %r166 };
cvt.rn.bf16x2.f32 %r167, %r259, %r258;
cvt.rn.bf16x2.f32 %r168, %r261, %r260;
cvt.rn.bf16x2.f32 %r169, %r263, %r262;
cvt.rn.bf16x2.f32 %r170, %r265, %r264;
@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r167, %r168, %r169, %r170 };
ret;
$L__tmp21:
$L__func_end0:
}
"#

def recognizes (mode : BackwardLayout) (ptx : String) : Bool :=
  QKVTTIR.normalizePTX ptx == match mode with
    | .gradientOnly => gradientAssembly
    | .withQuantization => encodeAssembly

structure Program (mode : BackwardLayout) where
  ptx : String
  checked : recognizes mode ptx = true

def Program.check (mode : BackwardLayout) (ptx : String) : Option (Program mode) :=
  if h : recognizes mode ptx then some ⟨ptx, h⟩ else none

def recognizesRetained (ptx : String) : Bool := QKVTTIR.normalizePTX ptx == retainedAssembly

structure RetainedProgram where
  ptx : String
  checked : recognizesRetained ptx = true

def RetainedProgram.check (ptx : String) : Option RetainedProgram :=
  if h : recognizesRetained ptx then some ⟨ptx, h⟩ else none

structure Pair where
  forward : MuddNorm.Binding.CachedProgram
  gradient : Program .gradientOnly
  encode : Program .withQuantization
  retained : Option RetainedProgram := none

def laneBits (mode : BackwardLayout) (column : Fin 768) : Nat :=
  let shift := match mode with | .gradientOnly => 3 | .withQuantization => 4
  (column.val >>> shift) &&& 31

theorem lane_bits_eq (mode : BackwardLayout) (column : Fin 768) :
    laneBits mode column = column.val / layout mode % 32 := by
  cases mode
  · have h : ∀ c : Fin 768, laneBits .withQuantization c = c.val / layout .withQuantization % 32 := by
      decide +kernel
    exact h column
  · have h : ∀ c : Fin 768, laneBits .gradientOnly c = c.val / layout .gradientOnly % 32 := by
      decide +kernel
    exact h column

/-- The signed row offset and the unsigned channel offset fit their
32-bit words before the separate wide address calculations. -/
theorem split_row_byte_offset (row : Fin 49152) (column : Fin 1024) (wordBytes : Nat) :
    let rowPart := (BitVec.ofNat 32 row.val * 768).toNat
    let colPart := (BitVec.ofNat 32 column.val).toNat
    rowPart < 2 ^ 31 ∧ wordBytes * rowPart + wordBytes * colPart =
      wordBytes * (768 * row.val + column.val) := by
  have rowBound := row.isLt
  have columnBound := column.isLt
  have rowFits : row.val < 4294967296 := by omega
  have columnFits : column.val < 4294967296 := by omega
  have rowPartFits : row.val * 768 < 4294967296 := by omega
  simp only [BitVec.toNat_mul, BitVec.toNat_ofNat]
  change (row.val % 4294967296 * 768) % 4294967296 < 2 ^ 31 ∧
    wordBytes * ((row.val % 4294967296 * 768) % 4294967296) +
      wordBytes * (column.val % 4294967296) = _
  rw [Nat.mod_eq_of_lt rowFits, Nat.mod_eq_of_lt rowPartFits, Nat.mod_eq_of_lt columnFits]
  constructor
  · omega
  · rw [Nat.mul_add, Nat.mul_comm 768 row.val]

#axiomsClean split_row_byte_offset

def Program.run {mode : BackwardLayout} (_ : Program mode) :=
  backwardInPaddedLayout mode (laneBits mode)

def Pair.backward (pair : Pair) : Run
  | .gradientOnly => pair.gradient.run
  | .withQuantization => pair.encode.run

theorem Pair.backward_eq (pair : Pair) : pair.backward = MuddBackward.backward := by
  have lanes (mode) : laneBits mode = (fun (c : Fin 768) => c.val / layout mode % 32) :=
    funext (lane_bits_eq mode)
  funext mode
  cases mode <;> simp only [Pair.backward, Program.run, backwardInPaddedLayout_eq, MuddBackward.backward, lanes]

def RetainedProgram.gradient (_ : RetainedProgram) (native : Native) (input : Input) (inverse : MuddNorm.F32) :=
  (backwardInPaddedLayout .withQuantization (laneBits .withQuantization) native input inverse).gradient

def Pair.gradients (pair : Pair) (retained : RetainedProgram) : GradientRun
  | .gradientOnly => fun native input inverse => (pair.gradient.run native input inverse).gradient
  | .withQuantization => retained.gradient

theorem Pair.gradients_eq (pair : Pair) (retained : RetainedProgram) :
    pair.gradients retained = fun layout native input inverse =>
      (MuddBackward.backward layout native input inverse).gradient := by
  have lanes (mode) : laneBits mode = (fun (c : Fin 768) => c.val / layout mode % 32) :=
    funext (lane_bits_eq mode)
  funext mode
  cases mode
  · funext native input inverse
    change (backwardInPaddedLayout .withQuantization (laneBits .withQuantization) native input inverse).gradient = _
    rw [backwardInPaddedLayout_eq, lanes]
    rfl
  · simp only [Pair.gradients, Program.run, backwardInPaddedLayout_eq, MuddBackward.backward, lanes]

def Pair.run (pair : Pair) : BatchRun :=
  match pair.retained with
  | none => execute pair.backward
  | some retained => executeRetained (pair.gradients retained)

theorem Pair.run_eq (pair : Pair) : pair.run = execute MuddBackward.backward := by
  unfold Pair.run
  cases pair.retained with
  | none =>
    change execute pair.backward = _
    rw [pair.backward_eq]
  | some retained =>
    change executeRetained (pair.gradients retained) = _
    rw [pair.gradients_eq, executeRetained_eq]

inductive Pipeline where
  | original
  | cached (pair : Pair)

def Pipeline.forward : Pipeline → MuddNorm.TrainingRun
  | .original => MuddNorm.RowProgram.staged.training .squareSum
  | .cached pair => pair.forward.training

def Pipeline.backward : Pipeline → BatchRun
  | .original => execute MuddBackward.backward
  | .cached pair => pair.run

def Pipeline.run (pipeline : Pipeline) (mode : BackwardLayout) (native : Native)
    (packet : MuddNorm.Packet) (gradient : Fin packet.shape.rows → Fin 768 → BF16) :=
  let output := pipeline.forward native packet
  let saved := output.saved mode
  (output.coefficients, saved.mixedResidual, output.activations,
    pipeline.backward packet.shape.rows saved gradient output.activations)

def samePipeline (left right : Pipeline) : Prop :=
  ∀ mode native packet gradient, left.run mode native packet gradient = right.run mode native packet gradient

instance : Spec.ProgramSyntax Pipeline where
  Statement := String
  Signature := Unit
  signature := fun _ => ()
  body
    | .original => []
    | .cached pair => [pair.forward.ptx, pair.gradient.ptx, pair.encode.ptx] ++
        (pair.retained.map RetainedProgram.ptx).toList
  structural := some samePipeline

specification cached_equiv (pair : Pair) :
    Pipeline.cached pair ≡[([] : Spec.Assumptions String)] Pipeline.original := by
  refine Spec.FloatingPoint.ofStructural (structural := samePipeline) rfl rfl ?_
  intro mode native packet gradient
  simp only [Pipeline.run, Pipeline.forward, Pipeline.backward, pair.forward.training_eq,
    MuddNorm.training_run_eq, pair.run_eq]

#print_fp_assumptions cached_equiv
#axiomsClean lane_bits_eq
#axiomsClean Pair.backward_eq
#axiomsClean Pair.gradients_eq
#axiomsClean Pair.run_eq
#axiomsClean cached_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.MuddBackward.Binding
