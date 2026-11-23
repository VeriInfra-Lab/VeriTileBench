import models.NanoGPT.Track1.MuddNormHeadReplay
import models.NanoGPT.Track1.QKVTTIRBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm.Binding
open VeriTile.Spec

/-- SM90a PTX for the one-warp launch. The signature includes ten
BF16 coefficient outputs, two FP32 saved buffers, and one E4M3 output. -/
private def assembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry mudd_final_norm(
.param .u64 .ptr .global .align 1 mudd_final_norm_param_0,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_1,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_2,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_3,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_4,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_5,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_6,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_7,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_8,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_9,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_10,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_11,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_12,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_13,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_14,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_15,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_16,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_17,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_18,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_19,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_20,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_21,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_22,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_23,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_24,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_25,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_26,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_27,
.param .u64 .ptr .global .align 1 mudd_final_norm_param_28
)
.reqntid 32
{
.reg .pred %p<6>;
.reg .b16 %rs<430>;
.reg .b32 %r<1150>;
.reg .b64 %rd<142>;
$L__func_begin0:
ld.param.b64 %rd106, [mudd_final_norm_param_0];
ld.param.b64 %rd107, [mudd_final_norm_param_1];
$L__tmp0:
mov.u32 %r326, %tid.x;
and.b32 %r327, %r326, 31;
ld.param.b64 %rd108, [mudd_final_norm_param_2];
ld.param.b64 %rd109, [mudd_final_norm_param_3];
bfe.u32 %r328, %r326, 2, 3;
ld.param.b64 %rd110, [mudd_final_norm_param_4];
or.b32 %r329, %r328, 8;
ld.param.b64 %rd111, [mudd_final_norm_param_5];
and.b32 %r330, %r326, 3;
ld.param.b64 %rd112, [mudd_final_norm_param_6];
shl.b32 %r331, %r330, 4;
ld.param.b64 %rd113, [mudd_final_norm_param_7];
shl.b32 %r332, %r328, 6;
ld.param.b64 %rd114, [mudd_final_norm_param_8];
shl.b32 %r333, %r329, 6;
ld.param.b64 %rd115, [mudd_final_norm_param_9];
ld.param.b64 %rd116, [mudd_final_norm_param_10];
mov.u32 %r334, %ctaid.x;
mul.lo.s32 %r335, %r334, 768;
ld.param.b64 %rd117, [mudd_final_norm_param_11];
or.b32 %r336, %r331, %r335;
ld.param.b64 %rd118, [mudd_final_norm_param_12];
add.s32 %r337, %r336, %r332;
ld.param.b64 %rd119, [mudd_final_norm_param_13];
or.b32 %r338, %r336, 4;
ld.param.b64 %rd120, [mudd_final_norm_param_14];
add.s32 %r339, %r338, %r332;
ld.param.b64 %rd121, [mudd_final_norm_param_15];
or.b32 %r340, %r336, 8;
ld.param.b64 %rd122, [mudd_final_norm_param_16];
add.s32 %r341, %r340, %r332;
ld.param.b64 %rd123, [mudd_final_norm_param_17];
or.b32 %r342, %r336, 12;
ld.param.b64 %rd124, [mudd_final_norm_param_18];
add.s32 %r343, %r342, %r332;
ld.param.b64 %rd125, [mudd_final_norm_param_19];
add.s32 %r344, %r336, %r333;
ld.param.b64 %rd126, [mudd_final_norm_param_20];
add.s32 %r345, %r338, %r333;
ld.param.b64 %rd127, [mudd_final_norm_param_21];
add.s32 %r346, %r340, %r333;
ld.param.b64 %rd128, [mudd_final_norm_param_22];
add.s32 %r347, %r342, %r333;
ld.param.b64 %rd129, [mudd_final_norm_param_23];
mul.lo.s32 %r348, %r334, 12;
ld.param.b64 %rd130, [mudd_final_norm_param_24];
add.s32 %r349, %r328, %r348;
ld.param.b64 %rd131, [mudd_final_norm_param_25];
add.s32 %r350, %r329, %r348;
ld.param.b64 %rd132, [mudd_final_norm_param_26];
setp.lt.u32 %p2, %r329, 12;
cvt.s64.s32 %rd133, %r337;
mul.wide.s32 %rd134, %r337, 2;
add.s64 %rd1, %rd111, %rd134;
mul.wide.s32 %rd135, %r341, 2;
add.s64 %rd2, %rd111, %rd135;
cvt.s64.s32 %rd136, %r344;
mul.wide.s32 %rd137, %r344, 2;
add.s64 %rd3, %rd111, %rd137;
mul.wide.s32 %rd138, %r346, 2;
add.s64 %rd4, %rd111, %rd138;
mov.b32 %r5, 0;
mov.pred %p1, -1;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs58, %rs59}, %r1;
mov.b32 {%rs60, %rs61}, %r2;
mov.b32 {%rs62, %rs63}, %r3;
mov.b32 {%rs64, %rs65}, %r4;
mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.b32 {%rs66, %rs67}, %r6;
mov.b32 {%rs68, %rs69}, %r7;
mov.b32 {%rs70, %rs71}, %r8;
mov.b32 {%rs72, %rs73}, %r9;
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs74, %rs75}, %r10;
mov.b32 {%rs76, %rs77}, %r11;
mov.b32 {%rs78, %rs79}, %r12;
mov.b32 {%rs80, %rs81}, %r13;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs82, %rs83}, %r14;
mov.b32 {%rs84, %rs85}, %r15;
mov.b32 {%rs86, %rs87}, %r16;
mov.b32 {%rs88, %rs89}, %r17;
cvt.f32.bf16 %r351, %rs58;
cvt.f32.bf16 %r352, %rs59;
cvt.f32.bf16 %r353, %rs60;
cvt.f32.bf16 %r354, %rs61;
cvt.f32.bf16 %r355, %rs62;
cvt.f32.bf16 %r356, %rs63;
cvt.f32.bf16 %r357, %rs64;
cvt.f32.bf16 %r358, %rs65;
cvt.f32.bf16 %r359, %rs66;
cvt.f32.bf16 %r360, %rs67;
cvt.f32.bf16 %r361, %rs68;
cvt.f32.bf16 %r362, %rs69;
cvt.f32.bf16 %r363, %rs70;
cvt.f32.bf16 %r364, %rs71;
cvt.f32.bf16 %r365, %rs72;
cvt.f32.bf16 %r366, %rs73;
cvt.f32.bf16 %r367, %rs74;
cvt.f32.bf16 %r368, %rs75;
cvt.f32.bf16 %r369, %rs76;
cvt.f32.bf16 %r370, %rs77;
cvt.f32.bf16 %r371, %rs78;
cvt.f32.bf16 %r372, %rs79;
cvt.f32.bf16 %r373, %rs80;
cvt.f32.bf16 %r374, %rs81;
cvt.f32.bf16 %r375, %rs82;
cvt.f32.bf16 %r376, %rs83;
cvt.f32.bf16 %r377, %rs84;
cvt.f32.bf16 %r378, %rs85;
cvt.f32.bf16 %r379, %rs86;
cvt.f32.bf16 %r380, %rs87;
cvt.f32.bf16 %r381, %rs88;
cvt.f32.bf16 %r382, %rs89;
add.s64 %rd5, %rd107, 28;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd5 + 0 ];
mov.b32 {%rs90, %rs91}, %r18;
mul.lo.s32 %r383, %r334, 10;
mad.wide.s32 %rd6, %r383, 2, %rd106;
mov.u32 %r19, 0x0;
ld.global.b32 { %r19 }, [ %rd6 + 0 ];
mov.b32 {%rs92, %rs93}, %r19;
cvt.f32.bf16 %r21, %rs90;
cvt.f32.bf16 %r22, %rs92;
mul.lo.s32 %r384, %r334, 120;
mad.wide.s32 %rd139, %r384, 2, %rd108;
mad.wide.u32 %rd7, %r328, 2, %rd139;
add.s64 %rd8, %rd7, 16;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd7 + 0 ];
mov.u16 %rs3, %rs2;
@%p2 ld.global.b16 { %rs3 }, [ %rd8 + 0 ];
cvt.f32.bf16 %r385, %rs1;
cvt.f32.bf16 %r386, %rs3;
add.rn.f32 %r20, %r21, %r22;
mul.f32 %r387, %r385, 0f3DCCCCCD;
mul.f32 %r388, %r386, 0f3DCCCCCD;
fma.rn.f32 %r389, %r20, 0f3DCCCCCD, %r387;
fma.rn.f32 %r390, %r20, 0f3DCCCCCD, %r388;
cvt.rn.bf16.f32 %rs4, %r389;
cvt.rn.bf16.f32 %rs5, %r390;
mul.wide.s32 %rd140, %r349, 2;
add.s64 %rd9, %rd112, %rd140;
mul.wide.s32 %rd141, %r350, 2;
add.s64 %rd10, %rd112, %rd141;
setp.eq.b32 %p3, %r330, 0;
@%p3 st.global.b16 [ %rd9 + 0 ], { %rs4 };
and.pred %p4, %p3, %p2;
@%p4 st.global.b16 [ %rd10 + 0 ], { %rs5 };
cvt.f32.bf16 %r391, %rs4;
cvt.f32.bf16 %r392, %rs5;
add.s64 %rd11, %rd113, %rd134;
add.s64 %rd12, %rd113, %rd135;
add.s64 %rd13, %rd113, %rd137;
add.s64 %rd14, %rd113, %rd138;
mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
mov.u32 %r26, %r5;
@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd11 + 0 ];
mov.b32 {%rs94, %rs95}, %r23;
mov.b32 {%rs96, %rs97}, %r24;
mov.b32 {%rs98, %rs99}, %r25;
mov.b32 {%rs100, %rs101}, %r26;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
mov.u32 %r30, %r5;
@%p1 ld.global.v4.b32 { %r27, %r28, %r29, %r30 }, [ %rd12 + 0 ];
mov.b32 {%rs102, %rs103}, %r27;
mov.b32 {%rs104, %rs105}, %r28;
mov.b32 {%rs106, %rs107}, %r29;
mov.b32 {%rs108, %rs109}, %r30;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
mov.u32 %r34, %r5;
@%p2 ld.global.v4.b32 { %r31, %r32, %r33, %r34 }, [ %rd13 + 0 ];
mov.b32 {%rs110, %rs111}, %r31;
mov.b32 {%rs112, %rs113}, %r32;
mov.b32 {%rs114, %rs115}, %r33;
mov.b32 {%rs116, %rs117}, %r34;
mov.u32 %r35, %r5;
mov.u32 %r36, %r5;
mov.u32 %r37, %r5;
mov.u32 %r38, %r5;
@%p2 ld.global.v4.b32 { %r35, %r36, %r37, %r38 }, [ %rd14 + 0 ];
mov.b32 {%rs118, %rs119}, %r35;
mov.b32 {%rs120, %rs121}, %r36;
mov.b32 {%rs122, %rs123}, %r37;
mov.b32 {%rs124, %rs125}, %r38;
cvt.f32.bf16 %r393, %rs94;
cvt.f32.bf16 %r394, %rs95;
cvt.f32.bf16 %r395, %rs96;
cvt.f32.bf16 %r396, %rs97;
cvt.f32.bf16 %r397, %rs98;
cvt.f32.bf16 %r398, %rs99;
cvt.f32.bf16 %r399, %rs100;
cvt.f32.bf16 %r400, %rs101;
cvt.f32.bf16 %r401, %rs102;
cvt.f32.bf16 %r402, %rs103;
cvt.f32.bf16 %r403, %rs104;
cvt.f32.bf16 %r404, %rs105;
cvt.f32.bf16 %r405, %rs106;
cvt.f32.bf16 %r406, %rs107;
cvt.f32.bf16 %r407, %rs108;
cvt.f32.bf16 %r408, %rs109;
cvt.f32.bf16 %r409, %rs110;
cvt.f32.bf16 %r410, %rs111;
cvt.f32.bf16 %r411, %rs112;
cvt.f32.bf16 %r412, %rs113;
cvt.f32.bf16 %r413, %rs114;
cvt.f32.bf16 %r414, %rs115;
cvt.f32.bf16 %r415, %rs116;
cvt.f32.bf16 %r416, %rs117;
cvt.f32.bf16 %r417, %rs118;
cvt.f32.bf16 %r418, %rs119;
cvt.f32.bf16 %r419, %rs120;
cvt.f32.bf16 %r420, %rs121;
cvt.f32.bf16 %r421, %rs122;
cvt.f32.bf16 %r422, %rs123;
cvt.f32.bf16 %r423, %rs124;
cvt.f32.bf16 %r424, %rs125;
fma.rn.f32 %r425, %r391, %r393, %r351;
fma.rn.f32 %r426, %r391, %r394, %r352;
fma.rn.f32 %r427, %r391, %r395, %r353;
fma.rn.f32 %r428, %r391, %r396, %r354;
fma.rn.f32 %r429, %r391, %r397, %r355;
fma.rn.f32 %r430, %r391, %r398, %r356;
fma.rn.f32 %r431, %r391, %r399, %r357;
fma.rn.f32 %r432, %r391, %r400, %r358;
fma.rn.f32 %r433, %r391, %r401, %r359;
fma.rn.f32 %r434, %r391, %r402, %r360;
fma.rn.f32 %r435, %r391, %r403, %r361;
fma.rn.f32 %r436, %r391, %r404, %r362;
fma.rn.f32 %r437, %r391, %r405, %r363;
fma.rn.f32 %r438, %r391, %r406, %r364;
fma.rn.f32 %r439, %r391, %r407, %r365;
fma.rn.f32 %r440, %r391, %r408, %r366;
fma.rn.f32 %r441, %r392, %r409, %r367;
fma.rn.f32 %r442, %r392, %r410, %r368;
fma.rn.f32 %r443, %r392, %r411, %r369;
fma.rn.f32 %r444, %r392, %r412, %r370;
fma.rn.f32 %r445, %r392, %r413, %r371;
fma.rn.f32 %r446, %r392, %r414, %r372;
fma.rn.f32 %r447, %r392, %r415, %r373;
fma.rn.f32 %r448, %r392, %r416, %r374;
fma.rn.f32 %r449, %r392, %r417, %r375;
fma.rn.f32 %r450, %r392, %r418, %r376;
fma.rn.f32 %r451, %r392, %r419, %r377;
fma.rn.f32 %r452, %r392, %r420, %r378;
fma.rn.f32 %r453, %r392, %r421, %r379;
fma.rn.f32 %r454, %r392, %r422, %r380;
fma.rn.f32 %r455, %r392, %r423, %r381;
fma.rn.f32 %r456, %r392, %r424, %r382;
cvt.f32.bf16 %r40, %rs91;
cvt.f32.bf16 %r41, %rs93;
add.s64 %rd15, %rd7, 24;
add.s64 %rd16, %rd7, 40;
mov.u16 %rs6, %rs2;
@%p1 ld.global.b16 { %rs6 }, [ %rd15 + 0 ];
mov.u16 %rs7, %rs2;
@%p2 ld.global.b16 { %rs7 }, [ %rd16 + 0 ];
cvt.f32.bf16 %r457, %rs6;
cvt.f32.bf16 %r458, %rs7;
add.rn.f32 %r39, %r40, %r41;
mul.f32 %r459, %r457, 0f3DCCCCCD;
mul.f32 %r460, %r458, 0f3DCCCCCD;
fma.rn.f32 %r461, %r39, 0f3DCCCCCD, %r459;
fma.rn.f32 %r462, %r39, 0f3DCCCCCD, %r460;
cvt.rn.bf16.f32 %rs8, %r461;
cvt.rn.bf16.f32 %rs9, %r462;
add.s64 %rd17, %rd114, %rd140;
add.s64 %rd18, %rd114, %rd141;
@%p3 st.global.b16 [ %rd17 + 0 ], { %rs8 };
@%p4 st.global.b16 [ %rd18 + 0 ], { %rs9 };
cvt.f32.bf16 %r463, %rs8;
cvt.f32.bf16 %r464, %rs9;
add.s64 %rd19, %rd115, %rd134;
add.s64 %rd20, %rd115, %rd135;
add.s64 %rd21, %rd115, %rd137;
add.s64 %rd22, %rd115, %rd138;
mov.u32 %r42, %r5;
mov.u32 %r43, %r5;
mov.u32 %r44, %r5;
mov.u32 %r45, %r5;
@%p1 ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd19 + 0 ];
mov.b32 {%rs126, %rs127}, %r42;
mov.b32 {%rs128, %rs129}, %r43;
mov.b32 {%rs130, %rs131}, %r44;
mov.b32 {%rs132, %rs133}, %r45;
mov.u32 %r46, %r5;
mov.u32 %r47, %r5;
mov.u32 %r48, %r5;
mov.u32 %r49, %r5;
@%p1 ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd20 + 0 ];
mov.b32 {%rs134, %rs135}, %r46;
mov.b32 {%rs136, %rs137}, %r47;
mov.b32 {%rs138, %rs139}, %r48;
mov.b32 {%rs140, %rs141}, %r49;
mov.u32 %r50, %r5;
mov.u32 %r51, %r5;
mov.u32 %r52, %r5;
mov.u32 %r53, %r5;
@%p2 ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd21 + 0 ];
mov.b32 {%rs142, %rs143}, %r50;
mov.b32 {%rs144, %rs145}, %r51;
mov.b32 {%rs146, %rs147}, %r52;
mov.b32 {%rs148, %rs149}, %r53;
mov.u32 %r54, %r5;
mov.u32 %r55, %r5;
mov.u32 %r56, %r5;
mov.u32 %r57, %r5;
@%p2 ld.global.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd22 + 0 ];
mov.b32 {%rs150, %rs151}, %r54;
mov.b32 {%rs152, %rs153}, %r55;
mov.b32 {%rs154, %rs155}, %r56;
mov.b32 {%rs156, %rs157}, %r57;
cvt.f32.bf16 %r465, %rs126;
cvt.f32.bf16 %r466, %rs127;
cvt.f32.bf16 %r467, %rs128;
cvt.f32.bf16 %r468, %rs129;
cvt.f32.bf16 %r469, %rs130;
cvt.f32.bf16 %r470, %rs131;
cvt.f32.bf16 %r471, %rs132;
cvt.f32.bf16 %r472, %rs133;
cvt.f32.bf16 %r473, %rs134;
cvt.f32.bf16 %r474, %rs135;
cvt.f32.bf16 %r475, %rs136;
cvt.f32.bf16 %r476, %rs137;
cvt.f32.bf16 %r477, %rs138;
cvt.f32.bf16 %r478, %rs139;
cvt.f32.bf16 %r479, %rs140;
cvt.f32.bf16 %r480, %rs141;
cvt.f32.bf16 %r481, %rs142;
cvt.f32.bf16 %r482, %rs143;
cvt.f32.bf16 %r483, %rs144;
cvt.f32.bf16 %r484, %rs145;
cvt.f32.bf16 %r485, %rs146;
cvt.f32.bf16 %r486, %rs147;
cvt.f32.bf16 %r487, %rs148;
cvt.f32.bf16 %r488, %rs149;
cvt.f32.bf16 %r489, %rs150;
cvt.f32.bf16 %r490, %rs151;
cvt.f32.bf16 %r491, %rs152;
cvt.f32.bf16 %r492, %rs153;
cvt.f32.bf16 %r493, %rs154;
cvt.f32.bf16 %r494, %rs155;
cvt.f32.bf16 %r495, %rs156;
cvt.f32.bf16 %r496, %rs157;
fma.rn.f32 %r497, %r463, %r465, %r425;
fma.rn.f32 %r498, %r463, %r466, %r426;
fma.rn.f32 %r499, %r463, %r467, %r427;
fma.rn.f32 %r500, %r463, %r468, %r428;
fma.rn.f32 %r501, %r463, %r469, %r429;
fma.rn.f32 %r502, %r463, %r470, %r430;
fma.rn.f32 %r503, %r463, %r471, %r431;
fma.rn.f32 %r504, %r463, %r472, %r432;
fma.rn.f32 %r505, %r463, %r473, %r433;
fma.rn.f32 %r506, %r463, %r474, %r434;
fma.rn.f32 %r507, %r463, %r475, %r435;
fma.rn.f32 %r508, %r463, %r476, %r436;
fma.rn.f32 %r509, %r463, %r477, %r437;
fma.rn.f32 %r510, %r463, %r478, %r438;
fma.rn.f32 %r511, %r463, %r479, %r439;
fma.rn.f32 %r512, %r463, %r480, %r440;
fma.rn.f32 %r513, %r464, %r481, %r441;
fma.rn.f32 %r514, %r464, %r482, %r442;
fma.rn.f32 %r515, %r464, %r483, %r443;
fma.rn.f32 %r516, %r464, %r484, %r444;
fma.rn.f32 %r517, %r464, %r485, %r445;
fma.rn.f32 %r518, %r464, %r486, %r446;
fma.rn.f32 %r519, %r464, %r487, %r447;
fma.rn.f32 %r520, %r464, %r488, %r448;
fma.rn.f32 %r521, %r464, %r489, %r449;
fma.rn.f32 %r522, %r464, %r490, %r450;
fma.rn.f32 %r523, %r464, %r491, %r451;
fma.rn.f32 %r524, %r464, %r492, %r452;
fma.rn.f32 %r525, %r464, %r493, %r453;
fma.rn.f32 %r526, %r464, %r494, %r454;
fma.rn.f32 %r527, %r464, %r495, %r455;
fma.rn.f32 %r528, %r464, %r496, %r456;
add.s64 %rd23, %rd107, 32;
mov.u32 %r58, 0x0;
ld.global.b32 { %r58 }, [ %rd23 + 0 ];
mov.b32 {%rs158, %rs159}, %r58;
add.s32 %r529, %r383, 2;
mad.wide.s32 %rd24, %r529, 2, %rd106;
mov.u32 %r59, 0x0;
ld.global.b32 { %r59 }, [ %rd24 + 0 ];
mov.b32 {%rs160, %rs161}, %r59;
cvt.f32.bf16 %r61, %rs158;
cvt.f32.bf16 %r62, %rs160;
add.s64 %rd25, %rd7, 48;
add.s64 %rd26, %rd7, 64;
mov.u16 %rs10, %rs2;
@%p1 ld.global.b16 { %rs10 }, [ %rd25 + 0 ];
mov.u16 %rs11, %rs2;
@%p2 ld.global.b16 { %rs11 }, [ %rd26 + 0 ];
cvt.f32.bf16 %r530, %rs10;
cvt.f32.bf16 %r531, %rs11;
add.rn.f32 %r60, %r61, %r62;
mul.f32 %r532, %r530, 0f3DCCCCCD;
mul.f32 %r533, %r531, 0f3DCCCCCD;
fma.rn.f32 %r534, %r60, 0f3DCCCCCD, %r532;
fma.rn.f32 %r535, %r60, 0f3DCCCCCD, %r533;
cvt.rn.bf16.f32 %rs12, %r534;
cvt.rn.bf16.f32 %rs13, %r535;
add.s64 %rd27, %rd116, %rd140;
add.s64 %rd28, %rd116, %rd141;
@%p3 st.global.b16 [ %rd27 + 0 ], { %rs12 };
@%p4 st.global.b16 [ %rd28 + 0 ], { %rs13 };
cvt.f32.bf16 %r536, %rs12;
cvt.f32.bf16 %r537, %rs13;
add.s64 %rd29, %rd117, %rd134;
add.s64 %rd30, %rd117, %rd135;
add.s64 %rd31, %rd117, %rd137;
add.s64 %rd32, %rd117, %rd138;
mov.u32 %r63, %r5;
mov.u32 %r64, %r5;
mov.u32 %r65, %r5;
mov.u32 %r66, %r5;
@%p1 ld.global.v4.b32 { %r63, %r64, %r65, %r66 }, [ %rd29 + 0 ];
mov.b32 {%rs162, %rs163}, %r63;
mov.b32 {%rs164, %rs165}, %r64;
mov.b32 {%rs166, %rs167}, %r65;
mov.b32 {%rs168, %rs169}, %r66;
mov.u32 %r67, %r5;
mov.u32 %r68, %r5;
mov.u32 %r69, %r5;
mov.u32 %r70, %r5;
@%p1 ld.global.v4.b32 { %r67, %r68, %r69, %r70 }, [ %rd30 + 0 ];
mov.b32 {%rs170, %rs171}, %r67;
mov.b32 {%rs172, %rs173}, %r68;
mov.b32 {%rs174, %rs175}, %r69;
mov.b32 {%rs176, %rs177}, %r70;
mov.u32 %r71, %r5;
mov.u32 %r72, %r5;
mov.u32 %r73, %r5;
mov.u32 %r74, %r5;
@%p2 ld.global.v4.b32 { %r71, %r72, %r73, %r74 }, [ %rd31 + 0 ];
mov.b32 {%rs178, %rs179}, %r71;
mov.b32 {%rs180, %rs181}, %r72;
mov.b32 {%rs182, %rs183}, %r73;
mov.b32 {%rs184, %rs185}, %r74;
mov.u32 %r75, %r5;
mov.u32 %r76, %r5;
mov.u32 %r77, %r5;
mov.u32 %r78, %r5;
@%p2 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd32 + 0 ];
mov.b32 {%rs186, %rs187}, %r75;
mov.b32 {%rs188, %rs189}, %r76;
mov.b32 {%rs190, %rs191}, %r77;
mov.b32 {%rs192, %rs193}, %r78;
cvt.f32.bf16 %r538, %rs162;
cvt.f32.bf16 %r539, %rs163;
cvt.f32.bf16 %r540, %rs164;
cvt.f32.bf16 %r541, %rs165;
cvt.f32.bf16 %r542, %rs166;
cvt.f32.bf16 %r543, %rs167;
cvt.f32.bf16 %r544, %rs168;
cvt.f32.bf16 %r545, %rs169;
cvt.f32.bf16 %r546, %rs170;
cvt.f32.bf16 %r547, %rs171;
cvt.f32.bf16 %r548, %rs172;
cvt.f32.bf16 %r549, %rs173;
cvt.f32.bf16 %r550, %rs174;
cvt.f32.bf16 %r551, %rs175;
cvt.f32.bf16 %r552, %rs176;
cvt.f32.bf16 %r553, %rs177;
cvt.f32.bf16 %r554, %rs178;
cvt.f32.bf16 %r555, %rs179;
cvt.f32.bf16 %r556, %rs180;
cvt.f32.bf16 %r557, %rs181;
cvt.f32.bf16 %r558, %rs182;
cvt.f32.bf16 %r559, %rs183;
cvt.f32.bf16 %r560, %rs184;
cvt.f32.bf16 %r561, %rs185;
cvt.f32.bf16 %r562, %rs186;
cvt.f32.bf16 %r563, %rs187;
cvt.f32.bf16 %r564, %rs188;
cvt.f32.bf16 %r565, %rs189;
cvt.f32.bf16 %r566, %rs190;
cvt.f32.bf16 %r567, %rs191;
cvt.f32.bf16 %r568, %rs192;
cvt.f32.bf16 %r569, %rs193;
fma.rn.f32 %r570, %r536, %r538, %r497;
fma.rn.f32 %r571, %r536, %r539, %r498;
fma.rn.f32 %r572, %r536, %r540, %r499;
fma.rn.f32 %r573, %r536, %r541, %r500;
fma.rn.f32 %r574, %r536, %r542, %r501;
fma.rn.f32 %r575, %r536, %r543, %r502;
fma.rn.f32 %r576, %r536, %r544, %r503;
fma.rn.f32 %r577, %r536, %r545, %r504;
fma.rn.f32 %r578, %r536, %r546, %r505;
fma.rn.f32 %r579, %r536, %r547, %r506;
fma.rn.f32 %r580, %r536, %r548, %r507;
fma.rn.f32 %r581, %r536, %r549, %r508;
fma.rn.f32 %r582, %r536, %r550, %r509;
fma.rn.f32 %r583, %r536, %r551, %r510;
fma.rn.f32 %r584, %r536, %r552, %r511;
fma.rn.f32 %r585, %r536, %r553, %r512;
fma.rn.f32 %r586, %r537, %r554, %r513;
fma.rn.f32 %r587, %r537, %r555, %r514;
fma.rn.f32 %r588, %r537, %r556, %r515;
fma.rn.f32 %r589, %r537, %r557, %r516;
fma.rn.f32 %r590, %r537, %r558, %r517;
fma.rn.f32 %r591, %r537, %r559, %r518;
fma.rn.f32 %r592, %r537, %r560, %r519;
fma.rn.f32 %r593, %r537, %r561, %r520;
fma.rn.f32 %r594, %r537, %r562, %r521;
fma.rn.f32 %r595, %r537, %r563, %r522;
fma.rn.f32 %r596, %r537, %r564, %r523;
fma.rn.f32 %r597, %r537, %r565, %r524;
fma.rn.f32 %r598, %r537, %r566, %r525;
fma.rn.f32 %r599, %r537, %r567, %r526;
fma.rn.f32 %r600, %r537, %r568, %r527;
fma.rn.f32 %r601, %r537, %r569, %r528;
cvt.f32.bf16 %r80, %rs159;
cvt.f32.bf16 %r81, %rs161;
add.s64 %rd33, %rd7, 72;
add.s64 %rd34, %rd7, 88;
mov.u16 %rs14, %rs2;
@%p1 ld.global.b16 { %rs14 }, [ %rd33 + 0 ];
mov.u16 %rs15, %rs2;
@%p2 ld.global.b16 { %rs15 }, [ %rd34 + 0 ];
cvt.f32.bf16 %r602, %rs14;
cvt.f32.bf16 %r603, %rs15;
add.rn.f32 %r79, %r80, %r81;
mul.f32 %r604, %r602, 0f3DCCCCCD;
mul.f32 %r605, %r603, 0f3DCCCCCD;
fma.rn.f32 %r606, %r79, 0f3DCCCCCD, %r604;
fma.rn.f32 %r607, %r79, 0f3DCCCCCD, %r605;
cvt.rn.bf16.f32 %rs16, %r606;
cvt.rn.bf16.f32 %rs17, %r607;
add.s64 %rd35, %rd118, %rd140;
add.s64 %rd36, %rd118, %rd141;
@%p3 st.global.b16 [ %rd35 + 0 ], { %rs16 };
@%p4 st.global.b16 [ %rd36 + 0 ], { %rs17 };
cvt.f32.bf16 %r608, %rs16;
cvt.f32.bf16 %r609, %rs17;
add.s64 %rd37, %rd119, %rd134;
add.s64 %rd38, %rd119, %rd135;
add.s64 %rd39, %rd119, %rd137;
add.s64 %rd40, %rd119, %rd138;
mov.u32 %r82, %r5;
mov.u32 %r83, %r5;
mov.u32 %r84, %r5;
mov.u32 %r85, %r5;
@%p1 ld.global.v4.b32 { %r82, %r83, %r84, %r85 }, [ %rd37 + 0 ];
mov.b32 {%rs194, %rs195}, %r82;
mov.b32 {%rs196, %rs197}, %r83;
mov.b32 {%rs198, %rs199}, %r84;
mov.b32 {%rs200, %rs201}, %r85;
mov.u32 %r86, %r5;
mov.u32 %r87, %r5;
mov.u32 %r88, %r5;
mov.u32 %r89, %r5;
@%p1 ld.global.v4.b32 { %r86, %r87, %r88, %r89 }, [ %rd38 + 0 ];
mov.b32 {%rs202, %rs203}, %r86;
mov.b32 {%rs204, %rs205}, %r87;
mov.b32 {%rs206, %rs207}, %r88;
mov.b32 {%rs208, %rs209}, %r89;
mov.u32 %r90, %r5;
mov.u32 %r91, %r5;
mov.u32 %r92, %r5;
mov.u32 %r93, %r5;
@%p2 ld.global.v4.b32 { %r90, %r91, %r92, %r93 }, [ %rd39 + 0 ];
mov.b32 {%rs210, %rs211}, %r90;
mov.b32 {%rs212, %rs213}, %r91;
mov.b32 {%rs214, %rs215}, %r92;
mov.b32 {%rs216, %rs217}, %r93;
mov.u32 %r94, %r5;
mov.u32 %r95, %r5;
mov.u32 %r96, %r5;
mov.u32 %r97, %r5;
@%p2 ld.global.v4.b32 { %r94, %r95, %r96, %r97 }, [ %rd40 + 0 ];
mov.b32 {%rs218, %rs219}, %r94;
mov.b32 {%rs220, %rs221}, %r95;
mov.b32 {%rs222, %rs223}, %r96;
mov.b32 {%rs224, %rs225}, %r97;
cvt.f32.bf16 %r610, %rs194;
cvt.f32.bf16 %r611, %rs195;
cvt.f32.bf16 %r612, %rs196;
cvt.f32.bf16 %r613, %rs197;
cvt.f32.bf16 %r614, %rs198;
cvt.f32.bf16 %r615, %rs199;
cvt.f32.bf16 %r616, %rs200;
cvt.f32.bf16 %r617, %rs201;
cvt.f32.bf16 %r618, %rs202;
cvt.f32.bf16 %r619, %rs203;
cvt.f32.bf16 %r620, %rs204;
cvt.f32.bf16 %r621, %rs205;
cvt.f32.bf16 %r622, %rs206;
cvt.f32.bf16 %r623, %rs207;
cvt.f32.bf16 %r624, %rs208;
cvt.f32.bf16 %r625, %rs209;
cvt.f32.bf16 %r626, %rs210;
cvt.f32.bf16 %r627, %rs211;
cvt.f32.bf16 %r628, %rs212;
cvt.f32.bf16 %r629, %rs213;
cvt.f32.bf16 %r630, %rs214;
cvt.f32.bf16 %r631, %rs215;
cvt.f32.bf16 %r632, %rs216;
cvt.f32.bf16 %r633, %rs217;
cvt.f32.bf16 %r634, %rs218;
cvt.f32.bf16 %r635, %rs219;
cvt.f32.bf16 %r636, %rs220;
cvt.f32.bf16 %r637, %rs221;
cvt.f32.bf16 %r638, %rs222;
cvt.f32.bf16 %r639, %rs223;
cvt.f32.bf16 %r640, %rs224;
cvt.f32.bf16 %r641, %rs225;
fma.rn.f32 %r642, %r608, %r610, %r570;
fma.rn.f32 %r643, %r608, %r611, %r571;
fma.rn.f32 %r644, %r608, %r612, %r572;
fma.rn.f32 %r645, %r608, %r613, %r573;
fma.rn.f32 %r646, %r608, %r614, %r574;
fma.rn.f32 %r647, %r608, %r615, %r575;
fma.rn.f32 %r648, %r608, %r616, %r576;
fma.rn.f32 %r649, %r608, %r617, %r577;
fma.rn.f32 %r650, %r608, %r618, %r578;
fma.rn.f32 %r651, %r608, %r619, %r579;
fma.rn.f32 %r652, %r608, %r620, %r580;
fma.rn.f32 %r653, %r608, %r621, %r581;
fma.rn.f32 %r654, %r608, %r622, %r582;
fma.rn.f32 %r655, %r608, %r623, %r583;
fma.rn.f32 %r656, %r608, %r624, %r584;
fma.rn.f32 %r657, %r608, %r625, %r585;
fma.rn.f32 %r658, %r609, %r626, %r586;
fma.rn.f32 %r659, %r609, %r627, %r587;
fma.rn.f32 %r660, %r609, %r628, %r588;
fma.rn.f32 %r661, %r609, %r629, %r589;
fma.rn.f32 %r662, %r609, %r630, %r590;
fma.rn.f32 %r663, %r609, %r631, %r591;
fma.rn.f32 %r664, %r609, %r632, %r592;
fma.rn.f32 %r665, %r609, %r633, %r593;
fma.rn.f32 %r666, %r609, %r634, %r594;
fma.rn.f32 %r667, %r609, %r635, %r595;
fma.rn.f32 %r668, %r609, %r636, %r596;
fma.rn.f32 %r669, %r609, %r637, %r597;
fma.rn.f32 %r670, %r609, %r638, %r598;
fma.rn.f32 %r671, %r609, %r639, %r599;
fma.rn.f32 %r672, %r609, %r640, %r600;
fma.rn.f32 %r673, %r609, %r641, %r601;
add.s64 %rd41, %rd107, 36;
mov.u32 %r98, 0x0;
ld.global.b32 { %r98 }, [ %rd41 + 0 ];
mov.b32 {%rs226, %rs227}, %r98;
add.s32 %r674, %r383, 4;
mad.wide.s32 %rd42, %r674, 2, %rd106;
mov.u32 %r99, 0x0;
ld.global.b32 { %r99 }, [ %rd42 + 0 ];
mov.b32 {%rs228, %rs229}, %r99;
cvt.f32.bf16 %r101, %rs226;
cvt.f32.bf16 %r102, %rs228;
add.s64 %rd43, %rd7, 96;
add.s64 %rd44, %rd7, 112;
mov.u16 %rs18, %rs2;
@%p1 ld.global.b16 { %rs18 }, [ %rd43 + 0 ];
mov.u16 %rs19, %rs2;
@%p2 ld.global.b16 { %rs19 }, [ %rd44 + 0 ];
cvt.f32.bf16 %r675, %rs18;
cvt.f32.bf16 %r676, %rs19;
add.rn.f32 %r100, %r101, %r102;
mul.f32 %r677, %r675, 0f3DCCCCCD;
mul.f32 %r678, %r676, 0f3DCCCCCD;
fma.rn.f32 %r679, %r100, 0f3DCCCCCD, %r677;
fma.rn.f32 %r680, %r100, 0f3DCCCCCD, %r678;
cvt.rn.bf16.f32 %rs20, %r679;
cvt.rn.bf16.f32 %rs21, %r680;
add.s64 %rd45, %rd120, %rd140;
add.s64 %rd46, %rd120, %rd141;
@%p3 st.global.b16 [ %rd45 + 0 ], { %rs20 };
@%p4 st.global.b16 [ %rd46 + 0 ], { %rs21 };
cvt.f32.bf16 %r681, %rs20;
cvt.f32.bf16 %r682, %rs21;
add.s64 %rd47, %rd121, %rd134;
add.s64 %rd48, %rd121, %rd135;
add.s64 %rd49, %rd121, %rd137;
add.s64 %rd50, %rd121, %rd138;
mov.u32 %r103, %r5;
mov.u32 %r104, %r5;
mov.u32 %r105, %r5;
mov.u32 %r106, %r5;
@%p1 ld.global.v4.b32 { %r103, %r104, %r105, %r106 }, [ %rd47 + 0 ];
mov.b32 {%rs230, %rs231}, %r103;
mov.b32 {%rs232, %rs233}, %r104;
mov.b32 {%rs234, %rs235}, %r105;
mov.b32 {%rs236, %rs237}, %r106;
mov.u32 %r107, %r5;
mov.u32 %r108, %r5;
mov.u32 %r109, %r5;
mov.u32 %r110, %r5;
@%p1 ld.global.v4.b32 { %r107, %r108, %r109, %r110 }, [ %rd48 + 0 ];
mov.b32 {%rs238, %rs239}, %r107;
mov.b32 {%rs240, %rs241}, %r108;
mov.b32 {%rs242, %rs243}, %r109;
mov.b32 {%rs244, %rs245}, %r110;
mov.u32 %r111, %r5;
mov.u32 %r112, %r5;
mov.u32 %r113, %r5;
mov.u32 %r114, %r5;
@%p2 ld.global.v4.b32 { %r111, %r112, %r113, %r114 }, [ %rd49 + 0 ];
mov.b32 {%rs246, %rs247}, %r111;
mov.b32 {%rs248, %rs249}, %r112;
mov.b32 {%rs250, %rs251}, %r113;
mov.b32 {%rs252, %rs253}, %r114;
mov.u32 %r115, %r5;
mov.u32 %r116, %r5;
mov.u32 %r117, %r5;
mov.u32 %r118, %r5;
@%p2 ld.global.v4.b32 { %r115, %r116, %r117, %r118 }, [ %rd50 + 0 ];
mov.b32 {%rs254, %rs255}, %r115;
mov.b32 {%rs256, %rs257}, %r116;
mov.b32 {%rs258, %rs259}, %r117;
mov.b32 {%rs260, %rs261}, %r118;
cvt.f32.bf16 %r683, %rs230;
cvt.f32.bf16 %r684, %rs231;
cvt.f32.bf16 %r685, %rs232;
cvt.f32.bf16 %r686, %rs233;
cvt.f32.bf16 %r687, %rs234;
cvt.f32.bf16 %r688, %rs235;
cvt.f32.bf16 %r689, %rs236;
cvt.f32.bf16 %r690, %rs237;
cvt.f32.bf16 %r691, %rs238;
cvt.f32.bf16 %r692, %rs239;
cvt.f32.bf16 %r693, %rs240;
cvt.f32.bf16 %r694, %rs241;
cvt.f32.bf16 %r695, %rs242;
cvt.f32.bf16 %r696, %rs243;
cvt.f32.bf16 %r697, %rs244;
cvt.f32.bf16 %r698, %rs245;
cvt.f32.bf16 %r699, %rs246;
cvt.f32.bf16 %r700, %rs247;
cvt.f32.bf16 %r701, %rs248;
cvt.f32.bf16 %r702, %rs249;
cvt.f32.bf16 %r703, %rs250;
cvt.f32.bf16 %r704, %rs251;
cvt.f32.bf16 %r705, %rs252;
cvt.f32.bf16 %r706, %rs253;
cvt.f32.bf16 %r707, %rs254;
cvt.f32.bf16 %r708, %rs255;
cvt.f32.bf16 %r709, %rs256;
cvt.f32.bf16 %r710, %rs257;
cvt.f32.bf16 %r711, %rs258;
cvt.f32.bf16 %r712, %rs259;
cvt.f32.bf16 %r713, %rs260;
cvt.f32.bf16 %r714, %rs261;
fma.rn.f32 %r715, %r681, %r683, %r642;
fma.rn.f32 %r716, %r681, %r684, %r643;
fma.rn.f32 %r717, %r681, %r685, %r644;
fma.rn.f32 %r718, %r681, %r686, %r645;
fma.rn.f32 %r719, %r681, %r687, %r646;
fma.rn.f32 %r720, %r681, %r688, %r647;
fma.rn.f32 %r721, %r681, %r689, %r648;
fma.rn.f32 %r722, %r681, %r690, %r649;
fma.rn.f32 %r723, %r681, %r691, %r650;
fma.rn.f32 %r724, %r681, %r692, %r651;
fma.rn.f32 %r725, %r681, %r693, %r652;
fma.rn.f32 %r726, %r681, %r694, %r653;
fma.rn.f32 %r727, %r681, %r695, %r654;
fma.rn.f32 %r728, %r681, %r696, %r655;
fma.rn.f32 %r729, %r681, %r697, %r656;
fma.rn.f32 %r730, %r681, %r698, %r657;
fma.rn.f32 %r731, %r682, %r699, %r658;
fma.rn.f32 %r732, %r682, %r700, %r659;
fma.rn.f32 %r733, %r682, %r701, %r660;
fma.rn.f32 %r734, %r682, %r702, %r661;
fma.rn.f32 %r735, %r682, %r703, %r662;
fma.rn.f32 %r736, %r682, %r704, %r663;
fma.rn.f32 %r737, %r682, %r705, %r664;
fma.rn.f32 %r738, %r682, %r706, %r665;
fma.rn.f32 %r739, %r682, %r707, %r666;
fma.rn.f32 %r740, %r682, %r708, %r667;
fma.rn.f32 %r741, %r682, %r709, %r668;
fma.rn.f32 %r742, %r682, %r710, %r669;
fma.rn.f32 %r743, %r682, %r711, %r670;
fma.rn.f32 %r744, %r682, %r712, %r671;
fma.rn.f32 %r745, %r682, %r713, %r672;
fma.rn.f32 %r746, %r682, %r714, %r673;
cvt.f32.bf16 %r120, %rs227;
cvt.f32.bf16 %r121, %rs229;
add.s64 %rd51, %rd7, 120;
add.s64 %rd52, %rd7, 136;
mov.u16 %rs22, %rs2;
@%p1 ld.global.b16 { %rs22 }, [ %rd51 + 0 ];
mov.u16 %rs23, %rs2;
@%p2 ld.global.b16 { %rs23 }, [ %rd52 + 0 ];
cvt.f32.bf16 %r747, %rs22;
cvt.f32.bf16 %r748, %rs23;
add.rn.f32 %r119, %r120, %r121;
mul.f32 %r749, %r747, 0f3DCCCCCD;
mul.f32 %r750, %r748, 0f3DCCCCCD;
fma.rn.f32 %r751, %r119, 0f3DCCCCCD, %r749;
fma.rn.f32 %r752, %r119, 0f3DCCCCCD, %r750;
cvt.rn.bf16.f32 %rs24, %r751;
cvt.rn.bf16.f32 %rs25, %r752;
add.s64 %rd53, %rd122, %rd140;
add.s64 %rd54, %rd122, %rd141;
@%p3 st.global.b16 [ %rd53 + 0 ], { %rs24 };
@%p4 st.global.b16 [ %rd54 + 0 ], { %rs25 };
cvt.f32.bf16 %r753, %rs24;
cvt.f32.bf16 %r754, %rs25;
add.s64 %rd55, %rd123, %rd134;
add.s64 %rd56, %rd123, %rd135;
add.s64 %rd57, %rd123, %rd137;
add.s64 %rd58, %rd123, %rd138;
mov.u32 %r122, %r5;
mov.u32 %r123, %r5;
mov.u32 %r124, %r5;
mov.u32 %r125, %r5;
@%p1 ld.global.v4.b32 { %r122, %r123, %r124, %r125 }, [ %rd55 + 0 ];
mov.b32 {%rs262, %rs263}, %r122;
mov.b32 {%rs264, %rs265}, %r123;
mov.b32 {%rs266, %rs267}, %r124;
mov.b32 {%rs268, %rs269}, %r125;
mov.u32 %r126, %r5;
mov.u32 %r127, %r5;
mov.u32 %r128, %r5;
mov.u32 %r129, %r5;
@%p1 ld.global.v4.b32 { %r126, %r127, %r128, %r129 }, [ %rd56 + 0 ];
mov.b32 {%rs270, %rs271}, %r126;
mov.b32 {%rs272, %rs273}, %r127;
mov.b32 {%rs274, %rs275}, %r128;
mov.b32 {%rs276, %rs277}, %r129;
mov.u32 %r130, %r5;
mov.u32 %r131, %r5;
mov.u32 %r132, %r5;
mov.u32 %r133, %r5;
@%p2 ld.global.v4.b32 { %r130, %r131, %r132, %r133 }, [ %rd57 + 0 ];
mov.b32 {%rs278, %rs279}, %r130;
mov.b32 {%rs280, %rs281}, %r131;
mov.b32 {%rs282, %rs283}, %r132;
mov.b32 {%rs284, %rs285}, %r133;
mov.u32 %r134, %r5;
mov.u32 %r135, %r5;
mov.u32 %r136, %r5;
mov.u32 %r137, %r5;
@%p2 ld.global.v4.b32 { %r134, %r135, %r136, %r137 }, [ %rd58 + 0 ];
mov.b32 {%rs286, %rs287}, %r134;
mov.b32 {%rs288, %rs289}, %r135;
mov.b32 {%rs290, %rs291}, %r136;
mov.b32 {%rs292, %rs293}, %r137;
cvt.f32.bf16 %r755, %rs262;
cvt.f32.bf16 %r756, %rs263;
cvt.f32.bf16 %r757, %rs264;
cvt.f32.bf16 %r758, %rs265;
cvt.f32.bf16 %r759, %rs266;
cvt.f32.bf16 %r760, %rs267;
cvt.f32.bf16 %r761, %rs268;
cvt.f32.bf16 %r762, %rs269;
cvt.f32.bf16 %r763, %rs270;
cvt.f32.bf16 %r764, %rs271;
cvt.f32.bf16 %r765, %rs272;
cvt.f32.bf16 %r766, %rs273;
cvt.f32.bf16 %r767, %rs274;
cvt.f32.bf16 %r768, %rs275;
cvt.f32.bf16 %r769, %rs276;
cvt.f32.bf16 %r770, %rs277;
cvt.f32.bf16 %r771, %rs278;
cvt.f32.bf16 %r772, %rs279;
cvt.f32.bf16 %r773, %rs280;
cvt.f32.bf16 %r774, %rs281;
cvt.f32.bf16 %r775, %rs282;
cvt.f32.bf16 %r776, %rs283;
cvt.f32.bf16 %r777, %rs284;
cvt.f32.bf16 %r778, %rs285;
cvt.f32.bf16 %r779, %rs286;
cvt.f32.bf16 %r780, %rs287;
cvt.f32.bf16 %r781, %rs288;
cvt.f32.bf16 %r782, %rs289;
cvt.f32.bf16 %r783, %rs290;
cvt.f32.bf16 %r784, %rs291;
cvt.f32.bf16 %r785, %rs292;
cvt.f32.bf16 %r786, %rs293;
fma.rn.f32 %r787, %r753, %r755, %r715;
fma.rn.f32 %r788, %r753, %r756, %r716;
fma.rn.f32 %r789, %r753, %r757, %r717;
fma.rn.f32 %r790, %r753, %r758, %r718;
fma.rn.f32 %r791, %r753, %r759, %r719;
fma.rn.f32 %r792, %r753, %r760, %r720;
fma.rn.f32 %r793, %r753, %r761, %r721;
fma.rn.f32 %r794, %r753, %r762, %r722;
fma.rn.f32 %r795, %r753, %r763, %r723;
fma.rn.f32 %r796, %r753, %r764, %r724;
fma.rn.f32 %r797, %r753, %r765, %r725;
fma.rn.f32 %r798, %r753, %r766, %r726;
fma.rn.f32 %r799, %r753, %r767, %r727;
fma.rn.f32 %r800, %r753, %r768, %r728;
fma.rn.f32 %r801, %r753, %r769, %r729;
fma.rn.f32 %r802, %r753, %r770, %r730;
fma.rn.f32 %r803, %r754, %r771, %r731;
fma.rn.f32 %r804, %r754, %r772, %r732;
fma.rn.f32 %r805, %r754, %r773, %r733;
fma.rn.f32 %r806, %r754, %r774, %r734;
fma.rn.f32 %r807, %r754, %r775, %r735;
fma.rn.f32 %r808, %r754, %r776, %r736;
fma.rn.f32 %r809, %r754, %r777, %r737;
fma.rn.f32 %r810, %r754, %r778, %r738;
fma.rn.f32 %r811, %r754, %r779, %r739;
fma.rn.f32 %r812, %r754, %r780, %r740;
fma.rn.f32 %r813, %r754, %r781, %r741;
fma.rn.f32 %r814, %r754, %r782, %r742;
fma.rn.f32 %r815, %r754, %r783, %r743;
fma.rn.f32 %r816, %r754, %r784, %r744;
fma.rn.f32 %r817, %r754, %r785, %r745;
fma.rn.f32 %r818, %r754, %r786, %r746;
add.s64 %rd59, %rd107, 40;
mov.u32 %r138, 0x0;
ld.global.b32 { %r138 }, [ %rd59 + 0 ];
mov.b32 {%rs294, %rs295}, %r138;
add.s32 %r819, %r383, 6;
mad.wide.s32 %rd60, %r819, 2, %rd106;
mov.u32 %r139, 0x0;
ld.global.b32 { %r139 }, [ %rd60 + 0 ];
mov.b32 {%rs296, %rs297}, %r139;
cvt.f32.bf16 %r141, %rs294;
cvt.f32.bf16 %r142, %rs296;
add.s64 %rd61, %rd7, 144;
add.s64 %rd62, %rd7, 160;
mov.u16 %rs26, %rs2;
@%p1 ld.global.b16 { %rs26 }, [ %rd61 + 0 ];
mov.u16 %rs27, %rs2;
@%p2 ld.global.b16 { %rs27 }, [ %rd62 + 0 ];
cvt.f32.bf16 %r820, %rs26;
cvt.f32.bf16 %r821, %rs27;
add.rn.f32 %r140, %r141, %r142;
mul.f32 %r822, %r820, 0f3DCCCCCD;
mul.f32 %r823, %r821, 0f3DCCCCCD;
fma.rn.f32 %r824, %r140, 0f3DCCCCCD, %r822;
fma.rn.f32 %r825, %r140, 0f3DCCCCCD, %r823;
cvt.rn.bf16.f32 %rs28, %r824;
cvt.rn.bf16.f32 %rs29, %r825;
add.s64 %rd63, %rd124, %rd140;
add.s64 %rd64, %rd124, %rd141;
@%p3 st.global.b16 [ %rd63 + 0 ], { %rs28 };
@%p4 st.global.b16 [ %rd64 + 0 ], { %rs29 };
cvt.f32.bf16 %r826, %rs28;
cvt.f32.bf16 %r827, %rs29;
add.s64 %rd65, %rd125, %rd134;
add.s64 %rd66, %rd125, %rd135;
add.s64 %rd67, %rd125, %rd137;
add.s64 %rd68, %rd125, %rd138;
mov.u32 %r143, %r5;
mov.u32 %r144, %r5;
mov.u32 %r145, %r5;
mov.u32 %r146, %r5;
@%p1 ld.global.v4.b32 { %r143, %r144, %r145, %r146 }, [ %rd65 + 0 ];
mov.b32 {%rs298, %rs299}, %r143;
mov.b32 {%rs300, %rs301}, %r144;
mov.b32 {%rs302, %rs303}, %r145;
mov.b32 {%rs304, %rs305}, %r146;
mov.u32 %r147, %r5;
mov.u32 %r148, %r5;
mov.u32 %r149, %r5;
mov.u32 %r150, %r5;
@%p1 ld.global.v4.b32 { %r147, %r148, %r149, %r150 }, [ %rd66 + 0 ];
mov.b32 {%rs306, %rs307}, %r147;
mov.b32 {%rs308, %rs309}, %r148;
mov.b32 {%rs310, %rs311}, %r149;
mov.b32 {%rs312, %rs313}, %r150;
mov.u32 %r151, %r5;
mov.u32 %r152, %r5;
mov.u32 %r153, %r5;
mov.u32 %r154, %r5;
@%p2 ld.global.v4.b32 { %r151, %r152, %r153, %r154 }, [ %rd67 + 0 ];
mov.b32 {%rs314, %rs315}, %r151;
mov.b32 {%rs316, %rs317}, %r152;
mov.b32 {%rs318, %rs319}, %r153;
mov.b32 {%rs320, %rs321}, %r154;
mov.u32 %r155, %r5;
mov.u32 %r156, %r5;
mov.u32 %r157, %r5;
mov.u32 %r158, %r5;
@%p2 ld.global.v4.b32 { %r155, %r156, %r157, %r158 }, [ %rd68 + 0 ];
mov.b32 {%rs322, %rs323}, %r155;
mov.b32 {%rs324, %rs325}, %r156;
mov.b32 {%rs326, %rs327}, %r157;
mov.b32 {%rs328, %rs329}, %r158;
cvt.f32.bf16 %r828, %rs298;
cvt.f32.bf16 %r829, %rs299;
cvt.f32.bf16 %r830, %rs300;
cvt.f32.bf16 %r831, %rs301;
cvt.f32.bf16 %r832, %rs302;
cvt.f32.bf16 %r833, %rs303;
cvt.f32.bf16 %r834, %rs304;
cvt.f32.bf16 %r835, %rs305;
cvt.f32.bf16 %r836, %rs306;
cvt.f32.bf16 %r837, %rs307;
cvt.f32.bf16 %r838, %rs308;
cvt.f32.bf16 %r839, %rs309;
cvt.f32.bf16 %r840, %rs310;
cvt.f32.bf16 %r841, %rs311;
cvt.f32.bf16 %r842, %rs312;
cvt.f32.bf16 %r843, %rs313;
cvt.f32.bf16 %r844, %rs314;
cvt.f32.bf16 %r845, %rs315;
cvt.f32.bf16 %r846, %rs316;
cvt.f32.bf16 %r847, %rs317;
cvt.f32.bf16 %r848, %rs318;
cvt.f32.bf16 %r849, %rs319;
cvt.f32.bf16 %r850, %rs320;
cvt.f32.bf16 %r851, %rs321;
cvt.f32.bf16 %r852, %rs322;
cvt.f32.bf16 %r853, %rs323;
cvt.f32.bf16 %r854, %rs324;
cvt.f32.bf16 %r855, %rs325;
cvt.f32.bf16 %r856, %rs326;
cvt.f32.bf16 %r857, %rs327;
cvt.f32.bf16 %r858, %rs328;
cvt.f32.bf16 %r859, %rs329;
fma.rn.f32 %r860, %r826, %r828, %r787;
fma.rn.f32 %r861, %r826, %r829, %r788;
fma.rn.f32 %r862, %r826, %r830, %r789;
fma.rn.f32 %r863, %r826, %r831, %r790;
fma.rn.f32 %r864, %r826, %r832, %r791;
fma.rn.f32 %r865, %r826, %r833, %r792;
fma.rn.f32 %r866, %r826, %r834, %r793;
fma.rn.f32 %r867, %r826, %r835, %r794;
fma.rn.f32 %r868, %r826, %r836, %r795;
fma.rn.f32 %r869, %r826, %r837, %r796;
fma.rn.f32 %r870, %r826, %r838, %r797;
fma.rn.f32 %r871, %r826, %r839, %r798;
fma.rn.f32 %r872, %r826, %r840, %r799;
fma.rn.f32 %r873, %r826, %r841, %r800;
fma.rn.f32 %r874, %r826, %r842, %r801;
fma.rn.f32 %r875, %r826, %r843, %r802;
fma.rn.f32 %r876, %r827, %r844, %r803;
fma.rn.f32 %r877, %r827, %r845, %r804;
fma.rn.f32 %r878, %r827, %r846, %r805;
fma.rn.f32 %r879, %r827, %r847, %r806;
fma.rn.f32 %r880, %r827, %r848, %r807;
fma.rn.f32 %r881, %r827, %r849, %r808;
fma.rn.f32 %r882, %r827, %r850, %r809;
fma.rn.f32 %r883, %r827, %r851, %r810;
fma.rn.f32 %r884, %r827, %r852, %r811;
fma.rn.f32 %r885, %r827, %r853, %r812;
fma.rn.f32 %r886, %r827, %r854, %r813;
fma.rn.f32 %r887, %r827, %r855, %r814;
fma.rn.f32 %r888, %r827, %r856, %r815;
fma.rn.f32 %r889, %r827, %r857, %r816;
fma.rn.f32 %r890, %r827, %r858, %r817;
fma.rn.f32 %r891, %r827, %r859, %r818;
cvt.f32.bf16 %r160, %rs295;
cvt.f32.bf16 %r161, %rs297;
add.s64 %rd69, %rd7, 168;
add.s64 %rd70, %rd7, 184;
mov.u16 %rs30, %rs2;
@%p1 ld.global.b16 { %rs30 }, [ %rd69 + 0 ];
mov.u16 %rs31, %rs2;
@%p2 ld.global.b16 { %rs31 }, [ %rd70 + 0 ];
cvt.f32.bf16 %r892, %rs30;
cvt.f32.bf16 %r893, %rs31;
add.rn.f32 %r159, %r160, %r161;
mul.f32 %r894, %r892, 0f3DCCCCCD;
mul.f32 %r895, %r893, 0f3DCCCCCD;
fma.rn.f32 %r896, %r159, 0f3DCCCCCD, %r894;
fma.rn.f32 %r897, %r159, 0f3DCCCCCD, %r895;
cvt.rn.bf16.f32 %rs32, %r896;
cvt.rn.bf16.f32 %rs33, %r897;
add.s64 %rd71, %rd126, %rd140;
add.s64 %rd72, %rd126, %rd141;
@%p3 st.global.b16 [ %rd71 + 0 ], { %rs32 };
@%p4 st.global.b16 [ %rd72 + 0 ], { %rs33 };
cvt.f32.bf16 %r898, %rs32;
cvt.f32.bf16 %r899, %rs33;
add.s64 %rd73, %rd127, %rd134;
add.s64 %rd74, %rd127, %rd135;
add.s64 %rd75, %rd127, %rd137;
add.s64 %rd76, %rd127, %rd138;
mov.u32 %r162, %r5;
mov.u32 %r163, %r5;
mov.u32 %r164, %r5;
mov.u32 %r165, %r5;
@%p1 ld.global.v4.b32 { %r162, %r163, %r164, %r165 }, [ %rd73 + 0 ];
mov.b32 {%rs330, %rs331}, %r162;
mov.b32 {%rs332, %rs333}, %r163;
mov.b32 {%rs334, %rs335}, %r164;
mov.b32 {%rs336, %rs337}, %r165;
mov.u32 %r166, %r5;
mov.u32 %r167, %r5;
mov.u32 %r168, %r5;
mov.u32 %r169, %r5;
@%p1 ld.global.v4.b32 { %r166, %r167, %r168, %r169 }, [ %rd74 + 0 ];
mov.b32 {%rs338, %rs339}, %r166;
mov.b32 {%rs340, %rs341}, %r167;
mov.b32 {%rs342, %rs343}, %r168;
mov.b32 {%rs344, %rs345}, %r169;
mov.u32 %r170, %r5;
mov.u32 %r171, %r5;
mov.u32 %r172, %r5;
mov.u32 %r173, %r5;
@%p2 ld.global.v4.b32 { %r170, %r171, %r172, %r173 }, [ %rd75 + 0 ];
mov.b32 {%rs346, %rs347}, %r170;
mov.b32 {%rs348, %rs349}, %r171;
mov.b32 {%rs350, %rs351}, %r172;
mov.b32 {%rs352, %rs353}, %r173;
mov.u32 %r174, %r5;
mov.u32 %r175, %r5;
mov.u32 %r176, %r5;
mov.u32 %r177, %r5;
@%p2 ld.global.v4.b32 { %r174, %r175, %r176, %r177 }, [ %rd76 + 0 ];
mov.b32 {%rs354, %rs355}, %r174;
mov.b32 {%rs356, %rs357}, %r175;
mov.b32 {%rs358, %rs359}, %r176;
mov.b32 {%rs360, %rs361}, %r177;
cvt.f32.bf16 %r900, %rs330;
cvt.f32.bf16 %r901, %rs331;
cvt.f32.bf16 %r902, %rs332;
cvt.f32.bf16 %r903, %rs333;
cvt.f32.bf16 %r904, %rs334;
cvt.f32.bf16 %r905, %rs335;
cvt.f32.bf16 %r906, %rs336;
cvt.f32.bf16 %r907, %rs337;
cvt.f32.bf16 %r908, %rs338;
cvt.f32.bf16 %r909, %rs339;
cvt.f32.bf16 %r910, %rs340;
cvt.f32.bf16 %r911, %rs341;
cvt.f32.bf16 %r912, %rs342;
cvt.f32.bf16 %r913, %rs343;
cvt.f32.bf16 %r914, %rs344;
cvt.f32.bf16 %r915, %rs345;
cvt.f32.bf16 %r916, %rs346;
cvt.f32.bf16 %r917, %rs347;
cvt.f32.bf16 %r918, %rs348;
cvt.f32.bf16 %r919, %rs349;
cvt.f32.bf16 %r920, %rs350;
cvt.f32.bf16 %r921, %rs351;
cvt.f32.bf16 %r922, %rs352;
cvt.f32.bf16 %r923, %rs353;
cvt.f32.bf16 %r924, %rs354;
cvt.f32.bf16 %r925, %rs355;
cvt.f32.bf16 %r926, %rs356;
cvt.f32.bf16 %r927, %rs357;
cvt.f32.bf16 %r928, %rs358;
cvt.f32.bf16 %r929, %rs359;
cvt.f32.bf16 %r930, %rs360;
cvt.f32.bf16 %r931, %rs361;
fma.rn.f32 %r932, %r898, %r900, %r860;
fma.rn.f32 %r933, %r898, %r901, %r861;
fma.rn.f32 %r934, %r898, %r902, %r862;
fma.rn.f32 %r935, %r898, %r903, %r863;
fma.rn.f32 %r936, %r898, %r904, %r864;
fma.rn.f32 %r937, %r898, %r905, %r865;
fma.rn.f32 %r938, %r898, %r906, %r866;
fma.rn.f32 %r939, %r898, %r907, %r867;
fma.rn.f32 %r940, %r898, %r908, %r868;
fma.rn.f32 %r941, %r898, %r909, %r869;
fma.rn.f32 %r942, %r898, %r910, %r870;
fma.rn.f32 %r943, %r898, %r911, %r871;
fma.rn.f32 %r944, %r898, %r912, %r872;
fma.rn.f32 %r945, %r898, %r913, %r873;
fma.rn.f32 %r946, %r898, %r914, %r874;
fma.rn.f32 %r947, %r898, %r915, %r875;
fma.rn.f32 %r948, %r899, %r916, %r876;
fma.rn.f32 %r949, %r899, %r917, %r877;
fma.rn.f32 %r950, %r899, %r918, %r878;
fma.rn.f32 %r951, %r899, %r919, %r879;
fma.rn.f32 %r952, %r899, %r920, %r880;
fma.rn.f32 %r953, %r899, %r921, %r881;
fma.rn.f32 %r954, %r899, %r922, %r882;
fma.rn.f32 %r955, %r899, %r923, %r883;
fma.rn.f32 %r956, %r899, %r924, %r884;
fma.rn.f32 %r957, %r899, %r925, %r885;
fma.rn.f32 %r958, %r899, %r926, %r886;
fma.rn.f32 %r959, %r899, %r927, %r887;
fma.rn.f32 %r960, %r899, %r928, %r888;
fma.rn.f32 %r961, %r899, %r929, %r889;
fma.rn.f32 %r962, %r899, %r930, %r890;
fma.rn.f32 %r963, %r899, %r931, %r891;
add.s64 %rd77, %rd107, 44;
mov.u32 %r178, 0x0;
ld.global.b32 { %r178 }, [ %rd77 + 0 ];
mov.b32 {%rs362, %rs363}, %r178;
add.s32 %r964, %r383, 8;
mad.wide.s32 %rd78, %r964, 2, %rd106;
mov.u32 %r179, 0x0;
ld.global.b32 { %r179 }, [ %rd78 + 0 ];
mov.b32 {%rs364, %rs365}, %r179;
cvt.f32.bf16 %r181, %rs362;
cvt.f32.bf16 %r182, %rs364;
add.s64 %rd79, %rd7, 192;
add.s64 %rd80, %rd7, 208;
mov.u16 %rs34, %rs2;
@%p1 ld.global.b16 { %rs34 }, [ %rd79 + 0 ];
mov.u16 %rs35, %rs2;
@%p2 ld.global.b16 { %rs35 }, [ %rd80 + 0 ];
cvt.f32.bf16 %r965, %rs34;
cvt.f32.bf16 %r966, %rs35;
add.rn.f32 %r180, %r181, %r182;
mul.f32 %r967, %r965, 0f3DCCCCCD;
mul.f32 %r968, %r966, 0f3DCCCCCD;
fma.rn.f32 %r969, %r180, 0f3DCCCCCD, %r967;
fma.rn.f32 %r970, %r180, 0f3DCCCCCD, %r968;
cvt.rn.bf16.f32 %rs36, %r969;
cvt.rn.bf16.f32 %rs37, %r970;
add.s64 %rd81, %rd128, %rd140;
add.s64 %rd82, %rd128, %rd141;
@%p3 st.global.b16 [ %rd81 + 0 ], { %rs36 };
@%p4 st.global.b16 [ %rd82 + 0 ], { %rs37 };
cvt.f32.bf16 %r971, %rs36;
cvt.f32.bf16 %r972, %rs37;
add.s64 %rd83, %rd129, %rd134;
add.s64 %rd84, %rd129, %rd135;
add.s64 %rd85, %rd129, %rd137;
add.s64 %rd86, %rd129, %rd138;
mov.u32 %r183, %r5;
mov.u32 %r184, %r5;
mov.u32 %r185, %r5;
mov.u32 %r186, %r5;
@%p1 ld.global.v4.b32 { %r183, %r184, %r185, %r186 }, [ %rd83 + 0 ];
mov.b32 {%rs366, %rs367}, %r183;
mov.b32 {%rs368, %rs369}, %r184;
mov.b32 {%rs370, %rs371}, %r185;
mov.b32 {%rs372, %rs373}, %r186;
mov.u32 %r187, %r5;
mov.u32 %r188, %r5;
mov.u32 %r189, %r5;
mov.u32 %r190, %r5;
@%p1 ld.global.v4.b32 { %r187, %r188, %r189, %r190 }, [ %rd84 + 0 ];
mov.b32 {%rs374, %rs375}, %r187;
mov.b32 {%rs376, %rs377}, %r188;
mov.b32 {%rs378, %rs379}, %r189;
mov.b32 {%rs380, %rs381}, %r190;
mov.u32 %r191, %r5;
mov.u32 %r192, %r5;
mov.u32 %r193, %r5;
mov.u32 %r194, %r5;
@%p2 ld.global.v4.b32 { %r191, %r192, %r193, %r194 }, [ %rd85 + 0 ];
mov.b32 {%rs382, %rs383}, %r191;
mov.b32 {%rs384, %rs385}, %r192;
mov.b32 {%rs386, %rs387}, %r193;
mov.b32 {%rs388, %rs389}, %r194;
mov.u32 %r195, %r5;
mov.u32 %r196, %r5;
mov.u32 %r197, %r5;
mov.u32 %r198, %r5;
@%p2 ld.global.v4.b32 { %r195, %r196, %r197, %r198 }, [ %rd86 + 0 ];
mov.b32 {%rs390, %rs391}, %r195;
mov.b32 {%rs392, %rs393}, %r196;
mov.b32 {%rs394, %rs395}, %r197;
mov.b32 {%rs396, %rs397}, %r198;
cvt.f32.bf16 %r973, %rs366;
cvt.f32.bf16 %r974, %rs367;
cvt.f32.bf16 %r975, %rs368;
cvt.f32.bf16 %r976, %rs369;
cvt.f32.bf16 %r977, %rs370;
cvt.f32.bf16 %r978, %rs371;
cvt.f32.bf16 %r979, %rs372;
cvt.f32.bf16 %r980, %rs373;
cvt.f32.bf16 %r981, %rs374;
cvt.f32.bf16 %r982, %rs375;
cvt.f32.bf16 %r983, %rs376;
cvt.f32.bf16 %r984, %rs377;
cvt.f32.bf16 %r985, %rs378;
cvt.f32.bf16 %r986, %rs379;
cvt.f32.bf16 %r987, %rs380;
cvt.f32.bf16 %r988, %rs381;
cvt.f32.bf16 %r989, %rs382;
cvt.f32.bf16 %r990, %rs383;
cvt.f32.bf16 %r991, %rs384;
cvt.f32.bf16 %r992, %rs385;
cvt.f32.bf16 %r993, %rs386;
cvt.f32.bf16 %r994, %rs387;
cvt.f32.bf16 %r995, %rs388;
cvt.f32.bf16 %r996, %rs389;
cvt.f32.bf16 %r997, %rs390;
cvt.f32.bf16 %r998, %rs391;
cvt.f32.bf16 %r999, %rs392;
cvt.f32.bf16 %r1000, %rs393;
cvt.f32.bf16 %r1001, %rs394;
cvt.f32.bf16 %r1002, %rs395;
cvt.f32.bf16 %r1003, %rs396;
cvt.f32.bf16 %r1004, %rs397;
fma.rn.f32 %r1005, %r971, %r973, %r932;
fma.rn.f32 %r1006, %r971, %r974, %r933;
fma.rn.f32 %r1007, %r971, %r975, %r934;
fma.rn.f32 %r1008, %r971, %r976, %r935;
fma.rn.f32 %r1009, %r971, %r977, %r936;
fma.rn.f32 %r1010, %r971, %r978, %r937;
fma.rn.f32 %r1011, %r971, %r979, %r938;
fma.rn.f32 %r1012, %r971, %r980, %r939;
fma.rn.f32 %r1013, %r971, %r981, %r940;
fma.rn.f32 %r1014, %r971, %r982, %r941;
fma.rn.f32 %r1015, %r971, %r983, %r942;
fma.rn.f32 %r1016, %r971, %r984, %r943;
fma.rn.f32 %r1017, %r971, %r985, %r944;
fma.rn.f32 %r1018, %r971, %r986, %r945;
fma.rn.f32 %r1019, %r971, %r987, %r946;
fma.rn.f32 %r1020, %r971, %r988, %r947;
fma.rn.f32 %r1021, %r972, %r989, %r948;
fma.rn.f32 %r1022, %r972, %r990, %r949;
fma.rn.f32 %r1023, %r972, %r991, %r950;
fma.rn.f32 %r1024, %r972, %r992, %r951;
fma.rn.f32 %r1025, %r972, %r993, %r952;
fma.rn.f32 %r1026, %r972, %r994, %r953;
fma.rn.f32 %r1027, %r972, %r995, %r954;
fma.rn.f32 %r1028, %r972, %r996, %r955;
fma.rn.f32 %r1029, %r972, %r997, %r956;
fma.rn.f32 %r1030, %r972, %r998, %r957;
fma.rn.f32 %r1031, %r972, %r999, %r958;
fma.rn.f32 %r1032, %r972, %r1000, %r959;
fma.rn.f32 %r1033, %r972, %r1001, %r960;
fma.rn.f32 %r1034, %r972, %r1002, %r961;
fma.rn.f32 %r1035, %r972, %r1003, %r962;
fma.rn.f32 %r1036, %r972, %r1004, %r963;
cvt.f32.bf16 %r200, %rs363;
cvt.f32.bf16 %r201, %rs365;
add.s64 %rd87, %rd7, 216;
add.s64 %rd88, %rd7, 232;
mov.u16 %rs38, %rs2;
@%p1 ld.global.b16 { %rs38 }, [ %rd87 + 0 ];
mov.u16 %rs39, %rs2;
@%p2 ld.global.b16 { %rs39 }, [ %rd88 + 0 ];
cvt.f32.bf16 %r1037, %rs38;
cvt.f32.bf16 %r1038, %rs39;
add.rn.f32 %r199, %r200, %r201;
mul.f32 %r1039, %r1037, 0f3DCCCCCD;
mul.f32 %r1040, %r1038, 0f3DCCCCCD;
fma.rn.f32 %r1041, %r199, 0f3DCCCCCD, %r1039;
fma.rn.f32 %r1042, %r199, 0f3DCCCCCD, %r1040;
cvt.rn.bf16.f32 %rs40, %r1041;
cvt.rn.bf16.f32 %rs41, %r1042;
add.s64 %rd89, %rd130, %rd140;
add.s64 %rd90, %rd130, %rd141;
@%p3 st.global.b16 [ %rd89 + 0 ], { %rs40 };
@%p4 st.global.b16 [ %rd90 + 0 ], { %rs41 };
cvt.f32.bf16 %r1043, %rs40;
cvt.f32.bf16 %r1044, %rs41;
add.s64 %rd91, %rd131, %rd134;
add.s64 %rd92, %rd131, %rd135;
add.s64 %rd93, %rd131, %rd137;
add.s64 %rd94, %rd131, %rd138;
mov.u32 %r202, %r5;
mov.u32 %r203, %r5;
mov.u32 %r204, %r5;
mov.u32 %r205, %r5;
@%p1 ld.global.v4.b32 { %r202, %r203, %r204, %r205 }, [ %rd91 + 0 ];
mov.b32 {%rs398, %rs399}, %r202;
mov.b32 {%rs400, %rs401}, %r203;
mov.b32 {%rs402, %rs403}, %r204;
mov.b32 {%rs404, %rs405}, %r205;
mov.u32 %r206, %r5;
mov.u32 %r207, %r5;
mov.u32 %r208, %r5;
mov.u32 %r209, %r5;
@%p1 ld.global.v4.b32 { %r206, %r207, %r208, %r209 }, [ %rd92 + 0 ];
mov.b32 {%rs406, %rs407}, %r206;
mov.b32 {%rs408, %rs409}, %r207;
mov.b32 {%rs410, %rs411}, %r208;
mov.b32 {%rs412, %rs413}, %r209;
mov.u32 %r210, %r5;
mov.u32 %r211, %r5;
mov.u32 %r212, %r5;
mov.u32 %r213, %r5;
@%p2 ld.global.v4.b32 { %r210, %r211, %r212, %r213 }, [ %rd93 + 0 ];
mov.b32 {%rs414, %rs415}, %r210;
mov.b32 {%rs416, %rs417}, %r211;
mov.b32 {%rs418, %rs419}, %r212;
mov.b32 {%rs420, %rs421}, %r213;
mov.u32 %r214, %r5;
mov.u32 %r215, %r5;
mov.u32 %r216, %r5;
mov.u32 %r217, %r5;
@%p2 ld.global.v4.b32 { %r214, %r215, %r216, %r217 }, [ %rd94 + 0 ];
mov.b32 {%rs422, %rs423}, %r214;
mov.b32 {%rs424, %rs425}, %r215;
mov.b32 {%rs426, %rs427}, %r216;
mov.b32 {%rs428, %rs429}, %r217;
cvt.f32.bf16 %r1045, %rs398;
cvt.f32.bf16 %r1046, %rs399;
cvt.f32.bf16 %r1047, %rs400;
cvt.f32.bf16 %r1048, %rs401;
cvt.f32.bf16 %r1049, %rs402;
cvt.f32.bf16 %r1050, %rs403;
cvt.f32.bf16 %r1051, %rs404;
cvt.f32.bf16 %r1052, %rs405;
cvt.f32.bf16 %r1053, %rs406;
cvt.f32.bf16 %r1054, %rs407;
cvt.f32.bf16 %r1055, %rs408;
cvt.f32.bf16 %r1056, %rs409;
cvt.f32.bf16 %r1057, %rs410;
cvt.f32.bf16 %r1058, %rs411;
cvt.f32.bf16 %r1059, %rs412;
cvt.f32.bf16 %r1060, %rs413;
cvt.f32.bf16 %r1061, %rs414;
cvt.f32.bf16 %r1062, %rs415;
cvt.f32.bf16 %r1063, %rs416;
cvt.f32.bf16 %r1064, %rs417;
cvt.f32.bf16 %r1065, %rs418;
cvt.f32.bf16 %r1066, %rs419;
cvt.f32.bf16 %r1067, %rs420;
cvt.f32.bf16 %r1068, %rs421;
cvt.f32.bf16 %r1069, %rs422;
cvt.f32.bf16 %r1070, %rs423;
cvt.f32.bf16 %r1071, %rs424;
cvt.f32.bf16 %r1072, %rs425;
cvt.f32.bf16 %r1073, %rs426;
cvt.f32.bf16 %r1074, %rs427;
cvt.f32.bf16 %r1075, %rs428;
cvt.f32.bf16 %r1076, %rs429;
fma.rn.f32 %r222, %r1043, %r1045, %r1005;
fma.rn.f32 %r225, %r1043, %r1046, %r1006;
fma.rn.f32 %r227, %r1043, %r1047, %r1007;
fma.rn.f32 %r229, %r1043, %r1048, %r1008;
fma.rn.f32 %r231, %r1043, %r1049, %r1009;
fma.rn.f32 %r233, %r1043, %r1050, %r1010;
fma.rn.f32 %r235, %r1043, %r1051, %r1011;
fma.rn.f32 %r237, %r1043, %r1052, %r1012;
fma.rn.f32 %r239, %r1043, %r1053, %r1013;
fma.rn.f32 %r241, %r1043, %r1054, %r1014;
fma.rn.f32 %r243, %r1043, %r1055, %r1015;
fma.rn.f32 %r245, %r1043, %r1056, %r1016;
fma.rn.f32 %r247, %r1043, %r1057, %r1017;
fma.rn.f32 %r249, %r1043, %r1058, %r1018;
fma.rn.f32 %r251, %r1043, %r1059, %r1019;
fma.rn.f32 %r253, %r1043, %r1060, %r1020;
fma.rn.f32 %r255, %r1044, %r1061, %r1021;
fma.rn.f32 %r257, %r1044, %r1062, %r1022;
fma.rn.f32 %r259, %r1044, %r1063, %r1023;
fma.rn.f32 %r261, %r1044, %r1064, %r1024;
fma.rn.f32 %r263, %r1044, %r1065, %r1025;
fma.rn.f32 %r265, %r1044, %r1066, %r1026;
fma.rn.f32 %r267, %r1044, %r1067, %r1027;
fma.rn.f32 %r269, %r1044, %r1068, %r1028;
fma.rn.f32 %r271, %r1044, %r1069, %r1029;
fma.rn.f32 %r273, %r1044, %r1070, %r1030;
fma.rn.f32 %r275, %r1044, %r1071, %r1031;
fma.rn.f32 %r277, %r1044, %r1072, %r1032;
fma.rn.f32 %r279, %r1044, %r1073, %r1033;
fma.rn.f32 %r281, %r1044, %r1074, %r1034;
fma.rn.f32 %r283, %r1044, %r1075, %r1035;
fma.rn.f32 %r285, %r1044, %r1076, %r1036;
mul.f32 %r1077, %r225, %r225;
mul.f32 %r1078, %r255, %r255;
mul.f32 %r1079, %r257, %r257;
mul.f32 %r1080, %r259, %r259;
mul.f32 %r1081, %r261, %r261;
mul.f32 %r1082, %r263, %r263;
mul.f32 %r1083, %r265, %r265;
mul.f32 %r1084, %r267, %r267;
mul.f32 %r1085, %r269, %r269;
mul.f32 %r1086, %r271, %r271;
mul.f32 %r1087, %r273, %r273;
mul.f32 %r1088, %r275, %r275;
mul.f32 %r1089, %r277, %r277;
mul.f32 %r1090, %r279, %r279;
mul.f32 %r1091, %r281, %r281;
mul.f32 %r1092, %r283, %r283;
mul.f32 %r1093, %r285, %r285;
selp.f32 %r1094, %r1078, 0f00000000, %p2;
selp.f32 %r1095, %r1079, 0f00000000, %p2;
selp.f32 %r1096, %r1080, 0f00000000, %p2;
selp.f32 %r1097, %r1081, 0f00000000, %p2;
selp.f32 %r1098, %r1082, 0f00000000, %p2;
selp.f32 %r1099, %r1083, 0f00000000, %p2;
selp.f32 %r1100, %r1084, 0f00000000, %p2;
selp.f32 %r1101, %r1085, 0f00000000, %p2;
selp.f32 %r1102, %r1086, 0f00000000, %p2;
selp.f32 %r1103, %r1087, 0f00000000, %p2;
selp.f32 %r1104, %r1088, 0f00000000, %p2;
selp.f32 %r1105, %r1089, 0f00000000, %p2;
selp.f32 %r1106, %r1090, 0f00000000, %p2;
selp.f32 %r1107, %r1091, 0f00000000, %p2;
selp.f32 %r1108, %r1092, 0f00000000, %p2;
selp.f32 %r1109, %r1093, 0f00000000, %p2;
$L__tmp1:
fma.rn.f32 %r1110, %r222, %r222, %r1077;
fma.rn.f32 %r1111, %r227, %r227, %r1110;
fma.rn.f32 %r1112, %r229, %r229, %r1111;
fma.rn.f32 %r1113, %r231, %r231, %r1112;
fma.rn.f32 %r1114, %r233, %r233, %r1113;
fma.rn.f32 %r1115, %r235, %r235, %r1114;
fma.rn.f32 %r1116, %r237, %r237, %r1115;
fma.rn.f32 %r1117, %r239, %r239, %r1116;
fma.rn.f32 %r1118, %r241, %r241, %r1117;
fma.rn.f32 %r1119, %r243, %r243, %r1118;
fma.rn.f32 %r1120, %r245, %r245, %r1119;
fma.rn.f32 %r1121, %r247, %r247, %r1120;
fma.rn.f32 %r1122, %r249, %r249, %r1121;
fma.rn.f32 %r1123, %r251, %r251, %r1122;
fma.rn.f32 %r1124, %r253, %r253, %r1123;
add.f32 %r1125, %r1094, %r1124;
add.f32 %r1126, %r1095, %r1125;
add.f32 %r1127, %r1096, %r1126;
add.f32 %r1128, %r1097, %r1127;
add.f32 %r1129, %r1098, %r1128;
add.f32 %r1130, %r1099, %r1129;
add.f32 %r1131, %r1100, %r1130;
add.f32 %r1132, %r1101, %r1131;
add.f32 %r1133, %r1102, %r1132;
add.f32 %r1134, %r1103, %r1133;
add.f32 %r1135, %r1104, %r1134;
add.f32 %r1136, %r1105, %r1135;
add.f32 %r1137, %r1106, %r1136;
add.f32 %r1138, %r1107, %r1137;
add.f32 %r1139, %r1108, %r1138;
add.f32 %r1140, %r1109, %r1139;
$L__tmp2:
shfl.sync.bfly.b32 %r1141, %r1140, 16, 31, -1;
$L__tmp3:
add.f32 %r1142, %r1140, %r1141;
$L__tmp4:
shfl.sync.bfly.b32 %r1143, %r1142, 8, 31, -1;
$L__tmp5:
add.f32 %r1144, %r1142, %r1143;
$L__tmp6:
shfl.sync.bfly.b32 %r1145, %r1144, 4, 31, -1;
$L__tmp7:
add.f32 %r1146, %r1144, %r1145;
$L__tmp8:
shfl.sync.bfly.b32 %r1147, %r1146, 2, 31, -1;
$L__tmp9:
add.f32 %r1148, %r1146, %r1147;
$L__tmp10:
shfl.sync.bfly.b32 %r1149, %r1148, 1, 31, -1;
$L__tmp11:
add.f32 %r220, %r1148, %r1149;
$L__tmp12:
div.full.f32 %r218, %r220, 0f44400000;
add.f32 %r219, %r218, 0f34000000;
rsqrt.approx.ftz.f32 %r223, %r219;
mad.wide.u32 %rd95, %r334, 4, %rd110;
setp.eq.b32 %p5, %r327, 0;
@%p5 st.global.b32 [ %rd95 + 0 ], { %r220 };
mad.wide.s32 %rd96, %r337, 4, %rd109;
mad.wide.s32 %rd97, %r339, 4, %rd109;
mad.wide.s32 %rd98, %r341, 4, %rd109;
mad.wide.s32 %rd99, %r343, 4, %rd109;
mad.wide.s32 %rd100, %r344, 4, %rd109;
mad.wide.s32 %rd101, %r345, 4, %rd109;
mad.wide.s32 %rd102, %r346, 4, %rd109;
mad.wide.s32 %rd103, %r347, 4, %rd109;
@%p1 st.global.v4.b32 [ %rd96 + 0 ], { %r222, %r225, %r227, %r229 };
@%p1 st.global.v4.b32 [ %rd97 + 0 ], { %r231, %r233, %r235, %r237 };
@%p1 st.global.v4.b32 [ %rd98 + 0 ], { %r239, %r241, %r243, %r245 };
@%p1 st.global.v4.b32 [ %rd99 + 0 ], { %r247, %r249, %r251, %r253 };
@%p2 st.global.v4.b32 [ %rd100 + 0 ], { %r255, %r257, %r259, %r261 };
@%p2 st.global.v4.b32 [ %rd101 + 0 ], { %r263, %r265, %r267, %r269 };
@%p2 st.global.v4.b32 [ %rd102 + 0 ], { %r271, %r273, %r275, %r277 };
@%p2 st.global.v4.b32 [ %rd103 + 0 ], { %r279, %r281, %r283, %r285 };
mul.rn.f32 %r221, %r222, %r223;
mul.rn.f32 %r224, %r225, %r223;
mul.rn.f32 %r226, %r227, %r223;
mul.rn.f32 %r228, %r229, %r223;
mul.rn.f32 %r230, %r231, %r223;
mul.rn.f32 %r232, %r233, %r223;
mul.rn.f32 %r234, %r235, %r223;
mul.rn.f32 %r236, %r237, %r223;
mul.rn.f32 %r238, %r239, %r223;
mul.rn.f32 %r240, %r241, %r223;
mul.rn.f32 %r242, %r243, %r223;
mul.rn.f32 %r244, %r245, %r223;
mul.rn.f32 %r246, %r247, %r223;
mul.rn.f32 %r248, %r249, %r223;
mul.rn.f32 %r250, %r251, %r223;
mul.rn.f32 %r252, %r253, %r223;
mul.rn.f32 %r254, %r255, %r223;
mul.rn.f32 %r256, %r257, %r223;
mul.rn.f32 %r258, %r259, %r223;
mul.rn.f32 %r260, %r261, %r223;
mul.rn.f32 %r262, %r263, %r223;
mul.rn.f32 %r264, %r265, %r223;
mul.rn.f32 %r266, %r267, %r223;
mul.rn.f32 %r268, %r269, %r223;
mul.rn.f32 %r270, %r271, %r223;
mul.rn.f32 %r272, %r273, %r223;
mul.rn.f32 %r274, %r275, %r223;
mul.rn.f32 %r276, %r277, %r223;
mul.rn.f32 %r278, %r279, %r223;
mul.rn.f32 %r280, %r281, %r223;
mul.rn.f32 %r282, %r283, %r223;
mul.rn.f32 %r284, %r285, %r223;
add.s64 %rd104, %rd132, %rd133;
add.s64 %rd105, %rd132, %rd136;
mul.f32 %r286, %r221, 0f408F5C29;
mul.f32 %r287, %r224, 0f408F5C29;
mul.f32 %r288, %r226, 0f408F5C29;
mul.f32 %r289, %r228, 0f408F5C29;
mul.f32 %r290, %r230, 0f408F5C29;
mul.f32 %r291, %r232, 0f408F5C29;
mul.f32 %r292, %r234, 0f408F5C29;
mul.f32 %r293, %r236, 0f408F5C29;
mul.f32 %r294, %r238, 0f408F5C29;
mul.f32 %r295, %r240, 0f408F5C29;
mul.f32 %r296, %r242, 0f408F5C29;
mul.f32 %r297, %r244, 0f408F5C29;
mul.f32 %r298, %r246, 0f408F5C29;
mul.f32 %r299, %r248, 0f408F5C29;
mul.f32 %r300, %r250, 0f408F5C29;
mul.f32 %r301, %r252, 0f408F5C29;
mul.f32 %r302, %r254, 0f408F5C29;
mul.f32 %r303, %r256, 0f408F5C29;
mul.f32 %r304, %r258, 0f408F5C29;
mul.f32 %r305, %r260, 0f408F5C29;
mul.f32 %r306, %r262, 0f408F5C29;
mul.f32 %r307, %r264, 0f408F5C29;
mul.f32 %r308, %r266, 0f408F5C29;
mul.f32 %r309, %r268, 0f408F5C29;
mul.f32 %r310, %r270, 0f408F5C29;
mul.f32 %r311, %r272, 0f408F5C29;
mul.f32 %r312, %r274, 0f408F5C29;
mul.f32 %r313, %r276, 0f408F5C29;
mul.f32 %r314, %r278, 0f408F5C29;
mul.f32 %r315, %r280, 0f408F5C29;
mul.f32 %r316, %r282, 0f408F5C29;
mul.f32 %r317, %r284, 0f408F5C29;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r287, %r286;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r289, %r288;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r291, %r290;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r293, %r292;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r295, %r294;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r297, %r296;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r299, %r298;
cvt.rn.satfinite.e4m3x2.f32 %rs49, %r301, %r300;
cvt.rn.satfinite.e4m3x2.f32 %rs50, %r303, %r302;
cvt.rn.satfinite.e4m3x2.f32 %rs51, %r305, %r304;
cvt.rn.satfinite.e4m3x2.f32 %rs52, %r307, %r306;
cvt.rn.satfinite.e4m3x2.f32 %rs53, %r309, %r308;
cvt.rn.satfinite.e4m3x2.f32 %rs54, %r311, %r310;
cvt.rn.satfinite.e4m3x2.f32 %rs55, %r313, %r312;
cvt.rn.satfinite.e4m3x2.f32 %rs56, %r315, %r314;
cvt.rn.satfinite.e4m3x2.f32 %rs57, %r317, %r316;
mov.b32 %r318, {%rs42, %rs43};
mov.b32 %r319, {%rs44, %rs45};
mov.b32 %r320, {%rs46, %rs47};
mov.b32 %r321, {%rs48, %rs49};
@%p1 st.global.v4.b32 [ %rd104 + 0 ], { %r318, %r319, %r320, %r321 };
mov.b32 %r322, {%rs50, %rs51};
mov.b32 %r323, {%rs52, %rs53};
mov.b32 %r324, {%rs54, %rs55};
mov.b32 %r325, {%rs56, %rs57};
@%p2 st.global.v4.b32 [ %rd105 + 0 ], { %r322, %r323, %r324, %r325 };
ret;
$L__tmp13:
$L__func_end0:
}
"#

def recognizes (ptx : String) : Bool := QKVTTIR.normalizePTX ptx == assembly

structure Program where
  ptx : String
  checked : recognizes ptx = true

def Program.check (ptx : String) : Option Program :=
  if h : recognizes ptx then some ⟨ptx, h⟩ else none

/-- The cached specialization renames the affected scalar registers and stores
the inverse RMS at the original statistic address. -/
private def cachedChanges : List (String × String) := [
  ("fma.rn.f32 %r222, %r1043, %r1045, %r1005;", "fma.rn.f32 %r223, %r1043, %r1045, %r1005;"),
  ("fma.rn.f32 %r1110, %r222, %r222, %r1077;", "fma.rn.f32 %r1110, %r223, %r223, %r1077;"),
  ("add.f32 %r220, %r1148, %r1149;", "add.f32 %r219, %r1148, %r1149;"),
  ("div.full.f32 %r218, %r220, 0f44400000;", "div.full.f32 %r218, %r219, 0f44400000;"),
  ("add.f32 %r219, %r218, 0f34000000;", "add.f32 %r220, %r218, 0f34000000;"),
  ("rsqrt.approx.ftz.f32 %r223, %r219;", "rsqrt.approx.ftz.f32 %r221, %r220;"),
  ("@%p5 st.global.b32 [ %rd95 + 0 ], { %r220 };", "@%p5 st.global.b32 [ %rd95 + 0 ], { %r221 };"),
  ("@%p1 st.global.v4.b32 [ %rd96 + 0 ], { %r222, %r225, %r227, %r229 };", "@%p1 st.global.v4.b32 [ %rd96 + 0 ], { %r223, %r225, %r227, %r229 };"),
  ("mul.rn.f32 %r221, %r222, %r223;", "mul.rn.f32 %r222, %r223, %r221;"),
  ("mul.rn.f32 %r224, %r225, %r223;", "mul.rn.f32 %r224, %r225, %r221;"),
  ("mul.rn.f32 %r226, %r227, %r223;", "mul.rn.f32 %r226, %r227, %r221;"),
  ("mul.rn.f32 %r228, %r229, %r223;", "mul.rn.f32 %r228, %r229, %r221;"),
  ("mul.rn.f32 %r230, %r231, %r223;", "mul.rn.f32 %r230, %r231, %r221;"),
  ("mul.rn.f32 %r232, %r233, %r223;", "mul.rn.f32 %r232, %r233, %r221;"),
  ("mul.rn.f32 %r234, %r235, %r223;", "mul.rn.f32 %r234, %r235, %r221;"),
  ("mul.rn.f32 %r236, %r237, %r223;", "mul.rn.f32 %r236, %r237, %r221;"),
  ("mul.rn.f32 %r238, %r239, %r223;", "mul.rn.f32 %r238, %r239, %r221;"),
  ("mul.rn.f32 %r240, %r241, %r223;", "mul.rn.f32 %r240, %r241, %r221;"),
  ("mul.rn.f32 %r242, %r243, %r223;", "mul.rn.f32 %r242, %r243, %r221;"),
  ("mul.rn.f32 %r244, %r245, %r223;", "mul.rn.f32 %r244, %r245, %r221;"),
  ("mul.rn.f32 %r246, %r247, %r223;", "mul.rn.f32 %r246, %r247, %r221;"),
  ("mul.rn.f32 %r248, %r249, %r223;", "mul.rn.f32 %r248, %r249, %r221;"),
  ("mul.rn.f32 %r250, %r251, %r223;", "mul.rn.f32 %r250, %r251, %r221;"),
  ("mul.rn.f32 %r252, %r253, %r223;", "mul.rn.f32 %r252, %r253, %r221;"),
  ("mul.rn.f32 %r254, %r255, %r223;", "mul.rn.f32 %r254, %r255, %r221;"),
  ("mul.rn.f32 %r256, %r257, %r223;", "mul.rn.f32 %r256, %r257, %r221;"),
  ("mul.rn.f32 %r258, %r259, %r223;", "mul.rn.f32 %r258, %r259, %r221;"),
  ("mul.rn.f32 %r260, %r261, %r223;", "mul.rn.f32 %r260, %r261, %r221;"),
  ("mul.rn.f32 %r262, %r263, %r223;", "mul.rn.f32 %r262, %r263, %r221;"),
  ("mul.rn.f32 %r264, %r265, %r223;", "mul.rn.f32 %r264, %r265, %r221;"),
  ("mul.rn.f32 %r266, %r267, %r223;", "mul.rn.f32 %r266, %r267, %r221;"),
  ("mul.rn.f32 %r268, %r269, %r223;", "mul.rn.f32 %r268, %r269, %r221;"),
  ("mul.rn.f32 %r270, %r271, %r223;", "mul.rn.f32 %r270, %r271, %r221;"),
  ("mul.rn.f32 %r272, %r273, %r223;", "mul.rn.f32 %r272, %r273, %r221;"),
  ("mul.rn.f32 %r274, %r275, %r223;", "mul.rn.f32 %r274, %r275, %r221;"),
  ("mul.rn.f32 %r276, %r277, %r223;", "mul.rn.f32 %r276, %r277, %r221;"),
  ("mul.rn.f32 %r278, %r279, %r223;", "mul.rn.f32 %r278, %r279, %r221;"),
  ("mul.rn.f32 %r280, %r281, %r223;", "mul.rn.f32 %r280, %r281, %r221;"),
  ("mul.rn.f32 %r282, %r283, %r223;", "mul.rn.f32 %r282, %r283, %r221;"),
  ("mul.rn.f32 %r284, %r285, %r223;", "mul.rn.f32 %r284, %r285, %r221;"),
  ("mul.f32 %r286, %r221, 0f408F5C29;", "mul.f32 %r286, %r222, 0f408F5C29;")]

def recognizesCached (ptx : String) : Bool :=
  QKVTTIR.normalizePTX ptx == assembly.map (fun line => (cachedChanges.lookup line).getD line)

structure CachedProgram where
  ptx : String
  checked : recognizesCached ptx = true

def CachedProgram.check (ptx : String) : Option CachedProgram :=
  if h : recognizesCached ptx then some ⟨ptx, h⟩ else none

/-- Bits 2 through 4 of the lane ID select the coefficient in the
first half-row. The second half-row sets bit 3 of the group index. -/
def groupFromLane (lane word : Fin 32) : Nat :=
  let low := (lane.val >>> 2) &&& 7
  if word.val < 16 then low else low ||| 8

theorem group_from_lane_eq (lane word : Fin 32) :
    groupFromLane lane word = directGroup lane word := by
  have checked : ∀ lane word : Fin 32, groupFromLane lane word = directGroup lane word := by
    decide +kernel
  exact checked lane word

/-- Gluon's projected layout writes in lane 0 of each four-lane group. -/
def storeLane (group : Fin 12) : Fin 32 :=
  ⟨4 * (group.val % 8), by have h := Nat.mod_lt group.val (by decide : 0 < 8); omega⟩

def storeWord (group : Fin 12) : Fin 32 :=
  ⟨16 * (group.val / 8), by have := group.isLt; omega⟩

theorem group_store_exact (group : Fin 12) :
    groupFromLane (storeLane group) (storeWord group) = group.val ∧
    channel (storeLane group).val (storeWord group).val = 64 * group.val ∧
    (storeLane group).val % 4 = 0 := by
  have checked : ∀ group : Fin 12,
      groupFromLane (storeLane group) (storeWord group) = group.val ∧
      channel (storeLane group).val (storeWord group).val = 64 * group.val ∧
      (storeLane group).val % 4 = 0 := by decide +kernel
  exact checked group

private def packedColumn (plane : Fin 10) : Nat :=
  let pair := plane.val / 2
  let halfWord := plane.val % 2
  2 * pair + halfWord

theorem packed_column_exact (plane : Fin 10) : packedColumn plane = plane.val :=
  Nat.div_add_mod plane.val 2

/-- Each aligned load takes a pair from one scalar bias interval and one MU row. -/
theorem coefficient_pair_in_row (row : Nat) (pair : Fin 5) :
    (14 + 2 * pair.val) % 2 = 0 ∧ (row * 10 + 2 * pair.val) % 2 = 0 ∧
      14 + 2 * pair.val + 1 < 24 ∧
      row * 10 + 2 * pair.val + 1 < (row + 1) * 10 := by
  have completePair := pair.isLt
  omega

#axiomsClean packed_column_exact
#axiomsClean coefficient_pair_in_row

def emittedCoefficients (packet : RowPacket) : RowCoefficients := fun plane group =>
  let row := packet.row.val
  let k := plane.val
  let index := groupFromLane (storeLane group) (storeWord group)
  let adjacent := packedColumn plane
  MuddCoefficients.applyCoefficient (packet.projection.bias (14 + adjacent))
    (packet.projection.mu (row * 10 + adjacent))
    (packet.projection.delta (row * 120 + k * 12 + index))

theorem emitted_coefficients_eq (packet : RowPacket) :
    emittedCoefficients packet = packet.combinedCoefficients := by
  funext plane group
  simp only [emittedCoefficients, (group_store_exact group).1, packed_column_exact, RowPacket.combinedCoefficients]

def emittedRow (native : Native) (packet : RowPacket) : RowOutput :=
  let coefficients := emittedCoefficients packet
  ⟨coefficients, MuddNorm.run groupFromLane native ⟨packet.residual, packet.sources, coefficients⟩⟩

theorem emitted_row_eq : emittedRow = RowProgram.combined.run := by
  funext native packet
  have layout : groupFromLane = directGroup := funext fun lane => funext (group_from_lane_eq lane)
  simp only [emittedRow, RowProgram.run, emitted_coefficients_eq, layout]

def Program.run (_ : Program) : Run := batchOf emittedRow

theorem Program.run_eq (program : Program) : program.run = RowProgram.combined.batch := by
  rw [Program.run, RowProgram.batch, emitted_row_eq]

def Program.training (program : Program) : TrainingRun :=
  program.run.training .squareSum (program.run_eq.symm ▸ RowProgram.combined.consistent)

theorem Program.training_eq (program : Program) :
    program.training = RowProgram.combined.training .inverse := by
  exact (Run.training_congr program.run_eq .squareSum _ _).trans (cached_store_load _ _).symm

private theorem cached_run_eq : batchOf emittedRow = RowProgram.combined.batch :=
  congrArg batchOf emitted_row_eq

def CachedProgram.training (_ : CachedProgram) : TrainingRun :=
  (batchOf emittedRow).training .inverse (cached_run_eq.symm ▸ RowProgram.combined.consistent)

theorem CachedProgram.training_eq (program : CachedProgram) :
    program.training = RowProgram.combined.training .inverse :=
  Run.training_congr cached_run_eq .inverse _ _

#axiomsClean Program.training_eq
#axiomsClean CachedProgram.training_eq
#axiomsClean group_from_lane_eq
#axiomsClean group_store_exact
#axiomsClean emitted_coefficients_eq
#axiomsClean emitted_row_eq
#axiomsClean Program.run_eq

end VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm.Binding
