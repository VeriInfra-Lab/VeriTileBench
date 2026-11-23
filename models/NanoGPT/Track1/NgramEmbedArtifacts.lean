/-! Normalized PTX for the aligned 768-channel launches on SM90a.
The verifier compares every instruction and independently replays the
compiled loads, arithmetic, and stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedArtifacts

def referenceForward : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _ngram_embed_fwd_kernel(
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_0,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_1,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_2,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_3,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_4,
.param .u32 _ngram_embed_fwd_kernel_param_5,
.param .u32 _ngram_embed_fwd_kernel_param_6,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_7,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_8
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b16 %rs<33>;
.reg .b32 %r<138>;
.reg .b64 %rd<64>;
$L__func_begin0:
ld.param.b64 %rd53, [_ngram_embed_fwd_kernel_param_0];
ld.param.b64 %rd54, [_ngram_embed_fwd_kernel_param_1];
$L__tmp0:
mov.u32 %r61, %ctaid.x;
shl.b32 %r62, %r61, 10;
ld.param.b64 %rd55, [_ngram_embed_fwd_kernel_param_2];
ld.param.b64 %rd56, [_ngram_embed_fwd_kernel_param_3];
mov.u32 %r63, %tid.x;
shl.b32 %r64, %r63, 3;
ld.param.b64 %rd57, [_ngram_embed_fwd_kernel_param_4];
and.b32 %r65, %r64, 1016;
ld.param.b32 %r66, [_ngram_embed_fwd_kernel_param_5];
or.b32 %r67, %r65, %r62;
ld.param.b32 %r68, [_ngram_embed_fwd_kernel_param_6];
setp.lt.s32 %p1, %r67, %r68;
mul.hi.s32 %r69, %r67, 715827883;
shr.u32 %r70, %r69, 31;
shr.s32 %r71, %r69, 7;
add.s32 %r72, %r71, %r70;
mul.lo.s32 %r73, %r72, 768;
sub.s32 %r74, %r67, %r73;
mul.wide.s32 %rd58, %r72, 4;
add.s64 %rd1, %rd54, %rd58;
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd1 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd1 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r4, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd1 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r5, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd1 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r6, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd1 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r7, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd1 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r8, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd1 + 0 ], %rd9;
add.s32 %r75, %r66, %r72;
mad.wide.s32 %rd10, %r75, 4, %rd54;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r9, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd10 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r10, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd10 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r11, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd10 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r12, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd10 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r13, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd10 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r14, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd10 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r15, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd10 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r16, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd10 + 0 ], %rd18;
$L__tmp1:
add.s64 %rd19, %rd56, %rd58;
mov.u64 %rd20, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd20, 1.0;
mov.u32 %r17, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd19 + 0 ], %rd20;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r18, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd19 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r19, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd19 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r20, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd19 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r21, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd19 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r22, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd19 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r23, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd19 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r24, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd19 + 0 ], %rd27;
add.s64 %rd28, %rd19, -4;
setp.gt.s32 %p4, %r67, 767;
and.pred %p2, %p4, %p1;
mov.u64 %rd29, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd29, 1.0;
mov.u32 %r25, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd28 + 0 ], %rd29;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r26, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd28 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r27, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd28 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r28, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd28 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r29, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd28 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r30, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd28 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r31, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd28 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r32, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd28 + 0 ], %rd36;
add.s64 %rd37, %rd19, -8;
setp.gt.s32 %p5, %r67, 1535;
and.pred %p3, %p5, %p1;
mov.u64 %rd38, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd38, 1.0;
mov.u32 %r33, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd37 + 0 ], %rd38;
mov.u64 %rd39, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd39, 1.0;
mov.u32 %r34, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd37 + 0 ], %rd39;
mov.u64 %rd40, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd40, 1.0;
mov.u32 %r35, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd37 + 0 ], %rd40;
mov.u64 %rd41, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd41, 1.0;
mov.u32 %r36, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd37 + 0 ], %rd41;
mov.u64 %rd42, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd42, 1.0;
mov.u32 %r37, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r37 }, [ %rd37 + 0 ], %rd42;
mov.u64 %rd43, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd43, 1.0;
mov.u32 %r38, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r38 }, [ %rd37 + 0 ], %rd43;
mov.u64 %rd44, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd44, 1.0;
mov.u32 %r39, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r39 }, [ %rd37 + 0 ], %rd44;
mov.u64 %rd45, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd45, 1.0;
mov.u32 %r40, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r40 }, [ %rd37 + 0 ], %rd45;
mul.lo.s32 %r76, %r25, 5435;
mul.lo.s32 %r77, %r17, 7311;
xor.b32 %r78, %r76, %r77;
and.b32 %r79, %r78, 8191;
mul.lo.s32 %r80, %r33, 2225;
mul.lo.s32 %r81, %r25, 7011;
mul.lo.s32 %r82, %r17, 1355;
xor.b32 %r83, %r81, %r82;
xor.b32 %r84, %r83, %r80;
and.b32 %r85, %r84, 8191;
$L__tmp2:
mad.wide.s32 %rd59, %r1, 1536, %rd53;
mul.wide.s32 %rd60, %r74, 2;
add.s64 %rd46, %rd59, %rd60;
mov.u32 %r41, 0x0;
mov.u32 %r42, 0x0;
mov.u32 %r43, 0x0;
mov.u32 %r44, 0x0;
@%p1 ld.global.v4.b32 { %r41, %r42, %r43, %r44 }, [ %rd46 + 0 ];
mad.wide.s32 %rd61, %r9, 1536, %rd53;
add.s64 %rd47, %rd61, %rd60;
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
@%p1 ld.global.v4.b32 { %r45, %r46, %r47, %r48 }, [ %rd47 + 0 ];
mul.lo.s32 %r86, %r79, 768;
$L__tmp3:
selp.b32 %r87, %r86, 0, %p4;
$L__tmp4:
mad.wide.u32 %rd62, %r87, 2, %rd55;
add.s64 %rd48, %rd62, %rd60;
mov.u64 %rd49, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd49, 1.0;
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r49, %r50, %r51, %r52 }, [ %rd48 + 0 ], %rd49;
mul.lo.s32 %r88, %r85, 768;
$L__tmp5:
selp.b32 %r89, %r88, 0, %p5;
$L__tmp6:
mad.wide.u32 %rd63, %r89, 2, %rd55;
add.s64 %rd50, %rd63, %rd60;
mov.u64 %rd51, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd51, 1.0;
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r53, %r54, %r55, %r56 }, [ %rd50 + 0 ], %rd51;
mad.wide.s32 %rd52, %r67, 2, %rd57;
mov.b32 {%rs1, %rs2}, %r41;
cvt.f32.bf16 %r90, %rs2;
cvt.f32.bf16 %r91, %rs1;
mov.b32 {%rs3, %rs4}, %r45;
cvt.f32.bf16 %r92, %rs3;
cvt.f32.bf16 %r93, %rs4;
mov.b32 {%rs5, %rs6}, %r49;
cvt.f32.bf16 %r94, %rs6;
cvt.f32.bf16 %r95, %rs5;
mov.b32 {%rs7, %rs8}, %r53;
cvt.f32.bf16 %r96, %rs7;
cvt.f32.bf16 %r97, %rs8;
mul.f32 %r98, %r93, %r97;
mul.f32 %r99, %r92, %r96;
fma.rn.f32 %r100, %r91, %r95, %r99;
fma.rn.f32 %r101, %r90, %r94, %r98;
cvt.rn.bf16x2.f32 %r57, %r101, %r100;
mov.b32 {%rs9, %rs10}, %r42;
cvt.f32.bf16 %r102, %rs10;
cvt.f32.bf16 %r103, %rs9;
mov.b32 {%rs11, %rs12}, %r46;
cvt.f32.bf16 %r104, %rs11;
cvt.f32.bf16 %r105, %rs12;
mov.b32 {%rs13, %rs14}, %r50;
cvt.f32.bf16 %r106, %rs14;
cvt.f32.bf16 %r107, %rs13;
mov.b32 {%rs15, %rs16}, %r54;
cvt.f32.bf16 %r108, %rs15;
cvt.f32.bf16 %r109, %rs16;
mul.f32 %r110, %r105, %r109;
mul.f32 %r111, %r104, %r108;
fma.rn.f32 %r112, %r103, %r107, %r111;
fma.rn.f32 %r113, %r102, %r106, %r110;
cvt.rn.bf16x2.f32 %r58, %r113, %r112;
mov.b32 {%rs17, %rs18}, %r43;
cvt.f32.bf16 %r114, %rs18;
cvt.f32.bf16 %r115, %rs17;
mov.b32 {%rs19, %rs20}, %r47;
cvt.f32.bf16 %r116, %rs19;
cvt.f32.bf16 %r117, %rs20;
mov.b32 {%rs21, %rs22}, %r51;
cvt.f32.bf16 %r118, %rs22;
cvt.f32.bf16 %r119, %rs21;
mov.b32 {%rs23, %rs24}, %r55;
cvt.f32.bf16 %r120, %rs23;
cvt.f32.bf16 %r121, %rs24;
mul.f32 %r122, %r117, %r121;
mul.f32 %r123, %r116, %r120;
fma.rn.f32 %r124, %r115, %r119, %r123;
fma.rn.f32 %r125, %r114, %r118, %r122;
cvt.rn.bf16x2.f32 %r59, %r125, %r124;
mov.b32 {%rs25, %rs26}, %r44;
cvt.f32.bf16 %r126, %rs26;
cvt.f32.bf16 %r127, %rs25;
mov.b32 {%rs27, %rs28}, %r48;
cvt.f32.bf16 %r128, %rs27;
cvt.f32.bf16 %r129, %rs28;
mov.b32 {%rs29, %rs30}, %r52;
cvt.f32.bf16 %r130, %rs30;
cvt.f32.bf16 %r131, %rs29;
mov.b32 {%rs31, %rs32}, %r56;
cvt.f32.bf16 %r132, %rs31;
cvt.f32.bf16 %r133, %rs32;
mul.f32 %r134, %r129, %r133;
mul.f32 %r135, %r128, %r132;
fma.rn.f32 %r136, %r127, %r131, %r135;
fma.rn.f32 %r137, %r126, %r130, %r134;
cvt.rn.bf16x2.f32 %r60, %r137, %r136;
@%p1 st.global.v4.b32 [ %rd52 + 0 ], { %r57, %r58, %r59, %r60 };
ret;
$L__tmp7:
$L__func_end0:
}
"#

def referenceBackward : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _ngram_embed_bwd_kernel(
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_0,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_1,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_2,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_3,
.param .u32 _ngram_embed_bwd_kernel_param_4,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_5,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_6
)
.reqntid 128
{
.reg .pred %p<6>;
.reg .b32 %r<73>;
.reg .b64 %rd<43>;
$L__func_begin0:
ld.param.b64 %rd35, [_ngram_embed_bwd_kernel_param_0];
ld.param.b64 %rd36, [_ngram_embed_bwd_kernel_param_1];
$L__tmp0:
mov.u32 %r45, %ctaid.x;
shl.b32 %r46, %r45, 10;
ld.param.b64 %rd37, [_ngram_embed_bwd_kernel_param_2];
ld.param.b64 %rd38, [_ngram_embed_bwd_kernel_param_3];
mov.u32 %r47, %tid.x;
shl.b32 %r48, %r47, 3;
ld.param.b32 %r49, [_ngram_embed_bwd_kernel_param_4];
and.b32 %r50, %r48, 1016;
or.b32 %r51, %r50, %r46;
setp.lt.s32 %p1, %r51, %r49;
mul.hi.s32 %r52, %r51, 715827883;
shr.u32 %r53, %r52, 31;
shr.s32 %r54, %r52, 7;
add.s32 %r55, %r54, %r53;
mul.lo.s32 %r56, %r55, 768;
sub.s32 %r57, %r51, %r56;
$L__tmp1:
mad.wide.s32 %rd1, %r55, 4, %rd37;
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd1 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd1 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r4, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd1 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r5, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd1 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r6, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd1 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r7, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd1 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r8, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd1 + 0 ], %rd9;
add.s64 %rd10, %rd1, -4;
setp.gt.s32 %p4, %r51, 767;
and.pred %p2, %p4, %p1;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r9, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd10 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r10, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd10 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r11, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd10 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r12, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd10 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r13, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd10 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r14, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd10 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r15, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd10 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r16, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd10 + 0 ], %rd18;
add.s64 %rd19, %rd1, -8;
setp.gt.s32 %p5, %r51, 1535;
and.pred %p3, %p5, %p1;
mov.u64 %rd20, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd20, 1.0;
mov.u32 %r17, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd19 + 0 ], %rd20;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r18, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd19 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r19, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd19 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r20, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd19 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r21, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd19 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r22, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd19 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r23, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd19 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r24, 0x0;
@%p3 ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd19 + 0 ], %rd27;
mul.lo.s32 %r58, %r9, 5435;
mul.lo.s32 %r59, %r1, 7311;
xor.b32 %r60, %r58, %r59;
and.b32 %r61, %r60, 8191;
mul.lo.s32 %r62, %r17, 2225;
mul.lo.s32 %r63, %r9, 7011;
mul.lo.s32 %r64, %r1, 1355;
xor.b32 %r65, %r63, %r64;
xor.b32 %r66, %r65, %r62;
and.b32 %r67, %r66, 8191;
$L__tmp2:
mul.wide.s32 %rd39, %r51, 2;
add.s64 %rd28, %rd35, %rd39;
mov.u32 %r25, 0x0;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
@%p1 ld.global.v4.b32 { %r25, %r26, %r27, %r28 }, [ %rd28 + 0 ];
mul.lo.s32 %r68, %r61, 768;
$L__tmp3:
selp.b32 %r69, %r68, 0, %p4;
$L__tmp4:
mad.wide.u32 %rd40, %r69, 2, %rd36;
mul.wide.s32 %rd41, %r57, 2;
add.s64 %rd29, %rd40, %rd41;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r29, 0x0;
mov.u32 %r30, 0x0;
mov.u32 %r31, 0x0;
mov.u32 %r32, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r29, %r30, %r31, %r32 }, [ %rd29 + 0 ], %rd30;
mul.lo.s32 %r70, %r67, 768;
$L__tmp5:
selp.b32 %r71, %r70, 0, %p5;
$L__tmp6:
mad.wide.u32 %rd42, %r71, 2, %rd36;
add.s64 %rd31, %rd42, %rd41;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r33, 0x0;
mov.u32 %r34, 0x0;
mov.u32 %r35, 0x0;
mov.u32 %r36, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r33, %r34, %r35, %r36 }, [ %rd31 + 0 ], %rd32;
add.s64 %rd33, %rd38, %rd39;
mul.bf16x2 %r37, %r25, %r29;
mul.bf16x2 %r38, %r26, %r30;
mul.bf16x2 %r39, %r27, %r31;
mul.bf16x2 %r40, %r28, %r32;
@%p1 st.global.v4.b32 [ %rd33 + 0 ], { %r37, %r38, %r39, %r40 };
add.s32 %r72, %r49, %r51;
mad.wide.s32 %rd34, %r72, 2, %rd38;
mul.bf16x2 %r41, %r25, %r33;
mul.bf16x2 %r42, %r26, %r34;
mul.bf16x2 %r43, %r27, %r35;
mul.bf16x2 %r44, %r28, %r36;
@%p1 st.global.v4.b32 [ %rd34 + 0 ], { %r41, %r42, %r43, %r44 };
ret;
$L__tmp7:
$L__func_end0:
}
"#

def fullForward : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _ngram_embed_fwd_kernel(
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_0,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_1,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_2,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_3,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_4,
.param .u32 _ngram_embed_fwd_kernel_param_5,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_6,
.param .u64 .ptr .global .align 1 _ngram_embed_fwd_kernel_param_7
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b16 %rs<97>;
.reg .b32 %r<239>;
.reg .b64 %rd<43>;
$L__func_begin0:
ld.param.b64 %rd32, [_ngram_embed_fwd_kernel_param_0];
ld.param.b64 %rd33, [_ngram_embed_fwd_kernel_param_1];
$L__tmp0:
mov.u32 %r66, %ctaid.x;
shl.b32 %r67, %r66, 2;
ld.param.b64 %rd34, [_ngram_embed_fwd_kernel_param_2];
ld.param.b64 %rd35, [_ngram_embed_fwd_kernel_param_3];
mov.u32 %r68, %tid.x;
ld.param.b64 %rd36, [_ngram_embed_fwd_kernel_param_4];
bfe.u32 %r69, %r68, 5, 2;
ld.param.b32 %r70, [_ngram_embed_fwd_kernel_param_5];
or.b32 %r71, %r69, %r67;
mul.wide.s32 %rd37, %r71, 4;
add.s64 %rd1, %rd33, %rd37;
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
add.s32 %r72, %r70, %r71;
mad.wide.s32 %rd3, %r72, 4, %rd33;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd3 + 0 ], %rd4;
$L__tmp1:
add.s64 %rd5, %rd35, %rd37;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd5 + 0 ], %rd6;
add.s32 %r73, %r71, -1;
mad.wide.s32 %rd7, %r73, 4, %rd35;
setp.gt.s32 %p1, %r71, 0;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r4, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd7 + 0 ], %rd8;
add.s32 %r74, %r71, -2;
mad.wide.s32 %rd9, %r74, 4, %rd35;
setp.gt.s32 %p2, %r71, 1;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r5, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd9 + 0 ], %rd10;
mul.lo.s32 %r75, %r4, 5435;
mul.lo.s32 %r76, %r3, 7311;
xor.b32 %r77, %r75, %r76;
and.b32 %r78, %r77, 8191;
mul.lo.s32 %r79, %r5, 2225;
mul.lo.s32 %r80, %r4, 7011;
mul.lo.s32 %r81, %r3, 1355;
xor.b32 %r82, %r80, %r81;
xor.b32 %r83, %r82, %r79;
and.b32 %r84, %r83, 8191;
$L__tmp2:
shl.b32 %r85, %r68, 3;
and.b32 %r86, %r85, 248;
mad.wide.s32 %rd38, %r1, 1536, %rd32;
mul.wide.u32 %rd39, %r86, 2;
add.s64 %rd11, %rd38, %rd39;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd11 + 0 ];
mad.wide.s32 %rd40, %r2, 1536, %rd32;
add.s64 %rd12, %rd40, %rd39;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd12 + 0 ];
mul.lo.s32 %r87, %r78, 768;
$L__tmp3:
selp.b32 %r88, %r87, 0, %p1;
$L__tmp4:
mad.wide.u32 %rd41, %r88, 2, %rd34;
add.s64 %rd13, %rd41, %rd39;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd13 + 0 ], %rd14;
mul.lo.s32 %r89, %r84, 768;
$L__tmp5:
selp.b32 %r90, %r89, 0, %p2;
$L__tmp6:
mad.wide.u32 %rd42, %r90, 2, %rd34;
add.s64 %rd15, %rd42, %rd39;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd15 + 0 ], %rd16;
mul.lo.s32 %r91, %r71, 768;
or.b32 %r92, %r91, %r86;
mad.wide.s32 %rd17, %r92, 2, %rd36;
mov.b32 {%rs1, %rs2}, %r6;
cvt.f32.bf16 %r93, %rs2;
cvt.f32.bf16 %r94, %rs1;
mov.b32 {%rs3, %rs4}, %r10;
cvt.f32.bf16 %r95, %rs3;
cvt.f32.bf16 %r96, %rs4;
mov.b32 {%rs5, %rs6}, %r14;
cvt.f32.bf16 %r97, %rs6;
cvt.f32.bf16 %r98, %rs5;
mov.b32 {%rs7, %rs8}, %r18;
cvt.f32.bf16 %r99, %rs7;
cvt.f32.bf16 %r100, %rs8;
mul.f32 %r101, %r96, %r100;
mul.f32 %r102, %r95, %r99;
fma.rn.f32 %r103, %r94, %r98, %r102;
fma.rn.f32 %r104, %r93, %r97, %r101;
cvt.rn.bf16x2.f32 %r22, %r104, %r103;
mov.b32 {%rs9, %rs10}, %r7;
cvt.f32.bf16 %r105, %rs10;
cvt.f32.bf16 %r106, %rs9;
mov.b32 {%rs11, %rs12}, %r11;
cvt.f32.bf16 %r107, %rs11;
cvt.f32.bf16 %r108, %rs12;
mov.b32 {%rs13, %rs14}, %r15;
cvt.f32.bf16 %r109, %rs14;
cvt.f32.bf16 %r110, %rs13;
mov.b32 {%rs15, %rs16}, %r19;
cvt.f32.bf16 %r111, %rs15;
cvt.f32.bf16 %r112, %rs16;
mul.f32 %r113, %r108, %r112;
mul.f32 %r114, %r107, %r111;
fma.rn.f32 %r115, %r106, %r110, %r114;
fma.rn.f32 %r116, %r105, %r109, %r113;
cvt.rn.bf16x2.f32 %r23, %r116, %r115;
mov.b32 {%rs17, %rs18}, %r8;
cvt.f32.bf16 %r117, %rs18;
cvt.f32.bf16 %r118, %rs17;
mov.b32 {%rs19, %rs20}, %r12;
cvt.f32.bf16 %r119, %rs19;
cvt.f32.bf16 %r120, %rs20;
mov.b32 {%rs21, %rs22}, %r16;
cvt.f32.bf16 %r121, %rs22;
cvt.f32.bf16 %r122, %rs21;
mov.b32 {%rs23, %rs24}, %r20;
cvt.f32.bf16 %r123, %rs23;
cvt.f32.bf16 %r124, %rs24;
mul.f32 %r125, %r120, %r124;
mul.f32 %r126, %r119, %r123;
fma.rn.f32 %r127, %r118, %r122, %r126;
fma.rn.f32 %r128, %r117, %r121, %r125;
cvt.rn.bf16x2.f32 %r24, %r128, %r127;
mov.b32 {%rs25, %rs26}, %r9;
cvt.f32.bf16 %r129, %rs26;
cvt.f32.bf16 %r130, %rs25;
mov.b32 {%rs27, %rs28}, %r13;
cvt.f32.bf16 %r131, %rs27;
cvt.f32.bf16 %r132, %rs28;
mov.b32 {%rs29, %rs30}, %r17;
cvt.f32.bf16 %r133, %rs30;
cvt.f32.bf16 %r134, %rs29;
mov.b32 {%rs31, %rs32}, %r21;
cvt.f32.bf16 %r135, %rs31;
cvt.f32.bf16 %r136, %rs32;
mul.f32 %r137, %r132, %r136;
mul.f32 %r138, %r131, %r135;
fma.rn.f32 %r139, %r130, %r134, %r138;
fma.rn.f32 %r140, %r129, %r133, %r137;
cvt.rn.bf16x2.f32 %r25, %r140, %r139;
st.global.v4.b32 [ %rd17 + 0 ], { %r22, %r23, %r24, %r25 };
add.s64 %rd18, %rd11, 512;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd18 + 0 ];
add.s64 %rd19, %rd12, 512;
mov.u32 %r30, 0x0;
mov.u32 %r31, 0x0;
mov.u32 %r32, 0x0;
mov.u32 %r33, 0x0;
ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd19 + 0 ];
add.s64 %rd20, %rd13, 512;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r34, 0x0;
mov.u32 %r35, 0x0;
mov.u32 %r36, 0x0;
mov.u32 %r37, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd20 + 0 ], %rd21;
add.s64 %rd22, %rd15, 512;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r38, 0x0;
mov.u32 %r39, 0x0;
mov.u32 %r40, 0x0;
mov.u32 %r41, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd22 + 0 ], %rd23;
add.s32 %r141, %r92, 256;
mad.wide.s32 %rd24, %r141, 2, %rd36;
mov.b32 {%rs33, %rs34}, %r26;
cvt.f32.bf16 %r142, %rs34;
cvt.f32.bf16 %r143, %rs33;
mov.b32 {%rs35, %rs36}, %r30;
cvt.f32.bf16 %r144, %rs35;
cvt.f32.bf16 %r145, %rs36;
mov.b32 {%rs37, %rs38}, %r34;
cvt.f32.bf16 %r146, %rs38;
cvt.f32.bf16 %r147, %rs37;
mov.b32 {%rs39, %rs40}, %r38;
cvt.f32.bf16 %r148, %rs39;
cvt.f32.bf16 %r149, %rs40;
mul.f32 %r150, %r145, %r149;
mul.f32 %r151, %r144, %r148;
fma.rn.f32 %r152, %r143, %r147, %r151;
fma.rn.f32 %r153, %r142, %r146, %r150;
cvt.rn.bf16x2.f32 %r42, %r153, %r152;
mov.b32 {%rs41, %rs42}, %r27;
cvt.f32.bf16 %r154, %rs42;
cvt.f32.bf16 %r155, %rs41;
mov.b32 {%rs43, %rs44}, %r31;
cvt.f32.bf16 %r156, %rs43;
cvt.f32.bf16 %r157, %rs44;
mov.b32 {%rs45, %rs46}, %r35;
cvt.f32.bf16 %r158, %rs46;
cvt.f32.bf16 %r159, %rs45;
mov.b32 {%rs47, %rs48}, %r39;
cvt.f32.bf16 %r160, %rs47;
cvt.f32.bf16 %r161, %rs48;
mul.f32 %r162, %r157, %r161;
mul.f32 %r163, %r156, %r160;
fma.rn.f32 %r164, %r155, %r159, %r163;
fma.rn.f32 %r165, %r154, %r158, %r162;
cvt.rn.bf16x2.f32 %r43, %r165, %r164;
mov.b32 {%rs49, %rs50}, %r28;
cvt.f32.bf16 %r166, %rs50;
cvt.f32.bf16 %r167, %rs49;
mov.b32 {%rs51, %rs52}, %r32;
cvt.f32.bf16 %r168, %rs51;
cvt.f32.bf16 %r169, %rs52;
mov.b32 {%rs53, %rs54}, %r36;
cvt.f32.bf16 %r170, %rs54;
cvt.f32.bf16 %r171, %rs53;
mov.b32 {%rs55, %rs56}, %r40;
cvt.f32.bf16 %r172, %rs55;
cvt.f32.bf16 %r173, %rs56;
mul.f32 %r174, %r169, %r173;
mul.f32 %r175, %r168, %r172;
fma.rn.f32 %r176, %r167, %r171, %r175;
fma.rn.f32 %r177, %r166, %r170, %r174;
cvt.rn.bf16x2.f32 %r44, %r177, %r176;
mov.b32 {%rs57, %rs58}, %r29;
cvt.f32.bf16 %r178, %rs58;
cvt.f32.bf16 %r179, %rs57;
mov.b32 {%rs59, %rs60}, %r33;
cvt.f32.bf16 %r180, %rs59;
cvt.f32.bf16 %r181, %rs60;
mov.b32 {%rs61, %rs62}, %r37;
cvt.f32.bf16 %r182, %rs62;
cvt.f32.bf16 %r183, %rs61;
mov.b32 {%rs63, %rs64}, %r41;
cvt.f32.bf16 %r184, %rs63;
cvt.f32.bf16 %r185, %rs64;
mul.f32 %r186, %r181, %r185;
mul.f32 %r187, %r180, %r184;
fma.rn.f32 %r188, %r179, %r183, %r187;
fma.rn.f32 %r189, %r178, %r182, %r186;
cvt.rn.bf16x2.f32 %r45, %r189, %r188;
st.global.v4.b32 [ %rd24 + 0 ], { %r42, %r43, %r44, %r45 };
add.s64 %rd25, %rd11, 1024;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
mov.u32 %r49, 0x0;
ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd25 + 0 ];
add.s64 %rd26, %rd12, 1024;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
mov.u32 %r53, 0x0;
ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd26 + 0 ];
add.s64 %rd27, %rd13, 1024;
mov.u64 %rd28, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd28, 1.0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
mov.u32 %r57, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd27 + 0 ], %rd28;
add.s64 %rd29, %rd15, 1024;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r58, 0x0;
mov.u32 %r59, 0x0;
mov.u32 %r60, 0x0;
mov.u32 %r61, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r58, %r59, %r60, %r61 }, [ %rd29 + 0 ], %rd30;
add.s32 %r190, %r92, 512;
mad.wide.s32 %rd31, %r190, 2, %rd36;
mov.b32 {%rs65, %rs66}, %r46;
cvt.f32.bf16 %r191, %rs66;
cvt.f32.bf16 %r192, %rs65;
mov.b32 {%rs67, %rs68}, %r50;
cvt.f32.bf16 %r193, %rs67;
cvt.f32.bf16 %r194, %rs68;
mov.b32 {%rs69, %rs70}, %r54;
cvt.f32.bf16 %r195, %rs70;
cvt.f32.bf16 %r196, %rs69;
mov.b32 {%rs71, %rs72}, %r58;
cvt.f32.bf16 %r197, %rs71;
cvt.f32.bf16 %r198, %rs72;
mul.f32 %r199, %r194, %r198;
mul.f32 %r200, %r193, %r197;
fma.rn.f32 %r201, %r192, %r196, %r200;
fma.rn.f32 %r202, %r191, %r195, %r199;
cvt.rn.bf16x2.f32 %r62, %r202, %r201;
mov.b32 {%rs73, %rs74}, %r47;
cvt.f32.bf16 %r203, %rs74;
cvt.f32.bf16 %r204, %rs73;
mov.b32 {%rs75, %rs76}, %r51;
cvt.f32.bf16 %r205, %rs75;
cvt.f32.bf16 %r206, %rs76;
mov.b32 {%rs77, %rs78}, %r55;
cvt.f32.bf16 %r207, %rs78;
cvt.f32.bf16 %r208, %rs77;
mov.b32 {%rs79, %rs80}, %r59;
cvt.f32.bf16 %r209, %rs79;
cvt.f32.bf16 %r210, %rs80;
mul.f32 %r211, %r206, %r210;
mul.f32 %r212, %r205, %r209;
fma.rn.f32 %r213, %r204, %r208, %r212;
fma.rn.f32 %r214, %r203, %r207, %r211;
cvt.rn.bf16x2.f32 %r63, %r214, %r213;
mov.b32 {%rs81, %rs82}, %r48;
cvt.f32.bf16 %r215, %rs82;
cvt.f32.bf16 %r216, %rs81;
mov.b32 {%rs83, %rs84}, %r52;
cvt.f32.bf16 %r217, %rs83;
cvt.f32.bf16 %r218, %rs84;
mov.b32 {%rs85, %rs86}, %r56;
cvt.f32.bf16 %r219, %rs86;
cvt.f32.bf16 %r220, %rs85;
mov.b32 {%rs87, %rs88}, %r60;
cvt.f32.bf16 %r221, %rs87;
cvt.f32.bf16 %r222, %rs88;
mul.f32 %r223, %r218, %r222;
mul.f32 %r224, %r217, %r221;
fma.rn.f32 %r225, %r216, %r220, %r224;
fma.rn.f32 %r226, %r215, %r219, %r223;
cvt.rn.bf16x2.f32 %r64, %r226, %r225;
mov.b32 {%rs89, %rs90}, %r49;
cvt.f32.bf16 %r227, %rs90;
cvt.f32.bf16 %r228, %rs89;
mov.b32 {%rs91, %rs92}, %r53;
cvt.f32.bf16 %r229, %rs91;
cvt.f32.bf16 %r230, %rs92;
mov.b32 {%rs93, %rs94}, %r57;
cvt.f32.bf16 %r231, %rs94;
cvt.f32.bf16 %r232, %rs93;
mov.b32 {%rs95, %rs96}, %r61;
cvt.f32.bf16 %r233, %rs95;
cvt.f32.bf16 %r234, %rs96;
mul.f32 %r235, %r230, %r234;
mul.f32 %r236, %r229, %r233;
fma.rn.f32 %r237, %r228, %r232, %r236;
fma.rn.f32 %r238, %r227, %r231, %r235;
cvt.rn.bf16x2.f32 %r65, %r238, %r237;
st.global.v4.b32 [ %rd31 + 0 ], { %r62, %r63, %r64, %r65 };
ret;
$L__tmp7:
$L__func_end0:
}
"#

def fullBackward : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _ngram_embed_bwd_kernel(
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_0,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_1,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_2,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_3,
.param .u32 _ngram_embed_bwd_kernel_param_4,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_5,
.param .u64 .ptr .global .align 1 _ngram_embed_bwd_kernel_param_6
)
.reqntid 128
{
.reg .pred %p<3>;
.reg .b32 %r<93>;
.reg .b64 %rd<39>;
$L__func_begin0:
ld.param.b64 %rd28, [_ngram_embed_bwd_kernel_param_0];
ld.param.b64 %rd29, [_ngram_embed_bwd_kernel_param_1];
$L__tmp0:
mov.u32 %r64, %ctaid.x;
shl.b32 %r65, %r64, 2;
ld.param.b64 %rd30, [_ngram_embed_bwd_kernel_param_2];
ld.param.b64 %rd31, [_ngram_embed_bwd_kernel_param_3];
mov.u32 %r66, %tid.x;
ld.param.b32 %r67, [_ngram_embed_bwd_kernel_param_4];
bfe.u32 %r68, %r66, 5, 2;
or.b32 %r69, %r68, %r65;
$L__tmp1:
mad.wide.s32 %rd1, %r69, 4, %rd30;
mov.u64 %rd2, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd2, 1.0;
mov.u32 %r1, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r1 }, [ %rd1 + 0 ], %rd2;
add.s32 %r70, %r69, -1;
mad.wide.s32 %rd3, %r70, 4, %rd30;
setp.gt.s32 %p1, %r69, 0;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r2, 0x0;
@%p1 ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd3 + 0 ], %rd4;
add.s32 %r71, %r69, -2;
mad.wide.s32 %rd5, %r71, 4, %rd30;
setp.gt.s32 %p2, %r69, 1;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r3, 0x0;
@%p2 ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd5 + 0 ], %rd6;
mul.lo.s32 %r72, %r2, 5435;
mul.lo.s32 %r73, %r1, 7311;
xor.b32 %r74, %r72, %r73;
and.b32 %r75, %r74, 8191;
mul.lo.s32 %r76, %r3, 2225;
mul.lo.s32 %r77, %r2, 7011;
mul.lo.s32 %r78, %r1, 1355;
xor.b32 %r79, %r77, %r78;
xor.b32 %r80, %r79, %r76;
and.b32 %r81, %r80, 8191;
$L__tmp2:
shl.b32 %r82, %r66, 3;
and.b32 %r83, %r82, 248;
mul.lo.s32 %r84, %r69, 768;
or.b32 %r85, %r84, %r83;
mul.wide.s32 %rd32, %r85, 2;
add.s64 %rd7, %rd28, %rd32;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
ld.global.v4.b32 { %r4, %r5, %r6, %r7 }, [ %rd7 + 0 ];
mul.lo.s32 %r86, %r75, 768;
$L__tmp3:
selp.b32 %r87, %r86, 0, %p1;
$L__tmp4:
mad.wide.u32 %rd33, %r87, 2, %rd29;
mul.wide.u32 %rd34, %r83, 2;
add.s64 %rd8, %rd33, %rd34;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd8 + 0 ], %rd9;
mul.lo.s32 %r88, %r81, 768;
$L__tmp5:
selp.b32 %r89, %r88, 0, %p2;
$L__tmp6:
mad.wide.u32 %rd35, %r89, 2, %rd29;
add.s64 %rd10, %rd35, %rd34;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd10 + 0 ], %rd11;
add.s64 %rd12, %rd31, %rd32;
mul.bf16x2 %r16, %r4, %r8;
mul.bf16x2 %r17, %r5, %r9;
mul.bf16x2 %r18, %r6, %r10;
mul.bf16x2 %r19, %r7, %r11;
st.global.v4.b32 [ %rd12 + 0 ], { %r16, %r17, %r18, %r19 };
mul.lo.s32 %r90, %r67, 768;
mad.wide.s32 %rd36, %r90, 2, %rd31;
add.s64 %rd13, %rd36, %rd32;
mul.bf16x2 %r20, %r4, %r12;
mul.bf16x2 %r21, %r5, %r13;
mul.bf16x2 %r22, %r6, %r14;
mul.bf16x2 %r23, %r7, %r15;
st.global.v4.b32 [ %rd13 + 0 ], { %r20, %r21, %r22, %r23 };
add.s32 %r91, %r85, 256;
mul.wide.s32 %rd37, %r91, 2;
add.s64 %rd14, %rd28, %rd37;
mov.u32 %r24, 0x0;
mov.u32 %r25, 0x0;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
ld.global.v4.b32 { %r24, %r25, %r26, %r27 }, [ %rd14 + 0 ];
add.s64 %rd15, %rd8, 512;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
mov.u32 %r30, 0x0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r28, %r29, %r30, %r31 }, [ %rd15 + 0 ], %rd16;
add.s64 %rd17, %rd10, 512;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r32, 0x0;
mov.u32 %r33, 0x0;
mov.u32 %r34, 0x0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r32, %r33, %r34, %r35 }, [ %rd17 + 0 ], %rd18;
add.s64 %rd19, %rd31, %rd37;
mul.bf16x2 %r36, %r24, %r28;
mul.bf16x2 %r37, %r25, %r29;
mul.bf16x2 %r38, %r26, %r30;
mul.bf16x2 %r39, %r27, %r31;
st.global.v4.b32 [ %rd19 + 0 ], { %r36, %r37, %r38, %r39 };
add.s64 %rd20, %rd36, %rd37;
mul.bf16x2 %r40, %r24, %r32;
mul.bf16x2 %r41, %r25, %r33;
mul.bf16x2 %r42, %r26, %r34;
mul.bf16x2 %r43, %r27, %r35;
st.global.v4.b32 [ %rd20 + 0 ], { %r40, %r41, %r42, %r43 };
add.s32 %r92, %r85, 512;
mul.wide.s32 %rd38, %r92, 2;
add.s64 %rd21, %rd28, %rd38;
mov.u32 %r44, 0x0;
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
ld.global.v4.b32 { %r44, %r45, %r46, %r47 }, [ %rd21 + 0 ];
add.s64 %rd22, %rd8, 1024;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r48, 0x0;
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r48, %r49, %r50, %r51 }, [ %rd22 + 0 ], %rd23;
add.s64 %rd24, %rd10, 1024;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r52, 0x0;
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r52, %r53, %r54, %r55 }, [ %rd24 + 0 ], %rd25;
add.s64 %rd26, %rd31, %rd38;
mul.bf16x2 %r56, %r44, %r48;
mul.bf16x2 %r57, %r45, %r49;
mul.bf16x2 %r58, %r46, %r50;
mul.bf16x2 %r59, %r47, %r51;
st.global.v4.b32 [ %rd26 + 0 ], { %r56, %r57, %r58, %r59 };
add.s64 %rd27, %rd36, %rd38;
mul.bf16x2 %r60, %r44, %r52;
mul.bf16x2 %r61, %r45, %r53;
mul.bf16x2 %r62, %r46, %r54;
mul.bf16x2 %r63, %r47, %r55;
st.global.v4.b32 [ %rd27 + 0 ], { %r60, %r61, %r62, %r63 };
ret;
$L__tmp7:
$L__func_end0:
}
"#

end VeriTile.Bench.Models.NanoGPT.Track1.NgramEmbedArtifacts
