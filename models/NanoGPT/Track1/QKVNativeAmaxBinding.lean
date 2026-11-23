import models.NanoGPT.Track1.QKVNative
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact TTIR graph matching for the native QK and V address calculation,
including the signed division by the number of amax tiles in a cropped head. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVNative
open QKVLaunch QKVAmax
open QKVTTIR (Bank)
open VeriTile.Triton.FP

private def nativeGraph (w : Width) : String :=
  r#"
module {
tt.func public @_aligned_packed_amax(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %partial_amax: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<2048xf32>
%batch = tt.get_program_id x : i32
%part = tt.get_program_id y : i32
%lane = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%first_count = arith.subi 319, %part : i32
"#
  | .wide => r#"
%first_count = arith.subi 607, %part : i32
"#) ++
  r#"
%first_count_0 = arith.divsi %first_count, 32 : i32
"# ++
  (match w with
  | .narrow => r#"
%total_count = arith.subi 607, %part : i32
"#
  | .halfValue => r#"
%total_count = arith.subi 463, %part : i32
"#
  | .wide => r#"
%total_count = arith.subi 895, %part : i32
"#) ++
  r#"
%total_count_1 = arith.divsi %total_count, 32 : i32
%local_amax = scf.for %i = 0 to %first_count_0 step 1 iter_args(%local_amax_3 = %cst) -> (tensor<2048xf32>) : i32 {
%tile = arith.muli %i, 32 : i32
%tile_4 = arith.addi %part, %tile : i32
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%tile_5 = arith.divsi %tile_4, 24 : i32
%tile_6 = arith.muli %tile_5, 24 : i32
%tile_7 = arith.addi %tile_4, %tile_6 : i32
"#
  | .wide => "") ++
  r#"
%values = arith.muli %batch, 1179648 : i32
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%values_8 = tt.addptr %first, %values : !tt.ptr<bf16>, i32
%values_9 = arith.muli %tile_7, 2048 : i32
%values_10 = tt.addptr %values_8, %values_9 : !tt.ptr<bf16>, i32
%values_11 = tt.splat %values_10 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_12 = tt.addptr %values_11, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_13 = tt.load %values_12 : tensor<2048x!tt.ptr<bf16>>
%values_14 = arith.extf %values_13 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_15 = arith.addf %values_14, %cst : tensor<2048xf32>
%local_amax_16 = math.absf %local_amax_15 : tensor<2048xf32>
%local_amax_17 = arith.maxnumf %local_amax_3, %local_amax_16 : tensor<2048xf32>
scf.yield %local_amax_17 : tensor<2048xf32>
"#
  | .wide => r#"
%values_5 = tt.addptr %first, %values : !tt.ptr<bf16>, i32
%values_6 = arith.muli %tile_4, 2048 : i32
%values_7 = tt.addptr %values_5, %values_6 : !tt.ptr<bf16>, i32
%values_8 = tt.splat %values_7 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_9 = tt.addptr %values_8, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_10 = tt.load %values_9 : tensor<2048x!tt.ptr<bf16>>
%values_11 = arith.extf %values_10 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_12 = arith.addf %values_11, %cst : tensor<2048xf32>
%local_amax_13 = math.absf %local_amax_12 : tensor<2048xf32>
%local_amax_14 = arith.maxnumf %local_amax_3, %local_amax_13 : tensor<2048xf32>
scf.yield %local_amax_14 : tensor<2048xf32>
"#) ++
  r#"
}
%local_amax_2 = scf.for %i = %first_count_0 to %total_count_1 step 1 iter_args(%local_amax_3 = %local_amax) -> (tensor<2048xf32>) : i32 {
%tile = arith.muli %i, 32 : i32
%tile_4 = arith.addi %part, %tile : i32
"# ++
  (match w with
  | .narrow => r#"
%tile_5 = arith.subi %tile_4, 288 : i32
"#
  | .halfValue => r#"
%tile_5 = arith.subi %tile_4, 288 : i32
%tile_6 = arith.divsi %tile_5, 24 : i32
%tile_7 = arith.muli %tile_6, 24 : i32
%tile_8 = arith.addi %tile_5, %tile_7 : i32
"#
  | .wide => r#"
%tile_5 = arith.subi %tile_4, 576 : i32
"#) ++
  r#"
%values = arith.muli %batch, 1179648 : i32
"# ++
  (match w with
  | .narrow => r#"
%values_6 = tt.addptr %second, %values : !tt.ptr<bf16>, i32
%values_7 = arith.muli %tile_5, 2048 : i32
%values_8 = tt.addptr %values_6, %values_7 : !tt.ptr<bf16>, i32
%values_9 = tt.splat %values_8 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_10 = tt.addptr %values_9, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_11 = tt.load %values_10 : tensor<2048x!tt.ptr<bf16>>
%values_12 = arith.extf %values_11 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_13 = arith.addf %values_12, %cst : tensor<2048xf32>
%local_amax_14 = math.absf %local_amax_13 : tensor<2048xf32>
%local_amax_15 = arith.maxnumf %local_amax_3, %local_amax_14 : tensor<2048xf32>
scf.yield %local_amax_15 : tensor<2048xf32>
"#
  | .halfValue => r#"
%values_9 = tt.addptr %second, %values : !tt.ptr<bf16>, i32
%values_10 = arith.muli %tile_8, 2048 : i32
%values_11 = tt.addptr %values_9, %values_10 : !tt.ptr<bf16>, i32
%values_12 = tt.splat %values_11 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_13 = tt.addptr %values_12, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_14 = tt.load %values_13 : tensor<2048x!tt.ptr<bf16>>
%values_15 = arith.extf %values_14 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_16 = arith.addf %values_15, %cst : tensor<2048xf32>
%local_amax_17 = math.absf %local_amax_16 : tensor<2048xf32>
%local_amax_18 = arith.maxnumf %local_amax_3, %local_amax_17 : tensor<2048xf32>
scf.yield %local_amax_18 : tensor<2048xf32>
"#
  | .wide => r#"
%values_6 = tt.addptr %second, %values : !tt.ptr<bf16>, i32
%values_7 = arith.muli %tile_5, 2048 : i32
%values_8 = tt.addptr %values_6, %values_7 : !tt.ptr<bf16>, i32
%values_9 = tt.splat %values_8 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_10 = tt.addptr %values_9, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_11 = tt.load %values_10 : tensor<2048x!tt.ptr<bf16>>
%values_12 = arith.extf %values_11 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_13 = arith.addf %values_12, %cst : tensor<2048xf32>
%local_amax_14 = math.absf %local_amax_13 : tensor<2048xf32>
%local_amax_15 = arith.maxnumf %local_amax_3, %local_amax_14 : tensor<2048xf32>
scf.yield %local_amax_15 : tensor<2048xf32>
"#) ++
  r#"
}
%0 = arith.muli %batch, 32 : i32
%1 = tt.addptr %partial_amax, %0 : !tt.ptr<f32>, i32
%2 = tt.addptr %1, %part : !tt.ptr<f32>, i32
%3 = "tt.reduce"(%local_amax_2) <{axis = 0 : i32}> ({
^bb0(%arg3: f32, %arg4: f32):
%4 = arith.maxnumf %arg3, %arg4 : f32
tt.reduce.return %4 : f32
}) : (tensor<2048xf32>) -> f32
tt.store %2, %3 : !tt.ptr<f32>
tt.return
}
}
"#

def matchesNativeScan (w : Width) (text : String) : Bool :=
  QKVTTIR.normalizeTTIR text == some ((nativeGraph w).splitOn "\n" |>.filter (!·.isEmpty))

structure ScanCertificate (w : Width) (baseline native : String) : Type where
  referenceMatched : QKVTTIR.matchesArtifact .reference w baseline = true
  nativeMatched : matchesNativeScan w native = true
  stores : ∀ (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part),
    nativeScan first second reduce w batch part =
      QKVTTIR.referenceStore (compactBank w true first) (compactBank w false second) reduce w batch part

def certifyScan (w : Width) (baseline native : String) : Option (ScanCertificate w baseline native) :=
  if hr : QKVTTIR.matchesArtifact .reference w baseline = true then
    if hn : matchesNativeScan w native = true then
      some ⟨hr, hn, fun first second reduce batch part => scan_equiv first second reduce w batch part⟩
    else none
  else none

/-- Exact reference scan PTX specialized to the three attention widths. -/
private def expectedReferencePTX : Width → String
  | .narrow => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _packed_batched_matrix_amax_partial_kernel(
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_0,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_1,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_2,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_3,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<26>;
.reg .b32 %r<109>;
.reg .b64 %rd<36>;
$L__func_begin0:
ld.param.b64 %rd6, [_packed_batched_matrix_amax_partial_kernel_param_2];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
setp.lt.u32 %p1, %r2, 576;
and.b32 %r99, %r3, 255;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd5, [_packed_batched_matrix_amax_partial_kernel_param_0];
ld.param.b64 %rd7, [_packed_batched_matrix_amax_partial_kernel_param_1];
mul.lo.s32 %r4, %r1, 589824;
cvt.s64.s32 %rd1, %r4;
mul.lo.s32 %r5, %r1, 1179648;
mad.wide.s32 %rd2, %r5, 2, %rd7;
shl.b32 %r6, %r3, 3;
and.b32 %r7, %r6, 2040;
cvt.u64.u32 %rd35, %r2;
cvt.u64.u32 %rd3, %r7;
mad.wide.u32 %rd34, %r2, 2048, -65536;
mul.wide.u32 %rd8, %r2, 4096;
mul.wide.u32 %rd9, %r99, 16;
or.b64 %rd10, %rd8, %rd9;
shl.b64 %rd11, %rd1, 1;
add.s64 %rd12, %rd10, %rd11;
add.s64 %rd33, %rd5, %rd12;
mov.b32 %r100, 0f00000000;
mov.b32 %r101, %r100;
mov.b32 %r102, %r100;
mov.b32 %r103, %r100;
mov.b32 %r104, %r100;
mov.b32 %r105, %r100;
mov.b32 %r106, %r100;
mov.b32 %r107, %r100;
$L__BB0_3:
add.s64 %rd4, %rd34, 65536;
setp.gt.u64 %p3, %rd4, 589823;
setp.lt.u64 %p2, %rd4, 589824;
mov.b32 %r12, 0;
mov.u32 %r8, %r12;
mov.u32 %r9, %r12;
mov.u32 %r10, %r12;
mov.u32 %r11, %r12;
@%p2 ld.global.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd33 + 0 ];
add.s64 %rd21, %rd3, %rd34;
cvt.u32.u64 %r13, %rd21;
add.s32 %r14, %r13, 65536;
max.u32 %r15, %r14, 589824;
add.s32 %r16, %r13, 65537;
max.u32 %r17, %r16, 589824;
add.s32 %r18, %r13, 65538;
max.u32 %r19, %r18, 589824;
add.s32 %r20, %r13, 65539;
max.u32 %r21, %r20, 589824;
add.s32 %r22, %r13, 65540;
max.u32 %r23, %r22, 589824;
add.s32 %r24, %r13, 65541;
max.u32 %r25, %r24, 589824;
add.s32 %r26, %r13, 65542;
max.u32 %r27, %r26, 589824;
add.s32 %r28, %r13, 65543;
max.u32 %r29, %r28, 589824;
mad.wide.u32 %rd22, %r15, 2, %rd2;
add.s64 %rd13, %rd22, -1179648;
mad.wide.u32 %rd23, %r17, 2, %rd2;
add.s64 %rd14, %rd23, -1179648;
mad.wide.u32 %rd24, %r19, 2, %rd2;
add.s64 %rd15, %rd24, -1179648;
mad.wide.u32 %rd25, %r21, 2, %rd2;
add.s64 %rd16, %rd25, -1179648;
mad.wide.u32 %rd26, %r23, 2, %rd2;
add.s64 %rd17, %rd26, -1179648;
mad.wide.u32 %rd27, %r25, 2, %rd2;
add.s64 %rd18, %rd27, -1179648;
mad.wide.u32 %rd28, %r27, 2, %rd2;
add.s64 %rd19, %rd28, -1179648;
mad.wide.u32 %rd29, %r29, 2, %rd2;
add.s64 %rd20, %rd29, -1179648;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p3 ld.global.b16 { %rs1 }, [ %rd13 + 0 ];
mov.u16 %rs3, %rs2;
@%p3 ld.global.b16 { %rs3 }, [ %rd14 + 0 ];
mov.u16 %rs4, %rs2;
@%p3 ld.global.b16 { %rs4 }, [ %rd15 + 0 ];
mov.u16 %rs5, %rs2;
@%p3 ld.global.b16 { %rs5 }, [ %rd16 + 0 ];
mov.u16 %rs6, %rs2;
@%p3 ld.global.b16 { %rs6 }, [ %rd17 + 0 ];
mov.u16 %rs7, %rs2;
@%p3 ld.global.b16 { %rs7 }, [ %rd18 + 0 ];
mov.u16 %rs8, %rs2;
@%p3 ld.global.b16 { %rs8 }, [ %rd19 + 0 ];
mov.u16 %rs9, %rs2;
@%p3 ld.global.b16 { %rs9 }, [ %rd20 + 0 ];
mov.b32 %r30, {%rs1, %rs3};
mov.b32 %r31, {%rs4, %rs5};
mov.b32 %r32, {%rs6, %rs7};
mov.b32 %r33, {%rs8, %rs9};
mov.b32 {%rs10, %rs11}, %r11;
cvt.f32.bf16 %r34, %rs10;
cvt.f32.bf16 %r35, %rs11;
mov.b32 {%rs12, %rs13}, %r10;
cvt.f32.bf16 %r36, %rs12;
cvt.f32.bf16 %r37, %rs13;
mov.b32 {%rs14, %rs15}, %r9;
cvt.f32.bf16 %r38, %rs14;
cvt.f32.bf16 %r39, %rs15;
mov.b32 {%rs16, %rs17}, %r8;
cvt.f32.bf16 %r40, %rs16;
cvt.f32.bf16 %r41, %rs17;
mov.b32 {%rs18, %rs19}, %r33;
cvt.f32.bf16 %r42, %rs18;
cvt.f32.bf16 %r43, %rs19;
mov.b32 {%rs20, %rs21}, %r32;
cvt.f32.bf16 %r44, %rs20;
cvt.f32.bf16 %r45, %rs21;
mov.b32 {%rs22, %rs23}, %r31;
cvt.f32.bf16 %r46, %rs22;
cvt.f32.bf16 %r47, %rs23;
mov.b32 {%rs24, %rs25}, %r30;
cvt.f32.bf16 %r48, %rs24;
cvt.f32.bf16 %r49, %rs25;
add.f32 %r50, %r41, %r49;
add.f32 %r51, %r40, %r48;
add.f32 %r52, %r39, %r47;
add.f32 %r53, %r38, %r46;
add.f32 %r54, %r37, %r45;
add.f32 %r55, %r36, %r44;
add.f32 %r56, %r35, %r43;
add.f32 %r57, %r34, %r42;
abs.f32 %r58, %r57;
abs.f32 %r59, %r56;
abs.f32 %r60, %r55;
abs.f32 %r61, %r54;
abs.f32 %r62, %r53;
abs.f32 %r63, %r52;
abs.f32 %r64, %r51;
abs.f32 %r65, %r50;
max.f32 %r101, %r101, %r65;
max.f32 %r100, %r100, %r64;
max.f32 %r103, %r103, %r63;
max.f32 %r102, %r102, %r62;
max.f32 %r105, %r105, %r61;
max.f32 %r104, %r104, %r60;
max.f32 %r107, %r107, %r59;
max.f32 %r106, %r106, %r58;
add.s64 %rd33, %rd33, 131072;
setp.lt.u64 %p4, %rd4, 1114112;
mov.b64 %rd34, %rd4;
@%p4 bra $L__BB0_3;
$L__tmp1:
max.f32 %r66, %r100, %r101;
max.f32 %r67, %r66, %r102;
max.f32 %r68, %r67, %r103;
max.f32 %r69, %r68, %r104;
max.f32 %r70, %r69, %r105;
max.f32 %r71, %r70, %r106;
max.f32 %r108, %r71, %r107;
bra.uni $L__BB0_5;
$L__tmp2:
$L__BB0_1:
cvt.u64.u32 %rd35, %r2;
mov.b32 %r108, 0f00000000;
$L__BB0_5:
and.b32 %r78, %r3, 31;
shl.b32 %r79, %r1, 5;
mad.wide.s32 %rd31, %r79, 4, %rd6;
shl.b64 %rd32, %rd35, 2;
add.s64 %rd30, %rd31, %rd32;
$L__tmp3:
shfl.sync.bfly.b32 %r80, %r108, 16, 31, -1;
$L__tmp4:
max.f32 %r81, %r108, %r80;
$L__tmp5:
shfl.sync.bfly.b32 %r82, %r81, 8, 31, -1;
$L__tmp6:
max.f32 %r83, %r81, %r82;
$L__tmp7:
shfl.sync.bfly.b32 %r84, %r83, 4, 31, -1;
$L__tmp8:
max.f32 %r85, %r83, %r84;
$L__tmp9:
shfl.sync.bfly.b32 %r86, %r85, 2, 31, -1;
$L__tmp10:
max.f32 %r87, %r85, %r86;
$L__tmp11:
shfl.sync.bfly.b32 %r88, %r87, 1, 31, -1;
$L__tmp12:
max.f32 %r73, %r87, %r88;
$L__tmp13:
setp.eq.b32 %p5, %r78, 0;
shr.u32 %r89, %r3, 3;
and.b32 %r90, %r89, 28;
mov.b32 %r91, global_smem;
add.s32 %r72, %r91, %r90;
@%p5 st.shared.b32 [ %r72 + 0 ], %r73;
bar.sync 0;
setp.lt.u32 %p6, %r99, 8;
shl.b32 %r92, %r99, 2;
add.s32 %r75, %r91, %r92;
@%p6 ld.shared.b32 %r74, [ %r75 + 0 ];
shfl.sync.bfly.b32 %r93, %r74, 4, 31, -1;
$L__tmp14:
max.f32 %r94, %r74, %r93;
$L__tmp15:
shfl.sync.bfly.b32 %r95, %r94, 2, 31, -1;
$L__tmp16:
max.f32 %r96, %r94, %r95;
$L__tmp17:
shfl.sync.bfly.b32 %r97, %r96, 1, 31, -1;
$L__tmp18:
max.f32 %r76, %r96, %r97;
$L__tmp19:
and.b32 %r98, %r3, 7;
setp.eq.b32 %p9, %r98, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r75 + 0 ], %r76;
bar.sync 0;
ld.shared.b32 %r77, [global_smem];
$L__tmp20:
setp.lt.u32 %p10, %r99, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd30 + 0 ], { %r77 };
ret;
$L__tmp21:
$L__func_end0:
}
"#
  | .halfValue => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _packed_batched_matrix_amax_partial_kernel(
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_0,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_1,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_2,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_3,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<26>;
.reg .b32 %r<109>;
.reg .b64 %rd<36>;
$L__func_begin0:
ld.param.b64 %rd6, [_packed_batched_matrix_amax_partial_kernel_param_2];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
setp.lt.u32 %p1, %r2, 432;
and.b32 %r99, %r3, 255;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd5, [_packed_batched_matrix_amax_partial_kernel_param_0];
ld.param.b64 %rd7, [_packed_batched_matrix_amax_partial_kernel_param_1];
mul.lo.s32 %r4, %r1, 589824;
cvt.s64.s32 %rd1, %r4;
mul.lo.s32 %r5, %r1, 294912;
mad.wide.s32 %rd2, %r5, 2, %rd7;
shl.b32 %r6, %r3, 3;
and.b32 %r7, %r6, 2040;
cvt.u64.u32 %rd35, %r2;
cvt.u64.u32 %rd3, %r7;
mad.wide.u32 %rd34, %r2, 2048, -65536;
mul.wide.u32 %rd8, %r2, 4096;
mul.wide.u32 %rd9, %r99, 16;
or.b64 %rd10, %rd8, %rd9;
shl.b64 %rd11, %rd1, 1;
add.s64 %rd12, %rd10, %rd11;
add.s64 %rd33, %rd5, %rd12;
mov.b32 %r100, 0f00000000;
mov.b32 %r101, %r100;
mov.b32 %r102, %r100;
mov.b32 %r103, %r100;
mov.b32 %r104, %r100;
mov.b32 %r105, %r100;
mov.b32 %r106, %r100;
mov.b32 %r107, %r100;
$L__BB0_3:
add.s64 %rd4, %rd34, 65536;
setp.gt.u64 %p3, %rd4, 589823;
setp.lt.u64 %p2, %rd4, 589824;
mov.b32 %r12, 0;
mov.u32 %r8, %r12;
mov.u32 %r9, %r12;
mov.u32 %r10, %r12;
mov.u32 %r11, %r12;
@%p2 ld.global.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd33 + 0 ];
add.s64 %rd21, %rd3, %rd34;
cvt.u32.u64 %r13, %rd21;
add.s32 %r14, %r13, 65536;
max.u32 %r15, %r14, 589824;
add.s32 %r16, %r13, 65537;
max.u32 %r17, %r16, 589824;
add.s32 %r18, %r13, 65538;
max.u32 %r19, %r18, 589824;
add.s32 %r20, %r13, 65539;
max.u32 %r21, %r20, 589824;
add.s32 %r22, %r13, 65540;
max.u32 %r23, %r22, 589824;
add.s32 %r24, %r13, 65541;
max.u32 %r25, %r24, 589824;
add.s32 %r26, %r13, 65542;
max.u32 %r27, %r26, 589824;
add.s32 %r28, %r13, 65543;
max.u32 %r29, %r28, 589824;
mad.wide.u32 %rd22, %r15, 2, %rd2;
add.s64 %rd13, %rd22, -1179648;
mad.wide.u32 %rd23, %r17, 2, %rd2;
add.s64 %rd14, %rd23, -1179648;
mad.wide.u32 %rd24, %r19, 2, %rd2;
add.s64 %rd15, %rd24, -1179648;
mad.wide.u32 %rd25, %r21, 2, %rd2;
add.s64 %rd16, %rd25, -1179648;
mad.wide.u32 %rd26, %r23, 2, %rd2;
add.s64 %rd17, %rd26, -1179648;
mad.wide.u32 %rd27, %r25, 2, %rd2;
add.s64 %rd18, %rd27, -1179648;
mad.wide.u32 %rd28, %r27, 2, %rd2;
add.s64 %rd19, %rd28, -1179648;
mad.wide.u32 %rd29, %r29, 2, %rd2;
add.s64 %rd20, %rd29, -1179648;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p3 ld.global.b16 { %rs1 }, [ %rd13 + 0 ];
mov.u16 %rs3, %rs2;
@%p3 ld.global.b16 { %rs3 }, [ %rd14 + 0 ];
mov.u16 %rs4, %rs2;
@%p3 ld.global.b16 { %rs4 }, [ %rd15 + 0 ];
mov.u16 %rs5, %rs2;
@%p3 ld.global.b16 { %rs5 }, [ %rd16 + 0 ];
mov.u16 %rs6, %rs2;
@%p3 ld.global.b16 { %rs6 }, [ %rd17 + 0 ];
mov.u16 %rs7, %rs2;
@%p3 ld.global.b16 { %rs7 }, [ %rd18 + 0 ];
mov.u16 %rs8, %rs2;
@%p3 ld.global.b16 { %rs8 }, [ %rd19 + 0 ];
mov.u16 %rs9, %rs2;
@%p3 ld.global.b16 { %rs9 }, [ %rd20 + 0 ];
mov.b32 %r30, {%rs1, %rs3};
mov.b32 %r31, {%rs4, %rs5};
mov.b32 %r32, {%rs6, %rs7};
mov.b32 %r33, {%rs8, %rs9};
mov.b32 {%rs10, %rs11}, %r11;
cvt.f32.bf16 %r34, %rs10;
cvt.f32.bf16 %r35, %rs11;
mov.b32 {%rs12, %rs13}, %r10;
cvt.f32.bf16 %r36, %rs12;
cvt.f32.bf16 %r37, %rs13;
mov.b32 {%rs14, %rs15}, %r9;
cvt.f32.bf16 %r38, %rs14;
cvt.f32.bf16 %r39, %rs15;
mov.b32 {%rs16, %rs17}, %r8;
cvt.f32.bf16 %r40, %rs16;
cvt.f32.bf16 %r41, %rs17;
mov.b32 {%rs18, %rs19}, %r33;
cvt.f32.bf16 %r42, %rs18;
cvt.f32.bf16 %r43, %rs19;
mov.b32 {%rs20, %rs21}, %r32;
cvt.f32.bf16 %r44, %rs20;
cvt.f32.bf16 %r45, %rs21;
mov.b32 {%rs22, %rs23}, %r31;
cvt.f32.bf16 %r46, %rs22;
cvt.f32.bf16 %r47, %rs23;
mov.b32 {%rs24, %rs25}, %r30;
cvt.f32.bf16 %r48, %rs24;
cvt.f32.bf16 %r49, %rs25;
add.f32 %r50, %r41, %r49;
add.f32 %r51, %r40, %r48;
add.f32 %r52, %r39, %r47;
add.f32 %r53, %r38, %r46;
add.f32 %r54, %r37, %r45;
add.f32 %r55, %r36, %r44;
add.f32 %r56, %r35, %r43;
add.f32 %r57, %r34, %r42;
abs.f32 %r58, %r57;
abs.f32 %r59, %r56;
abs.f32 %r60, %r55;
abs.f32 %r61, %r54;
abs.f32 %r62, %r53;
abs.f32 %r63, %r52;
abs.f32 %r64, %r51;
abs.f32 %r65, %r50;
max.f32 %r101, %r101, %r65;
max.f32 %r100, %r100, %r64;
max.f32 %r103, %r103, %r63;
max.f32 %r102, %r102, %r62;
max.f32 %r105, %r105, %r61;
max.f32 %r104, %r104, %r60;
max.f32 %r107, %r107, %r59;
max.f32 %r106, %r106, %r58;
add.s64 %rd33, %rd33, 131072;
setp.lt.u64 %p4, %rd4, 819200;
mov.b64 %rd34, %rd4;
@%p4 bra $L__BB0_3;
$L__tmp1:
max.f32 %r66, %r100, %r101;
max.f32 %r67, %r66, %r102;
max.f32 %r68, %r67, %r103;
max.f32 %r69, %r68, %r104;
max.f32 %r70, %r69, %r105;
max.f32 %r71, %r70, %r106;
max.f32 %r108, %r71, %r107;
bra.uni $L__BB0_5;
$L__tmp2:
$L__BB0_1:
cvt.u64.u32 %rd35, %r2;
mov.b32 %r108, 0f00000000;
$L__BB0_5:
and.b32 %r78, %r3, 31;
shl.b32 %r79, %r1, 5;
mad.wide.s32 %rd31, %r79, 4, %rd6;
shl.b64 %rd32, %rd35, 2;
add.s64 %rd30, %rd31, %rd32;
$L__tmp3:
shfl.sync.bfly.b32 %r80, %r108, 16, 31, -1;
$L__tmp4:
max.f32 %r81, %r108, %r80;
$L__tmp5:
shfl.sync.bfly.b32 %r82, %r81, 8, 31, -1;
$L__tmp6:
max.f32 %r83, %r81, %r82;
$L__tmp7:
shfl.sync.bfly.b32 %r84, %r83, 4, 31, -1;
$L__tmp8:
max.f32 %r85, %r83, %r84;
$L__tmp9:
shfl.sync.bfly.b32 %r86, %r85, 2, 31, -1;
$L__tmp10:
max.f32 %r87, %r85, %r86;
$L__tmp11:
shfl.sync.bfly.b32 %r88, %r87, 1, 31, -1;
$L__tmp12:
max.f32 %r73, %r87, %r88;
$L__tmp13:
setp.eq.b32 %p5, %r78, 0;
shr.u32 %r89, %r3, 3;
and.b32 %r90, %r89, 28;
mov.b32 %r91, global_smem;
add.s32 %r72, %r91, %r90;
@%p5 st.shared.b32 [ %r72 + 0 ], %r73;
bar.sync 0;
setp.lt.u32 %p6, %r99, 8;
shl.b32 %r92, %r99, 2;
add.s32 %r75, %r91, %r92;
@%p6 ld.shared.b32 %r74, [ %r75 + 0 ];
shfl.sync.bfly.b32 %r93, %r74, 4, 31, -1;
$L__tmp14:
max.f32 %r94, %r74, %r93;
$L__tmp15:
shfl.sync.bfly.b32 %r95, %r94, 2, 31, -1;
$L__tmp16:
max.f32 %r96, %r94, %r95;
$L__tmp17:
shfl.sync.bfly.b32 %r97, %r96, 1, 31, -1;
$L__tmp18:
max.f32 %r76, %r96, %r97;
$L__tmp19:
and.b32 %r98, %r3, 7;
setp.eq.b32 %p9, %r98, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r75 + 0 ], %r76;
bar.sync 0;
ld.shared.b32 %r77, [global_smem];
$L__tmp20:
setp.lt.u32 %p10, %r99, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd30 + 0 ], { %r77 };
ret;
$L__tmp21:
$L__func_end0:
}
"#
  | .wide => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _packed_batched_matrix_amax_partial_kernel(
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_0,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_1,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_2,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_3,
.param .u64 .ptr .global .align 1 _packed_batched_matrix_amax_partial_kernel_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<26>;
.reg .b32 %r<108>;
.reg .b64 %rd<36>;
$L__func_begin0:
ld.param.b64 %rd6, [_packed_batched_matrix_amax_partial_kernel_param_2];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
setp.lt.u32 %p1, %r2, 864;
and.b32 %r98, %r3, 255;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b64 %rd5, [_packed_batched_matrix_amax_partial_kernel_param_0];
ld.param.b64 %rd7, [_packed_batched_matrix_amax_partial_kernel_param_1];
mul.lo.s32 %r4, %r1, 1179648;
cvt.s64.s32 %rd1, %r4;
mad.wide.s32 %rd2, %r4, 2, %rd7;
shl.b32 %r5, %r3, 3;
and.b32 %r6, %r5, 2040;
cvt.u64.u32 %rd35, %r2;
cvt.u64.u32 %rd3, %r6;
mad.wide.u32 %rd34, %r2, 2048, -65536;
mul.wide.u32 %rd8, %r2, 4096;
mul.wide.u32 %rd9, %r98, 16;
or.b64 %rd10, %rd8, %rd9;
shl.b64 %rd11, %rd1, 1;
add.s64 %rd12, %rd10, %rd11;
add.s64 %rd33, %rd5, %rd12;
mov.b32 %r99, 0f00000000;
mov.b32 %r100, %r99;
mov.b32 %r101, %r99;
mov.b32 %r102, %r99;
mov.b32 %r103, %r99;
mov.b32 %r104, %r99;
mov.b32 %r105, %r99;
mov.b32 %r106, %r99;
$L__BB0_3:
add.s64 %rd4, %rd34, 65536;
setp.gt.u64 %p3, %rd4, 1179647;
setp.lt.u64 %p2, %rd4, 1179648;
mov.b32 %r11, 0;
mov.u32 %r7, %r11;
mov.u32 %r8, %r11;
mov.u32 %r9, %r11;
mov.u32 %r10, %r11;
@%p2 ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd33 + 0 ];
add.s64 %rd21, %rd3, %rd34;
cvt.u32.u64 %r12, %rd21;
add.s32 %r13, %r12, 65536;
max.u32 %r14, %r13, 1179648;
add.s32 %r15, %r12, 65537;
max.u32 %r16, %r15, 1179648;
add.s32 %r17, %r12, 65538;
max.u32 %r18, %r17, 1179648;
add.s32 %r19, %r12, 65539;
max.u32 %r20, %r19, 1179648;
add.s32 %r21, %r12, 65540;
max.u32 %r22, %r21, 1179648;
add.s32 %r23, %r12, 65541;
max.u32 %r24, %r23, 1179648;
add.s32 %r25, %r12, 65542;
max.u32 %r26, %r25, 1179648;
add.s32 %r27, %r12, 65543;
max.u32 %r28, %r27, 1179648;
mad.wide.u32 %rd22, %r14, 2, %rd2;
add.s64 %rd13, %rd22, -2359296;
mad.wide.u32 %rd23, %r16, 2, %rd2;
add.s64 %rd14, %rd23, -2359296;
mad.wide.u32 %rd24, %r18, 2, %rd2;
add.s64 %rd15, %rd24, -2359296;
mad.wide.u32 %rd25, %r20, 2, %rd2;
add.s64 %rd16, %rd25, -2359296;
mad.wide.u32 %rd26, %r22, 2, %rd2;
add.s64 %rd17, %rd26, -2359296;
mad.wide.u32 %rd27, %r24, 2, %rd2;
add.s64 %rd18, %rd27, -2359296;
mad.wide.u32 %rd28, %r26, 2, %rd2;
add.s64 %rd19, %rd28, -2359296;
mad.wide.u32 %rd29, %r28, 2, %rd2;
add.s64 %rd20, %rd29, -2359296;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p3 ld.global.b16 { %rs1 }, [ %rd13 + 0 ];
mov.u16 %rs3, %rs2;
@%p3 ld.global.b16 { %rs3 }, [ %rd14 + 0 ];
mov.u16 %rs4, %rs2;
@%p3 ld.global.b16 { %rs4 }, [ %rd15 + 0 ];
mov.u16 %rs5, %rs2;
@%p3 ld.global.b16 { %rs5 }, [ %rd16 + 0 ];
mov.u16 %rs6, %rs2;
@%p3 ld.global.b16 { %rs6 }, [ %rd17 + 0 ];
mov.u16 %rs7, %rs2;
@%p3 ld.global.b16 { %rs7 }, [ %rd18 + 0 ];
mov.u16 %rs8, %rs2;
@%p3 ld.global.b16 { %rs8 }, [ %rd19 + 0 ];
mov.u16 %rs9, %rs2;
@%p3 ld.global.b16 { %rs9 }, [ %rd20 + 0 ];
mov.b32 %r29, {%rs1, %rs3};
mov.b32 %r30, {%rs4, %rs5};
mov.b32 %r31, {%rs6, %rs7};
mov.b32 %r32, {%rs8, %rs9};
mov.b32 {%rs10, %rs11}, %r10;
cvt.f32.bf16 %r33, %rs10;
cvt.f32.bf16 %r34, %rs11;
mov.b32 {%rs12, %rs13}, %r9;
cvt.f32.bf16 %r35, %rs12;
cvt.f32.bf16 %r36, %rs13;
mov.b32 {%rs14, %rs15}, %r8;
cvt.f32.bf16 %r37, %rs14;
cvt.f32.bf16 %r38, %rs15;
mov.b32 {%rs16, %rs17}, %r7;
cvt.f32.bf16 %r39, %rs16;
cvt.f32.bf16 %r40, %rs17;
mov.b32 {%rs18, %rs19}, %r32;
cvt.f32.bf16 %r41, %rs18;
cvt.f32.bf16 %r42, %rs19;
mov.b32 {%rs20, %rs21}, %r31;
cvt.f32.bf16 %r43, %rs20;
cvt.f32.bf16 %r44, %rs21;
mov.b32 {%rs22, %rs23}, %r30;
cvt.f32.bf16 %r45, %rs22;
cvt.f32.bf16 %r46, %rs23;
mov.b32 {%rs24, %rs25}, %r29;
cvt.f32.bf16 %r47, %rs24;
cvt.f32.bf16 %r48, %rs25;
add.f32 %r49, %r40, %r48;
add.f32 %r50, %r39, %r47;
add.f32 %r51, %r38, %r46;
add.f32 %r52, %r37, %r45;
add.f32 %r53, %r36, %r44;
add.f32 %r54, %r35, %r43;
add.f32 %r55, %r34, %r42;
add.f32 %r56, %r33, %r41;
abs.f32 %r57, %r56;
abs.f32 %r58, %r55;
abs.f32 %r59, %r54;
abs.f32 %r60, %r53;
abs.f32 %r61, %r52;
abs.f32 %r62, %r51;
abs.f32 %r63, %r50;
abs.f32 %r64, %r49;
max.f32 %r100, %r100, %r64;
max.f32 %r99, %r99, %r63;
max.f32 %r102, %r102, %r62;
max.f32 %r101, %r101, %r61;
max.f32 %r104, %r104, %r60;
max.f32 %r103, %r103, %r59;
max.f32 %r106, %r106, %r58;
max.f32 %r105, %r105, %r57;
add.s64 %rd33, %rd33, 131072;
setp.lt.u64 %p4, %rd4, 1703936;
mov.b64 %rd34, %rd4;
@%p4 bra $L__BB0_3;
$L__tmp1:
max.f32 %r65, %r99, %r100;
max.f32 %r66, %r65, %r101;
max.f32 %r67, %r66, %r102;
max.f32 %r68, %r67, %r103;
max.f32 %r69, %r68, %r104;
max.f32 %r70, %r69, %r105;
max.f32 %r107, %r70, %r106;
bra.uni $L__BB0_5;
$L__tmp2:
$L__BB0_1:
cvt.u64.u32 %rd35, %r2;
mov.b32 %r107, 0f00000000;
$L__BB0_5:
and.b32 %r77, %r3, 31;
shl.b32 %r78, %r1, 5;
mad.wide.s32 %rd31, %r78, 4, %rd6;
shl.b64 %rd32, %rd35, 2;
add.s64 %rd30, %rd31, %rd32;
$L__tmp3:
shfl.sync.bfly.b32 %r79, %r107, 16, 31, -1;
$L__tmp4:
max.f32 %r80, %r107, %r79;
$L__tmp5:
shfl.sync.bfly.b32 %r81, %r80, 8, 31, -1;
$L__tmp6:
max.f32 %r82, %r80, %r81;
$L__tmp7:
shfl.sync.bfly.b32 %r83, %r82, 4, 31, -1;
$L__tmp8:
max.f32 %r84, %r82, %r83;
$L__tmp9:
shfl.sync.bfly.b32 %r85, %r84, 2, 31, -1;
$L__tmp10:
max.f32 %r86, %r84, %r85;
$L__tmp11:
shfl.sync.bfly.b32 %r87, %r86, 1, 31, -1;
$L__tmp12:
max.f32 %r72, %r86, %r87;
$L__tmp13:
setp.eq.b32 %p5, %r77, 0;
shr.u32 %r88, %r3, 3;
and.b32 %r89, %r88, 28;
mov.b32 %r90, global_smem;
add.s32 %r71, %r90, %r89;
@%p5 st.shared.b32 [ %r71 + 0 ], %r72;
bar.sync 0;
setp.lt.u32 %p6, %r98, 8;
shl.b32 %r91, %r98, 2;
add.s32 %r74, %r90, %r91;
@%p6 ld.shared.b32 %r73, [ %r74 + 0 ];
shfl.sync.bfly.b32 %r92, %r73, 4, 31, -1;
$L__tmp14:
max.f32 %r93, %r73, %r92;
$L__tmp15:
shfl.sync.bfly.b32 %r94, %r93, 2, 31, -1;
$L__tmp16:
max.f32 %r95, %r93, %r94;
$L__tmp17:
shfl.sync.bfly.b32 %r96, %r95, 1, 31, -1;
$L__tmp18:
max.f32 %r75, %r95, %r96;
$L__tmp19:
and.b32 %r97, %r3, 7;
setp.eq.b32 %p9, %r97, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r74 + 0 ], %r75;
bar.sync 0;
ld.shared.b32 %r76, [global_smem];
$L__tmp20:
setp.lt.u32 %p10, %r98, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd30 + 0 ], { %r76 };
ret;
$L__tmp21:
$L__func_end0:
}
"#

/-- Exact native scan PTX specialized to the three attention widths. -/
private def expectedNativePTX : Width → String
  | .narrow => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _aligned_packed_amax(
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_0,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_1,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_2,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_3,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<17>;
.reg .b32 %r<100>;
.reg .b64 %rd<17>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
shl.b32 %r7, %r3, 3;
and.b32 %r4, %r7, 2040;
sub.s32 %r8, 319, %r2;
shr.u32 %r9, %r8, 27;
add.s32 %r10, %r8, %r9;
shr.s32 %r5, %r10, 5;
sub.s32 %r11, 607, %r2;
shr.u32 %r12, %r11, 27;
add.s32 %r13, %r11, %r12;
shr.s32 %r6, %r13, 5;
mul.lo.s32 %r14, %r1, 1179648;
cvt.s64.s32 %rd1, %r14;
setp.gt.u32 %p1, %r2, 287;
mov.b32 %r92, 0f00000000;
shl.b64 %rd15, %rd1, 1;
shl.b32 %r87, %r2, 11;
mov.b32 %r93, %r92;
mov.b32 %r94, %r92;
mov.b32 %r95, %r92;
mov.b32 %r96, %r92;
mov.b32 %r97, %r92;
mov.b32 %r98, %r92;
mov.b32 %r99, %r92;
@%p1 bra $L__BB0_3;
ld.param.b64 %rd4, [_aligned_packed_amax_param_0];
add.s64 %rd7, %rd4, %rd15;
mad.wide.u32 %rd2, %r4, 2, %rd7;
max.s32 %r90, %r5, 1;
mov.b32 %r92, 0f00000000;
mov.b32 %r88, %r2;
mov.b32 %r89, %r87;
mov.b32 %r93, %r92;
mov.b32 %r94, %r92;
mov.b32 %r95, %r92;
mov.b32 %r96, %r92;
mov.b32 %r97, %r92;
mov.b32 %r98, %r92;
mov.b32 %r99, %r92;
$L__BB0_2:
mul.hi.u32 %r19, %r88, -1431655765;
shr.u32 %r20, %r19, 4;
mad.lo.s32 %r21, %r20, 49152, %r89;
mad.wide.s32 %rd8, %r21, 2, %rd2;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd8 + 0 ];
and.b32 %r22, %r15, 2147450879;
and.b32 %r23, %r16, 2147450879;
and.b32 %r24, %r17, 2147450879;
and.b32 %r25, %r18, 2147450879;
mov.b32 {%rs1, %rs2}, %r22;
cvt.f32.bf16 %r26, %rs1;
cvt.f32.bf16 %r27, %rs2;
mov.b32 {%rs3, %rs4}, %r23;
cvt.f32.bf16 %r28, %rs3;
cvt.f32.bf16 %r29, %rs4;
mov.b32 {%rs5, %rs6}, %r24;
cvt.f32.bf16 %r30, %rs5;
cvt.f32.bf16 %r31, %rs6;
mov.b32 {%rs7, %rs8}, %r25;
cvt.f32.bf16 %r32, %rs7;
cvt.f32.bf16 %r33, %rs8;
max.f32 %r99, %r99, %r33;
max.f32 %r98, %r98, %r32;
max.f32 %r97, %r97, %r31;
max.f32 %r96, %r96, %r30;
max.f32 %r95, %r95, %r29;
max.f32 %r94, %r94, %r28;
max.f32 %r93, %r93, %r27;
max.f32 %r92, %r92, %r26;
add.s32 %r90, %r90, -1;
add.s32 %r89, %r89, 65536;
add.s32 %r88, %r88, 32;
setp.ne.b32 %p2, %r90, 0;
@%p2 bra $L__BB0_2;
$L__BB0_3:
ld.param.b64 %rd6, [_aligned_packed_amax_param_2];
setp.ge.s32 %p3, %r5, %r6;
@%p3 bra $L__BB0_6;
ld.param.b64 %rd5, [_aligned_packed_amax_param_1];
add.s64 %rd9, %rd5, %rd15;
mad.wide.u32 %rd3, %r4, 2, %rd9;
cvt.s64.s32 %rd10, %r5;
cvt.s64.s32 %rd11, %r6;
sub.s64 %rd16, %rd11, %rd10;
shl.b32 %r34, %r5, 16;
add.s32 %r35, %r34, %r87;
add.s32 %r91, %r35, -589824;
$L__BB0_5:
mad.wide.s32 %rd12, %r91, 2, %rd3;
mov.u32 %r36, 0x0;
mov.u32 %r37, 0x0;
mov.u32 %r38, 0x0;
mov.u32 %r39, 0x0;
ld.global.v4.b32 { %r36, %r37, %r38, %r39 }, [ %rd12 + 0 ];
and.b32 %r40, %r36, 2147450879;
and.b32 %r41, %r37, 2147450879;
and.b32 %r42, %r38, 2147450879;
and.b32 %r43, %r39, 2147450879;
mov.b32 {%rs9, %rs10}, %r43;
cvt.f32.bf16 %r44, %rs10;
cvt.f32.bf16 %r45, %rs9;
mov.b32 {%rs11, %rs12}, %r40;
cvt.f32.bf16 %r46, %rs11;
cvt.f32.bf16 %r47, %rs12;
mov.b32 {%rs13, %rs14}, %r41;
cvt.f32.bf16 %r48, %rs13;
cvt.f32.bf16 %r49, %rs14;
mov.b32 {%rs15, %rs16}, %r42;
cvt.f32.bf16 %r50, %rs15;
cvt.f32.bf16 %r51, %rs16;
max.f32 %r97, %r97, %r51;
max.f32 %r96, %r96, %r50;
max.f32 %r95, %r95, %r49;
max.f32 %r94, %r94, %r48;
max.f32 %r93, %r93, %r47;
max.f32 %r92, %r92, %r46;
max.f32 %r98, %r98, %r45;
max.f32 %r99, %r99, %r44;
add.s64 %rd16, %rd16, -1;
add.s32 %r91, %r91, 65536;
setp.ne.b64 %p4, %rd16, 0;
@%p4 bra $L__BB0_5;
$L__BB0_6:
and.b32 %r58, %r3, 31;
and.b32 %r59, %r3, 255;
shl.b32 %r60, %r1, 5;
mad.wide.s32 %rd14, %r60, 4, %rd6;
mad.wide.u32 %rd13, %r2, 4, %rd14;
$L__tmp0:
max.f32 %r61, %r92, %r93;
max.f32 %r62, %r61, %r94;
max.f32 %r63, %r62, %r95;
max.f32 %r64, %r63, %r96;
max.f32 %r65, %r64, %r97;
max.f32 %r66, %r65, %r98;
max.f32 %r67, %r66, %r99;
$L__tmp1:
shfl.sync.bfly.b32 %r68, %r67, 16, 31, -1;
$L__tmp2:
max.f32 %r69, %r67, %r68;
$L__tmp3:
shfl.sync.bfly.b32 %r70, %r69, 8, 31, -1;
$L__tmp4:
max.f32 %r71, %r69, %r70;
$L__tmp5:
shfl.sync.bfly.b32 %r72, %r71, 4, 31, -1;
$L__tmp6:
max.f32 %r73, %r71, %r72;
$L__tmp7:
shfl.sync.bfly.b32 %r74, %r73, 2, 31, -1;
$L__tmp8:
max.f32 %r75, %r73, %r74;
$L__tmp9:
shfl.sync.bfly.b32 %r76, %r75, 1, 31, -1;
$L__tmp10:
max.f32 %r53, %r75, %r76;
$L__tmp11:
setp.eq.b32 %p5, %r58, 0;
shr.u32 %r77, %r3, 3;
and.b32 %r78, %r77, 28;
mov.b32 %r79, global_smem;
add.s32 %r52, %r79, %r78;
@%p5 st.shared.b32 [ %r52 + 0 ], %r53;
bar.sync 0;
setp.lt.u32 %p6, %r59, 8;
shl.b32 %r80, %r59, 2;
add.s32 %r55, %r79, %r80;
@%p6 ld.shared.b32 %r54, [ %r55 + 0 ];
shfl.sync.bfly.b32 %r81, %r54, 4, 31, -1;
$L__tmp12:
max.f32 %r82, %r54, %r81;
$L__tmp13:
shfl.sync.bfly.b32 %r83, %r82, 2, 31, -1;
$L__tmp14:
max.f32 %r84, %r82, %r83;
$L__tmp15:
shfl.sync.bfly.b32 %r85, %r84, 1, 31, -1;
$L__tmp16:
max.f32 %r56, %r84, %r85;
$L__tmp17:
and.b32 %r86, %r3, 7;
setp.eq.b32 %p9, %r86, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r55 + 0 ], %r56;
bar.sync 0;
ld.shared.b32 %r57, [global_smem];
$L__tmp18:
setp.lt.u32 %p10, %r59, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd13 + 0 ], { %r57 };
ret;
$L__tmp19:
$L__func_end0:
}
"#
  | .halfValue => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _aligned_packed_amax(
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_0,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_1,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_2,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_3,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<17>;
.reg .b32 %r<109>;
.reg .b64 %rd<14>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
shl.b32 %r7, %r3, 3;
and.b32 %r4, %r7, 2040;
sub.s32 %r8, 319, %r2;
shr.u32 %r9, %r8, 27;
add.s32 %r10, %r8, %r9;
shr.s32 %r5, %r10, 5;
sub.s32 %r11, 463, %r2;
shr.u32 %r12, %r11, 27;
add.s32 %r13, %r11, %r12;
shr.s32 %r6, %r13, 5;
mul.lo.s32 %r14, %r1, 1179648;
cvt.s64.s32 %rd1, %r14;
setp.gt.u32 %p1, %r2, 287;
mov.b32 %r101, 0f00000000;
shl.b64 %rd13, %rd1, 1;
mov.b32 %r102, %r101;
mov.b32 %r103, %r101;
mov.b32 %r104, %r101;
mov.b32 %r105, %r101;
mov.b32 %r106, %r101;
mov.b32 %r107, %r101;
mov.b32 %r108, %r101;
@%p1 bra $L__BB0_3;
ld.param.b64 %rd4, [_aligned_packed_amax_param_0];
add.s64 %rd7, %rd4, %rd13;
mad.wide.u32 %rd2, %r4, 2, %rd7;
max.s32 %r97, %r5, 1;
shl.b32 %r96, %r2, 11;
mov.b32 %r101, 0f00000000;
mov.b32 %r95, %r2;
mov.b32 %r102, %r101;
mov.b32 %r103, %r101;
mov.b32 %r104, %r101;
mov.b32 %r105, %r101;
mov.b32 %r106, %r101;
mov.b32 %r107, %r101;
mov.b32 %r108, %r101;
$L__BB0_2:
mul.hi.u32 %r19, %r95, -1431655765;
shr.u32 %r20, %r19, 4;
mad.lo.s32 %r21, %r20, 49152, %r96;
mad.wide.s32 %rd8, %r21, 2, %rd2;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd8 + 0 ];
and.b32 %r22, %r15, 2147450879;
and.b32 %r23, %r16, 2147450879;
and.b32 %r24, %r17, 2147450879;
and.b32 %r25, %r18, 2147450879;
mov.b32 {%rs1, %rs2}, %r22;
cvt.f32.bf16 %r26, %rs1;
cvt.f32.bf16 %r27, %rs2;
mov.b32 {%rs3, %rs4}, %r23;
cvt.f32.bf16 %r28, %rs3;
cvt.f32.bf16 %r29, %rs4;
mov.b32 {%rs5, %rs6}, %r24;
cvt.f32.bf16 %r30, %rs5;
cvt.f32.bf16 %r31, %rs6;
mov.b32 {%rs7, %rs8}, %r25;
cvt.f32.bf16 %r32, %rs7;
cvt.f32.bf16 %r33, %rs8;
max.f32 %r108, %r108, %r33;
max.f32 %r107, %r107, %r32;
max.f32 %r106, %r106, %r31;
max.f32 %r105, %r105, %r30;
max.f32 %r104, %r104, %r29;
max.f32 %r103, %r103, %r28;
max.f32 %r102, %r102, %r27;
max.f32 %r101, %r101, %r26;
add.s32 %r97, %r97, -1;
add.s32 %r96, %r96, 65536;
add.s32 %r95, %r95, 32;
setp.ne.b32 %p2, %r97, 0;
@%p2 bra $L__BB0_2;
$L__BB0_3:
ld.param.b64 %rd6, [_aligned_packed_amax_param_2];
setp.le.s32 %p3, %r6, %r5;
@%p3 bra $L__BB0_6;
ld.param.b64 %rd5, [_aligned_packed_amax_param_1];
add.s64 %rd9, %rd5, %rd13;
mad.wide.u32 %rd3, %r4, 2, %rd9;
sub.s32 %r100, %r6, %r5;
shl.b32 %r34, %r5, 5;
add.s32 %r35, %r2, %r34;
add.s32 %r98, %r35, -288;
shl.b32 %r99, %r98, 12;
$L__BB0_5:
mul.hi.s32 %r40, %r98, 715827883;
shr.u32 %r41, %r40, 31;
shr.u32 %r42, %r40, 2;
add.s32 %r43, %r42, %r41;
mul.lo.s32 %r44, %r43, 24;
sub.s32 %r45, %r98, %r44;
shl.b32 %r46, %r45, 11;
sub.s32 %r47, %r99, %r46;
mad.wide.s32 %rd10, %r47, 2, %rd3;
mov.u32 %r36, 0x0;
mov.u32 %r37, 0x0;
mov.u32 %r38, 0x0;
mov.u32 %r39, 0x0;
ld.global.v4.b32 { %r36, %r37, %r38, %r39 }, [ %rd10 + 0 ];
and.b32 %r48, %r36, 2147450879;
and.b32 %r49, %r37, 2147450879;
and.b32 %r50, %r38, 2147450879;
and.b32 %r51, %r39, 2147450879;
mov.b32 {%rs9, %rs10}, %r48;
cvt.f32.bf16 %r52, %rs9;
cvt.f32.bf16 %r53, %rs10;
mov.b32 {%rs11, %rs12}, %r49;
cvt.f32.bf16 %r54, %rs11;
cvt.f32.bf16 %r55, %rs12;
mov.b32 {%rs13, %rs14}, %r50;
cvt.f32.bf16 %r56, %rs13;
cvt.f32.bf16 %r57, %rs14;
mov.b32 {%rs15, %rs16}, %r51;
cvt.f32.bf16 %r58, %rs15;
cvt.f32.bf16 %r59, %rs16;
max.f32 %r108, %r108, %r59;
max.f32 %r107, %r107, %r58;
max.f32 %r106, %r106, %r57;
max.f32 %r105, %r105, %r56;
max.f32 %r104, %r104, %r55;
max.f32 %r103, %r103, %r54;
max.f32 %r102, %r102, %r53;
max.f32 %r101, %r101, %r52;
add.s32 %r100, %r100, -1;
add.s32 %r99, %r99, 131072;
add.s32 %r98, %r98, 32;
setp.ne.b32 %p4, %r100, 0;
@%p4 bra $L__BB0_5;
$L__BB0_6:
and.b32 %r66, %r3, 31;
and.b32 %r67, %r3, 255;
shl.b32 %r68, %r1, 5;
mad.wide.s32 %rd12, %r68, 4, %rd6;
mad.wide.u32 %rd11, %r2, 4, %rd12;
$L__tmp0:
max.f32 %r69, %r101, %r102;
max.f32 %r70, %r69, %r103;
max.f32 %r71, %r70, %r104;
max.f32 %r72, %r71, %r105;
max.f32 %r73, %r72, %r106;
max.f32 %r74, %r73, %r107;
max.f32 %r75, %r74, %r108;
$L__tmp1:
shfl.sync.bfly.b32 %r76, %r75, 16, 31, -1;
$L__tmp2:
max.f32 %r77, %r75, %r76;
$L__tmp3:
shfl.sync.bfly.b32 %r78, %r77, 8, 31, -1;
$L__tmp4:
max.f32 %r79, %r77, %r78;
$L__tmp5:
shfl.sync.bfly.b32 %r80, %r79, 4, 31, -1;
$L__tmp6:
max.f32 %r81, %r79, %r80;
$L__tmp7:
shfl.sync.bfly.b32 %r82, %r81, 2, 31, -1;
$L__tmp8:
max.f32 %r83, %r81, %r82;
$L__tmp9:
shfl.sync.bfly.b32 %r84, %r83, 1, 31, -1;
$L__tmp10:
max.f32 %r61, %r83, %r84;
$L__tmp11:
setp.eq.b32 %p5, %r66, 0;
shr.u32 %r85, %r3, 3;
and.b32 %r86, %r85, 28;
mov.b32 %r87, global_smem;
add.s32 %r60, %r87, %r86;
@%p5 st.shared.b32 [ %r60 + 0 ], %r61;
bar.sync 0;
setp.lt.u32 %p6, %r67, 8;
shl.b32 %r88, %r67, 2;
add.s32 %r63, %r87, %r88;
@%p6 ld.shared.b32 %r62, [ %r63 + 0 ];
shfl.sync.bfly.b32 %r89, %r62, 4, 31, -1;
$L__tmp12:
max.f32 %r90, %r62, %r89;
$L__tmp13:
shfl.sync.bfly.b32 %r91, %r90, 2, 31, -1;
$L__tmp14:
max.f32 %r92, %r90, %r91;
$L__tmp15:
shfl.sync.bfly.b32 %r93, %r92, 1, 31, -1;
$L__tmp16:
max.f32 %r64, %r92, %r93;
$L__tmp17:
and.b32 %r94, %r3, 7;
setp.eq.b32 %p9, %r94, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r63 + 0 ], %r64;
bar.sync 0;
ld.shared.b32 %r65, [global_smem];
$L__tmp18:
setp.lt.u32 %p10, %r67, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd11 + 0 ], { %r65 };
ret;
$L__tmp19:
$L__func_end0:
}
"#
  | .wide => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _aligned_packed_amax(
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_0,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_1,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_2,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_3,
.param .u64 .ptr .global .align 1 _aligned_packed_amax_param_4
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<17>;
.reg .b32 %r<95>;
.reg .b64 %rd<18>;
$L__func_begin0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
shl.b32 %r7, %r3, 3;
and.b32 %r4, %r7, 2040;
sub.s32 %r8, 607, %r2;
shr.u32 %r9, %r8, 27;
add.s32 %r10, %r8, %r9;
shr.s32 %r5, %r10, 5;
sub.s32 %r11, 895, %r2;
shr.u32 %r12, %r11, 27;
add.s32 %r13, %r11, %r12;
shr.s32 %r6, %r13, 5;
mul.lo.s32 %r14, %r1, 1179648;
cvt.s64.s32 %rd1, %r14;
setp.gt.u32 %p1, %r2, 575;
mov.b32 %r87, 0f00000000;
shl.b64 %rd14, %rd1, 1;
shl.b32 %r85, %r2, 11;
mov.b32 %r88, %r87;
mov.b32 %r89, %r87;
mov.b32 %r90, %r87;
mov.b32 %r91, %r87;
mov.b32 %r92, %r87;
mov.b32 %r93, %r87;
mov.b32 %r94, %r87;
@%p1 bra $L__BB0_3;
ld.param.b64 %rd3, [_aligned_packed_amax_param_0];
add.s64 %rd6, %rd3, %rd14;
mad.wide.u32 %rd7, %r4, 2, %rd6;
max.s32 %r15, %r5, 1;
cvt.u64.u32 %rd16, %r15;
mad.wide.u32 %rd15, %r85, 2, %rd7;
mov.b32 %r87, 0f00000000;
mov.b32 %r88, %r87;
mov.b32 %r89, %r87;
mov.b32 %r90, %r87;
mov.b32 %r91, %r87;
mov.b32 %r92, %r87;
mov.b32 %r93, %r87;
mov.b32 %r94, %r87;
$L__BB0_2:
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
ld.global.v4.b32 { %r16, %r17, %r18, %r19 }, [ %rd15 + 0 ];
and.b32 %r20, %r16, 2147450879;
and.b32 %r21, %r17, 2147450879;
and.b32 %r22, %r18, 2147450879;
and.b32 %r23, %r19, 2147450879;
mov.b32 {%rs1, %rs2}, %r23;
cvt.f32.bf16 %r24, %rs2;
cvt.f32.bf16 %r25, %rs1;
mov.b32 {%rs3, %rs4}, %r22;
cvt.f32.bf16 %r26, %rs4;
mov.b32 {%rs5, %rs6}, %r20;
cvt.f32.bf16 %r27, %rs5;
cvt.f32.bf16 %r28, %rs6;
mov.b32 {%rs7, %rs8}, %r21;
cvt.f32.bf16 %r29, %rs7;
cvt.f32.bf16 %r30, %rs8;
cvt.f32.bf16 %r31, %rs3;
max.f32 %r91, %r91, %r31;
max.f32 %r90, %r90, %r30;
max.f32 %r89, %r89, %r29;
max.f32 %r88, %r88, %r28;
max.f32 %r87, %r87, %r27;
max.f32 %r92, %r92, %r26;
max.f32 %r93, %r93, %r25;
max.f32 %r94, %r94, %r24;
add.s64 %rd16, %rd16, -1;
add.s64 %rd15, %rd15, 131072;
setp.ne.b64 %p2, %rd16, 0;
@%p2 bra $L__BB0_2;
$L__BB0_3:
ld.param.b64 %rd5, [_aligned_packed_amax_param_2];
setp.ge.s32 %p3, %r5, %r6;
@%p3 bra $L__BB0_6;
ld.param.b64 %rd4, [_aligned_packed_amax_param_1];
add.s64 %rd8, %rd4, %rd14;
mad.wide.u32 %rd2, %r4, 2, %rd8;
cvt.s64.s32 %rd9, %r5;
cvt.s64.s32 %rd10, %r6;
sub.s64 %rd17, %rd10, %rd9;
shl.b32 %r32, %r5, 16;
add.s32 %r33, %r32, %r85;
add.s32 %r86, %r33, -1179648;
$L__BB0_5:
mad.wide.s32 %rd11, %r86, 2, %rd2;
mov.u32 %r34, 0x0;
mov.u32 %r35, 0x0;
mov.u32 %r36, 0x0;
mov.u32 %r37, 0x0;
ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd11 + 0 ];
and.b32 %r38, %r34, 2147450879;
and.b32 %r39, %r35, 2147450879;
and.b32 %r40, %r36, 2147450879;
and.b32 %r41, %r37, 2147450879;
mov.b32 {%rs9, %rs10}, %r41;
cvt.f32.bf16 %r42, %rs10;
cvt.f32.bf16 %r43, %rs9;
mov.b32 {%rs11, %rs12}, %r38;
cvt.f32.bf16 %r44, %rs11;
cvt.f32.bf16 %r45, %rs12;
mov.b32 {%rs13, %rs14}, %r39;
cvt.f32.bf16 %r46, %rs13;
cvt.f32.bf16 %r47, %rs14;
mov.b32 {%rs15, %rs16}, %r40;
cvt.f32.bf16 %r48, %rs15;
cvt.f32.bf16 %r49, %rs16;
max.f32 %r92, %r92, %r49;
max.f32 %r91, %r91, %r48;
max.f32 %r90, %r90, %r47;
max.f32 %r89, %r89, %r46;
max.f32 %r88, %r88, %r45;
max.f32 %r87, %r87, %r44;
max.f32 %r93, %r93, %r43;
max.f32 %r94, %r94, %r42;
add.s64 %rd17, %rd17, -1;
add.s32 %r86, %r86, 65536;
setp.ne.b64 %p4, %rd17, 0;
@%p4 bra $L__BB0_5;
$L__BB0_6:
and.b32 %r56, %r3, 31;
and.b32 %r57, %r3, 255;
shl.b32 %r58, %r1, 5;
mad.wide.s32 %rd13, %r58, 4, %rd5;
mad.wide.u32 %rd12, %r2, 4, %rd13;
$L__tmp0:
max.f32 %r59, %r87, %r88;
max.f32 %r60, %r59, %r89;
max.f32 %r61, %r60, %r90;
max.f32 %r62, %r61, %r91;
max.f32 %r63, %r62, %r92;
max.f32 %r64, %r63, %r93;
max.f32 %r65, %r64, %r94;
$L__tmp1:
shfl.sync.bfly.b32 %r66, %r65, 16, 31, -1;
$L__tmp2:
max.f32 %r67, %r65, %r66;
$L__tmp3:
shfl.sync.bfly.b32 %r68, %r67, 8, 31, -1;
$L__tmp4:
max.f32 %r69, %r67, %r68;
$L__tmp5:
shfl.sync.bfly.b32 %r70, %r69, 4, 31, -1;
$L__tmp6:
max.f32 %r71, %r69, %r70;
$L__tmp7:
shfl.sync.bfly.b32 %r72, %r71, 2, 31, -1;
$L__tmp8:
max.f32 %r73, %r71, %r72;
$L__tmp9:
shfl.sync.bfly.b32 %r74, %r73, 1, 31, -1;
$L__tmp10:
max.f32 %r51, %r73, %r74;
$L__tmp11:
setp.eq.b32 %p5, %r56, 0;
shr.u32 %r75, %r3, 3;
and.b32 %r76, %r75, 28;
mov.b32 %r77, global_smem;
add.s32 %r50, %r77, %r76;
@%p5 st.shared.b32 [ %r50 + 0 ], %r51;
bar.sync 0;
setp.lt.u32 %p6, %r57, 8;
shl.b32 %r78, %r57, 2;
add.s32 %r53, %r77, %r78;
@%p6 ld.shared.b32 %r52, [ %r53 + 0 ];
shfl.sync.bfly.b32 %r79, %r52, 4, 31, -1;
$L__tmp12:
max.f32 %r80, %r52, %r79;
$L__tmp13:
shfl.sync.bfly.b32 %r81, %r80, 2, 31, -1;
$L__tmp14:
max.f32 %r82, %r80, %r81;
$L__tmp15:
shfl.sync.bfly.b32 %r83, %r82, 1, 31, -1;
$L__tmp16:
max.f32 %r54, %r82, %r83;
$L__tmp17:
and.b32 %r84, %r3, 7;
setp.eq.b32 %p9, %r84, 0;
and.pred %p7, %p6, %p9;
@%p7 st.shared.b32 [ %r53 + 0 ], %r54;
bar.sync 0;
ld.shared.b32 %r55, [global_smem];
$L__tmp18:
setp.lt.u32 %p10, %r57, 32;
and.pred %p8, %p5, %p10;
@%p8 st.global.b32 [ %rd12 + 0 ], { %r55 };
ret;
$L__tmp19:
$L__func_end0:
}
"#

private def ptxMatches (expected actual : String) : Bool :=
  QKVTTIR.normalizePTX actual == (expected.splitOn "\n").filter (!·.isEmpty)

structure ScanArtifacts where
  referenceIR : String
  referencePTX : String
  nativeIR : String
  nativePTX : String

structure ScanProgram (w : Width) where
  artifacts : ScanArtifacts
  ir : ScanCertificate w artifacts.referenceIR artifacts.nativeIR
  referencePTX : ptxMatches (expectedReferencePTX w) artifacts.referencePTX = true
  nativePTX : ptxMatches (expectedNativePTX w) artifacts.nativePTX = true

def ScanProgram.check (w : Width) (a : ScanArtifacts) : Option (ScanProgram w) := do
  let ir ← certifyScan w a.referenceIR a.nativeIR
  if hr : ptxMatches (expectedReferencePTX w) a.referencePTX = true then
    if hn : ptxMatches (expectedNativePTX w) a.nativePTX = true then some ⟨a, ir, hr, hn⟩
    else none
  else none

#axiomsClean ScanProgram.check
#axiomsClean certifyScan
end VeriTile.Bench.Models.NanoGPT.Track1.QKVNative
