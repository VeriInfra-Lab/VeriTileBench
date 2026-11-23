import models.NanoGPT.Track1.QKVScanSchedule
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact emitted fragments for the three Hopper launches, including the
group buffer selection, bitmask decoder, warp-shuffle scale, and warp-local
FP8 layout conversion. Values are interpreted by the proved store models. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Grouped
open VeriTile.Triton.FP QKVAmax QKVLaunch QKVPack QKVNative
open QKVTTIR (Bank)

private def scanGraph : String := r#"
module {
tt.func public @_grouped_qkv_scan(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %partial0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<2048xf32>
%second_count = arith.constant 16 : i32
%batch = tt.get_program_id x : i32
%part = tt.get_program_id y : i32
%narrow = arith.cmpi slt, %batch, 3 : i32
%wide = arith.cmpi sge, %batch, 5 : i32
%local_batch = arith.select %wide, 5, 3 : i32
%local_batch_0 = arith.select %narrow, 0, %local_batch : i32
%local_batch_1 = arith.subi %batch, %local_batch_0 : i32
%partial = arith.select %wide, %partial2, %partial1 : !tt.ptr<f32>
%partial_2 = arith.select %narrow, %partial0, %partial : !tt.ptr<f32>
%lane = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
%first_count = arith.select %wide, 18, 9 : i32
%second_count_3 = arith.ori %narrow, %wide : i1
%second_count_4 = arith.cmpi slt, %part, %second_count : i32
%second_count_5 = arith.extui %second_count_4 : i1 to i32
%second_count_6 = arith.addi %second_count_5, 4 : i32
%second_count_7 = arith.select %second_count_3, 9, %second_count_6 : i32
%first_8 = arith.muli %batch, 1179648 : i32
%first_9 = tt.addptr %first, %first_8 : !tt.ptr<bf16>, i32
%second_10 = tt.addptr %second, %first_8 : !tt.ptr<bf16>, i32
%local_amax = arith.cmpi slt, %batch, 5 : i32
%stride = arith.select %local_amax, 192, 96 : i32
%blocks = arith.divsi %first_count, 3 : i32
%local_amax_11 = scf.for %block = 0 to %blocks step 1 iter_args(%local_amax_20 = %cst) -> (tensor<2048xf32>) : i32 {
%initial = arith.divsi %part, 24 : i32
%initial_21 = arith.muli %initial, 24 : i32
%initial_22 = arith.select %local_amax, %initial_21, 0 : i32
%initial_23 = arith.addi %part, %initial_22 : i32
%offset = arith.muli %block, %stride : i32
%offset_24 = arith.addi %offset, %initial_23 : i32
%values = arith.muli %offset_24, 2048 : i32
%values_25 = tt.addptr %first_9, %values : !tt.ptr<bf16>, i32
%values_26 = tt.splat %values_25 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_27 = tt.addptr %values_26, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_28 = tt.load %values_27 : tensor<2048x!tt.ptr<bf16>>
%values_29 = arith.extf %values_28 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude = arith.addf %values_29, %cst : tensor<2048xf32>
%magnitude_30 = math.absf %magnitude : tensor<2048xf32>
%local_amax_31 = arith.maxnumf %local_amax_20, %magnitude_30 : tensor<2048xf32>
%tile = arith.addi %part, 32 : i32
%initial_32 = arith.divsi %tile, 24 : i32
%initial_33 = arith.muli %initial_32, 24 : i32
%initial_34 = arith.select %local_amax, %initial_33, 0 : i32
%initial_35 = arith.addi %tile, %initial_34 : i32
%offset_36 = arith.addi %offset, %initial_35 : i32
%values_37 = arith.muli %offset_36, 2048 : i32
%values_38 = tt.addptr %first_9, %values_37 : !tt.ptr<bf16>, i32
%values_39 = tt.splat %values_38 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_40 = tt.addptr %values_39, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_41 = tt.load %values_40 : tensor<2048x!tt.ptr<bf16>>
%values_42 = arith.extf %values_41 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude_43 = arith.addf %values_42, %cst : tensor<2048xf32>
%magnitude_44 = math.absf %magnitude_43 : tensor<2048xf32>
%local_amax_45 = arith.maxnumf %local_amax_31, %magnitude_44 : tensor<2048xf32>
%tile_46 = arith.addi %part, 64 : i32
%initial_47 = arith.divsi %tile_46, 24 : i32
%initial_48 = arith.muli %initial_47, 24 : i32
%initial_49 = arith.select %local_amax, %initial_48, 0 : i32
%initial_50 = arith.addi %tile_46, %initial_49 : i32
%offset_51 = arith.addi %offset, %initial_50 : i32
%values_52 = arith.muli %offset_51, 2048 : i32
%values_53 = tt.addptr %first_9, %values_52 : !tt.ptr<bf16>, i32
%values_54 = tt.splat %values_53 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_55 = tt.addptr %values_54, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_56 = tt.load %values_55 : tensor<2048x!tt.ptr<bf16>>
%values_57 = arith.extf %values_56 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude_58 = arith.addf %values_57, %cst : tensor<2048xf32>
%magnitude_59 = math.absf %magnitude_58 : tensor<2048xf32>
%local_amax_60 = arith.maxnumf %local_amax_45, %magnitude_59 : tensor<2048xf32>
scf.yield %local_amax_60 : tensor<2048xf32>
}
%stride_12 = arith.select %second_count_3, 96, 192 : i32
%blocks_13 = arith.divsi %second_count_7, 3 : i32
%local_amax_14 = scf.for %block = 0 to %blocks_13 step 1 iter_args(%local_amax_20 = %local_amax_11) -> (tensor<2048xf32>) : i32 {
%initial = arith.divsi %part, 24 : i32
%initial_21 = arith.muli %initial, 24 : i32
%initial_22 = arith.select %second_count_3, 0, %initial_21 : i32
%initial_23 = arith.addi %part, %initial_22 : i32
%offset = arith.muli %block, %stride_12 : i32
%offset_24 = arith.addi %offset, %initial_23 : i32
%values = arith.muli %offset_24, 2048 : i32
%values_25 = tt.addptr %second_10, %values : !tt.ptr<bf16>, i32
%values_26 = tt.splat %values_25 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_27 = tt.addptr %values_26, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_28 = tt.load %values_27 : tensor<2048x!tt.ptr<bf16>>
%values_29 = arith.extf %values_28 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude = arith.addf %values_29, %cst : tensor<2048xf32>
%magnitude_30 = math.absf %magnitude : tensor<2048xf32>
%local_amax_31 = arith.maxnumf %local_amax_20, %magnitude_30 : tensor<2048xf32>
%tile = arith.addi %part, 32 : i32
%initial_32 = arith.divsi %tile, 24 : i32
%initial_33 = arith.muli %initial_32, 24 : i32
%initial_34 = arith.select %second_count_3, 0, %initial_33 : i32
%initial_35 = arith.addi %tile, %initial_34 : i32
%offset_36 = arith.addi %offset, %initial_35 : i32
%values_37 = arith.muli %offset_36, 2048 : i32
%values_38 = tt.addptr %second_10, %values_37 : !tt.ptr<bf16>, i32
%values_39 = tt.splat %values_38 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_40 = tt.addptr %values_39, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_41 = tt.load %values_40 : tensor<2048x!tt.ptr<bf16>>
%values_42 = arith.extf %values_41 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude_43 = arith.addf %values_42, %cst : tensor<2048xf32>
%magnitude_44 = math.absf %magnitude_43 : tensor<2048xf32>
%local_amax_45 = arith.maxnumf %local_amax_31, %magnitude_44 : tensor<2048xf32>
%tile_46 = arith.addi %part, 64 : i32
%initial_47 = arith.divsi %tile_46, 24 : i32
%initial_48 = arith.muli %initial_47, 24 : i32
%initial_49 = arith.select %second_count_3, 0, %initial_48 : i32
%initial_50 = arith.addi %tile_46, %initial_49 : i32
%offset_51 = arith.addi %offset, %initial_50 : i32
%values_52 = arith.muli %offset_51, 2048 : i32
%values_53 = tt.addptr %second_10, %values_52 : !tt.ptr<bf16>, i32
%values_54 = tt.splat %values_53 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_55 = tt.addptr %values_54, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_56 = tt.load %values_55 : tensor<2048x!tt.ptr<bf16>>
%values_57 = arith.extf %values_56 : tensor<2048xbf16> to tensor<2048xf32>
%magnitude_58 = arith.addf %values_57, %cst : tensor<2048xf32>
%magnitude_59 = math.absf %magnitude_58 : tensor<2048xf32>
%local_amax_60 = arith.maxnumf %local_amax_45, %magnitude_59 : tensor<2048xf32>
scf.yield %local_amax_60 : tensor<2048xf32>
}
%local_amax_15 = arith.remsi %second_count_7, 3 : i32
%local_amax_16 = arith.cmpi sgt, %local_amax_15, 0 : i32
%local_amax_17 = scf.if %local_amax_16 -> (tensor<2048xf32>) {
%initial = arith.divsi %part, 24 : i32
%initial_20 = arith.muli %initial, 24 : i32
%initial_21 = arith.select %second_count_3, 0, %initial_20 : i32
%initial_22 = arith.addi %part, %initial_21 : i32
%offset = arith.muli %blocks_13, %stride_12 : i32
%offset_23 = arith.addi %offset, %initial_22 : i32
%values = arith.muli %offset_23, 2048 : i32
%values_24 = tt.addptr %second_10, %values : !tt.ptr<bf16>, i32
%values_25 = tt.splat %values_24 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_26 = tt.addptr %values_25, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_27 = tt.load %values_26 : tensor<2048x!tt.ptr<bf16>>
%values_28 = arith.extf %values_27 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_29 = arith.addf %values_28, %cst : tensor<2048xf32>
%local_amax_30 = math.absf %local_amax_29 : tensor<2048xf32>
%local_amax_31 = arith.maxnumf %local_amax_14, %local_amax_30 : tensor<2048xf32>
scf.yield %local_amax_31 : tensor<2048xf32>
} else {
scf.yield %local_amax_14 : tensor<2048xf32>
}
%local_amax_18 = arith.cmpi sgt, %local_amax_15, 1 : i32
%local_amax_19 = scf.if %local_amax_18 -> (tensor<2048xf32>) {
%tile = arith.addi %part, 32 : i32
%initial = arith.divsi %tile, 24 : i32
%initial_20 = arith.muli %initial, 24 : i32
%initial_21 = arith.select %second_count_3, 0, %initial_20 : i32
%initial_22 = arith.addi %tile, %initial_21 : i32
%offset = arith.muli %blocks_13, %stride_12 : i32
%offset_23 = arith.addi %offset, %initial_22 : i32
%values = arith.muli %offset_23, 2048 : i32
%values_24 = tt.addptr %second_10, %values : !tt.ptr<bf16>, i32
%values_25 = tt.splat %values_24 : !tt.ptr<bf16> -> tensor<2048x!tt.ptr<bf16>>
%values_26 = tt.addptr %values_25, %lane : tensor<2048x!tt.ptr<bf16>>, tensor<2048xi32>
%values_27 = tt.load %values_26 : tensor<2048x!tt.ptr<bf16>>
%values_28 = arith.extf %values_27 : tensor<2048xbf16> to tensor<2048xf32>
%local_amax_29 = arith.addf %values_28, %cst : tensor<2048xf32>
%local_amax_30 = math.absf %local_amax_29 : tensor<2048xf32>
%local_amax_31 = arith.maxnumf %local_amax_17, %local_amax_30 : tensor<2048xf32>
scf.yield %local_amax_31 : tensor<2048xf32>
} else {
scf.yield %local_amax_17 : tensor<2048xf32>
}
%0 = arith.muli %local_batch_1, 32 : i32
%1 = tt.addptr %partial_2, %0 : !tt.ptr<f32>, i32
%2 = tt.addptr %1, %part : !tt.ptr<f32>, i32
%3 = "tt.reduce"(%local_amax_19) <{axis = 0 : i32}> ({
^bb0(%arg5: f32, %arg6: f32):
%4 = arith.maxnumf %arg5, %arg6 : f32
tt.reduce.return %4 : f32
}) : (tensor<2048xf32>) -> f32
tt.store %2, %3 : !tt.ptr<f32>
tt.return
}
}
"#

private def scanPtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _grouped_qkv_scan(
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_0,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_1,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_2,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_3,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_4,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_5,
.param .u64 .ptr .global .align 1 _grouped_qkv_scan_param_6
)
.reqntid 256
{
.reg .pred %p<16>;
.reg .b16 %rs<75>;
.reg .b32 %r<243>;
.reg .b64 %rd<38>;
$L__func_begin0:
ld.param.b64 %rd15, [_grouped_qkv_scan_param_4];
ld.param.b64 %rd14, [_grouped_qkv_scan_param_3];
ld.param.b64 %rd13, [_grouped_qkv_scan_param_2];
ld.param.b64 %rd12, [_grouped_qkv_scan_param_1];
ld.param.b64 %rd16, [_grouped_qkv_scan_param_0];
$L__tmp0:
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
setp.gt.u32 %p1, %r1, 4;
mov.u32 %r3, %tid.x;
shl.b32 %r14, %r3, 3;
and.b32 %r15, %r14, 2040;
mul.lo.s32 %r16, %r1, 1179648;
cvt.s64.s32 %rd1, %r16;
mad.wide.s32 %rd17, %r16, 2, %rd16;
setp.lt.u32 %p2, %r1, 5;
$L__tmp1:
selp.b64 %rd4, 786432, 393216, %p2;
$L__tmp2:
selp.b64 %rd35, 6, 3, %p1;
$L__tmp3:
cvt.u16.u32 %rs3, %r2;
mul.hi.u16 %rs4, %rs3, -21845;
shr.u16 %rs5, %rs4, 4;
mul.lo.s16 %rs6, %rs5, 24;
sub.s16 %rs7, %rs3, %rs6;
cvt.u32.u16 %r17, %rs7;
sub.s32 %r4, %r2, %r17;
selp.b32 %r18, %r4, 0, %p2;
add.s32 %r19, %r18, %r2;
add.s32 %r5, %r2, 32;
mul.hi.u32 %r20, %r5, 178956971;
mul.lo.s32 %r6, %r20, 24;
selp.b32 %r21, %r6, 0, %p2;
add.s32 %r22, %r21, %r5;
add.s32 %r7, %r2, 64;
mul.hi.u32 %r23, %r7, 178956971;
mul.lo.s32 %r8, %r23, 24;
selp.b32 %r24, %r8, 0, %p2;
add.s32 %r25, %r24, %r7;
cvt.u64.u32 %rd2, %r15;
mad.wide.u32 %rd34, %r15, 2, %rd17;
mul.wide.u32 %rd3, %r25, 4096;
mul.wide.u32 %rd5, %r22, 4096;
mul.wide.u32 %rd6, %r19, 4096;
mov.b32 %r235, 0f00000000;
mov.b32 %r236, %r235;
mov.b32 %r237, %r235;
mov.b32 %r238, %r235;
mov.b32 %r239, %r235;
mov.b32 %r240, %r235;
mov.b32 %r241, %r235;
mov.b32 %r242, %r235;
$L__BB0_1:
add.s64 %rd18, %rd34, %rd6;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd18 + 0 ];
and.b32 %r38, %r26, 2147450879;
and.b32 %r39, %r27, 2147450879;
and.b32 %r40, %r28, 2147450879;
and.b32 %r41, %r29, 2147450879;
add.s64 %rd19, %rd34, %rd5;
mov.u32 %r30, 0x0;
mov.u32 %r31, 0x0;
mov.u32 %r32, 0x0;
mov.u32 %r33, 0x0;
ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd19 + 0 ];
and.b32 %r42, %r30, 2147450879;
and.b32 %r43, %r31, 2147450879;
and.b32 %r44, %r32, 2147450879;
and.b32 %r45, %r33, 2147450879;
add.s64 %rd20, %rd34, %rd3;
mov.u32 %r34, 0x0;
mov.u32 %r35, 0x0;
mov.u32 %r36, 0x0;
mov.u32 %r37, 0x0;
ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd20 + 0 ];
and.b32 %r46, %r34, 2147450879;
and.b32 %r47, %r35, 2147450879;
and.b32 %r48, %r36, 2147450879;
and.b32 %r49, %r37, 2147450879;
mov.b32 {%rs8, %rs9}, %r41;
cvt.f32.bf16 %r50, %rs9;
cvt.f32.bf16 %r51, %rs8;
mov.b32 {%rs10, %rs11}, %r38;
cvt.f32.bf16 %r52, %rs10;
cvt.f32.bf16 %r53, %rs11;
mov.b32 {%rs12, %rs13}, %r39;
cvt.f32.bf16 %r54, %rs12;
cvt.f32.bf16 %r55, %rs13;
mov.b32 {%rs14, %rs15}, %r40;
cvt.f32.bf16 %r56, %rs14;
cvt.f32.bf16 %r57, %rs15;
max.f32 %r58, %r240, %r57;
max.f32 %r59, %r239, %r56;
max.f32 %r60, %r238, %r55;
max.f32 %r61, %r237, %r54;
max.f32 %r62, %r236, %r53;
max.f32 %r63, %r235, %r52;
max.f32 %r64, %r241, %r51;
max.f32 %r65, %r242, %r50;
mov.b32 {%rs16, %rs17}, %r44;
cvt.f32.bf16 %r66, %rs17;
cvt.f32.bf16 %r67, %rs16;
mov.b32 {%rs18, %rs19}, %r43;
cvt.f32.bf16 %r68, %rs19;
cvt.f32.bf16 %r69, %rs18;
mov.b32 {%rs20, %rs21}, %r42;
cvt.f32.bf16 %r70, %rs21;
cvt.f32.bf16 %r71, %rs20;
mov.b32 {%rs22, %rs23}, %r45;
cvt.f32.bf16 %r72, %rs22;
cvt.f32.bf16 %r73, %rs23;
max.f32 %r74, %r65, %r73;
max.f32 %r75, %r64, %r72;
max.f32 %r76, %r63, %r71;
max.f32 %r77, %r62, %r70;
max.f32 %r78, %r61, %r69;
max.f32 %r79, %r60, %r68;
max.f32 %r80, %r59, %r67;
max.f32 %r81, %r58, %r66;
mov.b32 {%rs24, %rs25}, %r49;
cvt.f32.bf16 %r82, %rs25;
cvt.f32.bf16 %r83, %rs24;
mov.b32 {%rs26, %rs27}, %r46;
cvt.f32.bf16 %r84, %rs26;
cvt.f32.bf16 %r85, %rs27;
mov.b32 {%rs28, %rs29}, %r47;
cvt.f32.bf16 %r86, %rs28;
cvt.f32.bf16 %r87, %rs29;
mov.b32 {%rs30, %rs31}, %r48;
cvt.f32.bf16 %r88, %rs30;
cvt.f32.bf16 %r89, %rs31;
max.f32 %r240, %r81, %r89;
max.f32 %r239, %r80, %r88;
max.f32 %r238, %r79, %r87;
max.f32 %r237, %r78, %r86;
max.f32 %r236, %r77, %r85;
max.f32 %r235, %r76, %r84;
max.f32 %r241, %r75, %r83;
max.f32 %r242, %r74, %r82;
add.s64 %rd35, %rd35, -1;
add.s64 %rd34, %rd34, %rd4;
setp.ne.b64 %p3, %rd35, 0;
@%p3 bra $L__BB0_1;
$L__tmp4:
add.s32 %r90, %r1, -5;
setp.lt.u32 %p4, %r90, -2;
setp.lt.u32 %p5, %r2, 16;
selp.b32 %r91, 5, 4, %p5;
selp.b32 %r92, 9, %r91, %p4;
shl.b64 %rd21, %rd1, 1;
add.s64 %rd7, %rd12, %rd21;
$L__tmp5:
selp.b32 %r9, 96, 192, %p4;
cvt.u16.u32 %rs1, %r92;
mul.lo.s16 %rs32, %rs1, 86;
shr.u16 %rs33, %rs32, 8;
cvt.u32.u16 %r10, %rs33;
selp.b32 %r93, 0, %r4, %p4;
add.s32 %r11, %r93, %r2;
selp.b32 %r94, 0, %r6, %p4;
add.s32 %r12, %r94, %r5;
selp.b32 %r95, 0, %r8, %p4;
add.s32 %r96, %r95, %r7;
shl.b64 %rd22, %rd2, 1;
add.s64 %rd36, %rd7, %rd22;
cvt.u64.u16 %rd37, %rs33;
mul.wide.u32 %rd8, %r96, 4096;
mul.wide.u32 %rd9, %r9, 4096;
mul.wide.u32 %rd10, %r12, 4096;
mul.wide.u32 %rd11, %r11, 4096;
$L__BB0_3:
add.s64 %rd23, %rd36, %rd11;
mov.u32 %r97, 0x0;
mov.u32 %r98, 0x0;
mov.u32 %r99, 0x0;
mov.u32 %r100, 0x0;
ld.global.v4.b32 { %r97, %r98, %r99, %r100 }, [ %rd23 + 0 ];
and.b32 %r109, %r97, 2147450879;
and.b32 %r110, %r98, 2147450879;
and.b32 %r111, %r99, 2147450879;
and.b32 %r112, %r100, 2147450879;
add.s64 %rd24, %rd36, %rd10;
mov.u32 %r101, 0x0;
mov.u32 %r102, 0x0;
mov.u32 %r103, 0x0;
mov.u32 %r104, 0x0;
ld.global.v4.b32 { %r101, %r102, %r103, %r104 }, [ %rd24 + 0 ];
and.b32 %r113, %r101, 2147450879;
and.b32 %r114, %r102, 2147450879;
and.b32 %r115, %r103, 2147450879;
and.b32 %r116, %r104, 2147450879;
add.s64 %rd25, %rd36, %rd8;
mov.u32 %r105, 0x0;
mov.u32 %r106, 0x0;
mov.u32 %r107, 0x0;
mov.u32 %r108, 0x0;
ld.global.v4.b32 { %r105, %r106, %r107, %r108 }, [ %rd25 + 0 ];
and.b32 %r117, %r105, 2147450879;
and.b32 %r118, %r106, 2147450879;
and.b32 %r119, %r107, 2147450879;
and.b32 %r120, %r108, 2147450879;
mov.b32 {%rs34, %rs35}, %r112;
cvt.f32.bf16 %r121, %rs35;
cvt.f32.bf16 %r122, %rs34;
mov.b32 {%rs36, %rs37}, %r109;
cvt.f32.bf16 %r123, %rs36;
cvt.f32.bf16 %r124, %rs37;
mov.b32 {%rs38, %rs39}, %r110;
cvt.f32.bf16 %r125, %rs38;
cvt.f32.bf16 %r126, %rs39;
mov.b32 {%rs40, %rs41}, %r111;
cvt.f32.bf16 %r127, %rs40;
cvt.f32.bf16 %r128, %rs41;
max.f32 %r129, %r240, %r128;
max.f32 %r130, %r239, %r127;
max.f32 %r131, %r238, %r126;
max.f32 %r132, %r237, %r125;
max.f32 %r133, %r236, %r124;
max.f32 %r134, %r235, %r123;
max.f32 %r135, %r241, %r122;
max.f32 %r136, %r242, %r121;
mov.b32 {%rs42, %rs43}, %r115;
cvt.f32.bf16 %r137, %rs43;
cvt.f32.bf16 %r138, %rs42;
mov.b32 {%rs44, %rs45}, %r114;
cvt.f32.bf16 %r139, %rs45;
cvt.f32.bf16 %r140, %rs44;
mov.b32 {%rs46, %rs47}, %r113;
cvt.f32.bf16 %r141, %rs47;
cvt.f32.bf16 %r142, %rs46;
mov.b32 {%rs48, %rs49}, %r116;
cvt.f32.bf16 %r143, %rs48;
cvt.f32.bf16 %r144, %rs49;
max.f32 %r145, %r136, %r144;
max.f32 %r146, %r135, %r143;
max.f32 %r147, %r134, %r142;
max.f32 %r148, %r133, %r141;
max.f32 %r149, %r132, %r140;
max.f32 %r150, %r131, %r139;
max.f32 %r151, %r130, %r138;
max.f32 %r152, %r129, %r137;
mov.b32 {%rs50, %rs51}, %r120;
cvt.f32.bf16 %r153, %rs51;
cvt.f32.bf16 %r154, %rs50;
mov.b32 {%rs52, %rs53}, %r117;
cvt.f32.bf16 %r155, %rs52;
cvt.f32.bf16 %r156, %rs53;
mov.b32 {%rs54, %rs55}, %r118;
cvt.f32.bf16 %r157, %rs54;
cvt.f32.bf16 %r158, %rs55;
mov.b32 {%rs56, %rs57}, %r119;
cvt.f32.bf16 %r159, %rs56;
cvt.f32.bf16 %r160, %rs57;
max.f32 %r240, %r152, %r160;
max.f32 %r239, %r151, %r159;
max.f32 %r238, %r150, %r158;
max.f32 %r237, %r149, %r157;
max.f32 %r236, %r148, %r156;
max.f32 %r235, %r147, %r155;
max.f32 %r241, %r146, %r154;
max.f32 %r242, %r145, %r153;
add.s64 %rd37, %rd37, -1;
add.s64 %rd36, %rd36, %rd9;
setp.ne.b64 %p6, %rd37, 0;
@%p6 bra $L__BB0_3;
mul.lo.s16 %rs58, %rs33, 3;
sub.s16 %rs2, %rs1, %rs58;
setp.eq.b16 %p7, %rs2, 0;
@%p7 bra $L__BB0_7;
mul.lo.s32 %r13, %r9, %r10;
add.s32 %r165, %r11, %r13;
shl.b32 %r166, %r165, 11;
mad.wide.u32 %rd27, %r166, 2, %rd7;
add.s64 %rd26, %rd27, %rd22;
mov.u32 %r161, 0x0;
mov.u32 %r162, 0x0;
mov.u32 %r163, 0x0;
mov.u32 %r164, 0x0;
ld.global.v4.b32 { %r161, %r162, %r163, %r164 }, [ %rd26 + 0 ];
and.b32 %r167, %r161, 2147450879;
and.b32 %r168, %r162, 2147450879;
and.b32 %r169, %r163, 2147450879;
and.b32 %r170, %r164, 2147450879;
mov.b32 {%rs59, %rs60}, %r167;
cvt.f32.bf16 %r171, %rs59;
cvt.f32.bf16 %r172, %rs60;
mov.b32 {%rs61, %rs62}, %r168;
cvt.f32.bf16 %r173, %rs61;
cvt.f32.bf16 %r174, %rs62;
mov.b32 {%rs63, %rs64}, %r169;
cvt.f32.bf16 %r175, %rs63;
cvt.f32.bf16 %r176, %rs64;
mov.b32 {%rs65, %rs66}, %r170;
cvt.f32.bf16 %r177, %rs65;
cvt.f32.bf16 %r178, %rs66;
max.f32 %r242, %r242, %r178;
max.f32 %r241, %r241, %r177;
max.f32 %r240, %r240, %r176;
max.f32 %r239, %r239, %r175;
max.f32 %r238, %r238, %r174;
max.f32 %r237, %r237, %r173;
max.f32 %r236, %r236, %r172;
max.f32 %r235, %r235, %r171;
setp.eq.b16 %p8, %rs2, 1;
@%p8 bra $L__BB0_7;
add.s32 %r183, %r13, %r12;
shl.b32 %r184, %r183, 11;
mad.wide.u32 %rd29, %r184, 2, %rd7;
add.s64 %rd28, %rd29, %rd22;
mov.u32 %r179, 0x0;
mov.u32 %r180, 0x0;
mov.u32 %r181, 0x0;
mov.u32 %r182, 0x0;
ld.global.v4.b32 { %r179, %r180, %r181, %r182 }, [ %rd28 + 0 ];
and.b32 %r185, %r179, 2147450879;
and.b32 %r186, %r180, 2147450879;
and.b32 %r187, %r181, 2147450879;
and.b32 %r188, %r182, 2147450879;
mov.b32 {%rs67, %rs68}, %r185;
cvt.f32.bf16 %r189, %rs67;
cvt.f32.bf16 %r190, %rs68;
mov.b32 {%rs69, %rs70}, %r186;
cvt.f32.bf16 %r191, %rs69;
cvt.f32.bf16 %r192, %rs70;
mov.b32 {%rs71, %rs72}, %r187;
cvt.f32.bf16 %r193, %rs71;
cvt.f32.bf16 %r194, %rs72;
mov.b32 {%rs73, %rs74}, %r188;
cvt.f32.bf16 %r195, %rs73;
cvt.f32.bf16 %r196, %rs74;
max.f32 %r242, %r242, %r196;
max.f32 %r241, %r241, %r195;
max.f32 %r240, %r240, %r194;
max.f32 %r239, %r239, %r193;
max.f32 %r238, %r238, %r192;
max.f32 %r237, %r237, %r191;
max.f32 %r236, %r236, %r190;
max.f32 %r235, %r235, %r189;
$L__tmp6:
$L__BB0_7:
setp.lt.u32 %p13, %r1, 3;
and.b32 %r203, %r3, 31;
and.b32 %r204, %r3, 255;
selp.b64 %rd31, %rd15, %rd14, %p1;
selp.b64 %rd32, %rd13, %rd31, %p13;
selp.b32 %r205, 134217723, 134217725, %p1;
selp.b32 %r206, 0, %r205, %p13;
add.s32 %r207, %r206, %r1;
shl.b32 %r208, %r207, 5;
mad.wide.s32 %rd33, %r208, 4, %rd32;
mad.wide.u32 %rd30, %r2, 4, %rd33;
$L__tmp7:
max.f32 %r209, %r235, %r236;
max.f32 %r210, %r209, %r237;
max.f32 %r211, %r210, %r238;
max.f32 %r212, %r211, %r239;
max.f32 %r213, %r212, %r240;
max.f32 %r214, %r213, %r241;
max.f32 %r215, %r214, %r242;
$L__tmp8:
shfl.sync.bfly.b32 %r216, %r215, 16, 31, -1;
$L__tmp9:
max.f32 %r217, %r215, %r216;
$L__tmp10:
shfl.sync.bfly.b32 %r218, %r217, 8, 31, -1;
$L__tmp11:
max.f32 %r219, %r217, %r218;
$L__tmp12:
shfl.sync.bfly.b32 %r220, %r219, 4, 31, -1;
$L__tmp13:
max.f32 %r221, %r219, %r220;
$L__tmp14:
shfl.sync.bfly.b32 %r222, %r221, 2, 31, -1;
$L__tmp15:
max.f32 %r223, %r221, %r222;
$L__tmp16:
shfl.sync.bfly.b32 %r224, %r223, 1, 31, -1;
$L__tmp17:
max.f32 %r198, %r223, %r224;
$L__tmp18:
setp.eq.b32 %p9, %r203, 0;
shr.u32 %r225, %r3, 3;
and.b32 %r226, %r225, 28;
mov.b32 %r227, global_smem;
add.s32 %r197, %r227, %r226;
@%p9 st.shared.b32 [ %r197 + 0 ], %r198;
bar.sync 0;
setp.lt.u32 %p10, %r204, 8;
shl.b32 %r228, %r204, 2;
add.s32 %r200, %r227, %r228;
@%p10 ld.shared.b32 %r199, [ %r200 + 0 ];
shfl.sync.bfly.b32 %r229, %r199, 4, 31, -1;
$L__tmp19:
max.f32 %r230, %r199, %r229;
$L__tmp20:
shfl.sync.bfly.b32 %r231, %r230, 2, 31, -1;
$L__tmp21:
max.f32 %r232, %r230, %r231;
$L__tmp22:
shfl.sync.bfly.b32 %r233, %r232, 1, 31, -1;
$L__tmp23:
max.f32 %r201, %r232, %r233;
$L__tmp24:
and.b32 %r234, %r3, 7;
setp.eq.b32 %p14, %r234, 0;
and.pred %p11, %p10, %p14;
@%p11 st.shared.b32 [ %r200 + 0 ], %r201;
bar.sync 0;
ld.shared.b32 %r202, [global_smem];
$L__tmp25:
setp.lt.u32 %p15, %r204, 32;
and.pred %p12, %p9, %p15;
@%p12 st.global.b32 [ %rd30 + 0 ], { %r202 };
ret;
$L__tmp26:
$L__func_end0:
}
"#

private def scaleGraph : String := r#"
module {
tt.func public @_grouped_qkv_scale(%partial0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %partial2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant 0.00223214296 : f32
%cst_0 = arith.constant 9.99999996E-13 : f32
%batch = tt.get_program_id x : i32
%narrow = arith.cmpi slt, %batch, 3 : i32
%wide = arith.cmpi sge, %batch, 5 : i32
%local_batch = arith.select %wide, 5, 3 : i32
%local_batch_1 = arith.select %narrow, 0, %local_batch : i32
%local_batch_2 = arith.subi %batch, %local_batch_1 : i32
%partial = arith.select %wide, %partial2, %partial1 : !tt.ptr<f32>
%partial_3 = arith.select %narrow, %partial0, %partial : !tt.ptr<f32>
%scales = arith.select %wide, %scale2, %scale1 : !tt.ptr<f32>
%scales_4 = arith.select %narrow, %scale0, %scales : !tt.ptr<f32>
%parts = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%partials = arith.muli %local_batch_2, 32 : i32
%partials_5 = tt.addptr %partial_3, %partials : !tt.ptr<f32>, i32
%partials_6 = tt.splat %partials_5 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
%partials_7 = tt.addptr %partials_6, %parts : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
%partials_8 = tt.load %partials_7 : tensor<32x!tt.ptr<f32>>
%amax = "tt.reduce"(%partials_8) <{axis = 0 : i32}> ({
^bb0(%arg6: f32, %arg7: f32):
%amax_10 = arith.maximumf %arg6, %arg7 : f32
tt.reduce.return %amax_10 : f32
}) : (tensor<32xf32>) -> f32
%floored = arith.cmpf olt, %amax, %cst_0 : f32
%floored_9 = arith.select %floored, %cst_0, %amax : f32
%0 = tt.addptr %scales_4, %local_batch_2 : !tt.ptr<f32>, i32
%1 = arith.mulf %floored_9, %cst : f32
tt.store %0, %1 : !tt.ptr<f32>
tt.return
}
}
"#

private def scalePtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _grouped_qkv_scale(
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_0,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_1,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_2,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_3,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_4,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_5,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_6,
.param .u64 .ptr .global .align 1 _grouped_qkv_scale_param_7
)
.reqntid 32
{
.reg .pred %p<5>;
.reg .b32 %r<21>;
.reg .b64 %rd<14>;
$L__func_begin0:
ld.param.b64 %rd3, [_grouped_qkv_scale_param_0];
ld.param.b64 %rd4, [_grouped_qkv_scale_param_1];
$L__tmp0:
mov.u32 %r3, %ctaid.x;
setp.lt.u32 %p2, %r3, 3;
ld.param.b64 %rd5, [_grouped_qkv_scale_param_2];
setp.gt.u32 %p3, %r3, 4;
ld.param.b64 %rd6, [_grouped_qkv_scale_param_3];
selp.b32 %r4, -5, -3, %p3;
ld.param.b64 %rd7, [_grouped_qkv_scale_param_4];
selp.b32 %r5, 0, %r4, %p2;
ld.param.b64 %rd8, [_grouped_qkv_scale_param_5];
add.s32 %r6, %r5, %r3;
selp.b64 %rd9, %rd5, %rd4, %p3;
selp.b64 %rd10, %rd3, %rd9, %p2;
selp.b64 %rd11, %rd8, %rd7, %p3;
selp.b64 %rd12, %rd6, %rd11, %p2;
mov.u32 %r7, %tid.x;
and.b32 %r8, %r7, 31;
shl.b32 %r9, %r6, 5;
mad.wide.s32 %rd13, %r9, 4, %rd10;
mad.wide.u32 %rd1, %r8, 4, %rd13;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
shfl.sync.bfly.b32 %r10, %r1, 16, 31, -1;
$L__tmp1:
max.NaN.f32 %r11, %r1, %r10;
$L__tmp2:
shfl.sync.bfly.b32 %r12, %r11, 8, 31, -1;
$L__tmp3:
max.NaN.f32 %r13, %r11, %r12;
$L__tmp4:
shfl.sync.bfly.b32 %r14, %r13, 4, 31, -1;
$L__tmp5:
max.NaN.f32 %r15, %r13, %r14;
$L__tmp6:
shfl.sync.bfly.b32 %r16, %r15, 2, 31, -1;
$L__tmp7:
max.NaN.f32 %r17, %r15, %r16;
$L__tmp8:
shfl.sync.bfly.b32 %r18, %r17, 1, 31, -1;
$L__tmp9:
max.NaN.f32 %r19, %r17, %r18;
$L__tmp10:
setp.lt.f32 %p4, %r19, 0f2B8CBCCC;
selp.f32 %r20, 0f2B8CBCCC, %r19, %p4;
mad.wide.s32 %rd2, %r6, 4, %rd12;
mul.f32 %r2, %r20, 0f3B124925;
setp.eq.b32 %p1, %r8, 0;
@%p1 st.global.b32 [ %rd2 + 0 ], { %r2 };
ret;
$L__tmp11:
$L__func_end0:
}
"#

private def packGraph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_grouped_qkv_pack(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %row1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %row2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %col2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scale0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %scale2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%true = arith.constant true
%cst = arith.constant dense<768> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32, #blocked>
%cst_1 = arith.constant dense<-4.480000e+02> : tensor<64x128xf32, #blocked>
%band = tt.get_program_id z : i32
%start_n = tt.get_program_id y : i32
%start_n_2 = arith.muli %start_n, 128 : i32
%narrow = arith.cmpi slt, %band, 12 : i32
%wide = arith.cmpi sge, %band, 18 : i32
%eligible = arith.shli 2, %band : i32
%eligible_3 = arith.subi %eligible, 1 : i32
%eligible_4 = arith.andi %eligible_3, 17076497 : i32
%0:2 = tt.elementwise_inline_asm "{ popc.b32 $0, $2; bfind.u32 $1, $2; }" {constraints = "=r,=r,r", packed_element = 1 : i32, pure = true} %eligible_4 : i32 -> i32, i32
%batch = arith.subi %0#0, 1 : i32
%local_batch = arith.select %wide, 5, 3 : i32
%local_batch_5 = arith.select %narrow, 0, %local_batch : i32
%local_batch_6 = arith.subi %batch, %local_batch_5 : i32
%start_m = arith.subi %band, %0#1 : i32
%start_m_7 = arith.muli %start_m, 384 : i32
%start_m_8 = tt.get_program_id x : i32
%start_m_9 = arith.muli %start_m_8, 64 : i32
%start_m_10 = arith.addi %start_m_7, %start_m_9 : i32
%M = arith.select %wide, 2304, 1152 : i32
%M_11 = arith.select %narrow, 1536, %M : i32
%first_rows = arith.select %wide, 1536, 768 : i32
%qk = arith.cmpi slt, %start_m_10, %first_rows : i32
%bank = arith.select %qk, %first, %second : !tt.ptr<bf16>
%crop = arith.cmpi slt, %band, 18 : i32
%crop_12 = arith.ori %narrow, %wide : i1
%crop_13 = arith.xori %crop_12, %true : i1
%crop_14 = arith.select %qk, %crop, %crop_13 : i1
%bank_row = arith.select %qk, 0, %first_rows : i32
%bank_row_15 = arith.subi %start_m_10, %bank_row : i32
%bank_row_16 = arith.select %crop_14, 2, 1 : i32
%bank_row_17 = arith.muli %bank_row_15, %bank_row_16 : i32
%source = arith.muli %batch, 1179648 : i32
%source_18 = arith.muli %bank_row_17, 768 : i32
%source_19 = arith.addi %source, %source_18 : i32
%source_20 = arith.addi %source_19, %start_n_2 : i32
%source_21 = tt.addptr %bank, %source_20 : !tt.ptr<bf16>, i32
%rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%values = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%values_22 = arith.muli %values, %cst : tensor<64x1xi32, #blocked>
%values_23 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%values_24 = tt.broadcast %values_22 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%values_25 = tt.broadcast %values_23 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%values_26 = arith.addi %values_24, %values_25 : tensor<64x128xi32, #blocked>
%values_27 = tt.splat %source_21 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_28 = tt.addptr %values_27, %values_26 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%values_29 = tt.load %values_28 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_30 = arith.extf %values_29 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
%scale = arith.select %wide, %scale2, %scale1 : !tt.ptr<f32>
%scale_31 = arith.select %narrow, %scale0, %scale : !tt.ptr<f32>
%scaled = tt.addptr %scale_31, %local_batch_6 : !tt.ptr<f32>, i32
%scaled_32 = tt.load %scaled : !tt.ptr<f32>
%scaled_33 = tt.splat %scaled_32 : f32 -> tensor<64x128xf32, #blocked>
%scaled_34 = arith.divf %values_30, %scaled_33 : tensor<64x128xf32, #blocked>
%quantized = arith.minnumf %scaled_34, %cst_0 : tensor<64x128xf32, #blocked>
%quantized_35 = arith.maxnumf %quantized, %cst_1 : tensor<64x128xf32, #blocked>
%quantized_36 = tt.fp_to_fp %quantized_35, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
%row = arith.select %wide, %row2, %row1 : !tt.ptr<f8E4M3FN>
%row_37 = arith.select %narrow, %row0, %row : !tt.ptr<f8E4M3FN>
%col = arith.select %wide, %col2, %col1 : !tt.ptr<f8E4M3FN>
%col_38 = arith.select %narrow, %col0, %col : !tt.ptr<f8E4M3FN>
%out = arith.muli %local_batch_6, %M_11 : i32
%out_39 = arith.muli %out, 768 : i32
%1 = arith.muli %start_m_10, 768 : i32
%2 = arith.addi %out_39, %1 : i32
%3 = arith.addi %2, %start_n_2 : i32
%4 = tt.addptr %row_37, %3 : !tt.ptr<f8E4M3FN>, i32
%5 = tt.splat %4 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%6 = tt.addptr %5, %values_26 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %6, %quantized_36 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %quantized_36 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
%trans_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%7 = arith.muli %start_n_2, %M_11 : i32
%8 = arith.addi %out_39, %7 : i32
%9 = arith.addi %8, %start_m_10 : i32
%10 = tt.addptr %col_38, %9 : !tt.ptr<f8E4M3FN>, i32
%11 = tt.expand_dims %trans_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%12 = tt.splat %M_11 : i32 -> tensor<1x128xi32, #blocked1>
%13 = arith.muli %11, %12 : tensor<1x128xi32, #blocked1>
%14 = tt.expand_dims %trans_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%15 = tt.broadcast %13 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%16 = tt.broadcast %14 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%17 = arith.addi %15, %16 : tensor<64x128xi32, #blocked1>
%18 = tt.splat %10 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%19 = tt.addptr %18, %17 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %19, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
tt.return
}
}
"#

private def packPtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _grouped_qkv_pack(
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_0,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_1,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_2,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_3,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_4,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_5,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_6,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_7,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_8,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_9,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_10,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_11,
.param .u64 .ptr .global .align 1 _grouped_qkv_pack_param_12
)
.reqntid 256
{
.reg .pred %p<10>;
.reg .b16 %rs<49>;
.reg .b32 %r<240>;
.reg .b64 %rd<41>;
$L__func_begin0:
ld.param.b64 %rd12, [_grouped_qkv_pack_param_0];
$L__tmp0:
mov.u32 %r69, %ctaid.z;
ld.param.b64 %rd13, [_grouped_qkv_pack_param_1];
ld.param.b64 %rd14, [_grouped_qkv_pack_param_2];
mov.u32 %r70, %ctaid.y;
shl.b32 %r71, %r70, 7;
ld.param.b64 %rd15, [_grouped_qkv_pack_param_3];
setp.lt.u32 %p1, %r69, 12;
ld.param.b64 %rd16, [_grouped_qkv_pack_param_4];
setp.gt.u32 %p2, %r69, 17;
ld.param.b64 %rd17, [_grouped_qkv_pack_param_5];
mov.b32 %r72, 2;
shl.b32 %r73, %r72, %r69;
ld.param.b64 %rd18, [_grouped_qkv_pack_param_6];
add.s32 %r74, %r73, -1;
ld.param.b64 %rd19, [_grouped_qkv_pack_param_7];
and.b32 %r3, %r74, 17076497;
ld.param.b64 %rd20, [_grouped_qkv_pack_param_8];
{ popc.b32 %r1, %r3; bfind.u32 %r2, %r3; }
ld.param.b64 %rd21, [_grouped_qkv_pack_param_9];
ld.param.b64 %rd22, [_grouped_qkv_pack_param_10];
add.s32 %r75, %r1, -1;
selp.b32 %r76, -5, -3, %p2;
selp.b32 %r77, 0, %r76, %p1;
add.s32 %r78, %r75, %r77;
sub.s32 %r79, %r69, %r2;
mov.u32 %r80, %ctaid.x;
shl.b32 %r81, %r80, 6;
mad.lo.s32 %r82, %r79, 384, %r81;
selp.b32 %r83, 2304, 1152, %p2;
selp.b32 %r84, 1536, %r83, %p1;
selp.b32 %r85, 1536, 768, %p2;
setp.ge.s32 %p3, %r82, %r85;
setp.lt.s32 %p4, %r82, %r85;
selp.b64 %rd23, %rd12, %rd13, %p4;
setp.lt.u32 %p5, %r69, 18;
add.s32 %r86, %r69, -12;
setp.lt.u32 %p6, %r86, 6;
and.pred %p7, %p3, %p6;
and.pred %p8, %p4, %p5;
or.pred %p9, %p8, %p7;
selp.b32 %r87, 0, %r85, %p4;
sub.s32 %r88, %r82, %r87;
selp.b32 %r89, 1, 0, %p9;
shl.b32 %r90, %r88, %r89;
mad.lo.s32 %r91, %r75, 1179648, %r71;
mad.lo.s32 %r92, %r90, 768, %r91;
mad.wide.s32 %rd24, %r92, 2, %rd23;
mov.u32 %r93, %tid.x;
shr.u32 %r94, %r93, 2;
and.b32 %r95, %r94, 62;
shl.b32 %r96, %r93, 4;
and.b32 %r97, %r96, 112;
mul.lo.s32 %r98, %r95, 768;
or.b32 %r99, %r98, %r97;
cvt.u64.u32 %rd25, %r99;
mad.wide.u32 %rd1, %r99, 2, %rd24;
add.s64 %rd2, %rd1, 16;
add.s64 %rd3, %rd1, 1536;
or.b32 %r100, %r99, 8;
mad.wide.u32 %rd26, %r100, 2, %rd24;
add.s64 %rd4, %rd26, 1536;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
ld.global.v4.b32 { %r4, %r5, %r6, %r7 }, [ %rd1 + 0 ];
mov.b32 {%rs17, %rs18}, %r4;
mov.b32 {%rs19, %rs20}, %r5;
mov.b32 {%rs21, %rs22}, %r6;
mov.b32 {%rs23, %rs24}, %r7;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
ld.global.v4.b32 { %r8, %r9, %r10, %r11 }, [ %rd2 + 0 ];
mov.b32 {%rs25, %rs26}, %r8;
mov.b32 {%rs27, %rs28}, %r9;
mov.b32 {%rs29, %rs30}, %r10;
mov.b32 {%rs31, %rs32}, %r11;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd3 + 0 ];
mov.b32 {%rs33, %rs34}, %r12;
mov.b32 {%rs35, %rs36}, %r13;
mov.b32 {%rs37, %rs38}, %r14;
mov.b32 {%rs39, %rs40}, %r15;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
ld.global.v4.b32 { %r16, %r17, %r18, %r19 }, [ %rd4 + 0 ];
mov.b32 {%rs41, %rs42}, %r16;
mov.b32 {%rs43, %rs44}, %r17;
mov.b32 {%rs45, %rs46}, %r18;
mov.b32 {%rs47, %rs48}, %r19;
cvt.f32.bf16 %r101, %rs17;
cvt.f32.bf16 %r102, %rs18;
cvt.f32.bf16 %r103, %rs19;
cvt.f32.bf16 %r104, %rs20;
cvt.f32.bf16 %r105, %rs21;
cvt.f32.bf16 %r106, %rs22;
cvt.f32.bf16 %r107, %rs23;
cvt.f32.bf16 %r108, %rs24;
cvt.f32.bf16 %r109, %rs25;
cvt.f32.bf16 %r110, %rs26;
cvt.f32.bf16 %r111, %rs27;
cvt.f32.bf16 %r112, %rs28;
cvt.f32.bf16 %r113, %rs29;
cvt.f32.bf16 %r114, %rs30;
cvt.f32.bf16 %r115, %rs31;
cvt.f32.bf16 %r116, %rs32;
cvt.f32.bf16 %r117, %rs33;
cvt.f32.bf16 %r118, %rs34;
cvt.f32.bf16 %r119, %rs35;
cvt.f32.bf16 %r120, %rs36;
cvt.f32.bf16 %r121, %rs37;
cvt.f32.bf16 %r122, %rs38;
cvt.f32.bf16 %r123, %rs39;
cvt.f32.bf16 %r124, %rs40;
cvt.f32.bf16 %r125, %rs41;
cvt.f32.bf16 %r126, %rs42;
cvt.f32.bf16 %r127, %rs43;
cvt.f32.bf16 %r128, %rs44;
cvt.f32.bf16 %r129, %rs45;
cvt.f32.bf16 %r130, %rs46;
cvt.f32.bf16 %r131, %rs47;
cvt.f32.bf16 %r132, %rs48;
selp.b64 %rd27, %rd22, %rd21, %p2;
selp.b64 %rd28, %rd20, %rd27, %p1;
mad.wide.s32 %rd5, %r78, 4, %rd28;
mov.u32 %r20, 0x0;
ld.global.b32 { %r20 }, [ %rd5 + 0 ];
div.full.f32 %r133, %r101, %r20;
div.full.f32 %r134, %r102, %r20;
div.full.f32 %r135, %r103, %r20;
div.full.f32 %r136, %r104, %r20;
div.full.f32 %r137, %r105, %r20;
div.full.f32 %r138, %r106, %r20;
div.full.f32 %r139, %r107, %r20;
div.full.f32 %r140, %r108, %r20;
div.full.f32 %r141, %r109, %r20;
div.full.f32 %r142, %r110, %r20;
div.full.f32 %r143, %r111, %r20;
div.full.f32 %r144, %r112, %r20;
div.full.f32 %r145, %r113, %r20;
div.full.f32 %r146, %r114, %r20;
div.full.f32 %r147, %r115, %r20;
div.full.f32 %r148, %r116, %r20;
div.full.f32 %r149, %r117, %r20;
div.full.f32 %r150, %r118, %r20;
div.full.f32 %r151, %r119, %r20;
div.full.f32 %r152, %r120, %r20;
div.full.f32 %r153, %r121, %r20;
div.full.f32 %r154, %r122, %r20;
div.full.f32 %r155, %r123, %r20;
div.full.f32 %r156, %r124, %r20;
div.full.f32 %r157, %r125, %r20;
div.full.f32 %r158, %r126, %r20;
div.full.f32 %r159, %r127, %r20;
div.full.f32 %r160, %r128, %r20;
div.full.f32 %r161, %r129, %r20;
div.full.f32 %r162, %r130, %r20;
div.full.f32 %r163, %r131, %r20;
div.full.f32 %r164, %r132, %r20;
min.f32 %r165, %r133, 0f43E00000;
min.f32 %r166, %r134, 0f43E00000;
min.f32 %r167, %r135, 0f43E00000;
min.f32 %r168, %r136, 0f43E00000;
min.f32 %r169, %r137, 0f43E00000;
min.f32 %r170, %r138, 0f43E00000;
min.f32 %r171, %r139, 0f43E00000;
min.f32 %r172, %r140, 0f43E00000;
min.f32 %r173, %r141, 0f43E00000;
min.f32 %r174, %r142, 0f43E00000;
min.f32 %r175, %r143, 0f43E00000;
min.f32 %r176, %r144, 0f43E00000;
min.f32 %r177, %r145, 0f43E00000;
min.f32 %r178, %r146, 0f43E00000;
min.f32 %r179, %r147, 0f43E00000;
min.f32 %r180, %r148, 0f43E00000;
min.f32 %r181, %r149, 0f43E00000;
min.f32 %r182, %r150, 0f43E00000;
min.f32 %r183, %r151, 0f43E00000;
min.f32 %r184, %r152, 0f43E00000;
min.f32 %r185, %r153, 0f43E00000;
min.f32 %r186, %r154, 0f43E00000;
min.f32 %r187, %r155, 0f43E00000;
min.f32 %r188, %r156, 0f43E00000;
min.f32 %r189, %r157, 0f43E00000;
min.f32 %r190, %r158, 0f43E00000;
min.f32 %r191, %r159, 0f43E00000;
min.f32 %r192, %r160, 0f43E00000;
min.f32 %r193, %r161, 0f43E00000;
min.f32 %r194, %r162, 0f43E00000;
min.f32 %r195, %r163, 0f43E00000;
min.f32 %r196, %r164, 0f43E00000;
max.f32 %r21, %r165, 0fC3E00000;
max.f32 %r22, %r166, 0fC3E00000;
max.f32 %r23, %r167, 0fC3E00000;
max.f32 %r24, %r168, 0fC3E00000;
max.f32 %r25, %r169, 0fC3E00000;
max.f32 %r26, %r170, 0fC3E00000;
max.f32 %r27, %r171, 0fC3E00000;
max.f32 %r28, %r172, 0fC3E00000;
max.f32 %r29, %r173, 0fC3E00000;
max.f32 %r30, %r174, 0fC3E00000;
max.f32 %r31, %r175, 0fC3E00000;
max.f32 %r32, %r176, 0fC3E00000;
max.f32 %r33, %r177, 0fC3E00000;
max.f32 %r34, %r178, 0fC3E00000;
max.f32 %r35, %r179, 0fC3E00000;
max.f32 %r36, %r180, 0fC3E00000;
max.f32 %r37, %r181, 0fC3E00000;
max.f32 %r38, %r182, 0fC3E00000;
max.f32 %r39, %r183, 0fC3E00000;
max.f32 %r40, %r184, 0fC3E00000;
max.f32 %r41, %r185, 0fC3E00000;
max.f32 %r42, %r186, 0fC3E00000;
max.f32 %r43, %r187, 0fC3E00000;
max.f32 %r44, %r188, 0fC3E00000;
max.f32 %r45, %r189, 0fC3E00000;
max.f32 %r46, %r190, 0fC3E00000;
max.f32 %r47, %r191, 0fC3E00000;
max.f32 %r48, %r192, 0fC3E00000;
max.f32 %r49, %r193, 0fC3E00000;
max.f32 %r50, %r194, 0fC3E00000;
max.f32 %r51, %r195, 0fC3E00000;
max.f32 %r52, %r196, 0fC3E00000;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r22, %r21;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r24, %r23;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r26, %r25;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r28, %r27;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r30, %r29;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r32, %r31;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r34, %r33;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r50, %r49;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r52, %r51;
selp.b64 %rd29, %rd16, %rd15, %p2;
selp.b64 %rd30, %rd14, %rd29, %p1;
selp.b64 %rd31, %rd19, %rd18, %p2;
selp.b64 %rd32, %rd17, %rd31, %p1;
mul.lo.s32 %r197, %r78, %r84;
add.s32 %r198, %r197, %r82;
mad.lo.s32 %r199, %r198, 768, %r71;
cvt.s64.s32 %rd33, %r199;
add.s64 %rd34, %rd30, %rd33;
add.s64 %rd6, %rd34, %rd25;
add.s64 %rd7, %rd6, 768;
mov.b32 %r53, {%rs1, %rs2};
mov.b32 %r54, {%rs3, %rs4};
mov.b32 %r55, {%rs5, %rs6};
mov.b32 %r56, {%rs7, %rs8};
st.global.v4.b32 [ %rd6 + 0 ], { %r53, %r54, %r55, %r56 };
mov.b32 %r57, {%rs9, %rs10};
mov.b32 %r58, {%rs11, %rs12};
mov.b32 %r59, {%rs13, %rs14};
mov.b32 %r60, {%rs15, %rs16};
st.global.v4.b32 [ %rd7 + 0 ], { %r57, %r58, %r59, %r60 };
and.b32 %r200, %r96, 4080;
mov.b32 %r201, global_smem;
add.s32 %r202, %r201, %r200;
st.shared.v4.b32 [%r202], {%r53, %r54, %r55, %r56};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r203, %r204, %r205, %r206}, [%r202];
bar.warp.sync -1;
st.shared.v4.b32 [%r202], {%r57, %r58, %r59, %r60};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r207, %r208, %r209, %r210}, [%r202];
and.b32 %r211, %r94, 56;
shl.b32 %r212, %r93, 2;
and.b32 %r213, %r212, 124;
mad.lo.s32 %r214, %r84, %r71, %r82;
mad.lo.s32 %r215, %r197, 768, %r214;
cvt.s64.s32 %rd35, %r215;
add.s64 %rd36, %rd32, %rd35;
mul.lo.s32 %r216, %r84, %r213;
add.s32 %r217, %r216, %r84;
add.s32 %r218, %r217, %r84;
add.s32 %r219, %r218, %r84;
or.b32 %r220, %r216, %r211;
or.b32 %r221, %r217, %r211;
or.b32 %r222, %r218, %r211;
or.b32 %r223, %r219, %r211;
cvt.u64.u32 %rd37, %r220;
add.s64 %rd8, %rd36, %rd37;
cvt.u64.u32 %rd38, %r221;
add.s64 %rd9, %rd36, %rd38;
cvt.u64.u32 %rd39, %r222;
add.s64 %rd10, %rd36, %rd39;
cvt.u64.u32 %rd40, %r223;
add.s64 %rd11, %rd36, %rd40;
prmt.b32 %r224, %r203, %r207, 0x40U;
prmt.b32 %r225, %r224, %r204, 0x410U;
prmt.b32 %r61, %r225, %r208, 0x4210U;
prmt.b32 %r226, %r205, %r209, 0x40U;
prmt.b32 %r227, %r226, %r206, 0x410U;
prmt.b32 %r62, %r227, %r210, 0x4210U;
st.global.v2.b32 [ %rd8 + 0 ], { %r61, %r62 };
prmt.b32 %r228, %r203, %r207, 0x51U;
prmt.b32 %r229, %r228, %r204, 0x510U;
prmt.b32 %r63, %r229, %r208, 0x5210U;
prmt.b32 %r230, %r205, %r209, 0x51U;
prmt.b32 %r231, %r230, %r206, 0x510U;
prmt.b32 %r64, %r231, %r210, 0x5210U;
st.global.v2.b32 [ %rd9 + 0 ], { %r63, %r64 };
prmt.b32 %r232, %r203, %r207, 0x62U;
prmt.b32 %r233, %r232, %r204, 0x610U;
prmt.b32 %r65, %r233, %r208, 0x6210U;
prmt.b32 %r234, %r205, %r209, 0x62U;
prmt.b32 %r235, %r234, %r206, 0x610U;
prmt.b32 %r66, %r235, %r210, 0x6210U;
st.global.v2.b32 [ %rd10 + 0 ], { %r65, %r66 };
prmt.b32 %r236, %r203, %r207, 0x73U;
prmt.b32 %r237, %r236, %r204, 0x710U;
prmt.b32 %r67, %r237, %r208, 0x7210U;
prmt.b32 %r238, %r205, %r209, 0x73U;
prmt.b32 %r239, %r238, %r206, 0x710U;
prmt.b32 %r68, %r239, %r210, 0x7210U;
st.global.v2.b32 [ %rd11 + 0 ], { %r67, %r68 };
ret;
$L__tmp1:
$L__func_end0:
}
"#


structure Artifacts where
  scanIR : String
  scanPTX : String
  scaleIR : String
  scalePTX : String
  packIR : String
  packPTX : String

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.scanIR == some (lines scanGraph) &&
  QKVTTIR.normalizePTX a.scanPTX == lines scanPtx &&
  QKVTTIR.normalizeTTIR a.scaleIR == some (lines scaleGraph) &&
  QKVTTIR.normalizePTX a.scalePTX == lines scalePtx &&
  QKVTTIR.normalizeTTIR a.packIR == some (lines packGraph) &&
  QKVTTIR.normalizePTX a.packPTX == lines packPtx

structure Certificate (a : Artifacts) : Type where
  matched : matchesArtifacts a = true
  slots : ∀ s : Fin 7, let selected := decodeSlot s; encodeSlot selected.width selected.batch = s
  tiles : ∀ band lane, encodeTile (decodeTile band lane) = (band, lane)
  scan : ∀ (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
      (w : Width) (b : Fin (Layout.ofWidth w).batches) (part : Part),
    ScanSchedule.store first second reduce (encodeSlot w b) part =
      (w, QKVTTIR.referenceStore (compactBank w true (bankView (slotBase w) first))
        (compactBank w false (bankView (slotBase w) second)) reduce w b part)
  scale : ∀ (scratch : Width → Nat → Scalar) (w : Width) (b : Fin (Layout.ofWidth w).batches),
    scaleStore scratch (encodeSlot w b) = (w, QKVScale.referenceStore (scratch w) w b)
  pack : ∀ (op : Arithmetic) (first second : Bank) (scales : Width → Nat → Scalar)
      (w : Width) (p : Point (Geometry.ofWidth w)),
    let (band, lane) := encodeTile ⟨w, p.batch, p.rowTile⟩
    vectorPackStore op first second scales band lane p.colTile p.rowLane p.colLane =
      (w, reference op (compactInputs w (inputsView (slotBase w) ⟨first, second, scales w⟩)) p)

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : matchesArtifacts a = true then
    some ⟨h, encode_slot, encode_tile, ScanSchedule.store_equiv, scale_store_equiv, vector_pack_store_equiv⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.Grouped
