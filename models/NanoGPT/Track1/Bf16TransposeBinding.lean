import models.NanoGPT.Track1.Bf16Transpose
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The checked baseline and Hopper artifacts for the two tied-embedding
launches. The copy preserves halfwords. Both add kernels use the same
BF16x2 instruction with the old destination as its left operand. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Bf16Transpose
open VeriTile.Triton FP QKVPack

structure Artifacts where
  referenceIR : String
  referencePTX : String
  hopperIR : String
  hopperPTX : String
  deriving BEq

private def copyArtifacts : Artifacts where
  referenceIR := r#"
module {
tt.func public @_transpose_copy_kernel(%src_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %src_stride_m: i32 {tt.divisibility = 16 : i32}, %dst_stride_0: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%tile = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
%pid_m = tt.get_program_id x : i32
%pid_n = tt.get_program_id y : i32
%offs_m = arith.muli %pid_m, 64 : i32
%offs_m_0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%offs_m_1 = tt.splat %offs_m : i32 -> tensor<64xi32>
%offs_m_2 = arith.addi %offs_m_1, %offs_m_0 : tensor<64xi32>
%offs_m_3 = arith.extsi %offs_m_2 : tensor<64xi32> to tensor<64xi64>
%offs_n = arith.muli %pid_n, 128 : i32
%offs_n_4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
%offs_n_5 = tt.splat %offs_n : i32 -> tensor<128xi32>
%offs_n_6 = arith.addi %offs_n_5, %offs_n_4 : tensor<128xi32>
%offs_n_7 = arith.extsi %offs_n_6 : tensor<128xi32> to tensor<128xi64>
%mask = tt.expand_dims %offs_m_3 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64>
%mask_8 = arith.extsi %M : i32 to i64
%mask_9 = tt.splat %mask_8 : i64 -> tensor<64x1xi64>
%mask_10 = arith.cmpi slt, %mask, %mask_9 : tensor<64x1xi64>
%mask_11 = tt.expand_dims %offs_n_7 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
%mask_12 = arith.extsi %N : i32 to i64
%mask_13 = tt.splat %mask_12 : i64 -> tensor<1x128xi64>
%mask_14 = arith.cmpi slt, %mask_11, %mask_13 : tensor<1x128xi64>
%mask_15 = tt.broadcast %mask_10 : tensor<64x1xi1> -> tensor<64x128xi1>
%mask_16 = tt.broadcast %mask_14 : tensor<1x128xi1> -> tensor<64x128xi1>
%mask_17 = arith.andi %mask_15, %mask_16 : tensor<64x128xi1>
%tile_18 = arith.extsi %src_stride_m : i32 to i64
%tile_19 = tt.splat %tile_18 : i64 -> tensor<64x1xi64>
%tile_20 = arith.muli %mask, %tile_19 : tensor<64x1xi64>
%tile_21 = tt.splat %src_ptr : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
%tile_22 = tt.addptr %tile_21, %tile_20 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi64>
%tile_23 = tt.broadcast %tile_22 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
%tile_24 = tt.broadcast %mask_11 : tensor<1x128xi64> -> tensor<64x128xi64>
%tile_25 = tt.addptr %tile_23, %tile_24 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi64>
%tile_26 = tt.load %tile_25, %mask_17, %tile : tensor<64x128x!tt.ptr<bf16>>
%mask_T = tt.expand_dims %offs_n_7 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
%mask_T_27 = tt.splat %mask_12 : i64 -> tensor<128x1xi64>
%mask_T_28 = arith.cmpi slt, %mask_T, %mask_T_27 : tensor<128x1xi64>
%mask_T_29 = tt.expand_dims %offs_m_3 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
%mask_T_30 = tt.splat %mask_8 : i64 -> tensor<1x64xi64>
%mask_T_31 = arith.cmpi slt, %mask_T_29, %mask_T_30 : tensor<1x64xi64>
%mask_T_32 = tt.broadcast %mask_T_28 : tensor<128x1xi1> -> tensor<128x64xi1>
%mask_T_33 = tt.broadcast %mask_T_31 : tensor<1x64xi1> -> tensor<128x64xi1>
%mask_T_34 = arith.andi %mask_T_32, %mask_T_33 : tensor<128x64xi1>
%0 = arith.extsi %dst_stride_0 : i32 to i64
%1 = tt.splat %0 : i64 -> tensor<128x1xi64>
%2 = arith.muli %mask_T, %1 : tensor<128x1xi64>
%3 = tt.splat %dst_ptr : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
%4 = tt.addptr %3, %2 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi64>
%5 = tt.broadcast %4 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
%6 = tt.broadcast %mask_T_29 : tensor<1x64xi64> -> tensor<128x64xi64>
%7 = tt.addptr %5, %6 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi64>
%8 = tt.trans %tile_26 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
tt.store %7, %8, %mask_T_34 : tensor<128x64x!tt.ptr<bf16>>
tt.return
}
}
"#
  referencePTX := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _transpose_copy_kernel(
.param .u64 .ptr .global .align 1 _transpose_copy_kernel_param_0,
.param .u64 .ptr .global .align 1 _transpose_copy_kernel_param_1,
.param .u32 _transpose_copy_kernel_param_2,
.param .u32 _transpose_copy_kernel_param_3,
.param .u32 _transpose_copy_kernel_param_4,
.param .u32 _transpose_copy_kernel_param_5,
.param .u64 .ptr .global .align 1 _transpose_copy_kernel_param_6,
.param .u64 .ptr .global .align 1 _transpose_copy_kernel_param_7
)
.reqntid 256
{
.reg .pred %p<19>;
.reg .b32 %r<109>;
.reg .b64 %rd<42>;
$L__func_begin0:
ld.param.b64 %rd9, [_transpose_copy_kernel_param_0];
ld.param.b64 %rd10, [_transpose_copy_kernel_param_1];
$L__tmp0:
mov.u32 %r34, %ctaid.x;
ld.param.b32 %r35, [_transpose_copy_kernel_param_2];
mov.u32 %r36, %ctaid.y;
shl.b32 %r37, %r34, 6;
ld.param.b32 %r38, [_transpose_copy_kernel_param_3];
ld.param.b32 %r39, [_transpose_copy_kernel_param_4];
mov.u32 %r40, %tid.x;
shr.u32 %r41, %r40, 4;
ld.param.s32 %rd11, [_transpose_copy_kernel_param_5];
bfe.u32 %r42, %r40, 4, 4;
and.b32 %r43, %r40, 7;
shl.b32 %r44, %r43, 3;
or.b32 %r45, %r42, %r37;
or.b32 %r46, %r45, 16;
or.b32 %r47, %r45, 32;
or.b32 %r48, %r41, %r37;
or.b32 %r49, %r48, 48;
or.b32 %r50, %r44, %r37;
shl.b32 %r51, %r36, 7;
shl.b32 %r52, %r40, 3;
and.b32 %r53, %r52, 120;
shr.u32 %r54, %r40, 3;
bfe.u32 %r55, %r40, 3, 5;
or.b32 %r56, %r53, %r51;
or.b32 %r57, %r55, %r51;
or.b32 %r58, %r57, 32;
or.b32 %r59, %r57, 64;
or.b32 %r60, %r54, %r51;
or.b32 %r61, %r60, 96;
cvt.u64.u32 %rd12, %r57;
cvt.u64.u32 %rd13, %r58;
cvt.u64.u32 %rd14, %r59;
cvt.u64.u32 %rd15, %r61;
setp.lt.s32 %p9, %r45, %r35;
setp.lt.s32 %p10, %r46, %r35;
setp.lt.s32 %p11, %r47, %r35;
setp.lt.s32 %p12, %r49, %r35;
setp.lt.s32 %p13, %r56, %r38;
and.pred %p1, %p9, %p13;
and.pred %p2, %p10, %p13;
and.pred %p3, %p11, %p13;
and.pred %p4, %p12, %p13;
mul.wide.s32 %rd16, %r39, %r45;
mul.wide.s32 %rd17, %r39, %r46;
mul.wide.s32 %rd18, %r39, %r47;
mul.wide.s32 %rd19, %r39, %r49;
shl.b64 %rd20, %rd16, 1;
add.s64 %rd21, %rd9, %rd20;
shl.b64 %rd22, %rd17, 1;
add.s64 %rd23, %rd9, %rd22;
shl.b64 %rd24, %rd18, 1;
add.s64 %rd25, %rd9, %rd24;
shl.b64 %rd26, %rd19, 1;
add.s64 %rd27, %rd9, %rd26;
mul.wide.u32 %rd28, %r56, 2;
add.s64 %rd1, %rd21, %rd28;
add.s64 %rd2, %rd23, %rd28;
add.s64 %rd3, %rd25, %rd28;
add.s64 %rd4, %rd27, %rd28;
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
@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p3 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
prmt.b32 %r62, %r1, %r10, 0x7632U;
prmt.b32 %r63, %r2, %r11, 0x7632U;
prmt.b32 %r64, %r3, %r12, 0x7632U;
prmt.b32 %r65, %r4, %r13, 0x7632U;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p4 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
prmt.b32 %r66, %r6, %r14, 0x7632U;
prmt.b32 %r67, %r7, %r15, 0x7632U;
prmt.b32 %r68, %r8, %r16, 0x7632U;
prmt.b32 %r69, %r9, %r17, 0x7632U;
setp.lt.s32 %p14, %r57, %r38;
setp.lt.s32 %p15, %r58, %r38;
setp.lt.s32 %p16, %r59, %r38;
setp.lt.s32 %p17, %r61, %r38;
setp.lt.s32 %p18, %r50, %r35;
and.pred %p5, %p18, %p14;
and.pred %p6, %p18, %p15;
and.pred %p7, %p18, %p16;
and.pred %p8, %p18, %p17;
mul.lo.s64 %rd29, %rd11, %rd12;
mul.lo.s64 %rd30, %rd11, %rd13;
mul.lo.s64 %rd31, %rd11, %rd14;
mul.lo.s64 %rd32, %rd11, %rd15;
shl.b64 %rd33, %rd29, 1;
add.s64 %rd34, %rd10, %rd33;
shl.b64 %rd35, %rd30, 1;
add.s64 %rd36, %rd10, %rd35;
shl.b64 %rd37, %rd31, 1;
add.s64 %rd38, %rd10, %rd37;
shl.b64 %rd39, %rd32, 1;
add.s64 %rd40, %rd10, %rd39;
mul.wide.s32 %rd41, %r50, 2;
add.s64 %rd5, %rd34, %rd41;
add.s64 %rd6, %rd36, %rd41;
add.s64 %rd7, %rd38, %rd41;
add.s64 %rd8, %rd40, %rd41;
shl.b32 %r70, %r43, 11;
shl.b32 %r71, %r40, 4;
and.b32 %r72, %r71, 1648;
shl.b32 %r73, %r40, 5;
and.b32 %r74, %r73, 256;
and.b32 %r75, %r40, 16;
shr.u32 %r76, %r40, 2;
and.b32 %r77, %r76, 32;
or.b32 %r78, %r70, %r74;
xor.b32 %r79, %r72, %r75;
xor.b32 %r80, %r79, %r77;
or.b32 %r81, %r80, %r78;
mov.b32 %r82, global_smem;
add.s32 %r83, %r82, %r81;
prmt.b32 %r84, %r1, %r10, 0x5410U;
prmt.b32 %r85, %r2, %r11, 0x5410U;
st.shared.v4.b32 [%r83], {%r84, %r62, %r85, %r63};
prmt.b32 %r86, %r3, %r12, 0x5410U;
prmt.b32 %r87, %r4, %r13, 0x5410U;
st.shared.v4.b32 [%r83+128], {%r86, %r64, %r87, %r65};
xor.b32 %r88, %r81, 64;
add.s32 %r89, %r82, %r88;
prmt.b32 %r90, %r6, %r14, 0x5410U;
prmt.b32 %r91, %r7, %r15, 0x5410U;
st.shared.v4.b32 [%r89], {%r90, %r66, %r91, %r67};
prmt.b32 %r92, %r8, %r16, 0x5410U;
prmt.b32 %r93, %r9, %r17, 0x5410U;
st.shared.v4.b32 [%r89+128], {%r92, %r68, %r93, %r69};
bar.sync 0;
shl.b32 %r94, %r40, 6;
and.b32 %r95, %r94, 1536;
and.b32 %r96, %r40, 192;
shl.b32 %r97, %r96, 5;
shl.b32 %r98, %r43, 4;
shr.u32 %r99, %r96, 2;
shl.b32 %r100, %r40, 2;
and.b32 %r101, %r100, 128;
or.b32 %r102, %r95, %r97;
xor.b32 %r103, %r98, %r99;
or.b32 %r104, %r102, %r103;
or.b32 %r105, %r104, %r101;
add.s32 %r106, %r82, %r105;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r18, %r19, %r20, %r21}, [%r106];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r26, %r27, %r28, %r29}, [%r106+256];
xor.b32 %r107, %r105, 64;
add.s32 %r108, %r82, %r107;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r22, %r23, %r24, %r25}, [%r108+8192];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r30, %r31, %r32, %r33}, [%r108+8448];
@%p5 st.global.v4.b32 [ %rd5 + 0 ], { %r18, %r19, %r20, %r21 };
@%p6 st.global.v4.b32 [ %rd6 + 0 ], { %r22, %r23, %r24, %r25 };
@%p7 st.global.v4.b32 [ %rd7 + 0 ], { %r26, %r27, %r28, %r29 };
@%p8 st.global.v4.b32 [ %rd8 + 0 ], { %r30, %r31, %r32, %r33 };
ret;
$L__tmp1:
$L__func_end0:
}
"#
  hopperIR := r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_warp_bf16_transpose(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<50304> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1x128xi32, #blocked1>
%m = tt.get_program_id y : i32
%m_1 = arith.muli %m, 64 : i32
%n = tt.get_program_id x : i32
%n_2 = arith.muli %n, 128 : i32
%rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_3 = tt.splat %m_1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_4 = arith.addi %rows_3, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_5 = tt.splat %n_2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_6 = arith.addi %cols_5, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%tile = tt.expand_dims %rows_4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%tile_7 = arith.muli %tile, %cst : tensor<64x1xi32, #blocked>
%tile_8 = tt.splat %src : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
%tile_9 = tt.addptr %tile_8, %tile_7 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%tile_10 = tt.expand_dims %cols_6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%tile_11 = tt.broadcast %tile_9 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%tile_12 = tt.broadcast %tile_10 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%tile_13 = tt.addptr %tile_11, %tile_12 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%tile_14 = tt.load %tile_13 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%moved = ttg.convert_layout %tile_14 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked1>
%dst_rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_15 = tt.splat %n_2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_16 = arith.addi %dst_rows_15, %dst_rows : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_17 = tt.splat %m_1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_18 = arith.addi %dst_cols_17, %dst_cols : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%out = tt.expand_dims %dst_rows_16 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%out_19 = arith.muli %out, %cst_0 : tensor<1x128xi32, #blocked1>
%out_20 = tt.expand_dims %dst_cols_18 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%out_21 = tt.broadcast %out_19 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%out_22 = tt.broadcast %out_20 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%out_23 = arith.addi %out_21, %out_22 : tensor<64x128xi32, #blocked1>
%out_24 = tt.splat %dst : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked1>
%out_25 = tt.addptr %out_24, %out_23 : tensor<64x128x!tt.ptr<bf16>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %out_25, %moved : tensor<64x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
"#
  hopperPTX := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _warp_bf16_transpose(
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_0,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_1,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_2,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_3
)
.reqntid 256
{
.reg .b32 %r<67>;
.reg .b64 %rd<12>;
$L__func_begin0:
ld.param.b64 %rd9, [_warp_bf16_transpose_param_0];
ld.param.b64 %rd10, [_warp_bf16_transpose_param_1];
$L__tmp0:
mov.u32 %r33, %ctaid.y;
shl.b32 %r34, %r33, 6;
mov.u32 %r35, %ctaid.x;
shl.b32 %r36, %r35, 7;
mov.u32 %r37, %tid.x;
shr.u32 %r38, %r37, 2;
bfe.u32 %r39, %r37, 2, 6;
or.b32 %r40, %r39, %r34;
shl.b32 %r41, %r37, 3;
and.b32 %r42, %r41, 24;
or.b32 %r43, %r42, %r36;
mul.lo.s32 %r44, %r40, 50304;
mad.wide.s32 %rd11, %r44, 2, %rd9;
mad.wide.s32 %rd1, %r43, 2, %rd11;
add.s64 %rd2, %rd1, 64;
add.s64 %rd3, %rd1, 128;
add.s64 %rd4, %rd1, 192;
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
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
shl.b32 %r45, %r37, 7;
and.b32 %r46, %r45, 3072;
shl.b32 %r47, %r37, 4;
and.b32 %r48, %r47, 112;
shl.b32 %r49, %r37, 2;
and.b32 %r50, %r49, 992;
xor.b32 %r51, %r48, %r50;
mov.b32 %r52, global_smem;
add.s32 %r53, %r52, %r46;
add.s32 %r54, %r53, %r51;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r1, %r2, %r3, %r4};
bar.warp.sync -1;
shl.b32 %r55, %r37, 9;
and.b32 %r56, %r55, 3072;
add.s32 %r57, %r52, %r56;
add.s32 %r58, %r57, %r51;
ld.shared.v4.b32 {%r17, %r18, %r19, %r20}, [%r58];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r5, %r6, %r7, %r8};
bar.warp.sync -1;
ld.shared.v4.b32 {%r21, %r22, %r23, %r24}, [%r58];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r9, %r10, %r11, %r12};
bar.warp.sync -1;
ld.shared.v4.b32 {%r25, %r26, %r27, %r28}, [%r58];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r54], {%r13, %r14, %r15, %r16};
bar.warp.sync -1;
ld.shared.v4.b32 {%r29, %r30, %r31, %r32}, [%r58];
and.b32 %r59, %r37, 31;
or.b32 %r60, %r36, %r59;
and.b32 %r61, %r38, 56;
or.b32 %r62, %r61, %r34;
mad.lo.s32 %r63, %r60, 768, %r62;
add.s32 %r64, %r63, 24576;
add.s32 %r65, %r63, 49152;
add.s32 %r66, %r63, 73728;
mad.wide.s32 %rd5, %r63, 2, %rd10;
mad.wide.s32 %rd6, %r64, 2, %rd10;
mad.wide.s32 %rd7, %r65, 2, %rd10;
mad.wide.s32 %rd8, %r66, 2, %rd10;
st.global.v4.b32 [ %rd5 + 0 ], { %r17, %r18, %r19, %r20 };
st.global.v4.b32 [ %rd6 + 0 ], { %r21, %r22, %r23, %r24 };
st.global.v4.b32 [ %rd7 + 0 ], { %r25, %r26, %r27, %r28 };
st.global.v4.b32 [ %rd8 + 0 ], { %r29, %r30, %r31, %r32 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def addArtifacts : Artifacts where
  referenceIR := r#"
module {
tt.func public @_transpose_add_kernel(%src_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %src_stride_m: i32 {tt.divisibility = 16 : i32}, %dst_stride_0: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%dst_tile = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
%src_tile = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
%pid_m = tt.get_program_id x : i32
%pid_n = tt.get_program_id y : i32
%offs_m = arith.muli %pid_m, 64 : i32
%offs_m_0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%offs_m_1 = tt.splat %offs_m : i32 -> tensor<64xi32>
%offs_m_2 = arith.addi %offs_m_1, %offs_m_0 : tensor<64xi32>
%offs_n = arith.muli %pid_n, 128 : i32
%offs_n_3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
%offs_n_4 = tt.splat %offs_n : i32 -> tensor<128xi32>
%offs_n_5 = arith.addi %offs_n_4, %offs_n_3 : tensor<128xi32>
%mask = tt.expand_dims %offs_m_2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%mask_6 = tt.splat %M : i32 -> tensor<64x1xi32>
%mask_7 = arith.cmpi slt, %mask, %mask_6 : tensor<64x1xi32>
%mask_8 = tt.expand_dims %offs_n_5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
%mask_9 = tt.splat %N : i32 -> tensor<1x128xi32>
%mask_10 = arith.cmpi slt, %mask_8, %mask_9 : tensor<1x128xi32>
%mask_11 = tt.broadcast %mask_7 : tensor<64x1xi1> -> tensor<64x128xi1>
%mask_12 = tt.broadcast %mask_10 : tensor<1x128xi1> -> tensor<64x128xi1>
%mask_13 = arith.andi %mask_11, %mask_12 : tensor<64x128xi1>
%src_tile_14 = tt.splat %src_stride_m : i32 -> tensor<64x1xi32>
%src_tile_15 = arith.muli %mask, %src_tile_14 : tensor<64x1xi32>
%src_tile_16 = tt.splat %src_ptr : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
%src_tile_17 = tt.addptr %src_tile_16, %src_tile_15 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
%src_tile_18 = tt.broadcast %src_tile_17 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
%src_tile_19 = tt.broadcast %mask_8 : tensor<1x128xi32> -> tensor<64x128xi32>
%src_tile_20 = tt.addptr %src_tile_18, %src_tile_19 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
%src_tile_21 = tt.load %src_tile_20, %mask_13, %src_tile : tensor<64x128x!tt.ptr<bf16>>
%mask_T = tt.expand_dims %offs_n_5 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
%mask_T_22 = tt.splat %N : i32 -> tensor<128x1xi32>
%mask_T_23 = arith.cmpi slt, %mask_T, %mask_T_22 : tensor<128x1xi32>
%mask_T_24 = tt.expand_dims %offs_m_2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%mask_T_25 = tt.splat %M : i32 -> tensor<1x64xi32>
%mask_T_26 = arith.cmpi slt, %mask_T_24, %mask_T_25 : tensor<1x64xi32>
%mask_T_27 = tt.broadcast %mask_T_23 : tensor<128x1xi1> -> tensor<128x64xi1>
%mask_T_28 = tt.broadcast %mask_T_26 : tensor<1x64xi1> -> tensor<128x64xi1>
%mask_T_29 = arith.andi %mask_T_27, %mask_T_28 : tensor<128x64xi1>
%dst_ptrs = tt.splat %dst_stride_0 : i32 -> tensor<128x1xi32>
%dst_ptrs_30 = arith.muli %mask_T, %dst_ptrs : tensor<128x1xi32>
%dst_ptrs_31 = tt.splat %dst_ptr : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
%dst_ptrs_32 = tt.addptr %dst_ptrs_31, %dst_ptrs_30 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
%dst_ptrs_33 = tt.broadcast %dst_ptrs_32 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x64x!tt.ptr<bf16>>
%dst_ptrs_34 = tt.broadcast %mask_T_24 : tensor<1x64xi32> -> tensor<128x64xi32>
%dst_ptrs_35 = tt.addptr %dst_ptrs_33, %dst_ptrs_34 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
%dst_tile_36 = tt.load %dst_ptrs_35, %mask_T_29, %dst_tile : tensor<128x64x!tt.ptr<bf16>>
%0 = tt.trans %src_tile_21 {order = array<i32: 1, 0>} : tensor<64x128xbf16> -> tensor<128x64xbf16>
%1 = arith.addf %dst_tile_36, %0 : tensor<128x64xbf16>
tt.store %dst_ptrs_35, %1, %mask_T_29 : tensor<128x64x!tt.ptr<bf16>>
tt.return
}
}
"#
  referencePTX := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _transpose_add_kernel(
.param .u64 .ptr .global .align 1 _transpose_add_kernel_param_0,
.param .u64 .ptr .global .align 1 _transpose_add_kernel_param_1,
.param .u32 _transpose_add_kernel_param_2,
.param .u32 _transpose_add_kernel_param_3,
.param .u32 _transpose_add_kernel_param_4,
.param .u32 _transpose_add_kernel_param_5,
.param .u64 .ptr .global .align 1 _transpose_add_kernel_param_6,
.param .u64 .ptr .global .align 1 _transpose_add_kernel_param_7
)
.reqntid 256
{
.reg .pred %p<19>;
.reg .b32 %r<152>;
.reg .b64 %rd<21>;
$L__func_begin0:
ld.param.b64 %rd9, [_transpose_add_kernel_param_0];
ld.param.b64 %rd10, [_transpose_add_kernel_param_1];
$L__tmp0:
mov.u32 %r50, %ctaid.x;
ld.param.b32 %r51, [_transpose_add_kernel_param_2];
mov.u32 %r52, %ctaid.y;
shl.b32 %r53, %r50, 6;
ld.param.b32 %r54, [_transpose_add_kernel_param_3];
ld.param.b32 %r55, [_transpose_add_kernel_param_4];
mov.u32 %r56, %tid.x;
shr.u32 %r57, %r56, 4;
ld.param.b32 %r58, [_transpose_add_kernel_param_5];
bfe.u32 %r59, %r56, 4, 4;
and.b32 %r60, %r56, 7;
shl.b32 %r61, %r60, 3;
or.b32 %r62, %r59, %r53;
or.b32 %r63, %r62, 16;
or.b32 %r64, %r62, 32;
or.b32 %r65, %r57, %r53;
or.b32 %r66, %r65, 48;
or.b32 %r67, %r61, %r53;
shl.b32 %r68, %r52, 7;
shl.b32 %r69, %r56, 3;
and.b32 %r70, %r69, 120;
shr.u32 %r71, %r56, 3;
bfe.u32 %r72, %r56, 3, 5;
or.b32 %r73, %r70, %r68;
or.b32 %r74, %r72, %r68;
or.b32 %r75, %r74, 32;
or.b32 %r76, %r74, 64;
or.b32 %r77, %r71, %r68;
or.b32 %r78, %r77, 96;
setp.lt.s32 %p9, %r62, %r51;
setp.lt.s32 %p10, %r63, %r51;
setp.lt.s32 %p11, %r64, %r51;
setp.lt.s32 %p12, %r66, %r51;
setp.lt.s32 %p13, %r73, %r54;
and.pred %p1, %p9, %p13;
and.pred %p2, %p10, %p13;
and.pred %p3, %p11, %p13;
and.pred %p4, %p12, %p13;
mul.lo.s32 %r79, %r55, %r62;
shl.b32 %r80, %r55, 4;
add.s32 %r81, %r79, %r80;
add.s32 %r82, %r81, %r80;
mul.lo.s32 %r83, %r55, %r66;
mad.wide.s32 %rd11, %r79, 2, %rd9;
mad.wide.s32 %rd12, %r81, 2, %rd9;
mad.wide.s32 %rd13, %r82, 2, %rd9;
mad.wide.s32 %rd14, %r83, 2, %rd9;
mul.wide.u32 %rd15, %r73, 2;
add.s64 %rd1, %rd11, %rd15;
add.s64 %rd2, %rd12, %rd15;
add.s64 %rd3, %rd13, %rd15;
add.s64 %rd4, %rd14, %rd15;
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
@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
mov.u32 %r10, %r5;
mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
@%p3 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
prmt.b32 %r84, %r1, %r10, 0x7632U;
prmt.b32 %r85, %r2, %r11, 0x7632U;
prmt.b32 %r86, %r3, %r12, 0x7632U;
prmt.b32 %r87, %r4, %r13, 0x7632U;
mov.u32 %r14, %r5;
mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
@%p4 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
prmt.b32 %r88, %r6, %r14, 0x7632U;
prmt.b32 %r89, %r7, %r15, 0x7632U;
prmt.b32 %r90, %r8, %r16, 0x7632U;
prmt.b32 %r91, %r9, %r17, 0x7632U;
setp.lt.s32 %p14, %r74, %r54;
setp.lt.s32 %p15, %r75, %r54;
setp.lt.s32 %p16, %r76, %r54;
setp.lt.s32 %p17, %r78, %r54;
setp.lt.s32 %p18, %r67, %r51;
and.pred %p5, %p18, %p14;
and.pred %p6, %p18, %p15;
and.pred %p7, %p18, %p16;
and.pred %p8, %p18, %p17;
mul.lo.s32 %r92, %r58, %r74;
shl.b32 %r93, %r58, 5;
add.s32 %r94, %r92, %r93;
add.s32 %r95, %r94, %r93;
mul.lo.s32 %r96, %r58, %r78;
mad.wide.s32 %rd16, %r92, 2, %rd10;
mad.wide.s32 %rd17, %r94, 2, %rd10;
mad.wide.s32 %rd18, %r95, 2, %rd10;
mad.wide.s32 %rd19, %r96, 2, %rd10;
mul.wide.s32 %rd20, %r67, 2;
add.s64 %rd5, %rd16, %rd20;
add.s64 %rd6, %rd17, %rd20;
add.s64 %rd7, %rd18, %rd20;
add.s64 %rd8, %rd19, %rd20;
mov.u32 %r18, %r5;
mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
@%p5 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
mov.u32 %r22, %r5;
mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
@%p6 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
mov.u32 %r26, %r5;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
@%p7 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
mov.u32 %r30, %r5;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
@%p8 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
shl.b32 %r97, %r60, 11;
shl.b32 %r98, %r56, 4;
and.b32 %r99, %r98, 1648;
shl.b32 %r100, %r56, 5;
and.b32 %r101, %r100, 256;
and.b32 %r102, %r56, 16;
shr.u32 %r103, %r56, 2;
and.b32 %r104, %r103, 32;
or.b32 %r105, %r97, %r101;
xor.b32 %r106, %r99, %r102;
xor.b32 %r107, %r106, %r104;
or.b32 %r108, %r107, %r105;
mov.b32 %r109, global_smem;
add.s32 %r110, %r109, %r108;
prmt.b32 %r111, %r1, %r10, 0x5410U;
prmt.b32 %r112, %r2, %r11, 0x5410U;
st.shared.v4.b32 [%r110], {%r111, %r84, %r112, %r85};
prmt.b32 %r113, %r3, %r12, 0x5410U;
prmt.b32 %r114, %r4, %r13, 0x5410U;
st.shared.v4.b32 [%r110+128], {%r113, %r86, %r114, %r87};
xor.b32 %r115, %r108, 64;
add.s32 %r116, %r109, %r115;
prmt.b32 %r117, %r6, %r14, 0x5410U;
prmt.b32 %r118, %r7, %r15, 0x5410U;
st.shared.v4.b32 [%r116], {%r117, %r88, %r118, %r89};
prmt.b32 %r119, %r8, %r16, 0x5410U;
prmt.b32 %r120, %r9, %r17, 0x5410U;
st.shared.v4.b32 [%r116+128], {%r119, %r90, %r120, %r91};
bar.sync 0;
shl.b32 %r121, %r56, 6;
and.b32 %r122, %r121, 1536;
and.b32 %r123, %r56, 192;
shl.b32 %r124, %r123, 5;
shl.b32 %r125, %r60, 4;
shr.u32 %r126, %r123, 2;
shl.b32 %r127, %r56, 2;
and.b32 %r128, %r127, 128;
or.b32 %r129, %r122, %r124;
xor.b32 %r130, %r125, %r126;
or.b32 %r131, %r129, %r130;
or.b32 %r132, %r131, %r128;
add.s32 %r133, %r109, %r132;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r134, %r135, %r136, %r137}, [%r133];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r138, %r139, %r140, %r141}, [%r133+256];
xor.b32 %r142, %r132, 64;
add.s32 %r143, %r109, %r142;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r144, %r145, %r146, %r147}, [%r143+8192];
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r148, %r149, %r150, %r151}, [%r143+8448];
add.bf16x2 %r34, %r18, %r134;
add.bf16x2 %r35, %r19, %r135;
add.bf16x2 %r36, %r20, %r136;
add.bf16x2 %r37, %r21, %r137;
add.bf16x2 %r38, %r22, %r144;
add.bf16x2 %r39, %r23, %r145;
add.bf16x2 %r40, %r24, %r146;
add.bf16x2 %r41, %r25, %r147;
add.bf16x2 %r42, %r26, %r138;
add.bf16x2 %r43, %r27, %r139;
add.bf16x2 %r44, %r28, %r140;
add.bf16x2 %r45, %r29, %r141;
add.bf16x2 %r46, %r30, %r148;
add.bf16x2 %r47, %r31, %r149;
add.bf16x2 %r48, %r32, %r150;
add.bf16x2 %r49, %r33, %r151;
@%p5 st.global.v4.b32 [ %rd5 + 0 ], { %r34, %r35, %r36, %r37 };
@%p6 st.global.v4.b32 [ %rd6 + 0 ], { %r38, %r39, %r40, %r41 };
@%p7 st.global.v4.b32 [ %rd7 + 0 ], { %r42, %r43, %r44, %r45 };
@%p8 st.global.v4.b32 [ %rd8 + 0 ], { %r46, %r47, %r48, %r49 };
ret;
$L__tmp1:
$L__func_end0:
}
"#
  hopperIR := r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_warp_bf16_transpose(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<50304> : tensor<1x128xi32, #blocked1>
%m = tt.get_program_id y : i32
%m_1 = arith.muli %m, 64 : i32
%n = tt.get_program_id x : i32
%n_2 = arith.muli %n, 128 : i32
%rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_3 = tt.splat %m_1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_4 = arith.addi %rows_3, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_5 = tt.splat %n_2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_6 = arith.addi %cols_5, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%tile = tt.expand_dims %rows_4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%tile_7 = arith.muli %tile, %cst : tensor<64x1xi32, #blocked>
%tile_8 = tt.splat %src : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
%tile_9 = tt.addptr %tile_8, %tile_7 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%tile_10 = tt.expand_dims %cols_6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%tile_11 = tt.broadcast %tile_9 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%tile_12 = tt.broadcast %tile_10 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%tile_13 = tt.addptr %tile_11, %tile_12 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%tile_14 = tt.load %tile_13 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%moved = ttg.convert_layout %tile_14 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked1>
%dst_rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_15 = tt.splat %n_2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_16 = arith.addi %dst_rows_15, %dst_rows : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_17 = tt.splat %m_1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_18 = arith.addi %dst_cols_17, %dst_cols : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%out = tt.expand_dims %dst_rows_16 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%out_19 = arith.muli %out, %cst_0 : tensor<1x128xi32, #blocked1>
%out_20 = tt.expand_dims %dst_cols_18 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%out_21 = tt.broadcast %out_19 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%out_22 = tt.broadcast %out_20 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%out_23 = arith.addi %out_21, %out_22 : tensor<64x128xi32, #blocked1>
%out_24 = tt.splat %dst : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked1>
%out_25 = tt.addptr %out_24, %out_23 : tensor<64x128x!tt.ptr<bf16>, #blocked1>, tensor<64x128xi32, #blocked1>
%old = tt.load %out_25 : tensor<64x128x!tt.ptr<bf16>, #blocked1>
%moved_26 = tt.elementwise_inline_asm "add.bf16x2 $0, $1, $2;" {constraints = "=r,r,r", packed_element = 2 : i32, pure = true} %old, %moved : tensor<64x128xbf16, #blocked1>, tensor<64x128xbf16, #blocked1> -> tensor<64x128xbf16, #blocked1>
tt.store %out_25, %moved_26 : tensor<64x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
"#
  hopperPTX := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _warp_bf16_transpose(
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_0,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_1,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_2,
.param .u64 .ptr .global .align 1 _warp_bf16_transpose_param_3
)
.reqntid 256
{
.reg .b32 %r<99>;
.reg .b64 %rd<12>;
$L__func_begin0:
ld.param.b64 %rd9, [_warp_bf16_transpose_param_0];
ld.param.b64 %rd10, [_warp_bf16_transpose_param_1];
$L__tmp0:
mov.u32 %r65, %ctaid.y;
shl.b32 %r66, %r65, 6;
mov.u32 %r67, %ctaid.x;
shl.b32 %r68, %r67, 7;
mov.u32 %r69, %tid.x;
shr.u32 %r70, %r69, 2;
bfe.u32 %r71, %r69, 2, 6;
or.b32 %r72, %r71, %r66;
shl.b32 %r73, %r69, 3;
and.b32 %r74, %r73, 24;
or.b32 %r75, %r74, %r68;
mul.lo.s32 %r76, %r72, 768;
mad.wide.s32 %rd11, %r76, 2, %rd9;
mad.wide.s32 %rd1, %r75, 2, %rd11;
add.s64 %rd2, %rd1, 64;
add.s64 %rd3, %rd1, 128;
add.s64 %rd4, %rd1, 192;
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
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
shl.b32 %r77, %r69, 7;
and.b32 %r78, %r77, 3072;
shl.b32 %r79, %r69, 4;
and.b32 %r80, %r79, 112;
shl.b32 %r81, %r69, 2;
and.b32 %r82, %r81, 992;
xor.b32 %r83, %r80, %r82;
mov.b32 %r84, global_smem;
add.s32 %r85, %r84, %r78;
add.s32 %r86, %r85, %r83;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r86], {%r1, %r2, %r3, %r4};
bar.warp.sync -1;
shl.b32 %r87, %r69, 9;
and.b32 %r88, %r87, 3072;
add.s32 %r89, %r84, %r88;
add.s32 %r90, %r89, %r83;
ld.shared.v4.b32 {%r18, %r20, %r22, %r24}, [%r90];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r86], {%r5, %r6, %r7, %r8};
bar.warp.sync -1;
ld.shared.v4.b32 {%r26, %r28, %r30, %r32}, [%r90];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r86], {%r9, %r10, %r11, %r12};
bar.warp.sync -1;
ld.shared.v4.b32 {%r34, %r36, %r38, %r40}, [%r90];
bar.warp.sync -1;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r86], {%r13, %r14, %r15, %r16};
bar.warp.sync -1;
ld.shared.v4.b32 {%r42, %r44, %r46, %r48}, [%r90];
and.b32 %r91, %r69, 31;
or.b32 %r92, %r68, %r91;
and.b32 %r93, %r70, 56;
or.b32 %r94, %r93, %r66;
mad.lo.s32 %r95, %r92, 50304, %r94;
add.s32 %r96, %r95, 1609728;
add.s32 %r97, %r95, 3219456;
add.s32 %r98, %r95, 4829184;
mad.wide.s32 %rd5, %r95, 2, %rd10;
mad.wide.s32 %rd6, %r96, 2, %rd10;
mad.wide.s32 %rd7, %r97, 2, %rd10;
mad.wide.s32 %rd8, %r98, 2, %rd10;
mov.u32 %r17, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r21, 0x0;
mov.u32 %r23, 0x0;
ld.global.v4.b32 { %r17, %r19, %r21, %r23 }, [ %rd5 + 0 ];
mov.u32 %r25, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r29, 0x0;
mov.u32 %r31, 0x0;
ld.global.v4.b32 { %r25, %r27, %r29, %r31 }, [ %rd6 + 0 ];
mov.u32 %r33, 0x0;
mov.u32 %r35, 0x0;
mov.u32 %r37, 0x0;
mov.u32 %r39, 0x0;
ld.global.v4.b32 { %r33, %r35, %r37, %r39 }, [ %rd7 + 0 ];
mov.u32 %r41, 0x0;
mov.u32 %r43, 0x0;
mov.u32 %r45, 0x0;
mov.u32 %r47, 0x0;
ld.global.v4.b32 { %r41, %r43, %r45, %r47 }, [ %rd8 + 0 ];
add.bf16x2 %r49, %r17, %r18;
add.bf16x2 %r50, %r19, %r20;
add.bf16x2 %r51, %r21, %r22;
add.bf16x2 %r52, %r23, %r24;
add.bf16x2 %r53, %r25, %r26;
add.bf16x2 %r54, %r27, %r28;
add.bf16x2 %r55, %r29, %r30;
add.bf16x2 %r56, %r31, %r32;
add.bf16x2 %r57, %r33, %r34;
add.bf16x2 %r58, %r35, %r36;
add.bf16x2 %r59, %r37, %r38;
add.bf16x2 %r60, %r39, %r40;
add.bf16x2 %r61, %r41, %r42;
add.bf16x2 %r62, %r43, %r44;
add.bf16x2 %r63, %r45, %r46;
add.bf16x2 %r64, %r47, %r48;
st.global.v4.b32 [ %rd5 + 0 ], { %r49, %r50, %r51, %r52 };
st.global.v4.b32 [ %rd6 + 0 ], { %r53, %r54, %r55, %r56 };
st.global.v4.b32 [ %rd7 + 0 ], { %r57, %r58, %r59, %r60 };
st.global.v4.b32 [ %rd8 + 0 ], { %r61, %r62, %r63, %r64 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (mode : Mode) (a : Artifacts) : Bool :=
  let expected := match mode with | .copy => copyArtifacts | .add => addArtifacts
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines expected.referenceIR) &&
    QKVTTIR.normalizePTX a.referencePTX == lines expected.referencePTX &&
    QKVTTIR.normalizeTTIR a.hopperIR == some (lines expected.hopperIR) &&
    QKVTTIR.normalizePTX a.hopperPTX == lines expected.hopperPTX

structure Certificate (mode : Mode) (a : Artifacts) : Type where
  matched : matchesArtifacts mode a = true
  phases : GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3]
  shared : ∀ t, matrixStoreAddress (inputThread t) = vectorLoadAddress t ∧
    (inputThread t).warp = t.warp ∧ (inputThread t).reg.val / 8 = t.reg.val / 8
  sharedUnique : ∀ (a b : Thread), a.reg.val / 8 = b.reg.val / 8 →
    matrixStoreAddress a = matrixStoreAddress b → a = b
  storeRegion : ∀ t, matrixStoreAddress t + 1 < 4096
  loadRegion : ∀ t, vectorLoadAddress t + 1 < 4096
  coverage : ∀ (r : Fin mode.rows) (c : Fin mode.cols),
    (point mode r c).row = r ∧ (point mode r c).col = c
  unique : ∀ (p q : Point mode), destOffset p = destOffset q → p = q
  loadAddress : ∀ (tile : Tile mode) (t : Thread),
    compiledSource tile t = sourceOffset (inputPoint tile t)
  storeAddress : ∀ (tile : Tile mode) (t : Thread),
    compiledDestination tile t = destOffset (outputPoint tile t)
  stores : ∀ (cfg : Config) (src dst : Bank) (p : Point mode),
    emitted cfg src dst p = reference cfg src dst p

def certify (mode : Mode) (a : Artifacts) : Option (Certificate mode a) :=
  if h : matchesArtifacts mode a = true then
    some ⟨h, phases_safe, shared_load_matches, shared_writer_unique,
      fun t => (matrix_store_location t).1, fun t => (vector_load_location t).1, tiles_cover mode,
      unique_destination, compiled_source_exact, compiled_destination_exact,
      fun cfg => emitted_equal cfg (mode := mode)⟩
  else none

structure Program (mode : Mode) where
  artifacts : Artifacts
  checked : Certificate mode artifacts

def Program.check (mode : Mode) (a : Artifacts) : Option (Program mode) := do
  return ⟨a, ← certify mode a⟩

abbrev Programs := (mode : Mode) → Program mode

def Programs.run (programs : Programs) : Dispatch := fun fast cfg mode src dst point =>
  if fast && matchesArtifacts mode (programs mode).artifacts then emitted cfg src dst point
  else reference cfg src dst point

theorem Programs.run_eq (programs : Programs) : programs.run = fun _ cfg _ => reference cfg := by
  funext fast cfg mode src dst point
  simp only [Programs.run, (programs mode).checked.matched, Bool.and_true]
  cases fast
  · rfl
  · exact (programs mode).checked.stores cfg src dst point

#axiomsClean certify
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.Bf16Transpose
