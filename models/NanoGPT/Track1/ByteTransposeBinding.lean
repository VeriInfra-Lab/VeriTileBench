import models.NanoGPT.Track1.ByteTranspose
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Check the pinned masked transpose and the dimension-generic Hopper lowering.
The artifact certificate supplies whole-grid byte equivalence and the
ownership and synchronization facts used by the warp exchange. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ByteTranspose
open SampledRows QKVPack

private def referenceIR : String := r#"
module {
tt.func public @_transpose_copy_kernel(%src_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %src_stride_m: i32 {tt.divisibility = 16 : i32}, %dst_stride_0: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%tile = arith.constant dense<0> : tensor<128x128xi8>
%pid_m = tt.get_program_id x : i32
%pid_n = tt.get_program_id y : i32
%offs_m = arith.muli %pid_m, 128 : i32
%offs_m_0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
%offs_m_1 = tt.splat %offs_m : i32 -> tensor<128xi32>
%offs_m_2 = arith.addi %offs_m_1, %offs_m_0 : tensor<128xi32>
%offs_m_3 = arith.extsi %offs_m_2 : tensor<128xi32> to tensor<128xi64>
%offs_n = arith.muli %pid_n, 128 : i32
%offs_n_4 = tt.splat %offs_n : i32 -> tensor<128xi32>
%offs_n_5 = arith.addi %offs_n_4, %offs_m_0 : tensor<128xi32>
%offs_n_6 = arith.extsi %offs_n_5 : tensor<128xi32> to tensor<128xi64>
%mask = tt.expand_dims %offs_m_3 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
%mask_7 = arith.extsi %M : i32 to i64
%mask_8 = tt.splat %mask_7 : i64 -> tensor<128x1xi64>
%mask_9 = arith.cmpi slt, %mask, %mask_8 : tensor<128x1xi64>
%mask_10 = tt.expand_dims %offs_n_6 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
%mask_11 = arith.extsi %N : i32 to i64
%mask_12 = tt.splat %mask_11 : i64 -> tensor<1x128xi64>
%mask_13 = arith.cmpi slt, %mask_10, %mask_12 : tensor<1x128xi64>
%mask_14 = tt.broadcast %mask_9 : tensor<128x1xi1> -> tensor<128x128xi1>
%mask_15 = tt.broadcast %mask_13 : tensor<1x128xi1> -> tensor<128x128xi1>
%mask_16 = arith.andi %mask_14, %mask_15 : tensor<128x128xi1>
%tile_17 = arith.extsi %src_stride_m : i32 to i64
%tile_18 = tt.splat %tile_17 : i64 -> tensor<128x1xi64>
%tile_19 = arith.muli %mask, %tile_18 : tensor<128x1xi64>
%tile_20 = tt.splat %src_ptr : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>>
%tile_21 = tt.addptr %tile_20, %tile_19 : tensor<128x1x!tt.ptr<i8>>, tensor<128x1xi64>
%tile_22 = tt.broadcast %tile_21 : tensor<128x1x!tt.ptr<i8>> -> tensor<128x128x!tt.ptr<i8>>
%tile_23 = tt.broadcast %mask_10 : tensor<1x128xi64> -> tensor<128x128xi64>
%tile_24 = tt.addptr %tile_22, %tile_23 : tensor<128x128x!tt.ptr<i8>>, tensor<128x128xi64>
%tile_25 = tt.load %tile_24, %mask_16, %tile : tensor<128x128x!tt.ptr<i8>>
%mask_T = tt.expand_dims %offs_n_6 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
%mask_T_26 = tt.splat %mask_11 : i64 -> tensor<128x1xi64>
%mask_T_27 = arith.cmpi slt, %mask_T, %mask_T_26 : tensor<128x1xi64>
%mask_T_28 = tt.expand_dims %offs_m_3 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
%mask_T_29 = tt.splat %mask_7 : i64 -> tensor<1x128xi64>
%mask_T_30 = arith.cmpi slt, %mask_T_28, %mask_T_29 : tensor<1x128xi64>
%mask_T_31 = tt.broadcast %mask_T_27 : tensor<128x1xi1> -> tensor<128x128xi1>
%mask_T_32 = tt.broadcast %mask_T_30 : tensor<1x128xi1> -> tensor<128x128xi1>
%mask_T_33 = arith.andi %mask_T_31, %mask_T_32 : tensor<128x128xi1>
%0 = arith.extsi %dst_stride_0 : i32 to i64
%1 = tt.splat %0 : i64 -> tensor<128x1xi64>
%2 = arith.muli %mask_T, %1 : tensor<128x1xi64>
%3 = tt.splat %dst_ptr : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>>
%4 = tt.addptr %3, %2 : tensor<128x1x!tt.ptr<i8>>, tensor<128x1xi64>
%5 = tt.broadcast %4 : tensor<128x1x!tt.ptr<i8>> -> tensor<128x128x!tt.ptr<i8>>
%6 = tt.broadcast %mask_T_28 : tensor<1x128xi64> -> tensor<128x128xi64>
%7 = tt.addptr %5, %6 : tensor<128x128x!tt.ptr<i8>>, tensor<128x128xi64>
%8 = tt.trans %tile_25 {order = array<i32: 1, 0>} : tensor<128x128xi8> -> tensor<128x128xi8>
tt.store %7, %8, %mask_T_33 : tensor<128x128x!tt.ptr<i8>>
tt.return
}
}
"#

private def hopperIR : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_byte_transpose(%src: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %M_tiles: i32, %N_tiles: i32) attributes {noinline = false} {
%0 = arith.muli %M_tiles, 128 : i32
%1 = arith.muli %N_tiles, 128 : i32
%2 = tt.get_program_id y : i32
%3 = arith.muli %2, 128 : i32
%4 = tt.get_program_id x : i32
%5 = arith.muli %4, 128 : i32
%rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_0 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_1 = arith.addi %rows_0, %rows : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_2 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_3 = arith.addi %cols_2, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%tile = tt.expand_dims %rows_1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%tile_4 = tt.splat %1 : i32 -> tensor<128x1xi32, #blocked>
%tile_5 = arith.muli %tile, %tile_4 : tensor<128x1xi32, #blocked>
%tile_6 = tt.splat %src : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
%tile_7 = tt.addptr %tile_6, %tile_5 : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi32, #blocked>
%tile_8 = tt.expand_dims %cols_3 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%tile_9 = tt.broadcast %tile_7 : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x128x!tt.ptr<i8>, #blocked>
%tile_10 = tt.broadcast %tile_8 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%tile_11 = tt.addptr %tile_9, %tile_10 : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi32, #blocked>
%tile_12 = tt.load %tile_11 : tensor<128x128x!tt.ptr<i8>, #blocked>
%moved = ttg.convert_layout %tile_12 : tensor<128x128xi8, #blocked> -> tensor<128x128xi8, #blocked1>
%dst_rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_13 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_14 = arith.addi %dst_rows_13, %dst_rows : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_15 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_16 = arith.addi %dst_cols_15, %dst_cols : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%6 = tt.expand_dims %dst_rows_14 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%7 = tt.splat %0 : i32 -> tensor<1x128xi32, #blocked1>
%8 = arith.muli %6, %7 : tensor<1x128xi32, #blocked1>
%9 = tt.splat %dst : !tt.ptr<i8> -> tensor<1x128x!tt.ptr<i8>, #blocked1>
%10 = tt.addptr %9, %8 : tensor<1x128x!tt.ptr<i8>, #blocked1>, tensor<1x128xi32, #blocked1>
%11 = tt.expand_dims %dst_cols_16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%12 = tt.broadcast %10 : tensor<1x128x!tt.ptr<i8>, #blocked1> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
%13 = tt.broadcast %11 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%14 = tt.addptr %12, %13 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
tt.store %14, %moved : tensor<128x128x!tt.ptr<i8>, #blocked1>
tt.return
}
}
"#

private def hopperPTX : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _byte_transpose(
.param .u64 .ptr .global .align 1 _byte_transpose_param_0,
.param .u64 .ptr .global .align 1 _byte_transpose_param_1,
.param .u32 _byte_transpose_param_2,
.param .u32 _byte_transpose_param_3,
.param .u64 .ptr .global .align 1 _byte_transpose_param_4,
.param .u64 .ptr .global .align 1 _byte_transpose_param_5
)
.reqntid 256
{
.reg .b32 %r<97>;
.reg .b64 %rd<33>;
$L__func_begin0:
ld.param.b64 %rd13, [_byte_transpose_param_0];
ld.param.b32 %r33, [_byte_transpose_param_2];
$L__tmp0:
shl.b32 %r34, %r33, 7;
ld.param.b64 %rd14, [_byte_transpose_param_1];
ld.param.b32 %r35, [_byte_transpose_param_3];
shl.b32 %r36, %r35, 7;
mov.u32 %r37, %ctaid.y;
shl.b32 %r38, %r37, 7;
mov.u32 %r39, %ctaid.x;
shl.b32 %r40, %r39, 7;
$L__tmp1:
mov.u32 %r41, %tid.x;
shr.u32 %r42, %r41, 1;
and.b32 %r43, %r42, 124;
or.b32 %r44, %r43, %r38;
shl.b32 %r45, %r41, 3;
and.b32 %r46, %r45, 56;
or.b32 %r47, %r46, %r40;
mul.lo.s32 %r48, %r36, %r44;
add.s32 %r49, %r48, %r36;
add.s32 %r50, %r49, %r36;
add.s32 %r51, %r50, %r36;
cvt.s64.s32 %rd15, %r48;
add.s64 %rd16, %rd13, %rd15;
cvt.s64.s32 %rd17, %r49;
add.s64 %rd18, %rd13, %rd17;
cvt.s64.s32 %rd19, %r50;
add.s64 %rd20, %rd13, %rd19;
cvt.s64.s32 %rd21, %r51;
add.s64 %rd22, %rd13, %rd21;
cvt.s64.s32 %rd23, %r47;
add.s64 %rd1, %rd16, %rd23;
add.s64 %rd2, %rd18, %rd23;
add.s64 %rd3, %rd20, %rd23;
add.s64 %rd4, %rd22, %rd23;
add.s64 %rd5, %rd1, 64;
add.s64 %rd6, %rd2, 64;
add.s64 %rd7, %rd3, 64;
add.s64 %rd8, %rd4, 64;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v2.b32 { %r3, %r4 }, [ %rd2 + 0 ];
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
ld.global.v2.b32 { %r5, %r6 }, [ %rd3 + 0 ];
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v2.b32 { %r7, %r8 }, [ %rd4 + 0 ];
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
ld.global.v2.b32 { %r9, %r10 }, [ %rd5 + 0 ];
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v2.b32 { %r11, %r12 }, [ %rd6 + 0 ];
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
ld.global.v2.b32 { %r13, %r14 }, [ %rd7 + 0 ];
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v2.b32 { %r15, %r16 }, [ %rd8 + 0 ];
shl.b32 %r52, %r41, 4;
and.b32 %r53, %r52, 4080;
mov.b32 %r54, global_smem;
add.s32 %r55, %r54, %r53;
prmt.b32 %r56, %r1, %r3, 0x5410U;
prmt.b32 %r57, %r1, %r3, 0x7632U;
prmt.b32 %r58, %r2, %r4, 0x5410U;
prmt.b32 %r59, %r2, %r4, 0x7632U;
st.shared.v4.b32 [%r55], {%r56, %r57, %r58, %r59};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r60, %r61, %r62, %r63}, [%r55];
bar.warp.sync -1;
prmt.b32 %r64, %r5, %r7, 0x5410U;
prmt.b32 %r65, %r5, %r7, 0x7632U;
prmt.b32 %r66, %r6, %r8, 0x5410U;
prmt.b32 %r67, %r6, %r8, 0x7632U;
st.shared.v4.b32 [%r55], {%r64, %r65, %r66, %r67};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r68, %r69, %r70, %r71}, [%r55];
bar.warp.sync -1;
prmt.b32 %r72, %r9, %r11, 0x5410U;
prmt.b32 %r73, %r9, %r11, 0x7632U;
prmt.b32 %r74, %r10, %r12, 0x5410U;
prmt.b32 %r75, %r10, %r12, 0x7632U;
st.shared.v4.b32 [%r55], {%r72, %r73, %r74, %r75};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r76, %r77, %r78, %r79}, [%r55];
bar.warp.sync -1;
prmt.b32 %r80, %r13, %r15, 0x5410U;
prmt.b32 %r81, %r13, %r15, 0x7632U;
prmt.b32 %r82, %r14, %r16, 0x5410U;
prmt.b32 %r83, %r14, %r16, 0x7632U;
st.shared.v4.b32 [%r55], {%r80, %r81, %r82, %r83};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r84, %r85, %r86, %r87}, [%r55];
shl.b32 %r88, %r41, 1;
and.b32 %r89, %r88, 62;
or.b32 %r90, %r89, %r40;
and.b32 %r91, %r42, 112;
or.b32 %r92, %r91, %r38;
mul.lo.s32 %r93, %r34, %r90;
add.s32 %r94, %r93, %r34;
mad.lo.s32 %r95, %r33, 8064, %r94;
add.s32 %r96, %r95, %r34;
cvt.s64.s32 %rd24, %r93;
add.s64 %rd25, %rd14, %rd24;
cvt.s64.s32 %rd26, %r94;
add.s64 %rd27, %rd14, %rd26;
cvt.s64.s32 %rd28, %r95;
add.s64 %rd29, %rd14, %rd28;
cvt.s64.s32 %rd30, %r96;
add.s64 %rd31, %rd14, %rd30;
cvt.u64.u32 %rd32, %r92;
add.s64 %rd9, %rd25, %rd32;
add.s64 %rd10, %rd27, %rd32;
add.s64 %rd11, %rd29, %rd32;
add.s64 %rd12, %rd31, %rd32;
prmt.b32 %r17, %r60, %r68, 0x6420U;
prmt.b32 %r18, %r61, %r69, 0x6420U;
prmt.b32 %r19, %r62, %r70, 0x6420U;
prmt.b32 %r20, %r63, %r71, 0x6420U;
st.global.v4.b32 [ %rd9 + 0 ], { %r17, %r18, %r19, %r20 };
prmt.b32 %r21, %r60, %r68, 0x7531U;
prmt.b32 %r22, %r61, %r69, 0x7531U;
prmt.b32 %r23, %r62, %r70, 0x7531U;
prmt.b32 %r24, %r63, %r71, 0x7531U;
st.global.v4.b32 [ %rd10 + 0 ], { %r21, %r22, %r23, %r24 };
prmt.b32 %r25, %r76, %r84, 0x6420U;
prmt.b32 %r26, %r77, %r85, 0x6420U;
prmt.b32 %r27, %r78, %r86, 0x6420U;
prmt.b32 %r28, %r79, %r87, 0x6420U;
st.global.v4.b32 [ %rd11 + 0 ], { %r25, %r26, %r27, %r28 };
prmt.b32 %r29, %r76, %r84, 0x7531U;
prmt.b32 %r30, %r77, %r85, 0x7531U;
prmt.b32 %r31, %r78, %r86, 0x7531U;
prmt.b32 %r32, %r79, %r87, 0x7531U;
st.global.v4.b32 [ %rd12 + 0 ], { %r29, %r30, %r31, %r32 };
$L__tmp2:
ret;
$L__tmp3:
$L__func_end0:
}
"#

structure Artifacts where
  referenceIR : String
  hopperIR : String
  hopperPTX : String

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines referenceIR) &&
    QKVTTIR.normalizeTTIR a.hopperIR == some (lines hopperIR) &&
    QKVTTIR.normalizePTX a.hopperPTX == lines hopperPTX

structure Certificate (a : Artifacts) : Type where
  recognized : matchesArtifacts a = true
  phases : GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3]
  shared : ∀ (r c : Fin 128),
    storeLocation (wideRowOwner (r, c)) = loadLocation (wideColOwner (r, c))
  disjoint : ∀ (p q : WideThread), p.warp ≠ q.warp →
    (storeLocation p).address ≠ (loadLocation q).address
  coverage : ∀ (s : Shape) (r : Fin s.rows) (c : Fin s.cols),
    (point s r c).row = r ∧ (point s r c).col = c
  unique : ∀ {s : Shape} (p q : Point s), destOffset p = destOffset q → p = q
  rowOwner : ∀ t : WideThread, wideRowOwner (wideRowIndex t) = t
  colOwner : ∀ t : WideThread, wideColOwner (wideColIndex t) = t
  loadAddress : ∀ {s : Shape} (p : Point s) (t : WideThread),
    compiledSource p t = sourceOffset {p with cell := wideRowIndex t}
  storeAddress : ∀ {s : Shape} (p : Point s) (t : WideThread),
    compiledDestination p t = destOffset {p with cell := wideColIndex t}
  stores : ∀ {s : Shape} (bytes : Nat → UInt8) (p : Point s),
    emitted bytes p = reference bytes p

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : matchesArtifacts a = true then
    some ⟨h, phases_safe, shared_copy_matches, shared_between_warps, tiles_cover,
      unique_destination, row_owner_index, col_owner_index, compiled_source_exact,
      compiled_destination_exact, emitted_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.ByteTranspose
