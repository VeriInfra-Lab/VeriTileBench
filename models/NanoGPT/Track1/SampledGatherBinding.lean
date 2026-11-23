import models.NanoGPT.Track1.SampledGatherVector
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Exact TTGIR and executable PTX for all three sampled gather launches.
The decoder binds the candidate-class indirection, vector byte copies,
four warp-local exchange phases and both output strides. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SampledRows
open QKVPack


private def gatherGraph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_gather_candidate_rows(%weights: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %candidates: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %rows: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %transposed: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<128x1xi32, #blocked>
%cst_0 = arith.constant dense<@P@> : tensor<1x128xi32, #blocked1>
%0 = tt.get_program_id y : i32
%1 = tt.get_program_id x : i32
%start_m = arith.muli %0, 128 : i32
%start_n = arith.muli %1, 128 : i32
%rm = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%vocab = tt.addptr %candidates, %start_m : !tt.ptr<i64>, i32
%vocab_1 = tt.splat %vocab : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
%vocab_2 = tt.addptr %vocab_1, %rm : tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%vocab_3 = tt.load %vocab_2 : tensor<128x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked}>>
%vocab_4 = arith.trunci %vocab_3 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%tile = tt.expand_dims %vocab_4 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%tile_5 = arith.muli %tile, %cst : tensor<128x1xi32, #blocked>
%tile_6 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%tile_7 = tt.splat %start_n : i32 -> tensor<1x128xi32, #blocked>
%tile_8 = arith.addi %tile_7, %tile_6 : tensor<1x128xi32, #blocked>
%tile_9 = tt.broadcast %tile_5 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%tile_10 = tt.broadcast %tile_8 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%tile_11 = arith.addi %tile_9, %tile_10 : tensor<128x128xi32, #blocked>
%tile_12 = tt.splat %weights : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked>
%tile_13 = tt.addptr %tile_12, %tile_11 : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi32, #blocked>
%tile_14 = tt.load %tile_13 : tensor<128x128x!tt.ptr<i8>, #blocked>
%2 = arith.muli %0, 98304 : i32
%3 = arith.addi %2, %start_n : i32
%4 = tt.addptr %rows, %3 : !tt.ptr<i8>, i32
%5 = tt.expand_dims %rm {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%6 = arith.muli %5, %cst : tensor<128x1xi32, #blocked>
%7 = tt.broadcast %6 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%8 = tt.broadcast %tile_6 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%9 = arith.addi %7, %8 : tensor<128x128xi32, #blocked>
%10 = tt.splat %4 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked>
%11 = tt.addptr %10, %9 : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi32, #blocked>
tt.store %11, %tile_14 : tensor<128x128x!tt.ptr<i8>, #blocked>
%moved = ttg.convert_layout %tile_14 : tensor<128x128xi8, #blocked> -> tensor<128x128xi8, #blocked1>
%tm = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%tn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%12 = arith.muli %1, @128P@ : i32
%13 = arith.addi %12, %start_m : i32
%14 = tt.addptr %transposed, %13 : !tt.ptr<i8>, i32
%15 = tt.expand_dims %tn {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%16 = arith.muli %15, %cst_0 : tensor<1x128xi32, #blocked1>
%17 = tt.expand_dims %tm {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%18 = tt.broadcast %16 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%19 = tt.broadcast %17 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%20 = arith.addi %18, %19 : tensor<128x128xi32, #blocked1>
%21 = tt.splat %14 : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
%22 = tt.addptr %21, %20 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi32, #blocked1>
tt.store %22, %moved : tensor<128x128x!tt.ptr<i8>, #blocked1>
tt.return
}
}
"#

private def gatherPtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _gather_candidate_rows(
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_0,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_1,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_2,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_3,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_4,
.param .u64 .ptr .global .align 1 _gather_candidate_rows_param_5
)
.reqntid 256
{
.reg .b32 %r<112>;
.reg .b64 %rd<35>;
$L__func_begin0:
ld.param.b64 %rd19, [_gather_candidate_rows_param_0];
ld.param.b64 %rd20, [_gather_candidate_rows_param_1];
$L__tmp0:
mov.u32 %r33, %ctaid.y;
ld.param.b64 %rd21, [_gather_candidate_rows_param_2];
mov.u32 %r34, %ctaid.x;
$L__tmp1:
shl.b32 %r35, %r33, 7;
ld.param.b64 %rd22, [_gather_candidate_rows_param_3];
shl.b32 %r36, %r34, 7;
mov.u32 %r37, %tid.x;
shr.u32 %r38, %r37, 1;
and.b32 %r39, %r38, 124;
shl.b32 %r40, %r37, 4;
and.b32 %r41, %r40, 112;
mad.wide.u32 %rd23, %r35, 8, %rd20;
mad.wide.u32 %rd3, %r39, 8, %rd23;
add.s64 %rd6, %rd3, 16;
mov.u64 %rd1, 0x0;
mov.u64 %rd2, 0x0;
ld.global.v2.b64 { %rd1, %rd2 }, [ %rd3 + 0 ];
mov.u64 %rd4, 0x0;
mov.u64 %rd5, 0x0;
ld.global.v2.b64 { %rd4, %rd5 }, [ %rd6 + 0 ];
cvt.u32.u64 %r42, %rd1;
cvt.u32.u64 %r43, %rd2;
cvt.u32.u64 %r44, %rd4;
cvt.u32.u64 %r45, %rd5;
or.b32 %r46, %r41, %r36;
mad.lo.s32 %r47, %r42, 768, %r46;
mad.lo.s32 %r48, %r43, 768, %r46;
mad.lo.s32 %r49, %r44, 768, %r46;
mad.lo.s32 %r50, %r45, 768, %r46;
cvt.s64.s32 %rd24, %r47;
add.s64 %rd7, %rd19, %rd24;
cvt.s64.s32 %rd25, %r48;
add.s64 %rd8, %rd19, %rd25;
cvt.s64.s32 %rd26, %r49;
add.s64 %rd9, %rd19, %rd26;
cvt.s64.s32 %rd27, %r50;
add.s64 %rd10, %rd19, %rd27;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd7 + 0 ];
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd8 + 0 ];
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd9 + 0 ];
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd10 + 0 ];
mad.lo.s32 %r51, %r33, 98304, %r36;
cvt.s64.s32 %rd28, %r51;
add.s64 %rd29, %rd21, %rd28;
mul.lo.s32 %r52, %r39, 768;
or.b32 %r53, %r52, %r41;
add.s32 %r54, %r53, 1536;
cvt.u64.u32 %rd30, %r53;
add.s64 %rd11, %rd29, %rd30;
add.s64 %rd12, %rd11, 768;
cvt.u64.u32 %rd31, %r54;
add.s64 %rd13, %rd29, %rd31;
add.s64 %rd14, %rd11, 2304;
st.global.v4.b32 [ %rd11 + 0 ], { %r1, %r2, %r3, %r4 };
st.global.v4.b32 [ %rd12 + 0 ], { %r5, %r6, %r7, %r8 };
st.global.v4.b32 [ %rd13 + 0 ], { %r9, %r10, %r11, %r12 };
st.global.v4.b32 [ %rd14 + 0 ], { %r13, %r14, %r15, %r16 };
and.b32 %r55, %r40, 4080;
mov.b32 %r56, global_smem;
add.s32 %r57, %r56, %r55;
st.shared.v4.b32 [%r57], {%r1, %r2, %r3, %r4};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r58, %r59, %r60, %r61}, [%r57];
bar.warp.sync -1;
st.shared.v4.b32 [%r57], {%r5, %r6, %r7, %r8};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r62, %r63, %r64, %r65}, [%r57];
bar.warp.sync -1;
st.shared.v4.b32 [%r57], {%r9, %r10, %r11, %r12};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r66, %r67, %r68, %r69}, [%r57];
bar.warp.sync -1;
st.shared.v4.b32 [%r57], {%r13, %r14, %r15, %r16};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r70, %r71, %r72, %r73}, [%r57];
and.b32 %r74, %r38, 112;
shl.b32 %r75, %r37, 2;
and.b32 %r76, %r75, 124;
mad.lo.s32 %r77, %r34, @128P@, %r35;
cvt.s64.s32 %rd32, %r77;
add.s64 %rd33, %rd22, %rd32;
mul.lo.s32 %r78, %r76, @P@;
or.b32 %r79, %r78, %r74;
cvt.u64.u32 %rd34, %r79;
add.s64 %rd15, %rd33, %rd34;
add.s64 %rd16, %rd15, @P@;
add.s64 %rd17, %rd15, @2P@;
add.s64 %rd18, %rd15, @3P@;
prmt.b32 %r80, %r58, %r62, 0x40U;
prmt.b32 %r81, %r80, %r66, 0x410U;
prmt.b32 %r17, %r81, %r70, 0x4210U;
prmt.b32 %r82, %r59, %r63, 0x40U;
prmt.b32 %r83, %r82, %r67, 0x410U;
prmt.b32 %r18, %r83, %r71, 0x4210U;
prmt.b32 %r84, %r60, %r64, 0x40U;
prmt.b32 %r85, %r84, %r68, 0x410U;
prmt.b32 %r19, %r85, %r72, 0x4210U;
prmt.b32 %r86, %r61, %r65, 0x40U;
prmt.b32 %r87, %r86, %r69, 0x410U;
prmt.b32 %r20, %r87, %r73, 0x4210U;
st.global.v4.b32 [ %rd15 + 0 ], { %r17, %r18, %r19, %r20 };
prmt.b32 %r88, %r58, %r62, 0x51U;
prmt.b32 %r89, %r88, %r66, 0x510U;
prmt.b32 %r21, %r89, %r70, 0x5210U;
prmt.b32 %r90, %r59, %r63, 0x51U;
prmt.b32 %r91, %r90, %r67, 0x510U;
prmt.b32 %r22, %r91, %r71, 0x5210U;
prmt.b32 %r92, %r60, %r64, 0x51U;
prmt.b32 %r93, %r92, %r68, 0x510U;
prmt.b32 %r23, %r93, %r72, 0x5210U;
prmt.b32 %r94, %r61, %r65, 0x51U;
prmt.b32 %r95, %r94, %r69, 0x510U;
prmt.b32 %r24, %r95, %r73, 0x5210U;
st.global.v4.b32 [ %rd16 + 0 ], { %r21, %r22, %r23, %r24 };
prmt.b32 %r96, %r58, %r62, 0x62U;
prmt.b32 %r97, %r96, %r66, 0x610U;
prmt.b32 %r25, %r97, %r70, 0x6210U;
prmt.b32 %r98, %r59, %r63, 0x62U;
prmt.b32 %r99, %r98, %r67, 0x610U;
prmt.b32 %r26, %r99, %r71, 0x6210U;
prmt.b32 %r100, %r60, %r64, 0x62U;
prmt.b32 %r101, %r100, %r68, 0x610U;
prmt.b32 %r27, %r101, %r72, 0x6210U;
prmt.b32 %r102, %r61, %r65, 0x62U;
prmt.b32 %r103, %r102, %r69, 0x610U;
prmt.b32 %r28, %r103, %r73, 0x6210U;
st.global.v4.b32 [ %rd17 + 0 ], { %r25, %r26, %r27, %r28 };
prmt.b32 %r104, %r58, %r62, 0x73U;
prmt.b32 %r105, %r104, %r66, 0x710U;
prmt.b32 %r29, %r105, %r70, 0x7210U;
prmt.b32 %r106, %r59, %r63, 0x73U;
prmt.b32 %r107, %r106, %r67, 0x710U;
prmt.b32 %r30, %r107, %r71, 0x7210U;
prmt.b32 %r108, %r60, %r64, 0x73U;
prmt.b32 %r109, %r108, %r68, 0x710U;
prmt.b32 %r31, %r109, %r72, 0x7210U;
prmt.b32 %r110, %r61, %r65, 0x73U;
prmt.b32 %r111, %r110, %r69, 0x710U;
prmt.b32 %r32, %r111, %r73, 0x7210U;
st.global.v4.b32 [ %rd18 + 0 ], { %r29, %r30, %r31, %r32 };
$L__tmp2:
ret;
$L__tmp3:
$L__func_end0:
}
"#

def artifactLines (c : Count) (text : String) : List String :=
  let patched := text.replace "@P@" (toString c.rows)
    |>.replace "@2P@" (toString (2 * c.rows))
    |>.replace "@3P@" (toString (3 * c.rows))
    |>.replace "@128P@" (toString (128 * c.rows))
    |>.replace "@B@" (toString (c.rows / 16))
  (patched.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (c : Count) (ttgir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttgir == some (artifactLines c gatherGraph) &&
    QKVTTIR.normalizePTX ptx == artifactLines c gatherPtx

structure Certificate (c : Count) (ttgir ptx : String) : Type where
  matched : matchesArtifacts c ttgir ptx = true
  phases : GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3]
  coordinates : ∀ (row : Fin c.rows) (col : Channel),
    (point c row col).row = row ∧ (point c row col).column = col
  unique : ∀ (p q : Point c),
    (p.row.val * 768 + p.column.val = q.row.val * 768 + q.column.val → p = q) ∧
      (p.column.val * c.rows + p.row.val = q.column.val * c.rows + q.row.val → p = q)
  shared : ∀ (r col : Fin 128),
    Vector.storeLocation (Vector.rowOwner (r, col)) = Vector.loadLocation (Vector.columnOwner (r, col))
  stores : ∀ (bank : Nat → UInt8) (ids : Fin c.rows → ClassID) (p : Point c),
    Vector.observe bank ids p = reference c bank ids p.row p.column

def certify (c : Count) (ttgir ptx : String) : Option (Certificate c ttgir ptx) :=
  if h : matchesArtifacts c ttgir ptx = true then
    some ⟨h, phases_safe, point_covers c, output_owners_unique, fun r col => Vector.shared_copy_matches (r, col), Vector.observe_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.SampledRows
