import models.NanoGPT.Track1.CETranspose
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Hopper artifacts for the shared transpose of the activation
and gradient operands. The program's column tile selects both buffer
addresses and the source row stride before the warp-local exchange. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.CETranspose

private def graph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_ce_backward_transposes(%x: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %xt: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %grad: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %gt: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %M_tiles: i32, %N_tiles: i32) attributes {noinline = false} {
%column = tt.get_program_id x : i32
%activation = arith.cmpi ult, %column, 6 : i32
%src = arith.select %activation, %x, %grad : !tt.ptr<i8>
%dst = arith.select %activation, %xt, %gt : !tt.ptr<i8>
%width = arith.muli %N_tiles, 128 : i32
%width_0 = arith.select %activation, 768, %width : i32
%tile = arith.subi %column, 6 : i32
%tile_1 = arith.select %activation, %column, %tile : i32
%0 = arith.muli %M_tiles, 128 : i32
%1 = tt.get_program_id y : i32
%2 = arith.muli %1, 128 : i32
%3 = arith.muli %tile_1, 128 : i32
%rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_2 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%rows_3 = arith.addi %rows_2, %rows : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_4 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cols_5 = arith.addi %cols_4, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%tile_6 = tt.expand_dims %rows_3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%tile_7 = tt.splat %width_0 : i32 -> tensor<128x1xi32, #blocked>
%tile_8 = arith.muli %tile_6, %tile_7 : tensor<128x1xi32, #blocked>
%tile_9 = tt.splat %src : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
%tile_10 = arith.extui %tile_8 : tensor<128x1xi32, #blocked> to tensor<128x1xi64, #blocked>
%tile_11 = tt.addptr %tile_9, %tile_10 : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi64, #blocked>
%tile_12 = tt.expand_dims %cols_5 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%tile_13 = tt.broadcast %tile_11 : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x128x!tt.ptr<i8>, #blocked>
%tile_14 = tt.broadcast %tile_12 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%tile_15 = arith.extui %tile_14 : tensor<128x128xi32, #blocked> to tensor<128x128xi64, #blocked>
%tile_16 = tt.addptr %tile_13, %tile_15 : tensor<128x128x!tt.ptr<i8>, #blocked>, tensor<128x128xi64, #blocked>
%tile_17 = tt.load %tile_16 : tensor<128x128x!tt.ptr<i8>, #blocked>
%moved = ttg.convert_layout %tile_17 : tensor<128x128xi8, #blocked> -> tensor<128x128xi8, #blocked1>
%dst_rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_18 = tt.splat %3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_rows_19 = arith.addi %dst_rows_18, %dst_rows : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%dst_cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_20 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dst_cols_21 = arith.addi %dst_cols_20, %dst_cols : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%4 = tt.expand_dims %dst_rows_19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%5 = tt.splat %0 : i32 -> tensor<1x128xi32, #blocked1>
%6 = arith.muli %4, %5 : tensor<1x128xi32, #blocked1>
%7 = tt.expand_dims %dst_cols_21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%8 = tt.broadcast %6 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%9 = tt.broadcast %7 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%10 = arith.addi %8, %9 : tensor<128x128xi32, #blocked1>
%11 = tt.splat %dst : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #blocked1>
%12 = arith.extui %10 : tensor<128x128xi32, #blocked1> to tensor<128x128xi64, #blocked1>
%13 = tt.addptr %11, %12 : tensor<128x128x!tt.ptr<i8>, #blocked1>, tensor<128x128xi64, #blocked1>
tt.store %13, %moved : tensor<128x128x!tt.ptr<i8>, #blocked1>
tt.return
}
}
"#

private def assembly : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _ce_backward_transposes(
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_0,
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_1,
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_2,
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_3,
.param .u32 _ce_backward_transposes_param_4,
.param .u32 _ce_backward_transposes_param_5,
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_6,
.param .u64 .ptr .global .align 1 _ce_backward_transposes_param_7
)
.reqntid 256
{
.reg .pred %p<2>;
.reg .b32 %r<116>;
.reg .b64 %rd<24>;
$L__func_begin0:
ld.param.b64 %rd9, [_ce_backward_transposes_param_0];
ld.param.b64 %rd10, [_ce_backward_transposes_param_1];
$L__tmp0:
mov.u32 %r33, %ctaid.x;
setp.lt.u32 %p1, %r33, 6;
ld.param.b64 %rd11, [_ce_backward_transposes_param_2];
selp.b64 %rd12, %rd9, %rd11, %p1;
ld.param.b64 %rd13, [_ce_backward_transposes_param_3];
selp.b64 %rd14, %rd10, %rd13, %p1;
ld.param.b32 %r34, [_ce_backward_transposes_param_4];
ld.param.b32 %r35, [_ce_backward_transposes_param_5];
shl.b32 %r36, %r35, 7;
selp.b32 %r37, 768, %r36, %p1;
shl.b32 %r38, %r34, 7;
mov.u32 %r39, %ctaid.y;
shl.b32 %r40, %r39, 7;
shl.b32 %r41, %r33, 7;
add.s32 %r42, %r41, -768;
selp.b32 %r43, %r41, %r42, %p1;
$L__tmp1:
mov.u32 %r44, %tid.x;
shr.u32 %r45, %r44, 1;
and.b32 %r46, %r45, 126;
or.b32 %r47, %r46, %r40;
and.b32 %r48, %r44, 3;
shl.b32 %r49, %r48, 4;
or.b32 %r50, %r43, %r49;
mul.lo.s32 %r51, %r37, %r47;
add.s32 %r52, %r51, %r37;
cvt.u64.u32 %rd15, %r51;
cvt.u64.u32 %rd16, %r52;
add.s64 %rd17, %rd12, %rd15;
add.s64 %rd18, %rd12, %rd16;
cvt.u64.u32 %rd19, %r50;
add.s64 %rd1, %rd17, %rd19;
add.s64 %rd2, %rd18, %rd19;
add.s64 %rd3, %rd1, 64;
add.s64 %rd4, %rd2, 64;
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
shl.b32 %r53, %r48, 5;
shl.b32 %r54, %r44, 4;
and.b32 %r55, %r54, 3968;
bfe.s32 %r56, %r44, 2, 1;
and.b32 %r57, %r56, 4112;
or.b32 %r58, %r53, %r55;
or.b32 %r59, %r58, %r57;
mov.b32 %r60, global_smem;
add.s32 %r61, %r60, %r59;
prmt.b32 %r62, %r1, %r5, 0x5410U;
prmt.b32 %r63, %r1, %r5, 0x7632U;
prmt.b32 %r64, %r2, %r6, 0x5410U;
prmt.b32 %r65, %r2, %r6, 0x7632U;
st.shared.v4.b32 [%r61], {%r62, %r63, %r64, %r65};
xor.b32 %r66, %r59, 16;
add.s32 %r67, %r60, %r66;
prmt.b32 %r68, %r3, %r7, 0x5410U;
prmt.b32 %r69, %r3, %r7, 0x7632U;
prmt.b32 %r70, %r4, %r8, 0x5410U;
prmt.b32 %r71, %r4, %r8, 0x7632U;
st.shared.v4.b32 [%r67], {%r68, %r69, %r70, %r71};
bar.warp.sync -1;
and.b32 %r72, %r54, 3696;
bfe.s32 %r73, %r44, 3, 1;
and.b32 %r74, %r73, 4112;
shl.b32 %r75, %r44, 3;
and.b32 %r76, %r75, 128;
xor.b32 %r77, %r74, %r72;
add.s32 %r78, %r60, %r76;
add.s32 %r79, %r78, %r77;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r80, %r81, %r82, %r83}, [%r79];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r84, %r85, %r86, %r87}, [%r79+256];
bar.warp.sync -1;
prmt.b32 %r88, %r9, %r13, 0x5410U;
prmt.b32 %r89, %r9, %r13, 0x7632U;
prmt.b32 %r90, %r10, %r14, 0x5410U;
prmt.b32 %r91, %r10, %r14, 0x7632U;
st.shared.v4.b32 [%r61], {%r88, %r89, %r90, %r91};
prmt.b32 %r92, %r11, %r15, 0x5410U;
prmt.b32 %r93, %r11, %r15, 0x7632U;
prmt.b32 %r94, %r12, %r16, 0x5410U;
prmt.b32 %r95, %r12, %r16, 0x7632U;
st.shared.v4.b32 [%r67], {%r92, %r93, %r94, %r95};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r96, %r97, %r98, %r99}, [%r79];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r100, %r101, %r102, %r103}, [%r79+256];
shl.b32 %r104, %r44, 1;
and.b32 %r105, %r104, 62;
or.b32 %r106, %r43, %r105;
or.b32 %r107, %r106, 1;
or.b32 %r108, %r106, 64;
or.b32 %r109, %r106, 65;
and.b32 %r110, %r45, 112;
or.b32 %r111, %r110, %r40;
mad.lo.s32 %r112, %r38, %r106, %r111;
mad.lo.s32 %r113, %r38, %r107, %r111;
mad.lo.s32 %r114, %r38, %r108, %r111;
mad.lo.s32 %r115, %r38, %r109, %r111;
cvt.u64.u32 %rd20, %r112;
cvt.u64.u32 %rd21, %r113;
cvt.u64.u32 %rd22, %r114;
cvt.u64.u32 %rd23, %r115;
add.s64 %rd5, %rd14, %rd20;
add.s64 %rd6, %rd14, %rd21;
add.s64 %rd7, %rd14, %rd22;
add.s64 %rd8, %rd14, %rd23;
prmt.b32 %r17, %r80, %r81, 0x6420U;
prmt.b32 %r18, %r82, %r83, 0x6420U;
prmt.b32 %r19, %r84, %r85, 0x6420U;
prmt.b32 %r20, %r86, %r87, 0x6420U;
st.global.v4.b32 [ %rd5 + 0 ], { %r17, %r18, %r19, %r20 };
prmt.b32 %r21, %r80, %r81, 0x7531U;
prmt.b32 %r22, %r82, %r83, 0x7531U;
prmt.b32 %r23, %r84, %r85, 0x7531U;
prmt.b32 %r24, %r86, %r87, 0x7531U;
st.global.v4.b32 [ %rd6 + 0 ], { %r21, %r22, %r23, %r24 };
prmt.b32 %r25, %r96, %r97, 0x6420U;
prmt.b32 %r26, %r98, %r99, 0x6420U;
prmt.b32 %r27, %r100, %r101, 0x6420U;
prmt.b32 %r28, %r102, %r103, 0x6420U;
st.global.v4.b32 [ %rd7 + 0 ], { %r25, %r26, %r27, %r28 };
prmt.b32 %r29, %r96, %r97, 0x7531U;
prmt.b32 %r30, %r98, %r99, 0x7531U;
prmt.b32 %r31, %r100, %r101, 0x7531U;
prmt.b32 %r32, %r102, %r103, 0x7531U;
st.global.v4.b32 [ %rd8 + 0 ], { %r29, %r30, %r31, %r32 };
$L__tmp2:
ret;
$L__tmp3:
$L__func_end0:
}
"#

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (ttgir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttgir == some (lines graph) &&
    QKVTTIR.normalizePTX ptx == lines assembly

structure Certificate (ttgir ptx : String) : Type where
  matched : matchesArtifacts ttgir ptx = true
  phases : QKVPack.GluonShared.checkedPhases Wide.phases = some [0, 1]
  sharedStores : ∀ (left right : Wide.InputByte), left.half = right.half →
    Wide.sharedStoreAddress left = Wide.sharedStoreAddress right → left = right
  sharedReads : ∀ (output : SampledRows.WideThread),
    Wide.sharedStoreAddress (Wide.readInput output) = Wide.sharedLoadAddress output ∧
      (Wide.readInput output).thread.val / 32 = output.warp.val ∧
      Wide.loadedCell (Wide.readInput output) = SampledRows.wideColIndex output
  routing : ∀ {s : Shape} (p : Cell s), route s (launchX p) (launchY p) p.2.cell = p
  coverage : ∀ (s : Shape) (x : X s) (y : Fin s.tokens) (cell : SampledRows.Cell),
    launchX (route s x y cell) = x ∧ launchY (route s x y cell) = y
  unique : ∀ {s : Shape} (p q : Cell s),
    (p.1, ByteTranspose.destOffset p.2) = (q.1, ByteTranspose.destOffset q.2) → p = q
  stores : paired = staged
  unsignedStores : unsigned = staged
  unsignedPitches : ∀ {s : ByteTranspose.Shape} (p : ByteTranspose.Point s),
    byteOffset (p.row.val * s.cols) + p.col.val = ByteTranspose.sourceOffset p ∧
    byteOffset (p.col.val * s.rows) + p.row.val = ByteTranspose.destOffset p

def certify (ttgir ptx : String) : Option (Certificate ttgir ptx) :=
  if h : matchesArtifacts ttgir ptx = true then
    some ⟨h, Wide.phases_ready, Wide.unique_shared_store, Wide.shared_reads,
      route_exact, launch_covers, unique_store, stores_equal, unsigned_stores_equal,
      extended_pitches_exact⟩
  else none

structure Program where
  ttgir : String
  ptx : String
  checked : Certificate ttgir ptx

def Program.check (ttgir ptx : String) : Option Program :=
  (certify ttgir ptx).map (⟨ttgir, ptx, ·⟩)

def Program.run (program : Program) : Dispatch := fun fastPath =>
  if fastPath && matchesArtifacts program.ttgir program.ptx then unsigned else staged

theorem Program.run_eq (program : Program) : program.run = fun _ => staged := by
  funext fastPath
  cases fastPath <;> simp [Program.run, program.checked.matched, program.checked.unsignedStores]

#axiomsClean certify
#axiomsClean Program.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.CETranspose
