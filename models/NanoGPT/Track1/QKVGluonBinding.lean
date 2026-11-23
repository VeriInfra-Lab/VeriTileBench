import models.NanoGPT.Track1.QKVNative
import models.NanoGPT.Track1.QKVPackBinding

/-! Bind the warp-local QKV model to the installed kernel's TTGIR and
executable PTX. The PTX includes the vector packing, two ldmatrix loads,
warp barriers, byte permutation and the two cache stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
open QKVLaunch



private def gluonGraph (w : Width) : String :=
  r#"
#blocked = #ttg.blocked<{sizePerThread = [2, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_quantize_qkv_tiles_gluon(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<768> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32, #blocked>
%cst_1 = arith.constant dense<-4.480000e+02> : tensor<64x128xf32, #blocked>
"# ++
  (match w with
  | .narrow => r#"
%cst_2 = arith.constant dense<1536> : tensor<1x128xi32, #blocked1>
"#
  | .halfValue => r#"
%cst_2 = arith.constant dense<1152> : tensor<1x128xi32, #blocked1>
"#
  | .wide => r#"
%cst_2 = arith.constant dense<2304> : tensor<1x128xi32, #blocked1>
"#) ++
  r#"
%batch = tt.get_program_id z : i32
%start_m = tt.get_program_id x : i32
%start_m_3 = arith.muli %start_m, 64 : i32
%start_n = tt.get_program_id y : i32
%start_n_4 = arith.muli %start_n, 128 : i32
%rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%qk = arith.cmpi slt, %start_m_3, 768 : i32
"#
  | .wide => r#"
%qk = arith.cmpi slt, %start_m_3, 1536 : i32
"#) ++
  r#"
%bank = arith.select %qk, %first, %second : !tt.ptr<bf16>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%bank_row = arith.select %qk, 0, 768 : i32
"#
  | .wide => r#"
%bank_row = arith.select %qk, 0, 1536 : i32
"#) ++
  r#"
%bank_row_5 = arith.subi %start_m_3, %bank_row : i32
"# ++
  (match w with
  | .narrow => r#"
%bank_row_6 = arith.select %qk, 2, 1 : i32
%bank_row_7 = arith.muli %bank_row_5, %bank_row_6 : i32
"#
  | .halfValue => ""
  | .wide => "") ++
  r#"
%source = arith.muli %batch, 1179648 : i32
"# ++
  (match w with
  | .narrow => r#"
%source_8 = tt.addptr %bank, %source : !tt.ptr<bf16>, i32
%source_9 = arith.muli %bank_row_7, 768 : i32
%source_10 = tt.addptr %source_8, %source_9 : !tt.ptr<bf16>, i32
"#
  | .halfValue => r#"
%source_6 = tt.addptr %bank, %source : !tt.ptr<bf16>, i32
%source_7 = arith.muli %bank_row_5, 1536 : i32
%source_8 = tt.addptr %source_6, %source_7 : !tt.ptr<bf16>, i32
"#
  | .wide => r#"
%source_6 = tt.addptr %bank, %source : !tt.ptr<bf16>, i32
%source_7 = arith.muli %bank_row_5, 768 : i32
%source_8 = tt.addptr %source_6, %source_7 : !tt.ptr<bf16>, i32
"#) ++
  r#"
%values = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"# ++
  (match w with
  | .narrow => r#"
%values_11 = arith.muli %values, %cst : tensor<64x1xi32, #blocked>
%values_12 = tt.splat %source_10 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
%values_13 = tt.addptr %values_12, %values_11 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_14 = tt.splat %start_n_4 : i32 -> tensor<64x1xi32, #blocked>
%values_15 = tt.addptr %values_13, %values_14 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_16 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%values_17 = tt.broadcast %values_15 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_18 = tt.broadcast %values_16 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%values_19 = tt.addptr %values_17, %values_18 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%values_20 = tt.load %values_19 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_21 = arith.extf %values_20 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
"#
  | .halfValue => r#"
%values_9 = arith.muli %values, %cst : tensor<64x1xi32, #blocked>
%values_10 = tt.splat %source_8 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
%values_11 = tt.addptr %values_10, %values_9 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_12 = tt.splat %start_n_4 : i32 -> tensor<64x1xi32, #blocked>
%values_13 = tt.addptr %values_11, %values_12 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_14 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%values_15 = tt.broadcast %values_13 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_16 = tt.broadcast %values_14 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%values_17 = tt.addptr %values_15, %values_16 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%values_18 = tt.load %values_17 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_19 = arith.extf %values_18 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
"#
  | .wide => r#"
%values_9 = arith.muli %values, %cst : tensor<64x1xi32, #blocked>
%values_10 = tt.splat %source_8 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
%values_11 = tt.addptr %values_10, %values_9 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_12 = tt.splat %start_n_4 : i32 -> tensor<64x1xi32, #blocked>
%values_13 = tt.addptr %values_11, %values_12 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
%values_14 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%values_15 = tt.broadcast %values_13 : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_16 = tt.broadcast %values_14 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%values_17 = tt.addptr %values_15, %values_16 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%values_18 = tt.load %values_17 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%values_19 = arith.extf %values_18 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
"#) ++
  r#"
%scaled = tt.addptr %scale_ptr, %batch : !tt.ptr<f32>, i32
"# ++
  (match w with
  | .narrow => r#"
%scaled_22 = tt.load %scaled : !tt.ptr<f32>
%scaled_23 = tt.splat %scaled_22 : f32 -> tensor<64x128xf32, #blocked>
%scaled_24 = arith.divf %values_21, %scaled_23 : tensor<64x128xf32, #blocked>
%quantized = arith.minnumf %scaled_24, %cst_0 : tensor<64x128xf32, #blocked>
%quantized_25 = arith.maxnumf %quantized, %cst_1 : tensor<64x128xf32, #blocked>
%quantized_26 = tt.fp_to_fp %quantized_25, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
"#
  | .halfValue => r#"
%scaled_20 = tt.load %scaled : !tt.ptr<f32>
%scaled_21 = tt.splat %scaled_20 : f32 -> tensor<64x128xf32, #blocked>
%scaled_22 = arith.divf %values_19, %scaled_21 : tensor<64x128xf32, #blocked>
%quantized = arith.minnumf %scaled_22, %cst_0 : tensor<64x128xf32, #blocked>
%quantized_23 = arith.maxnumf %quantized, %cst_1 : tensor<64x128xf32, #blocked>
%quantized_24 = tt.fp_to_fp %quantized_23, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
"#
  | .wide => r#"
%scaled_20 = tt.load %scaled : !tt.ptr<f32>
%scaled_21 = tt.splat %scaled_20 : f32 -> tensor<64x128xf32, #blocked>
%scaled_22 = arith.divf %values_19, %scaled_21 : tensor<64x128xf32, #blocked>
%quantized = arith.minnumf %scaled_22, %cst_0 : tensor<64x128xf32, #blocked>
%quantized_23 = arith.maxnumf %quantized, %cst_1 : tensor<64x128xf32, #blocked>
%quantized_24 = tt.fp_to_fp %quantized_23, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
"#) ++
  r#"
%offs_m = tt.splat %start_m_3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"# ++
  (match w with
  | .narrow => r#"
%offs_m_27 = arith.addi %offs_m, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"#
  | .halfValue => r#"
%offs_m_25 = arith.addi %offs_m, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"#
  | .wide => r#"
%offs_m_25 = arith.addi %offs_m, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"#) ++
  r#"
%offs_n = tt.splat %start_n_4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"# ++
  (match w with
  | .narrow => r#"
%offs_n_28 = arith.addi %offs_n, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%0 = tt.addptr %row, %source : !tt.ptr<f8E4M3FN>, i32
%1 = tt.expand_dims %offs_m_27 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%2 = arith.muli %1, %cst : tensor<64x1xi32, #blocked>
%3 = tt.splat %0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>
%4 = tt.addptr %3, %2 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
%5 = tt.expand_dims %offs_n_28 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%6 = tt.broadcast %4 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%7 = tt.broadcast %5 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%8 = tt.addptr %6, %7 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %8, %quantized_26 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %quantized_26 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
"#
  | .halfValue => r#"
%offs_n_26 = arith.addi %offs_n, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%0 = arith.muli %batch, 884736 : i32
%1 = tt.addptr %row, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%3 = arith.muli %2, %cst : tensor<64x1xi32, #blocked>
%4 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>
%5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
%6 = tt.expand_dims %offs_n_26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%8 = tt.broadcast %6 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%9 = tt.addptr %7, %8 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %9, %quantized_24 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %quantized_24 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
"#
  | .wide => r#"
%offs_n_26 = arith.addi %offs_n, %cols : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%0 = arith.muli %batch, 1769472 : i32
%1 = tt.addptr %row, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%3 = arith.muli %2, %cst : tensor<64x1xi32, #blocked>
%4 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>
%5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
%6 = tt.expand_dims %offs_n_26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%8 = tt.broadcast %6 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%9 = tt.addptr %7, %8 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %9, %quantized_24 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %quantized_24 : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
"#) ++
  r#"
%trans_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (match w with
  | .narrow => r#"
%trans_m_29 = tt.splat %start_m_3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_m_30 = arith.addi %trans_m_29, %trans_m : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#
  | .halfValue => r#"
%trans_m_27 = tt.splat %start_m_3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_m_28 = arith.addi %trans_m_27, %trans_m : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#
  | .wide => r#"
%trans_m_27 = tt.splat %start_m_3 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_m_28 = arith.addi %trans_m_27, %trans_m : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#) ++
  r#"
%trans_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
"# ++
  (match w with
  | .narrow => r#"
%trans_n_31 = tt.splat %start_n_4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%trans_n_32 = arith.addi %trans_n_31, %trans_n : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%9 = tt.addptr %transposed, %source : !tt.ptr<f8E4M3FN>, i32
%10 = tt.expand_dims %trans_n_32 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%11 = arith.muli %10, %cst_2 : tensor<1x128xi32, #blocked1>
%12 = tt.splat %9 : !tt.ptr<f8E4M3FN> -> tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>
%13 = tt.addptr %12, %11 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<1x128xi32, #blocked1>
%14 = tt.expand_dims %trans_m_30 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%15 = tt.broadcast %13 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%16 = tt.broadcast %14 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%17 = tt.addptr %15, %16 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %17, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
"#
  | .halfValue => r#"
%trans_n_29 = tt.splat %start_n_4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%trans_n_30 = arith.addi %trans_n_29, %trans_n : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%10 = tt.addptr %transposed, %0 : !tt.ptr<f8E4M3FN>, i32
%11 = tt.expand_dims %trans_n_30 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%12 = arith.muli %11, %cst_2 : tensor<1x128xi32, #blocked1>
%13 = tt.splat %10 : !tt.ptr<f8E4M3FN> -> tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>
%14 = tt.addptr %13, %12 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<1x128xi32, #blocked1>
%15 = tt.expand_dims %trans_m_28 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%16 = tt.broadcast %14 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%17 = tt.broadcast %15 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%18 = tt.addptr %16, %17 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %18, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
"#
  | .wide => r#"
%trans_n_29 = tt.splat %start_n_4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%trans_n_30 = arith.addi %trans_n_29, %trans_n : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%10 = tt.addptr %transposed, %0 : !tt.ptr<f8E4M3FN>, i32
%11 = tt.expand_dims %trans_n_30 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%12 = arith.muli %11, %cst_2 : tensor<1x128xi32, #blocked1>
%13 = tt.splat %10 : !tt.ptr<f8E4M3FN> -> tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>
%14 = tt.addptr %13, %12 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<1x128xi32, #blocked1>
%15 = tt.expand_dims %trans_m_28 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%16 = tt.broadcast %14 : tensor<1x128x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%17 = tt.broadcast %15 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%18 = tt.addptr %16, %17 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %18, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
"#) ++
  r#"
tt.return
}
}
"#

private def emittedPtx (w : Width) : String :=
  r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _quantize_qkv_tiles_gluon(
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_0,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_1,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_2,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_3,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_4,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_5,
.param .u64 .ptr .global .align 1 _quantize_qkv_tiles_gluon_param_6
)
.reqntid 256
{
.reg .pred %p<2>;
.reg .b16 %rs<49>;
"# ++
  (match w with
  | .narrow => r#"
.reg .b32 %r<213>;
"#
  | .halfValue => r#"
.reg .b32 %r<212>;
"#
  | .wide => r#"
.reg .b32 %r<212>;
"#) ++
  r#"
.reg .b64 %rd<41>;
$L__func_begin0:
ld.param.b64 %rd14, [_quantize_qkv_tiles_gluon_param_0];
$L__tmp0:
mov.u32 %r66, %ctaid.z;
ld.param.b64 %rd15, [_quantize_qkv_tiles_gluon_param_1];
ld.param.b64 %rd16, [_quantize_qkv_tiles_gluon_param_2];
mov.u32 %r67, %ctaid.x;
shl.b32 %r68, %r67, 6;
ld.param.b64 %rd17, [_quantize_qkv_tiles_gluon_param_3];
ld.param.b64 %rd18, [_quantize_qkv_tiles_gluon_param_4];
mov.u32 %r69, %ctaid.y;
shl.b32 %r70, %r69, 7;
mov.u32 %r71, %tid.x;
shr.u32 %r72, %r71, 2;
and.b32 %r73, %r72, 62;
shl.b32 %r74, %r71, 3;
and.b32 %r75, %r74, 56;
"# ++
  (match w with
  | .narrow | .halfValue => r#"
setp.lt.s32 %p1, %r68, 768;
"#
  | .wide => r#"
setp.lt.s32 %p1, %r68, 1536;
"#) ++
  r#"
selp.b64 %rd19, %rd14, %rd15, %p1;
"# ++
  (match w with
  | .narrow | .halfValue => r#"
selp.b32 %r76, 0, -768, %p1;
"#
  | .wide => r#"
selp.b32 %r76, 0, -1536, %p1;
"#) ++
  r#"
add.s32 %r77, %r76, %r68;
"# ++
  (match w with
  | .narrow => r#"
selp.b32 %r78, 1, 0, %p1;
shl.b32 %r79, %r77, %r78;
mul.lo.s32 %r80, %r66, 1179648;
cvt.s64.s32 %rd20, %r80;
mad.wide.s32 %rd21, %r80, 2, %rd19;
mul.lo.s32 %r81, %r79, 768;
mad.wide.s32 %rd22, %r81, 2, %rd21;
mul.lo.s32 %r82, %r73, 768;
mad.wide.u32 %rd23, %r82, 2, %rd22;
mad.wide.u32 %rd24, %r70, 2, %rd23;
mad.wide.u32 %rd1, %r75, 2, %rd24;
"#
  | .halfValue => r#"
mul.lo.s32 %r78, %r66, 1179648;
mad.wide.s32 %rd20, %r78, 2, %rd19;
mul.lo.s32 %r79, %r77, 1536;
mad.wide.s32 %rd21, %r79, 2, %rd20;
mul.lo.s32 %r80, %r73, 768;
mad.wide.u32 %rd22, %r80, 2, %rd21;
mad.wide.u32 %rd23, %r70, 2, %rd22;
mad.wide.u32 %rd1, %r75, 2, %rd23;
"#
  | .wide => r#"
mul.lo.s32 %r78, %r66, 1179648;
mad.wide.s32 %rd20, %r78, 2, %rd19;
mul.lo.s32 %r79, %r77, 768;
mad.wide.s32 %rd21, %r79, 2, %rd20;
mul.lo.s32 %r80, %r73, 768;
mad.wide.u32 %rd22, %r80, 2, %rd21;
mad.wide.u32 %rd23, %r70, 2, %rd22;
mad.wide.u32 %rd1, %r75, 2, %rd23;
"#) ++
  r#"
add.s64 %rd2, %rd1, 1536;
add.s64 %rd3, %rd1, 128;
add.s64 %rd4, %rd1, 1664;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mov.b32 {%rs17, %rs18}, %r1;
mov.b32 {%rs19, %rs20}, %r2;
mov.b32 {%rs21, %rs22}, %r3;
mov.b32 {%rs23, %rs24}, %r4;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mov.b32 {%rs25, %rs26}, %r5;
mov.b32 {%rs27, %rs28}, %r6;
mov.b32 {%rs29, %rs30}, %r7;
mov.b32 {%rs31, %rs32}, %r8;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd3 + 0 ];
mov.b32 {%rs33, %rs34}, %r9;
mov.b32 {%rs35, %rs36}, %r10;
mov.b32 {%rs37, %rs38}, %r11;
mov.b32 {%rs39, %rs40}, %r12;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd4 + 0 ];
mov.b32 {%rs41, %rs42}, %r13;
mov.b32 {%rs43, %rs44}, %r14;
mov.b32 {%rs45, %rs46}, %r15;
mov.b32 {%rs47, %rs48}, %r16;
"# ++
  (match w with
  | .narrow => r#"
cvt.f32.bf16 %r83, %rs17;
cvt.f32.bf16 %r84, %rs18;
cvt.f32.bf16 %r85, %rs19;
cvt.f32.bf16 %r86, %rs20;
cvt.f32.bf16 %r87, %rs21;
cvt.f32.bf16 %r88, %rs22;
cvt.f32.bf16 %r89, %rs23;
cvt.f32.bf16 %r90, %rs24;
cvt.f32.bf16 %r91, %rs25;
cvt.f32.bf16 %r92, %rs26;
cvt.f32.bf16 %r93, %rs27;
cvt.f32.bf16 %r94, %rs28;
cvt.f32.bf16 %r95, %rs29;
cvt.f32.bf16 %r96, %rs30;
cvt.f32.bf16 %r97, %rs31;
cvt.f32.bf16 %r98, %rs32;
cvt.f32.bf16 %r99, %rs33;
cvt.f32.bf16 %r100, %rs34;
cvt.f32.bf16 %r101, %rs35;
cvt.f32.bf16 %r102, %rs36;
cvt.f32.bf16 %r103, %rs37;
cvt.f32.bf16 %r104, %rs38;
cvt.f32.bf16 %r105, %rs39;
cvt.f32.bf16 %r106, %rs40;
cvt.f32.bf16 %r107, %rs41;
cvt.f32.bf16 %r108, %rs42;
cvt.f32.bf16 %r109, %rs43;
cvt.f32.bf16 %r110, %rs44;
cvt.f32.bf16 %r111, %rs45;
cvt.f32.bf16 %r112, %rs46;
cvt.f32.bf16 %r113, %rs47;
cvt.f32.bf16 %r114, %rs48;
"#
  | .halfValue => r#"
cvt.f32.bf16 %r81, %rs17;
cvt.f32.bf16 %r82, %rs18;
cvt.f32.bf16 %r83, %rs19;
cvt.f32.bf16 %r84, %rs20;
cvt.f32.bf16 %r85, %rs21;
cvt.f32.bf16 %r86, %rs22;
cvt.f32.bf16 %r87, %rs23;
cvt.f32.bf16 %r88, %rs24;
cvt.f32.bf16 %r89, %rs25;
cvt.f32.bf16 %r90, %rs26;
cvt.f32.bf16 %r91, %rs27;
cvt.f32.bf16 %r92, %rs28;
cvt.f32.bf16 %r93, %rs29;
cvt.f32.bf16 %r94, %rs30;
cvt.f32.bf16 %r95, %rs31;
cvt.f32.bf16 %r96, %rs32;
cvt.f32.bf16 %r97, %rs33;
cvt.f32.bf16 %r98, %rs34;
cvt.f32.bf16 %r99, %rs35;
cvt.f32.bf16 %r100, %rs36;
cvt.f32.bf16 %r101, %rs37;
cvt.f32.bf16 %r102, %rs38;
cvt.f32.bf16 %r103, %rs39;
cvt.f32.bf16 %r104, %rs40;
cvt.f32.bf16 %r105, %rs41;
cvt.f32.bf16 %r106, %rs42;
cvt.f32.bf16 %r107, %rs43;
cvt.f32.bf16 %r108, %rs44;
cvt.f32.bf16 %r109, %rs45;
cvt.f32.bf16 %r110, %rs46;
cvt.f32.bf16 %r111, %rs47;
cvt.f32.bf16 %r112, %rs48;
"#
  | .wide => r#"
cvt.f32.bf16 %r81, %rs17;
cvt.f32.bf16 %r82, %rs18;
cvt.f32.bf16 %r83, %rs19;
cvt.f32.bf16 %r84, %rs20;
cvt.f32.bf16 %r85, %rs21;
cvt.f32.bf16 %r86, %rs22;
cvt.f32.bf16 %r87, %rs23;
cvt.f32.bf16 %r88, %rs24;
cvt.f32.bf16 %r89, %rs25;
cvt.f32.bf16 %r90, %rs26;
cvt.f32.bf16 %r91, %rs27;
cvt.f32.bf16 %r92, %rs28;
cvt.f32.bf16 %r93, %rs29;
cvt.f32.bf16 %r94, %rs30;
cvt.f32.bf16 %r95, %rs31;
cvt.f32.bf16 %r96, %rs32;
cvt.f32.bf16 %r97, %rs33;
cvt.f32.bf16 %r98, %rs34;
cvt.f32.bf16 %r99, %rs35;
cvt.f32.bf16 %r100, %rs36;
cvt.f32.bf16 %r101, %rs37;
cvt.f32.bf16 %r102, %rs38;
cvt.f32.bf16 %r103, %rs39;
cvt.f32.bf16 %r104, %rs40;
cvt.f32.bf16 %r105, %rs41;
cvt.f32.bf16 %r106, %rs42;
cvt.f32.bf16 %r107, %rs43;
cvt.f32.bf16 %r108, %rs44;
cvt.f32.bf16 %r109, %rs45;
cvt.f32.bf16 %r110, %rs46;
cvt.f32.bf16 %r111, %rs47;
cvt.f32.bf16 %r112, %rs48;
"#) ++
  r#"
mad.wide.u32 %rd5, %r66, 4, %rd18;
mov.u32 %r17, 0x0;
ld.global.b32 { %r17 }, [ %rd5 + 0 ];
"# ++
  (match w with
  | .narrow => ""
  | .halfValue => r#"
div.full.f32 %r113, %r81, %r17;
div.full.f32 %r114, %r82, %r17;
"#
  | .wide => r#"
div.full.f32 %r113, %r81, %r17;
div.full.f32 %r114, %r82, %r17;
"#) ++
  r#"
div.full.f32 %r115, %r83, %r17;
div.full.f32 %r116, %r84, %r17;
div.full.f32 %r117, %r85, %r17;
div.full.f32 %r118, %r86, %r17;
div.full.f32 %r119, %r87, %r17;
div.full.f32 %r120, %r88, %r17;
div.full.f32 %r121, %r89, %r17;
div.full.f32 %r122, %r90, %r17;
div.full.f32 %r123, %r91, %r17;
div.full.f32 %r124, %r92, %r17;
div.full.f32 %r125, %r93, %r17;
div.full.f32 %r126, %r94, %r17;
div.full.f32 %r127, %r95, %r17;
div.full.f32 %r128, %r96, %r17;
div.full.f32 %r129, %r97, %r17;
div.full.f32 %r130, %r98, %r17;
div.full.f32 %r131, %r99, %r17;
div.full.f32 %r132, %r100, %r17;
div.full.f32 %r133, %r101, %r17;
div.full.f32 %r134, %r102, %r17;
div.full.f32 %r135, %r103, %r17;
div.full.f32 %r136, %r104, %r17;
div.full.f32 %r137, %r105, %r17;
div.full.f32 %r138, %r106, %r17;
div.full.f32 %r139, %r107, %r17;
div.full.f32 %r140, %r108, %r17;
div.full.f32 %r141, %r109, %r17;
div.full.f32 %r142, %r110, %r17;
div.full.f32 %r143, %r111, %r17;
div.full.f32 %r144, %r112, %r17;
"# ++
  (match w with
  | .narrow => r#"
div.full.f32 %r145, %r113, %r17;
div.full.f32 %r146, %r114, %r17;
"#
  | .halfValue => r#"
min.f32 %r145, %r113, 0f43E00000;
min.f32 %r146, %r114, 0f43E00000;
"#
  | .wide => r#"
min.f32 %r145, %r113, 0f43E00000;
min.f32 %r146, %r114, 0f43E00000;
"#) ++
  r#"
min.f32 %r147, %r115, 0f43E00000;
min.f32 %r148, %r116, 0f43E00000;
min.f32 %r149, %r117, 0f43E00000;
min.f32 %r150, %r118, 0f43E00000;
min.f32 %r151, %r119, 0f43E00000;
min.f32 %r152, %r120, 0f43E00000;
min.f32 %r153, %r121, 0f43E00000;
min.f32 %r154, %r122, 0f43E00000;
min.f32 %r155, %r123, 0f43E00000;
min.f32 %r156, %r124, 0f43E00000;
min.f32 %r157, %r125, 0f43E00000;
min.f32 %r158, %r126, 0f43E00000;
min.f32 %r159, %r127, 0f43E00000;
min.f32 %r160, %r128, 0f43E00000;
min.f32 %r161, %r129, 0f43E00000;
min.f32 %r162, %r130, 0f43E00000;
min.f32 %r163, %r131, 0f43E00000;
min.f32 %r164, %r132, 0f43E00000;
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
"# ++
  (match w with
  | .narrow => r#"
min.f32 %r177, %r145, 0f43E00000;
min.f32 %r178, %r146, 0f43E00000;
max.f32 %r18, %r147, 0fC3E00000;
max.f32 %r19, %r148, 0fC3E00000;
max.f32 %r20, %r149, 0fC3E00000;
max.f32 %r21, %r150, 0fC3E00000;
max.f32 %r22, %r151, 0fC3E00000;
max.f32 %r23, %r152, 0fC3E00000;
max.f32 %r24, %r153, 0fC3E00000;
max.f32 %r25, %r154, 0fC3E00000;
max.f32 %r26, %r155, 0fC3E00000;
max.f32 %r27, %r156, 0fC3E00000;
max.f32 %r28, %r157, 0fC3E00000;
max.f32 %r29, %r158, 0fC3E00000;
max.f32 %r30, %r159, 0fC3E00000;
max.f32 %r31, %r160, 0fC3E00000;
max.f32 %r32, %r161, 0fC3E00000;
max.f32 %r33, %r162, 0fC3E00000;
max.f32 %r34, %r163, 0fC3E00000;
max.f32 %r35, %r164, 0fC3E00000;
max.f32 %r36, %r165, 0fC3E00000;
max.f32 %r37, %r166, 0fC3E00000;
max.f32 %r38, %r167, 0fC3E00000;
max.f32 %r39, %r168, 0fC3E00000;
max.f32 %r40, %r169, 0fC3E00000;
max.f32 %r41, %r170, 0fC3E00000;
max.f32 %r42, %r171, 0fC3E00000;
max.f32 %r43, %r172, 0fC3E00000;
max.f32 %r44, %r173, 0fC3E00000;
max.f32 %r45, %r174, 0fC3E00000;
max.f32 %r46, %r175, 0fC3E00000;
max.f32 %r47, %r176, 0fC3E00000;
max.f32 %r48, %r177, 0fC3E00000;
max.f32 %r49, %r178, 0fC3E00000;
"#
  | .halfValue => r#"
max.f32 %r18, %r145, 0fC3E00000;
max.f32 %r19, %r146, 0fC3E00000;
max.f32 %r20, %r147, 0fC3E00000;
max.f32 %r21, %r148, 0fC3E00000;
max.f32 %r22, %r149, 0fC3E00000;
max.f32 %r23, %r150, 0fC3E00000;
max.f32 %r24, %r151, 0fC3E00000;
max.f32 %r25, %r152, 0fC3E00000;
max.f32 %r26, %r153, 0fC3E00000;
max.f32 %r27, %r154, 0fC3E00000;
max.f32 %r28, %r155, 0fC3E00000;
max.f32 %r29, %r156, 0fC3E00000;
max.f32 %r30, %r157, 0fC3E00000;
max.f32 %r31, %r158, 0fC3E00000;
max.f32 %r32, %r159, 0fC3E00000;
max.f32 %r33, %r160, 0fC3E00000;
max.f32 %r34, %r161, 0fC3E00000;
max.f32 %r35, %r162, 0fC3E00000;
max.f32 %r36, %r163, 0fC3E00000;
max.f32 %r37, %r164, 0fC3E00000;
max.f32 %r38, %r165, 0fC3E00000;
max.f32 %r39, %r166, 0fC3E00000;
max.f32 %r40, %r167, 0fC3E00000;
max.f32 %r41, %r168, 0fC3E00000;
max.f32 %r42, %r169, 0fC3E00000;
max.f32 %r43, %r170, 0fC3E00000;
max.f32 %r44, %r171, 0fC3E00000;
max.f32 %r45, %r172, 0fC3E00000;
max.f32 %r46, %r173, 0fC3E00000;
max.f32 %r47, %r174, 0fC3E00000;
max.f32 %r48, %r175, 0fC3E00000;
max.f32 %r49, %r176, 0fC3E00000;
"#
  | .wide => r#"
max.f32 %r18, %r145, 0fC3E00000;
max.f32 %r19, %r146, 0fC3E00000;
max.f32 %r20, %r147, 0fC3E00000;
max.f32 %r21, %r148, 0fC3E00000;
max.f32 %r22, %r149, 0fC3E00000;
max.f32 %r23, %r150, 0fC3E00000;
max.f32 %r24, %r151, 0fC3E00000;
max.f32 %r25, %r152, 0fC3E00000;
max.f32 %r26, %r153, 0fC3E00000;
max.f32 %r27, %r154, 0fC3E00000;
max.f32 %r28, %r155, 0fC3E00000;
max.f32 %r29, %r156, 0fC3E00000;
max.f32 %r30, %r157, 0fC3E00000;
max.f32 %r31, %r158, 0fC3E00000;
max.f32 %r32, %r159, 0fC3E00000;
max.f32 %r33, %r160, 0fC3E00000;
max.f32 %r34, %r161, 0fC3E00000;
max.f32 %r35, %r162, 0fC3E00000;
max.f32 %r36, %r163, 0fC3E00000;
max.f32 %r37, %r164, 0fC3E00000;
max.f32 %r38, %r165, 0fC3E00000;
max.f32 %r39, %r166, 0fC3E00000;
max.f32 %r40, %r167, 0fC3E00000;
max.f32 %r41, %r168, 0fC3E00000;
max.f32 %r42, %r169, 0fC3E00000;
max.f32 %r43, %r170, 0fC3E00000;
max.f32 %r44, %r171, 0fC3E00000;
max.f32 %r45, %r172, 0fC3E00000;
max.f32 %r46, %r173, 0fC3E00000;
max.f32 %r47, %r174, 0fC3E00000;
max.f32 %r48, %r175, 0fC3E00000;
max.f32 %r49, %r176, 0fC3E00000;
"#) ++
  r#"
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r19, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r21, %r20;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r23, %r22;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r25, %r24;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r27, %r26;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r29, %r28;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r31, %r30;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r33, %r32;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r35, %r34;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r37, %r36;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r39, %r38;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r41, %r40;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r43, %r42;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r45, %r44;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r47, %r46;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r49, %r48;
"# ++
  (match w with
  | .narrow => r#"
or.b32 %r179, %r73, %r68;
or.b32 %r180, %r179, 1;
or.b32 %r181, %r75, %r70;
add.s64 %rd25, %rd16, %rd20;
mul.lo.s32 %r182, %r179, 768;
mul.lo.s32 %r183, %r180, 768;
cvt.s64.s32 %rd26, %r182;
"#
  | .halfValue => r#"
or.b32 %r177, %r73, %r68;
or.b32 %r178, %r177, 1;
or.b32 %r179, %r75, %r70;
mul.lo.s32 %r180, %r66, 884736;
cvt.s64.s32 %rd24, %r180;
add.s64 %rd25, %rd16, %rd24;
mul.lo.s32 %r181, %r177, 768;
mul.lo.s32 %r182, %r178, 768;
cvt.s64.s32 %rd26, %r181;
"#
  | .wide => r#"
or.b32 %r177, %r73, %r68;
or.b32 %r178, %r177, 1;
or.b32 %r179, %r75, %r70;
mul.lo.s32 %r180, %r66, 1769472;
cvt.s64.s32 %rd24, %r180;
add.s64 %rd25, %rd16, %rd24;
mul.lo.s32 %r181, %r177, 768;
mul.lo.s32 %r182, %r178, 768;
cvt.s64.s32 %rd26, %r181;
"#) ++
  r#"
add.s64 %rd27, %rd25, %rd26;
"# ++
  (match w with
  | .narrow => r#"
cvt.s64.s32 %rd28, %r183;
"#
  | .halfValue => r#"
cvt.s64.s32 %rd28, %r182;
"#
  | .wide => r#"
cvt.s64.s32 %rd28, %r182;
"#) ++
  r#"
add.s64 %rd29, %rd25, %rd28;
"# ++
  (match w with
  | .narrow => r#"
cvt.u64.u32 %rd30, %r181;
"#
  | .halfValue => r#"
cvt.u64.u32 %rd30, %r179;
"#
  | .wide => r#"
cvt.u64.u32 %rd30, %r179;
"#) ++
  r#"
add.s64 %rd6, %rd27, %rd30;
add.s64 %rd7, %rd29, %rd30;
add.s64 %rd8, %rd6, 64;
add.s64 %rd9, %rd7, 64;
mov.b32 %r50, {%rs1, %rs2};
mov.b32 %r51, {%rs3, %rs4};
st.global.v2.b32 [ %rd6 + 0 ], { %r50, %r51 };
mov.b32 %r52, {%rs5, %rs6};
mov.b32 %r53, {%rs7, %rs8};
st.global.v2.b32 [ %rd7 + 0 ], { %r52, %r53 };
mov.b32 %r54, {%rs9, %rs10};
mov.b32 %r55, {%rs11, %rs12};
st.global.v2.b32 [ %rd8 + 0 ], { %r54, %r55 };
mov.b32 %r56, {%rs13, %rs14};
mov.b32 %r57, {%rs15, %rs16};
st.global.v2.b32 [ %rd9 + 0 ], { %r56, %r57 };
"# ++
  (match w with
  | .narrow => r#"
shl.b32 %r184, %r71, 4;
and.b32 %r185, %r184, 4080;
mov.b32 %r186, global_smem;
add.s32 %r187, %r186, %r185;
mov.b32 %r188, {%rs1, %rs5};
mov.b32 %r189, {%rs3, %rs7};
mov.b32 %r190, {%rs2, %rs6};
mov.b32 %r191, {%rs4, %rs8};
st.shared.v4.b32 [%r187], {%r188, %r190, %r189, %r191};
"#
  | .halfValue => r#"
shl.b32 %r183, %r71, 4;
and.b32 %r184, %r183, 4080;
mov.b32 %r185, global_smem;
add.s32 %r186, %r185, %r184;
mov.b32 %r187, {%rs1, %rs5};
mov.b32 %r188, {%rs3, %rs7};
mov.b32 %r189, {%rs2, %rs6};
mov.b32 %r190, {%rs4, %rs8};
st.shared.v4.b32 [%r186], {%r187, %r189, %r188, %r190};
"#
  | .wide => r#"
shl.b32 %r183, %r71, 4;
and.b32 %r184, %r183, 4080;
mov.b32 %r185, global_smem;
add.s32 %r186, %r185, %r184;
mov.b32 %r187, {%rs1, %rs5};
mov.b32 %r188, {%rs3, %rs7};
mov.b32 %r189, {%rs2, %rs6};
mov.b32 %r190, {%rs4, %rs8};
st.shared.v4.b32 [%r186], {%r187, %r189, %r188, %r190};
"#) ++
  r#"
bar.warp.sync -1;
"# ++
  (match w with
  | .narrow => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r192, %r193, %r194, %r195}, [%r187];
"#
  | .halfValue => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r191, %r192, %r193, %r194}, [%r186];
"#
  | .wide => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r191, %r192, %r193, %r194}, [%r186];
"#) ++
  r#"
bar.warp.sync -1;
"# ++
  (match w with
  | .narrow => r#"
mov.b32 %r196, {%rs9, %rs13};
mov.b32 %r197, {%rs11, %rs15};
mov.b32 %r198, {%rs10, %rs14};
mov.b32 %r199, {%rs12, %rs16};
st.shared.v4.b32 [%r187], {%r196, %r198, %r197, %r199};
"#
  | .halfValue => r#"
mov.b32 %r195, {%rs9, %rs13};
mov.b32 %r196, {%rs11, %rs15};
mov.b32 %r197, {%rs10, %rs14};
mov.b32 %r198, {%rs12, %rs16};
st.shared.v4.b32 [%r186], {%r195, %r197, %r196, %r198};
"#
  | .wide => r#"
mov.b32 %r195, {%rs9, %rs13};
mov.b32 %r196, {%rs11, %rs15};
mov.b32 %r197, {%rs10, %rs14};
mov.b32 %r198, {%rs12, %rs16};
st.shared.v4.b32 [%r186], {%r195, %r197, %r196, %r198};
"#) ++
  r#"
bar.warp.sync -1;
"# ++
  (match w with
  | .narrow => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r200, %r201, %r202, %r203}, [%r187];
and.b32 %r204, %r72, 56;
or.b32 %r205, %r204, %r68;
shl.b32 %r206, %r71, 1;
and.b32 %r207, %r206, 62;
or.b32 %r208, %r207, %r70;
add.s64 %rd31, %rd17, %rd20;
mul.lo.s32 %r209, %r208, 1536;
add.s32 %r210, %r209, 1536;
add.s32 %r211, %r209, 98304;
add.s32 %r212, %r209, 99840;
cvt.s64.s32 %rd32, %r209;
"#
  | .halfValue => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r201, %r202}, [%r186];
and.b32 %r203, %r72, 56;
or.b32 %r204, %r203, %r68;
shl.b32 %r205, %r71, 1;
and.b32 %r206, %r205, 62;
or.b32 %r207, %r206, %r70;
add.s64 %rd31, %rd17, %rd24;
mul.lo.s32 %r208, %r207, 1152;
add.s32 %r209, %r208, 1152;
add.s32 %r210, %r208, 73728;
add.s32 %r211, %r208, 74880;
cvt.s64.s32 %rd32, %r208;
"#
  | .wide => r#"
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r201, %r202}, [%r186];
and.b32 %r203, %r72, 56;
or.b32 %r204, %r203, %r68;
shl.b32 %r205, %r71, 1;
and.b32 %r206, %r205, 62;
or.b32 %r207, %r206, %r70;
add.s64 %rd31, %rd17, %rd24;
mul.lo.s32 %r208, %r207, 2304;
add.s32 %r209, %r208, 2304;
add.s32 %r210, %r208, 147456;
add.s32 %r211, %r208, 149760;
cvt.s64.s32 %rd32, %r208;
"#) ++
  r#"
add.s64 %rd33, %rd31, %rd32;
"# ++
  (match w with
  | .narrow => r#"
cvt.s64.s32 %rd34, %r210;
"#
  | .halfValue => r#"
cvt.s64.s32 %rd34, %r209;
"#
  | .wide => r#"
cvt.s64.s32 %rd34, %r209;
"#) ++
  r#"
add.s64 %rd35, %rd31, %rd34;
"# ++
  (match w with
  | .narrow => r#"
cvt.s64.s32 %rd36, %r211;
"#
  | .halfValue => r#"
cvt.s64.s32 %rd36, %r210;
"#
  | .wide => r#"
cvt.s64.s32 %rd36, %r210;
"#) ++
  r#"
add.s64 %rd37, %rd31, %rd36;
"# ++
  (match w with
  | .narrow => r#"
cvt.s64.s32 %rd38, %r212;
"#
  | .halfValue => r#"
cvt.s64.s32 %rd38, %r211;
"#
  | .wide => r#"
cvt.s64.s32 %rd38, %r211;
"#) ++
  r#"
add.s64 %rd39, %rd31, %rd38;
"# ++
  (match w with
  | .narrow => r#"
cvt.s64.s32 %rd40, %r205;
"#
  | .halfValue => r#"
cvt.s64.s32 %rd40, %r204;
"#
  | .wide => r#"
cvt.s64.s32 %rd40, %r204;
"#) ++
  r#"
add.s64 %rd10, %rd33, %rd40;
add.s64 %rd11, %rd35, %rd40;
add.s64 %rd12, %rd37, %rd40;
add.s64 %rd13, %rd39, %rd40;
"# ++
  (match w with
  | .narrow => r#"
prmt.b32 %r58, %r192, %r193, 0x6420U;
prmt.b32 %r59, %r194, %r195, 0x6420U;
"#
  | .halfValue => r#"
prmt.b32 %r58, %r191, %r192, 0x6420U;
prmt.b32 %r59, %r193, %r194, 0x6420U;
"#
  | .wide => r#"
prmt.b32 %r58, %r191, %r192, 0x6420U;
prmt.b32 %r59, %r193, %r194, 0x6420U;
"#) ++
  r#"
st.global.v2.b32 [ %rd10 + 0 ], { %r58, %r59 };
"# ++
  (match w with
  | .narrow => r#"
prmt.b32 %r60, %r192, %r193, 0x7531U;
prmt.b32 %r61, %r194, %r195, 0x7531U;
"#
  | .halfValue => r#"
prmt.b32 %r60, %r191, %r192, 0x7531U;
prmt.b32 %r61, %r193, %r194, 0x7531U;
"#
  | .wide => r#"
prmt.b32 %r60, %r191, %r192, 0x7531U;
prmt.b32 %r61, %r193, %r194, 0x7531U;
"#) ++
  r#"
st.global.v2.b32 [ %rd11 + 0 ], { %r60, %r61 };
"# ++
  (match w with
  | .narrow => r#"
prmt.b32 %r62, %r200, %r201, 0x6420U;
prmt.b32 %r63, %r202, %r203, 0x6420U;
"#
  | .halfValue => r#"
prmt.b32 %r62, %r199, %r200, 0x6420U;
prmt.b32 %r63, %r201, %r202, 0x6420U;
"#
  | .wide => r#"
prmt.b32 %r62, %r199, %r200, 0x6420U;
prmt.b32 %r63, %r201, %r202, 0x6420U;
"#) ++
  r#"
st.global.v2.b32 [ %rd12 + 0 ], { %r62, %r63 };
"# ++
  (match w with
  | .narrow => r#"
prmt.b32 %r64, %r200, %r201, 0x7531U;
prmt.b32 %r65, %r202, %r203, 0x7531U;
"#
  | .halfValue => r#"
prmt.b32 %r64, %r199, %r200, 0x7531U;
prmt.b32 %r65, %r201, %r202, 0x7531U;
"#
  | .wide => r#"
prmt.b32 %r64, %r199, %r200, 0x7531U;
prmt.b32 %r65, %r201, %r202, 0x7531U;
"#) ++
  r#"
st.global.v2.b32 [ %rd13 + 0 ], { %r64, %r65 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesGluon (w : Width) (baseline ttgir ptx : String) : Bool :=
  matchesReference w baseline &&
    QKVTTIR.normalizeTTIR ttgir == some (lines (gluonGraph w)) &&
    QKVTTIR.normalizePTX ptx == lines (emittedPtx w)

structure GluonCertificate (w : Width) (baseline ttgir ptx : String) : Type where
  matched : matchesGluon w baseline ttgir ptx = true
  phases : GluonShared.checkedPhases GluonShared.warpEvents = some [0, 1]
  stores : ∀ (op : Arithmetic) (inputs : Inputs) (p : Point (Geometry.ofWidth w)),
    observeGluon (QKVNative.nativePack op inputs w) p = reference op (QKVNative.compactInputs w inputs) p

def certifyGluon (w : Width) (baseline ttgir ptx : String) :
    Option (GluonCertificate w baseline ttgir ptx) :=
  if h : matchesGluon w baseline ttgir ptx = true then
    some ⟨h, GluonShared.phase_order, fun op inputs p => QKVNative.pack_equiv op inputs w p⟩
  else none

#axiomsClean certifyGluon
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
