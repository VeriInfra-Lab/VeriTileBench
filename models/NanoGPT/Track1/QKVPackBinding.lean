import models.NanoGPT.Track1.QKVPack
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Recognize the complete quantization TTIR for the three launch shapes.
The reference's M and N are launch arguments. The candidate specializes
them to the same dimensions. Shared multiplication results account for
the differences between width groups with equal strides. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
open QKVLaunch

private def referenceGraph (w : Width) : String :=
  r#"
module {
tt.func public @_quantize_dual_layout_packed_batched_kernel(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%values = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
%quantized = arith.constant dense<-4.480000e+02> : tensor<64x128xf32>
%quantized_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32>
%second_rows = arith.constant dense<0> : tensor<64xi32>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%second_rows_1 = arith.constant dense<768> : tensor<64xi32>
%in_first = arith.constant dense<768> : tensor<64x1xi32>
"#
  | .wide => r#"
%second_rows_1 = arith.constant dense<1536> : tensor<64xi32>
%in_first = arith.constant dense<1536> : tensor<64x1xi32>
"#) ++
  r#"
%batch = tt.get_program_id z : i32
%offs_m = tt.get_program_id x : i32
%offs_m_2 = arith.muli %offs_m, 64 : i32
%offs_m_3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%offs_m_4 = tt.splat %offs_m_2 : i32 -> tensor<64xi32>
%offs_m_5 = arith.addi %offs_m_4, %offs_m_3 : tensor<64xi32>
%offs_n = tt.get_program_id y : i32
%offs_n_6 = arith.muli %offs_n, 128 : i32
%offs_n_7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
%offs_n_8 = tt.splat %offs_n_6 : i32 -> tensor<128xi32>
%offs_n_9 = arith.addi %offs_n_8, %offs_n_7 : tensor<128xi32>
%valid = tt.expand_dims %offs_m_5 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%valid_10 = tt.splat %M : i32 -> tensor<64x1xi32>
%valid_11 = arith.cmpi slt, %valid, %valid_10 : tensor<64x1xi32>
%valid_12 = tt.expand_dims %offs_n_9 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
%valid_13 = tt.splat %N : i32 -> tensor<1x128xi32>
%valid_14 = arith.cmpi slt, %valid_12, %valid_13 : tensor<1x128xi32>
%valid_15 = tt.broadcast %valid_11 : tensor<64x1xi1> -> tensor<64x128xi1>
%valid_16 = tt.broadcast %valid_14 : tensor<1x128xi1> -> tensor<64x128xi1>
%valid_17 = arith.andi %valid_15, %valid_16 : tensor<64x128xi1>
%in_first_18 = arith.cmpi slt, %valid, %in_first : tensor<64x1xi32>
%first_offsets = tt.splat %N : i32 -> tensor<64x1xi32>
%first_offsets_19 = arith.muli %valid, %first_offsets : tensor<64x1xi32>
%first_offsets_20 = tt.broadcast %first_offsets_19 : tensor<64x1xi32> -> tensor<64x128xi32>
%first_offsets_21 = tt.broadcast %valid_12 : tensor<1x128xi32> -> tensor<64x128xi32>
%first_offsets_22 = arith.addi %first_offsets_20, %first_offsets_21 : tensor<64x128xi32>
%second_rows_23 = arith.subi %offs_m_5, %second_rows_1 : tensor<64xi32>
%second_rows_24 = arith.maxsi %second_rows_23, %second_rows : tensor<64xi32>
%second_offsets = tt.expand_dims %second_rows_24 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%second_offsets_25 = arith.muli %second_offsets, %first_offsets : tensor<64x1xi32>
%second_offsets_26 = tt.broadcast %second_offsets_25 : tensor<64x1xi32> -> tensor<64x128xi32>
%second_offsets_27 = arith.addi %second_offsets_26, %first_offsets_21 : tensor<64x128xi32>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%source_ptrs = arith.muli %batch, 589824 : i32
"#
  | .wide => r#"
%source_ptrs = arith.muli %batch, 1179648 : i32
"#) ++
  r#"
%source_ptrs_28 = tt.addptr %first, %source_ptrs : !tt.ptr<bf16>, i32
%source_ptrs_29 = tt.splat %source_ptrs_28 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
%source_ptrs_30 = tt.addptr %source_ptrs_29, %first_offsets_22 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
"# ++
  (match w with
  | .narrow => r#"
%source_ptrs_31 = arith.muli %batch, 1179648 : i32
%source_ptrs_32 = tt.addptr %second, %source_ptrs_31 : !tt.ptr<bf16>, i32
%source_ptrs_33 = tt.splat %source_ptrs_32 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
%source_ptrs_34 = tt.addptr %source_ptrs_33, %second_offsets_27 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
%source_ptrs_35 = tt.broadcast %in_first_18 : tensor<64x1xi1> -> tensor<64x128xi1>
%source_ptrs_36 = arith.select %source_ptrs_35, %source_ptrs_30, %source_ptrs_34 : tensor<64x128xi1>, tensor<64x128x!tt.ptr<bf16>>
%values_37 = tt.load %source_ptrs_36, %valid_17, %values : tensor<64x128x!tt.ptr<bf16>>
%values_38 = arith.extf %values_37 : tensor<64x128xbf16> to tensor<64x128xf32>
"#
  | .halfValue => r#"
%source_ptrs_31 = arith.muli %batch, 294912 : i32
%source_ptrs_32 = tt.addptr %second, %source_ptrs_31 : !tt.ptr<bf16>, i32
%source_ptrs_33 = tt.splat %source_ptrs_32 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
%source_ptrs_34 = tt.addptr %source_ptrs_33, %second_offsets_27 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
%source_ptrs_35 = tt.broadcast %in_first_18 : tensor<64x1xi1> -> tensor<64x128xi1>
%source_ptrs_36 = arith.select %source_ptrs_35, %source_ptrs_30, %source_ptrs_34 : tensor<64x128xi1>, tensor<64x128x!tt.ptr<bf16>>
%values_37 = tt.load %source_ptrs_36, %valid_17, %values : tensor<64x128x!tt.ptr<bf16>>
%values_38 = arith.extf %values_37 : tensor<64x128xbf16> to tensor<64x128xf32>
"#
  | .wide => r#"
%source_ptrs_31 = tt.addptr %second, %source_ptrs : !tt.ptr<bf16>, i32
%source_ptrs_32 = tt.splat %source_ptrs_31 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
%source_ptrs_33 = tt.addptr %source_ptrs_32, %second_offsets_27 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
%source_ptrs_34 = tt.broadcast %in_first_18 : tensor<64x1xi1> -> tensor<64x128xi1>
%source_ptrs_35 = arith.select %source_ptrs_34, %source_ptrs_30, %source_ptrs_33 : tensor<64x128xi1>, tensor<64x128x!tt.ptr<bf16>>
%values_36 = tt.load %source_ptrs_35, %valid_17, %values : tensor<64x128x!tt.ptr<bf16>>
%values_37 = arith.extf %values_36 : tensor<64x128xbf16> to tensor<64x128xf32>
"#) ++
  r#"
%scaled = tt.addptr %scale_ptr, %batch : !tt.ptr<f32>, i32
"# ++
  (match w with
  | .narrow => r#"
%scaled_39 = tt.load %scaled : !tt.ptr<f32>
%scaled_40 = tt.splat %scaled_39 : f32 -> tensor<64x128xf32>
%scaled_41 = arith.divf %values_38, %scaled_40 : tensor<64x128xf32>
%quantized_42 = arith.minnumf %scaled_41, %quantized_0 : tensor<64x128xf32>
%quantized_43 = arith.maxnumf %quantized_42, %quantized : tensor<64x128xf32>
%quantized_44 = tt.fp_to_fp %quantized_43, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
%0 = tt.addptr %row, %source_ptrs_31 : !tt.ptr<f8E4M3FN>, i32
%1 = tt.splat %0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%2 = tt.addptr %1, %first_offsets_19 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%3 = tt.broadcast %2 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
%4 = tt.addptr %3, %first_offsets_21 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
tt.store %4, %quantized_44, %valid_17 : tensor<64x128x!tt.ptr<f8E4M3FN>>
"#
  | .halfValue => r#"
%scaled_39 = tt.load %scaled : !tt.ptr<f32>
%scaled_40 = tt.splat %scaled_39 : f32 -> tensor<64x128xf32>
%scaled_41 = arith.divf %values_38, %scaled_40 : tensor<64x128xf32>
%quantized_42 = arith.minnumf %scaled_41, %quantized_0 : tensor<64x128xf32>
%quantized_43 = arith.maxnumf %quantized_42, %quantized : tensor<64x128xf32>
%quantized_44 = tt.fp_to_fp %quantized_43, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
%0 = arith.muli %batch, 884736 : i32
%1 = tt.addptr %row, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%3 = tt.addptr %2, %first_offsets_19 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%4 = tt.broadcast %3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
%5 = tt.addptr %4, %first_offsets_21 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
tt.store %5, %quantized_44, %valid_17 : tensor<64x128x!tt.ptr<f8E4M3FN>>
"#
  | .wide => r#"
%scaled_38 = tt.load %scaled : !tt.ptr<f32>
%scaled_39 = tt.splat %scaled_38 : f32 -> tensor<64x128xf32>
%scaled_40 = arith.divf %values_37, %scaled_39 : tensor<64x128xf32>
%quantized_41 = arith.minnumf %scaled_40, %quantized_0 : tensor<64x128xf32>
%quantized_42 = arith.maxnumf %quantized_41, %quantized : tensor<64x128xf32>
%quantized_43 = tt.fp_to_fp %quantized_42, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
%0 = arith.muli %batch, 1769472 : i32
%1 = tt.addptr %row, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%3 = tt.addptr %2, %first_offsets_19 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%4 = tt.broadcast %3 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
%5 = tt.addptr %4, %first_offsets_21 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
tt.store %5, %quantized_43, %valid_17 : tensor<64x128x!tt.ptr<f8E4M3FN>>
"#) ++
  r#"
%transposed_mask = tt.expand_dims %offs_n_9 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
"# ++
  (match w with
  | .narrow => r#"
%transposed_mask_45 = tt.splat %N : i32 -> tensor<128x1xi32>
%transposed_mask_46 = arith.cmpi slt, %transposed_mask, %transposed_mask_45 : tensor<128x1xi32>
%transposed_mask_47 = tt.expand_dims %offs_m_5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%transposed_mask_48 = tt.splat %M : i32 -> tensor<1x64xi32>
%transposed_mask_49 = arith.cmpi slt, %transposed_mask_47, %transposed_mask_48 : tensor<1x64xi32>
%transposed_mask_50 = tt.broadcast %transposed_mask_46 : tensor<128x1xi1> -> tensor<128x64xi1>
%transposed_mask_51 = tt.broadcast %transposed_mask_49 : tensor<1x64xi1> -> tensor<128x64xi1>
%transposed_mask_52 = arith.andi %transposed_mask_50, %transposed_mask_51 : tensor<128x64xi1>
%5 = tt.addptr %transposed, %source_ptrs_31 : !tt.ptr<f8E4M3FN>, i32
%6 = tt.splat %M : i32 -> tensor<128x1xi32>
%7 = arith.muli %transposed_mask, %6 : tensor<128x1xi32>
%8 = tt.splat %5 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
%9 = tt.addptr %8, %7 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
%10 = tt.broadcast %9 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x64x!tt.ptr<f8E4M3FN>>
%11 = tt.broadcast %transposed_mask_47 : tensor<1x64xi32> -> tensor<128x64xi32>
%12 = tt.addptr %10, %11 : tensor<128x64x!tt.ptr<f8E4M3FN>>, tensor<128x64xi32>
%13 = tt.trans %quantized_44 {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
tt.store %12, %13, %transposed_mask_52 : tensor<128x64x!tt.ptr<f8E4M3FN>>
"#
  | .halfValue => r#"
%transposed_mask_45 = tt.splat %N : i32 -> tensor<128x1xi32>
%transposed_mask_46 = arith.cmpi slt, %transposed_mask, %transposed_mask_45 : tensor<128x1xi32>
%transposed_mask_47 = tt.expand_dims %offs_m_5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%transposed_mask_48 = tt.splat %M : i32 -> tensor<1x64xi32>
%transposed_mask_49 = arith.cmpi slt, %transposed_mask_47, %transposed_mask_48 : tensor<1x64xi32>
%transposed_mask_50 = tt.broadcast %transposed_mask_46 : tensor<128x1xi1> -> tensor<128x64xi1>
%transposed_mask_51 = tt.broadcast %transposed_mask_49 : tensor<1x64xi1> -> tensor<128x64xi1>
%transposed_mask_52 = arith.andi %transposed_mask_50, %transposed_mask_51 : tensor<128x64xi1>
%6 = tt.addptr %transposed, %0 : !tt.ptr<f8E4M3FN>, i32
%7 = tt.splat %M : i32 -> tensor<128x1xi32>
%8 = arith.muli %transposed_mask, %7 : tensor<128x1xi32>
%9 = tt.splat %6 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
%10 = tt.addptr %9, %8 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
%11 = tt.broadcast %10 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x64x!tt.ptr<f8E4M3FN>>
%12 = tt.broadcast %transposed_mask_47 : tensor<1x64xi32> -> tensor<128x64xi32>
%13 = tt.addptr %11, %12 : tensor<128x64x!tt.ptr<f8E4M3FN>>, tensor<128x64xi32>
%14 = tt.trans %quantized_44 {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
tt.store %13, %14, %transposed_mask_52 : tensor<128x64x!tt.ptr<f8E4M3FN>>
"#
  | .wide => r#"
%transposed_mask_44 = tt.splat %N : i32 -> tensor<128x1xi32>
%transposed_mask_45 = arith.cmpi slt, %transposed_mask, %transposed_mask_44 : tensor<128x1xi32>
%transposed_mask_46 = tt.expand_dims %offs_m_5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%transposed_mask_47 = tt.splat %M : i32 -> tensor<1x64xi32>
%transposed_mask_48 = arith.cmpi slt, %transposed_mask_46, %transposed_mask_47 : tensor<1x64xi32>
%transposed_mask_49 = tt.broadcast %transposed_mask_45 : tensor<128x1xi1> -> tensor<128x64xi1>
%transposed_mask_50 = tt.broadcast %transposed_mask_48 : tensor<1x64xi1> -> tensor<128x64xi1>
%transposed_mask_51 = arith.andi %transposed_mask_49, %transposed_mask_50 : tensor<128x64xi1>
%6 = tt.addptr %transposed, %0 : !tt.ptr<f8E4M3FN>, i32
%7 = tt.splat %M : i32 -> tensor<128x1xi32>
%8 = arith.muli %transposed_mask, %7 : tensor<128x1xi32>
%9 = tt.splat %6 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
%10 = tt.addptr %9, %8 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
%11 = tt.broadcast %10 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x64x!tt.ptr<f8E4M3FN>>
%12 = tt.broadcast %transposed_mask_46 : tensor<1x64xi32> -> tensor<128x64xi32>
%13 = tt.addptr %11, %12 : tensor<128x64x!tt.ptr<f8E4M3FN>>, tensor<128x64xi32>
%14 = tt.trans %quantized_43 {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
tt.store %13, %14, %transposed_mask_51 : tensor<128x64x!tt.ptr<f8E4M3FN>>
"#) ++
  r#"
tt.return
}
}
"#

private def alignedGraph (w : Width) : String :=
  r#"
module {
tt.func public @_quantize_aligned_qkv_tiles(%first: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %second: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
"# ++
  (match w with
  | .narrow => r#"
%cst = arith.constant dense<1536> : tensor<128x1xi32>
"#
  | .halfValue => r#"
%cst = arith.constant dense<1152> : tensor<128x1xi32>
"#
  | .wide => r#"
%cst = arith.constant dense<2304> : tensor<128x1xi32>
"#) ++
  r#"
%quantized = arith.constant dense<-4.480000e+02> : tensor<64x128xf32>
%quantized_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32>
%cst_1 = arith.constant dense<768> : tensor<64x1xi32>
%batch = tt.get_program_id z : i32
%start_m = tt.get_program_id x : i32
%start_m_2 = arith.muli %start_m, 64 : i32
%start_n = tt.get_program_id y : i32
%start_n_3 = arith.muli %start_n, 128 : i32
%rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%cols = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%in_first = arith.cmpi slt, %start_m_2, 768 : i32
"#
  | .wide => r#"
%in_first = arith.cmpi slt, %start_m_2, 1536 : i32
"#) ++
  r#"
%bank = arith.select %in_first, %first, %second : !tt.ptr<bf16>
"# ++
  (match w with
  | .narrow => r#"
%pitch = arith.select %in_first, 589824, 1179648 : i32
%bank_row = arith.select %in_first, 0, 768 : i32
"#
  | .halfValue => r#"
%pitch = arith.select %in_first, 589824, 294912 : i32
%bank_row = arith.select %in_first, 0, 768 : i32
"#
  | .wide => r#"
%bank_row = arith.select %in_first, 0, 1536 : i32
"#) ++
  r#"
%bank_row_4 = arith.subi %start_m_2, %bank_row : i32
"# ++
  (match w with
  | .narrow | .halfValue => r#"
%source = arith.muli %batch, %pitch : i32
"#
  | .wide => r#"
%source = arith.muli %batch, 1179648 : i32
"#) ++
  r#"
%source_5 = tt.addptr %bank, %source : !tt.ptr<bf16>, i32
%source_6 = arith.muli %bank_row_4, 768 : i32
%source_7 = tt.addptr %source_5, %source_6 : !tt.ptr<bf16>, i32
%values = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%values_8 = arith.muli %values, %cst_1 : tensor<64x1xi32>
%values_9 = tt.splat %source_7 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
%values_10 = tt.addptr %values_9, %values_8 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
%values_11 = tt.splat %start_n_3 : i32 -> tensor<64x1xi32>
%values_12 = tt.addptr %values_10, %values_11 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
%values_13 = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
%values_14 = tt.broadcast %values_12 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
%values_15 = tt.broadcast %values_13 : tensor<1x128xi32> -> tensor<64x128xi32>
%values_16 = tt.addptr %values_14, %values_15 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
%values_17 = tt.load %values_16 : tensor<64x128x!tt.ptr<bf16>>
%values_18 = arith.extf %values_17 : tensor<64x128xbf16> to tensor<64x128xf32>
%scaled = tt.addptr %scale_ptr, %batch : !tt.ptr<f32>, i32
%scaled_19 = tt.load %scaled : !tt.ptr<f32>
%scaled_20 = tt.splat %scaled_19 : f32 -> tensor<64x128xf32>
%scaled_21 = arith.divf %values_18, %scaled_20 : tensor<64x128xf32>
%quantized_22 = arith.minnumf %scaled_21, %quantized_0 : tensor<64x128xf32>
%quantized_23 = arith.maxnumf %quantized_22, %quantized : tensor<64x128xf32>
%quantized_24 = tt.fp_to_fp %quantized_23, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
%offs_m = tt.splat %start_m_2 : i32 -> tensor<64xi32>
%offs_m_25 = arith.addi %offs_m, %rows : tensor<64xi32>
%offs_n = tt.splat %start_n_3 : i32 -> tensor<128xi32>
%offs_n_26 = arith.addi %offs_n, %cols : tensor<128xi32>
"# ++
  (match w with
  | .narrow => r#"
%0 = arith.muli %batch, 1179648 : i32
"#
  | .halfValue => r#"
%0 = arith.muli %batch, 884736 : i32
"#
  | .wide => r#"
%0 = arith.muli %batch, 1769472 : i32
"#) ++
  r#"
%1 = tt.addptr %row, %0 : !tt.ptr<f8E4M3FN>, i32
%2 = tt.expand_dims %offs_m_25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
%3 = arith.muli %2, %cst_1 : tensor<64x1xi32>
%4 = tt.splat %1 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
%6 = tt.expand_dims %offs_n_26 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
%7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
%8 = tt.broadcast %6 : tensor<1x128xi32> -> tensor<64x128xi32>
%9 = tt.addptr %7, %8 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi32>
tt.store %9, %quantized_24 : tensor<64x128x!tt.ptr<f8E4M3FN>>
%10 = tt.addptr %transposed, %0 : !tt.ptr<f8E4M3FN>, i32
%11 = tt.expand_dims %offs_n_26 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
%12 = arith.muli %11, %cst : tensor<128x1xi32>
%13 = tt.splat %10 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
%14 = tt.addptr %13, %12 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi32>
%15 = tt.expand_dims %offs_m_25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
%16 = tt.broadcast %14 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x64x!tt.ptr<f8E4M3FN>>
%17 = tt.broadcast %15 : tensor<1x64xi32> -> tensor<128x64xi32>
%18 = tt.addptr %16, %17 : tensor<128x64x!tt.ptr<f8E4M3FN>>, tensor<128x64xi32>
%19 = tt.trans %quantized_24 {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN> -> tensor<128x64xf8E4M3FN>
tt.store %18, %19 : tensor<128x64x!tt.ptr<f8E4M3FN>>
tt.return
}
}
"#

private def expected (s : String) : Option (List String) :=
  some ((s.splitOn "\n").filter (!·.isEmpty))

def matchesReference (w : Width) (baseline : String) : Bool :=
  QKVTTIR.normalizeTTIR baseline == expected (referenceGraph w)

def matchesPair (w : Width) (baseline candidate : String) : Bool :=
  matchesReference w baseline && QKVTTIR.normalizeTTIR candidate == expected (alignedGraph w)

structure Certificate (w : Width) (baseline candidate : String) : Type where
  matched : matchesPair w baseline candidate = true
  stores : ∀ (op : Arithmetic) (inputs : Inputs) (p : Point (Geometry.ofWidth w)),
    some (emitted op inputs w p) = reference op inputs p

def certify (w : Width) (baseline candidate : String) : Option (Certificate w baseline candidate) :=
  if h : matchesPair w baseline candidate = true then
    some ⟨h, fun op inputs p => emitted_equiv op inputs w p⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
