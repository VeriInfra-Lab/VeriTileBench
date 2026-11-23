import models.NanoGPT.Track1.RopeBackward
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized Hopper artifacts for the two 64-channel backward layouts.
The templates share code across the four training lengths. Debug metadata
is removed by the common TTGIR and PTX normalizers. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward.Narrow
open RopeShift (Length Head)

private def pairedTTGIR (vector : Bool) : String :=
  r#"#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
"# ++
  (if vector then r#"#blocked3 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
"# else r#"#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
"#) ++
  r#"module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_qkv_norm_rope_pack_fp8_backward_kernel(%grad_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_v: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %qk: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<1536> : tensor<32x1xi32, #blocked>
%cst_0 = arith.constant dense<1536> : tensor<32x1xi32, #blocked1>
%cst_1 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
@EXTENT_0@
%cst_3 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
%cst_4 = arith.constant dense<128> : tensor<128x1xi32, #blocked2>
@EXTENT_1@
%cst_6 = arith.constant dense<192> : tensor<32x1xi32, #blocked1>
%cst_7 = arith.constant dense<768> : tensor<32x1xi32, #blocked>
%cst_8 = arith.constant dense<768> : tensor<32x1xi32, #blocked1>
"# ++
  (if vector then r#"%cst_9 = arith.constant dense<128> : tensor<32x1xi32, #blocked1>
%cst_10 = arith.constant dense<2> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_11 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
@EXTENT_3@
%cst_15 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked1>
%cst_16 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #blocked>
%cst_17 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_18 = arith.constant dense<6.400000e+01> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# else r#"%cst_9 = arith.constant dense<128> : tensor<32x1xi32, #blocked3>
%cst_10 = arith.constant dense<128> : tensor<32x1xi32, #blocked1>
%cst_11 = arith.constant dense<2> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_12 = arith.constant dense<1> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
%cst_13 = arith.constant dense<64> : tensor<1x64xi32, #blocked3>
%cst_14 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
@EXTENT_2@
%cst_19 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked1>
%cst_20 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked3>
%cst_21 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #blocked>
%cst_22 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_23 = arith.constant dense<6.400000e+01> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#) ++
  r#"%logical_head = tt.get_program_id x : i32
%token = tt.get_program_id y : i32
"# ++
  (if vector then r#"%token_19 = arith.muli %token, 32 : i32
%token_20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_21 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%token_23 = tt.splat %token_19 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_24 = tt.splat %token_19 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_25 = tt.splat %token_19 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%token_26 = arith.addi %token_23, %token_20 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_27 = arith.addi %token_24, %token_21 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_28 = arith.addi %token_25, %token_22 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offs_d = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_mask = arith.cmpi slt, %token_26, %cst_14 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_30 = arith.cmpi slt, %token_27, %cst_13 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask_31 = arith.cmpi slt, %token_28, %cst_12 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
"# else r#"%token_24 = arith.muli %token, 32 : i32
%token_25 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%token_27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_28 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%token_29 = tt.splat %token_24 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_30 = tt.splat %token_24 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%token_31 = tt.splat %token_24 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_32 = tt.splat %token_24 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%token_33 = arith.addi %token_29, %token_25 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_34 = arith.addi %token_30, %token_26 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%token_35 = arith.addi %token_31, %token_27 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_36 = arith.addi %token_32, %token_28 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offs_d = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
%offs_d_37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_mask = arith.cmpi slt, %token_33, %cst_18 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_39 = arith.cmpi slt, %token_34, %cst_17 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%token_mask_40 = arith.cmpi slt, %token_35, %cst_16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask_41 = arith.cmpi slt, %token_36, %cst_15 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
"#) ++
  r#"%qk_mask = tt.expand_dims %token_mask {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi1, #blocked1>
"# ++
  (if vector then r#"%qk_mask_32 = tt.expand_dims %token_mask_30 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
%qk_mask_33 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%qk_mask_34 = arith.cmpi slt, %qk_mask_33, %cst_11 : tensor<1x64xi32, #blocked1>
%qk_mask_35 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_36 = tt.broadcast %qk_mask_34 : tensor<1x64xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_37 = arith.andi %qk_mask_35, %qk_mask_36 : tensor<32x64xi1, #blocked1>
%output_token = arith.muli %token_26, %cst_10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%output_token_38 = arith.divsi %logical_head, 3 : i32
%output_token_39 = tt.splat %output_token_38 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%output_token_40 = arith.addi %output_token, %output_token_39 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# else r#"%qk_mask_42 = tt.expand_dims %token_mask_39 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi1, #blocked3>
%qk_mask_43 = tt.expand_dims %token_mask_40 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi1, #blocked>
%qk_mask_44 = tt.expand_dims %offs_d_37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%qk_mask_45 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
%qk_mask_46 = arith.cmpi slt, %qk_mask_44, %cst_14 : tensor<1x64xi32, #blocked1>
%qk_mask_47 = arith.cmpi slt, %qk_mask_45, %cst_13 : tensor<1x64xi32, #blocked3>
%qk_mask_48 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_49 = tt.broadcast %qk_mask_42 : tensor<32x1xi1, #blocked3> -> tensor<32x64xi1, #blocked3>
%qk_mask_50 = tt.broadcast %qk_mask_46 : tensor<1x64xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_51 = tt.broadcast %qk_mask_47 : tensor<1x64xi1, #blocked3> -> tensor<32x64xi1, #blocked3>
%qk_mask_52 = arith.andi %qk_mask_48, %qk_mask_50 : tensor<32x64xi1, #blocked1>
%qk_mask_53 = arith.andi %qk_mask_49, %qk_mask_51 : tensor<32x64xi1, #blocked3>
%offs_flip = arith.xori %offs_d, %cst_12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
%output_token = arith.muli %token_33, %cst_11 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%output_token_54 = arith.divsi %logical_head, 3 : i32
%output_token_55 = tt.splat %output_token_54 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%output_token_56 = arith.addi %output_token, %output_token_55 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#) ++
  r#"%output_head = arith.remsi %logical_head, 3 : i32
%factor_offset = arith.remsi %logical_head, 2 : i32
"# ++
  (if vector then r#"%factor_offset_41 = arith.muli %factor_offset, 64 : i32
%f1 = tt.expand_dims %token_26 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_42 = tt.expand_dims %token_27 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
%f1_43 = arith.muli %f1, %cst_9 : tensor<32x1xi32, #blocked1>
%f1_44 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_45 = tt.addptr %f1_44, %f1_43 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_46 = tt.splat %factor_offset_41 : i32 -> tensor<32x1xi32, #blocked1>
%f1_47 = tt.addptr %f1_45, %f1_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_48 = tt.broadcast %f1_47 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_49 = tt.broadcast %qk_mask_33 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%f1_50 = tt.addptr %f1_48, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f1_51 = tt.load %f1_50, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_52 = arith.extf %f1_51 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%f2 = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f2_53 = tt.addptr %f2, %f1_43 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f2_54 = tt.addptr %f2_53, %f1_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f2_55 = tt.broadcast %f2_54 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f2_56 = tt.addptr %f2_55, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f2_57 = tt.load %f2_56, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f2_flip = tt.reshape %f2_57 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%y = tt.bitcast %f2_flip : tensor<32x32x2xbf16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_58 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_194: i16, %y_195: i16):
%y_196 = arith.xori %y_194, %y_195 : i16
tt.reduce.return %y_196 : i16
}) : (tensor<32x32x2xi16, #blocked3>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_59 = tt.expand_dims %y_58 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x32x1xi16, #blocked3>
%y_60 = tt.broadcast %y_59 : tensor<32x32x1xi16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_61 = arith.xori %y, %y_60 : tensor<32x32x2xi16, #blocked3>
%x = tt.bitcast %y_61 : tensor<32x32x2xi16, #blocked3> -> tensor<32x32x2xbf16, #blocked3>
%f2_flip_62 = tt.reshape %x : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%f2_flip_63 = arith.extf %f2_flip_62 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"# else r#"%factor_offset_57 = arith.muli %factor_offset, 64 : i32
%f1 = tt.expand_dims %token_33 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_58 = tt.expand_dims %token_34 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi32, #blocked3>
%f1_59 = tt.expand_dims %token_35 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
%f1_60 = arith.muli %f1, %cst_10 : tensor<32x1xi32, #blocked1>
%f1_61 = arith.muli %f1_58, %cst_9 : tensor<32x1xi32, #blocked3>
%f1_62 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_63 = tt.addptr %f1_62, %f1_60 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_64 = tt.splat %factor_offset_57 : i32 -> tensor<32x1xi32, #blocked1>
%f1_65 = tt.splat %factor_offset_57 : i32 -> tensor<32x1xi32, #blocked3>
%f1_66 = tt.addptr %f1_63, %f1_64 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_67 = tt.broadcast %f1_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_68 = tt.broadcast %qk_mask_44 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%f1_69 = tt.addptr %f1_67, %f1_68 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f1_70 = tt.load %f1_69, %qk_mask_52, %cst_19 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_71 = arith.extf %f1_70 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%f2_flip = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked3>
%f2_flip_72 = tt.addptr %f2_flip, %f1_61 : tensor<32x1x!tt.ptr<bf16>, #blocked3>, tensor<32x1xi32, #blocked3>
%f2_flip_73 = tt.addptr %f2_flip_72, %f1_65 : tensor<32x1x!tt.ptr<bf16>, #blocked3>, tensor<32x1xi32, #blocked3>
%f2_flip_74 = tt.expand_dims %offs_flip {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
%f2_flip_75 = tt.broadcast %f2_flip_73 : tensor<32x1x!tt.ptr<bf16>, #blocked3> -> tensor<32x64x!tt.ptr<bf16>, #blocked3>
%f2_flip_76 = tt.broadcast %f2_flip_74 : tensor<1x64xi32, #blocked3> -> tensor<32x64xi32, #blocked3>
%f2_flip_77 = tt.addptr %f2_flip_75, %f2_flip_76 : tensor<32x64x!tt.ptr<bf16>, #blocked3>, tensor<32x64xi32, #blocked3>
%f2_flip_78 = tt.load %f2_flip_77, %qk_mask_53, %cst_20 : tensor<32x64x!tt.ptr<bf16>, #blocked3>
%f2_flip_79 = ttg.convert_layout %f2_flip_78 : tensor<32x64xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%f2_flip_80 = arith.extf %f2_flip_79 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"#) ++
  r#"%scale = tt.load %grad_scale_ptr : !tt.ptr<f32>
"# ++
  (if vector then r#"%x_64 = arith.muli %f1, %cst_8 : tensor<32x1xi32, #blocked1>
%x_65 = arith.muli %f1_42, %cst_7 : tensor<32x1xi32, #blocked>
%x_66 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_67 = tt.addptr %x_66, %x_64 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_68 = arith.muli %logical_head, 64 : i32
%x_69 = tt.splat %x_68 : i32 -> tensor<32x1xi32, #blocked1>
%x_70 = tt.addptr %x_67, %x_69 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_71 = tt.broadcast %x_70 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_72 = tt.addptr %x_71, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_73 = tt.load %x_72, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_74 = arith.extf %x_73 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr = tt.expand_dims %output_token_40 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%grad_ptr_75 = arith.muli %grad_ptr, %cst_6 : tensor<32x1xi32, #blocked1>
%grad_ptr_76 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_77 = tt.addptr %grad_ptr_76, %grad_ptr_75 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_78 = arith.muli %output_head, 64 : i32
%grad_ptr_79 = tt.splat %grad_ptr_78 : i32 -> tensor<32x1xi32, #blocked1>
%grad_ptr_80 = tt.addptr %grad_ptr_77, %grad_ptr_79 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_81 = tt.broadcast %grad_ptr_80 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_82 = tt.addptr %grad_ptr_81, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad = tt.load %grad_ptr_82, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%y_83 = tt.bitcast %grad_flip : tensor<32x32x2xbf16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_84 = "tt.reduce"(%y_83) <{axis = 2 : i32}> ({
^bb0(%y_194: i16, %y_195: i16):
%y_196 = arith.xori %y_194, %y_195 : i16
tt.reduce.return %y_196 : i16
}) : (tensor<32x32x2xi16, #blocked3>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_85 = tt.expand_dims %y_84 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x32x1xi16, #blocked3>
%y_86 = tt.broadcast %y_85 : tensor<32x32x1xi16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_87 = arith.xori %y_83, %y_86 : tensor<32x32x2xi16, #blocked3>
%x_88 = tt.bitcast %y_87 : tensor<32x32x2xi16, #blocked3> -> tensor<32x32x2xbf16, #blocked3>
%grad_flip_89 = tt.reshape %x_88 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%grad_90 = arith.extf %grad : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_91 = arith.extf %grad_flip_89 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd = arith.mulf %x_74, %x_74 : tensor<32x64xf32, #blocked1>
%rstd_92 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_194: f32, %rstd_195: f32):
%rstd_196 = arith.addf %rstd_194, %rstd_195 : f32
tt.reduce.return %rstd_196 : f32
"# else r#"%x = arith.muli %f1, %cst_8 : tensor<32x1xi32, #blocked1>
%x_81 = arith.muli %f1_59, %cst_7 : tensor<32x1xi32, #blocked>
%x_82 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_83 = tt.addptr %x_82, %x : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_84 = arith.muli %logical_head, 64 : i32
%x_85 = tt.splat %x_84 : i32 -> tensor<32x1xi32, #blocked1>
%x_86 = tt.addptr %x_83, %x_85 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_87 = tt.broadcast %x_86 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_88 = tt.addptr %x_87, %f1_68 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_89 = tt.load %x_88, %qk_mask_52, %cst_19 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_90 = arith.extf %x_89 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr = tt.expand_dims %output_token_56 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%grad_ptr_91 = arith.muli %grad_ptr, %cst_6 : tensor<32x1xi32, #blocked1>
%grad_ptr_92 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_93 = tt.addptr %grad_ptr_92, %grad_ptr_91 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_94 = arith.muli %output_head, 64 : i32
%grad_ptr_95 = tt.splat %grad_ptr_94 : i32 -> tensor<32x1xi32, #blocked1>
%grad_ptr_96 = tt.addptr %grad_ptr_93, %grad_ptr_95 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_97 = tt.broadcast %grad_ptr_96 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_98 = tt.addptr %grad_ptr_97, %f1_68 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad = tt.load %grad_ptr_98, %qk_mask_52, %cst_19 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked4>
%y = tt.bitcast %grad_flip : tensor<32x32x2xbf16, #blocked4> -> tensor<32x32x2xi16, #blocked4>
%y_99 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_209: i16, %y_210: i16):
%y_211 = arith.xori %y_209, %y_210 : i16
tt.reduce.return %y_211 : i16
}) : (tensor<32x32x2xi16, #blocked4>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked4}>>
%y_100 = tt.expand_dims %y_99 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked4}>> -> tensor<32x32x1xi16, #blocked4>
%y_101 = tt.broadcast %y_100 : tensor<32x32x1xi16, #blocked4> -> tensor<32x32x2xi16, #blocked4>
%y_102 = arith.xori %y, %y_101 : tensor<32x32x2xi16, #blocked4>
%x_103 = tt.bitcast %y_102 : tensor<32x32x2xi16, #blocked4> -> tensor<32x32x2xbf16, #blocked4>
%grad_flip_104 = tt.reshape %x_103 : tensor<32x32x2xbf16, #blocked4> -> tensor<32x64xbf16, #blocked1>
%grad_105 = arith.extf %grad : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_106 = arith.extf %grad_flip_104 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd = arith.mulf %x_90, %x_90 : tensor<32x64xf32, #blocked1>
%rstd_107 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_209: f32, %rstd_210: f32):
%rstd_211 = arith.addf %rstd_209, %rstd_210 : f32
tt.reduce.return %rstd_211 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%rstd_93 = arith.divf %rstd_92, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_94 = arith.addf %rstd_93, %cst_17 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_95 = math.rsqrt %rstd_94 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_95 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_96 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_97 = arith.mulf %x_74, %normalized_96 : tensor<32x64xf32, #blocked1>
%grad_normalized = arith.mulf %f1_52, %grad_90 : tensor<32x64xf32, #blocked1>
%grad_normalized_98 = arith.mulf %f2_flip_63, %grad_flip_91 : tensor<32x64xf32, #blocked1>
%grad_normalized_99 = arith.addf %grad_normalized, %grad_normalized_98 : tensor<32x64xf32, #blocked1>
%correction = arith.mulf %grad_normalized_99, %normalized_97 : tensor<32x64xf32, #blocked1>
%correction_100 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_194: f32, %correction_195: f32):
%correction_196 = arith.addf %correction_194, %correction_195 : f32
tt.reduce.return %correction_196 : f32
"# else r#"%rstd_108 = arith.divf %rstd_107, %cst_23 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_109 = arith.addf %rstd_108, %cst_22 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_110 = math.rsqrt %rstd_109 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_110 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_111 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_112 = arith.mulf %x_90, %normalized_111 : tensor<32x64xf32, #blocked1>
%grad_normalized = arith.mulf %f1_71, %grad_105 : tensor<32x64xf32, #blocked1>
%grad_normalized_113 = arith.mulf %f2_flip_80, %grad_flip_106 : tensor<32x64xf32, #blocked1>
%grad_normalized_114 = arith.addf %grad_normalized, %grad_normalized_113 : tensor<32x64xf32, #blocked1>
%correction = arith.mulf %grad_normalized_114, %normalized_112 : tensor<32x64xf32, #blocked1>
%correction_115 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_209: f32, %correction_210: f32):
%correction_211 = arith.addf %correction_209, %correction_210 : f32
tt.reduce.return %correction_211 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%correction_101 = arith.divf %correction_100, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_101 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_102 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_103 = arith.mulf %normalized_97, %dx_102 : tensor<32x64xf32, #blocked1>
%dx_104 = arith.subf %grad_normalized_99, %dx_103 : tensor<32x64xf32, #blocked1>
%dx_105 = arith.mulf %normalized_96, %dx_104 : tensor<32x64xf32, #blocked1>
%q = arith.truncf %dx_105 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked4>
%q_106 = arith.extf %0 : tensor<32x64xbf16, #blocked4> to tensor<32x64xf32, #blocked4>
%q_107 = arith.extf %q : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_108 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked4>
%q_109 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked1>
%q_110 = arith.divf %q_106, %q_108 : tensor<32x64xf32, #blocked4>
%q_111 = arith.divf %q_107, %q_109 : tensor<32x64xf32, #blocked1>
%q_112 = tt.fp_to_fp %q_110, rounding = rtne : tensor<32x64xf32, #blocked4> -> tensor<32x64xf8E4M3FN, #blocked4>
%1 = tt.trans %q_112 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked4> -> tensor<64x32xf8E4M3FN, #blocked2>
%q_113 = tt.fp_to_fp %q_111, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature = tt.splat %x_68 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_114 = tt.splat %x_68 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%feature_115 = arith.addi %feature, %offs_d : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_116 = arith.addi %feature_114, %offs_d_29 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
"# else r#"%correction_116 = arith.divf %correction_115, %cst_23 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_116 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_117 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_118 = arith.mulf %normalized_112, %dx_117 : tensor<32x64xf32, #blocked1>
%dx_119 = arith.subf %grad_normalized_114, %dx_118 : tensor<32x64xf32, #blocked1>
%dx_120 = arith.mulf %normalized_111, %dx_119 : tensor<32x64xf32, #blocked1>
%q = arith.truncf %dx_120 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked5>
%q_121 = arith.extf %0 : tensor<32x64xbf16, #blocked5> to tensor<32x64xf32, #blocked5>
%q_122 = arith.extf %q : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_123 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked5>
%q_124 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked1>
%q_125 = arith.divf %q_121, %q_123 : tensor<32x64xf32, #blocked5>
%q_126 = arith.divf %q_122, %q_124 : tensor<32x64xf32, #blocked1>
%q_127 = tt.fp_to_fp %q_125, rounding = rtne : tensor<32x64xf32, #blocked5> -> tensor<32x64xf8E4M3FN, #blocked5>
%1 = tt.trans %q_127 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked5> -> tensor<64x32xf8E4M3FN, #blocked2>
%q_128 = tt.fp_to_fp %q_126, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature = tt.splat %x_84 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_129 = tt.splat %x_84 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%feature_130 = arith.addi %feature, %offs_d_37 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_131 = arith.addi %feature_129, %offs_d_38 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
"#) ++
  r#"%2 = arith.muli %f1, %cst_0 : tensor<32x1xi32, #blocked1>
"# ++
  (if vector then r#"%3 = arith.muli %f1_42, %cst : tensor<32x1xi32, #blocked>
"# else r#"%3 = arith.muli %f1_59, %cst : tensor<32x1xi32, #blocked>
"#) ++
  r#"%4 = tt.splat %grad_row : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>
%5 = tt.splat %grad_row : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked>
%6 = tt.addptr %4, %2 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x1xi32, #blocked1>
%7 = tt.addptr %5, %3 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<32x1xi32, #blocked>
"# ++
  (if vector then r#"%8 = tt.expand_dims %feature_115 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"# else r#"%8 = tt.expand_dims %feature_130 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"#) ++
  r#"%9 = tt.broadcast %6 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%10 = tt.broadcast %8 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%11 = tt.addptr %9, %10 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x64xi32, #blocked1>
"# ++
  (if vector then r#"tt.store %11, %q_113, %qk_mask_37 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%12 = tt.expand_dims %offs_d_29 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"# else r#"tt.store %11, %q_128, %qk_mask_52 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%12 = tt.expand_dims %offs_d_38 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"#) ++
  r#"%13 = arith.cmpi slt, %12, %cst_1 : tensor<64x1xi32, #blocked2>
"# ++
  (if vector then r#"%14 = tt.expand_dims %token_mask_31 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi1, #blocked2>
"# else r#"%14 = tt.expand_dims %token_mask_41 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi1, #blocked2>
"#) ++
  r#"%15 = tt.broadcast %13 : tensor<64x1xi1, #blocked2> -> tensor<64x32xi1, #blocked2>
%16 = tt.broadcast %14 : tensor<1x32xi1, #blocked2> -> tensor<64x32xi1, #blocked2>
%17 = arith.andi %15, %16 : tensor<64x32xi1, #blocked2>
"# ++
  (if vector then r#"%18 = tt.expand_dims %feature_116 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"# else r#"%18 = tt.expand_dims %feature_131 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"#) ++
  r#"%19 = arith.muli %18, %cst_2 : tensor<64x1xi32, #blocked2>
%20 = tt.splat %grad_transposed : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked2>
%21 = tt.addptr %20, %19 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<64x1xi32, #blocked2>
"# ++
  (if vector then r#"%22 = tt.expand_dims %token_28 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
"# else r#"%22 = tt.expand_dims %token_36 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
"#) ++
  r#"%23 = tt.broadcast %21 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>
%24 = tt.broadcast %22 : tensor<1x32xi32, #blocked2> -> tensor<64x32xi32, #blocked2>
%25 = tt.addptr %23, %24 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<64x32xi32, #blocked2>
tt.store %25, %1, %17 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>
%input_head = arith.addi %logical_head, 6 : i32
"# ++
  (if vector then r#"%x_117 = arith.muli %input_head, 64 : i32
%x_118 = tt.splat %x_117 : i32 -> tensor<32x1xi32, #blocked1>
%x_119 = tt.addptr %x_67, %x_118 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_120 = tt.broadcast %x_119 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_121 = tt.addptr %x_120, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_122 = tt.load %x_121, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_123 = arith.extf %x_122 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr_124 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_125 = tt.addptr %grad_ptr_124, %grad_ptr_75 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_126 = tt.addptr %grad_ptr_125, %grad_ptr_79 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_127 = tt.broadcast %grad_ptr_126 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_128 = tt.addptr %grad_ptr_127, %f1_49 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_129 = tt.load %grad_ptr_128, %qk_mask_37, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip_130 = tt.reshape %grad_129 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%y_131 = tt.bitcast %grad_flip_130 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_132 = "tt.reduce"(%y_131) <{axis = 2 : i32}> ({
^bb0(%y_194: i16, %y_195: i16):
%y_196 = arith.xori %y_194, %y_195 : i16
tt.reduce.return %y_196 : i16
}) : (tensor<32x32x2xi16, #blocked3>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_133 = tt.expand_dims %y_132 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x32x1xi16, #blocked3>
%y_134 = tt.broadcast %y_133 : tensor<32x32x1xi16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_135 = arith.xori %y_131, %y_134 : tensor<32x32x2xi16, #blocked3>
%x_136 = tt.bitcast %y_135 : tensor<32x32x2xi16, #blocked3> -> tensor<32x32x2xbf16, #blocked3>
%grad_flip_137 = tt.reshape %x_136 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%grad_138 = arith.extf %grad_129 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_139 = arith.extf %grad_flip_137 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd_140 = arith.mulf %x_123, %x_123 : tensor<32x64xf32, #blocked1>
%rstd_141 = "tt.reduce"(%rstd_140) <{axis = 1 : i32}> ({
^bb0(%rstd_194: f32, %rstd_195: f32):
%rstd_196 = arith.addf %rstd_194, %rstd_195 : f32
tt.reduce.return %rstd_196 : f32
"# else r#"%x_132 = arith.muli %input_head, 64 : i32
%x_133 = tt.splat %x_132 : i32 -> tensor<32x1xi32, #blocked1>
%x_134 = tt.addptr %x_83, %x_133 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_135 = tt.broadcast %x_134 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_136 = tt.addptr %x_135, %f1_68 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_137 = tt.load %x_136, %qk_mask_52, %cst_19 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_138 = arith.extf %x_137 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr_139 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_140 = tt.addptr %grad_ptr_139, %grad_ptr_91 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_141 = tt.addptr %grad_ptr_140, %grad_ptr_95 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_142 = tt.broadcast %grad_ptr_141 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_143 = tt.addptr %grad_ptr_142, %f1_68 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_144 = tt.load %grad_ptr_143, %qk_mask_52, %cst_19 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip_145 = tt.reshape %grad_144 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked4>
%y_146 = tt.bitcast %grad_flip_145 : tensor<32x32x2xbf16, #blocked4> -> tensor<32x32x2xi16, #blocked4>
%y_147 = "tt.reduce"(%y_146) <{axis = 2 : i32}> ({
^bb0(%y_209: i16, %y_210: i16):
%y_211 = arith.xori %y_209, %y_210 : i16
tt.reduce.return %y_211 : i16
}) : (tensor<32x32x2xi16, #blocked4>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked4}>>
%y_148 = tt.expand_dims %y_147 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked4}>> -> tensor<32x32x1xi16, #blocked4>
%y_149 = tt.broadcast %y_148 : tensor<32x32x1xi16, #blocked4> -> tensor<32x32x2xi16, #blocked4>
%y_150 = arith.xori %y_146, %y_149 : tensor<32x32x2xi16, #blocked4>
%x_151 = tt.bitcast %y_150 : tensor<32x32x2xi16, #blocked4> -> tensor<32x32x2xbf16, #blocked4>
%grad_flip_152 = tt.reshape %x_151 : tensor<32x32x2xbf16, #blocked4> -> tensor<32x64xbf16, #blocked1>
%grad_153 = arith.extf %grad_144 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_154 = arith.extf %grad_flip_152 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd_155 = arith.mulf %x_138, %x_138 : tensor<32x64xf32, #blocked1>
%rstd_156 = "tt.reduce"(%rstd_155) <{axis = 1 : i32}> ({
^bb0(%rstd_209: f32, %rstd_210: f32):
%rstd_211 = arith.addf %rstd_209, %rstd_210 : f32
tt.reduce.return %rstd_211 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%rstd_142 = arith.divf %rstd_141, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_143 = arith.addf %rstd_142, %cst_17 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_144 = math.rsqrt %rstd_143 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_145 = tt.expand_dims %rstd_144 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_146 = tt.broadcast %normalized_145 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_147 = arith.mulf %x_123, %normalized_146 : tensor<32x64xf32, #blocked1>
%grad_normalized_148 = arith.mulf %f1_52, %grad_138 : tensor<32x64xf32, #blocked1>
%grad_normalized_149 = arith.mulf %f2_flip_63, %grad_flip_139 : tensor<32x64xf32, #blocked1>
%grad_normalized_150 = arith.addf %grad_normalized_148, %grad_normalized_149 : tensor<32x64xf32, #blocked1>
%correction_151 = arith.mulf %grad_normalized_150, %normalized_147 : tensor<32x64xf32, #blocked1>
%correction_152 = "tt.reduce"(%correction_151) <{axis = 1 : i32}> ({
^bb0(%correction_194: f32, %correction_195: f32):
%correction_196 = arith.addf %correction_194, %correction_195 : f32
tt.reduce.return %correction_196 : f32
"# else r#"%rstd_157 = arith.divf %rstd_156, %cst_23 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_158 = arith.addf %rstd_157, %cst_22 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_159 = math.rsqrt %rstd_158 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_160 = tt.expand_dims %rstd_159 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_161 = tt.broadcast %normalized_160 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_162 = arith.mulf %x_138, %normalized_161 : tensor<32x64xf32, #blocked1>
%grad_normalized_163 = arith.mulf %f1_71, %grad_153 : tensor<32x64xf32, #blocked1>
%grad_normalized_164 = arith.mulf %f2_flip_80, %grad_flip_154 : tensor<32x64xf32, #blocked1>
%grad_normalized_165 = arith.addf %grad_normalized_163, %grad_normalized_164 : tensor<32x64xf32, #blocked1>
%correction_166 = arith.mulf %grad_normalized_165, %normalized_162 : tensor<32x64xf32, #blocked1>
%correction_167 = "tt.reduce"(%correction_166) <{axis = 1 : i32}> ({
^bb0(%correction_209: f32, %correction_210: f32):
%correction_211 = arith.addf %correction_209, %correction_210 : f32
tt.reduce.return %correction_211 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%correction_153 = arith.divf %correction_152, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_154 = tt.expand_dims %correction_153 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_155 = tt.broadcast %dx_154 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_156 = arith.mulf %normalized_147, %dx_155 : tensor<32x64xf32, #blocked1>
%dx_157 = arith.subf %grad_normalized_150, %dx_156 : tensor<32x64xf32, #blocked1>
%dx_158 = arith.mulf %normalized_146, %dx_157 : tensor<32x64xf32, #blocked1>
%q_159 = arith.truncf %dx_158 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%26 = ttg.convert_layout %q_159 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked4>
%q_160 = arith.extf %26 : tensor<32x64xbf16, #blocked4> to tensor<32x64xf32, #blocked4>
%q_161 = arith.extf %q_159 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_162 = arith.divf %q_160, %q_108 : tensor<32x64xf32, #blocked4>
%q_163 = arith.divf %q_161, %q_109 : tensor<32x64xf32, #blocked1>
%q_164 = tt.fp_to_fp %q_162, rounding = rtne : tensor<32x64xf32, #blocked4> -> tensor<32x64xf8E4M3FN, #blocked4>
%27 = tt.trans %q_164 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked4> -> tensor<64x32xf8E4M3FN, #blocked2>
%q_165 = tt.fp_to_fp %q_163, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature_166 = tt.splat %x_117 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_167 = tt.splat %x_117 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%feature_168 = arith.addi %feature_166, %offs_d : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_169 = arith.addi %feature_167, %offs_d_29 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%28 = tt.expand_dims %feature_168 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"# else r#"%correction_168 = arith.divf %correction_167, %cst_23 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_169 = tt.expand_dims %correction_168 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_170 = tt.broadcast %dx_169 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_171 = arith.mulf %normalized_162, %dx_170 : tensor<32x64xf32, #blocked1>
%dx_172 = arith.subf %grad_normalized_165, %dx_171 : tensor<32x64xf32, #blocked1>
%dx_173 = arith.mulf %normalized_161, %dx_172 : tensor<32x64xf32, #blocked1>
%q_174 = arith.truncf %dx_173 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%26 = ttg.convert_layout %q_174 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked5>
%q_175 = arith.extf %26 : tensor<32x64xbf16, #blocked5> to tensor<32x64xf32, #blocked5>
%q_176 = arith.extf %q_174 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_177 = arith.divf %q_175, %q_123 : tensor<32x64xf32, #blocked5>
%q_178 = arith.divf %q_176, %q_124 : tensor<32x64xf32, #blocked1>
%q_179 = tt.fp_to_fp %q_177, rounding = rtne : tensor<32x64xf32, #blocked5> -> tensor<32x64xf8E4M3FN, #blocked5>
%27 = tt.trans %q_179 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked5> -> tensor<64x32xf8E4M3FN, #blocked2>
%q_180 = tt.fp_to_fp %q_178, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature_181 = tt.splat %x_132 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_182 = tt.splat %x_132 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%feature_183 = arith.addi %feature_181, %offs_d_37 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_184 = arith.addi %feature_182, %offs_d_38 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%28 = tt.expand_dims %feature_183 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"#) ++
  r#"%29 = tt.broadcast %28 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%30 = tt.addptr %9, %29 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x64xi32, #blocked1>
"# ++
  (if vector then r#"tt.store %30, %q_165, %qk_mask_37 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%31 = tt.expand_dims %feature_169 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"# else r#"tt.store %30, %q_180, %qk_mask_52 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%31 = tt.expand_dims %feature_184 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
"#) ++
  r#"%32 = arith.muli %31, %cst_2 : tensor<64x1xi32, #blocked2>
%33 = tt.addptr %20, %32 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<64x1xi32, #blocked2>
%34 = tt.broadcast %33 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>
%35 = tt.addptr %34, %24 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<64x32xi32, #blocked2>
tt.store %35, %27, %17 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked2>
%offs_v = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"# ++
  (if vector then r#"%offs_v_170 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
"# else r#"%offs_v_185 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
"#) ++
  r#"%v_mask = tt.expand_dims %offs_v {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
"# ++
  (if vector then r#"%v_mask_171 = arith.cmpi slt, %v_mask, %cst_3 : tensor<1x128xi32, #blocked>
%v_mask_172 = tt.broadcast %qk_mask_32 : tensor<32x1xi1, #blocked> -> tensor<32x128xi1, #blocked>
%v_mask_173 = tt.broadcast %v_mask_171 : tensor<1x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
%v_mask_174 = arith.andi %v_mask_172, %v_mask_173 : tensor<32x128xi1, #blocked>
"# else r#"%v_mask_186 = arith.cmpi slt, %v_mask, %cst_3 : tensor<1x128xi32, #blocked>
%v_mask_187 = tt.broadcast %qk_mask_43 : tensor<32x1xi1, #blocked> -> tensor<32x128xi1, #blocked>
%v_mask_188 = tt.broadcast %v_mask_186 : tensor<1x128xi1, #blocked> -> tensor<32x128xi1, #blocked>
%v_mask_189 = arith.andi %v_mask_187, %v_mask_188 : tensor<32x128xi1, #blocked>
"#) ++
  r#"%grad_v_value = tt.splat %grad_v : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked>
"# ++
  (if vector then r#"%grad_v_value_175 = tt.addptr %grad_v_value, %x_65 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
%grad_v_value_176 = arith.muli %logical_head, 128 : i32
%grad_v_value_177 = tt.splat %grad_v_value_176 : i32 -> tensor<32x1xi32, #blocked>
%grad_v_value_178 = tt.addptr %grad_v_value_175, %grad_v_value_177 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
%grad_v_value_179 = tt.broadcast %grad_v_value_178 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x128x!tt.ptr<bf16>, #blocked>
%grad_v_value_180 = tt.broadcast %v_mask : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
%grad_v_value_181 = tt.addptr %grad_v_value_179, %grad_v_value_180 : tensor<32x128x!tt.ptr<bf16>, #blocked>, tensor<32x128xi32, #blocked>
%grad_v_value_182 = tt.load %grad_v_value_181, %v_mask_174, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked>
%36 = ttg.convert_layout %grad_v_value_182 : tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #blocked4>
%grad_v_value_183 = arith.extf %36 : tensor<32x128xbf16, #blocked4> to tensor<32x128xf32, #blocked4>
%grad_v_value_184 = arith.extf %grad_v_value_182 : tensor<32x128xbf16, #blocked> to tensor<32x128xf32, #blocked>
%qv = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked4>
%qv_185 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked>
%qv_186 = arith.divf %grad_v_value_183, %qv : tensor<32x128xf32, #blocked4>
%qv_187 = arith.divf %grad_v_value_184, %qv_185 : tensor<32x128xf32, #blocked>
%qv_188 = tt.fp_to_fp %qv_186, rounding = rtne : tensor<32x128xf32, #blocked4> -> tensor<32x128xf8E4M3FN, #blocked4>
%37 = tt.trans %qv_188 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked4> -> tensor<128x32xf8E4M3FN, #blocked2>
%qv_189 = tt.fp_to_fp %qv_187, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FN, #blocked>
%v_feature = arith.addi %grad_v_value_176, 768 : i32
%v_feature_190 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_feature_191 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%v_feature_192 = arith.addi %v_feature_190, %offs_v : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_feature_193 = arith.addi %v_feature_191, %offs_v_170 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%38 = tt.expand_dims %v_feature_192 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
"# else r#"%grad_v_value_190 = tt.addptr %grad_v_value, %x_81 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
%grad_v_value_191 = arith.muli %logical_head, 128 : i32
%grad_v_value_192 = tt.splat %grad_v_value_191 : i32 -> tensor<32x1xi32, #blocked>
%grad_v_value_193 = tt.addptr %grad_v_value_190, %grad_v_value_192 : tensor<32x1x!tt.ptr<bf16>, #blocked>, tensor<32x1xi32, #blocked>
%grad_v_value_194 = tt.broadcast %grad_v_value_193 : tensor<32x1x!tt.ptr<bf16>, #blocked> -> tensor<32x128x!tt.ptr<bf16>, #blocked>
%grad_v_value_195 = tt.broadcast %v_mask : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
%grad_v_value_196 = tt.addptr %grad_v_value_194, %grad_v_value_195 : tensor<32x128x!tt.ptr<bf16>, #blocked>, tensor<32x128xi32, #blocked>
%grad_v_value_197 = tt.load %grad_v_value_196, %v_mask_189, %cst_21 : tensor<32x128x!tt.ptr<bf16>, #blocked>
%36 = ttg.convert_layout %grad_v_value_197 : tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #blocked5>
%grad_v_value_198 = arith.extf %36 : tensor<32x128xbf16, #blocked5> to tensor<32x128xf32, #blocked5>
%grad_v_value_199 = arith.extf %grad_v_value_197 : tensor<32x128xbf16, #blocked> to tensor<32x128xf32, #blocked>
%qv = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked5>
%qv_200 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked>
%qv_201 = arith.divf %grad_v_value_198, %qv : tensor<32x128xf32, #blocked5>
%qv_202 = arith.divf %grad_v_value_199, %qv_200 : tensor<32x128xf32, #blocked>
%qv_203 = tt.fp_to_fp %qv_201, rounding = rtne : tensor<32x128xf32, #blocked5> -> tensor<32x128xf8E4M3FN, #blocked5>
%37 = tt.trans %qv_203 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked5> -> tensor<128x32xf8E4M3FN, #blocked2>
%qv_204 = tt.fp_to_fp %qv_202, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FN, #blocked>
%v_feature = arith.addi %grad_v_value_191, 768 : i32
%v_feature_205 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_feature_206 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%v_feature_207 = arith.addi %v_feature_205, %offs_v : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%v_feature_208 = arith.addi %v_feature_206, %offs_v_185 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%38 = tt.expand_dims %v_feature_207 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
"#) ++
  r#"%39 = tt.broadcast %7 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked>
%40 = tt.broadcast %38 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
%41 = tt.addptr %39, %40 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<32x128xi32, #blocked>
"# ++
  (if vector then r#"tt.store %41, %qv_189, %v_mask_174 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked>
%42 = tt.expand_dims %offs_v_170 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
"# else r#"tt.store %41, %qv_204, %v_mask_189 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked>
%42 = tt.expand_dims %offs_v_185 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
"#) ++
  r#"%43 = arith.cmpi slt, %42, %cst_4 : tensor<128x1xi32, #blocked2>
%44 = tt.broadcast %43 : tensor<128x1xi1, #blocked2> -> tensor<128x32xi1, #blocked2>
%45 = tt.broadcast %14 : tensor<1x32xi1, #blocked2> -> tensor<128x32xi1, #blocked2>
%46 = arith.andi %44, %45 : tensor<128x32xi1, #blocked2>
"# ++
  (if vector then r#"%47 = tt.expand_dims %v_feature_193 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
"# else r#"%47 = tt.expand_dims %v_feature_208 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
"#) ++
  r#"%48 = arith.muli %47, %cst_5 : tensor<128x1xi32, #blocked2>
%49 = tt.splat %grad_transposed : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
%50 = tt.addptr %49, %48 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
%51 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked2>
%52 = tt.broadcast %22 : tensor<1x32xi32, #blocked2> -> tensor<128x32xi32, #blocked2>
%53 = tt.addptr %51, %52 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x32xi32, #blocked2>
tt.store %53, %37, %46 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked2>
tt.return
}
}
"#

private def pairedPTX (vector : Bool) : String :=
  r#".version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry _qkv_norm_rope_pack_fp8_backward_kernel(
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_0,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_1,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_2,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_3,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_4,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_5,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_6,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_7,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_8,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_9,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_10
)
.reqntid 128
{
.reg .pred %p<2>;
"# ++
  (if vector then r#".reg .b16 %rs<297>;
@EXTENT_11@
"# else r#".reg .b16 %rs<314>;
@EXTENT_4@
"#) ++
  r#"$L__func_begin0:
"# ++
  (if vector then r#"ld.param.b64 %rd26, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd27, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"# else r#"ld.param.b64 %rd40, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd41, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"#) ++
  r#"$L__tmp0:
"# ++
  (if vector then r#"mov.u32 %r227, %ctaid.x;
ld.param.b64 %rd28, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r228, %ctaid.y;
shl.b32 %r229, %r228, 5;
ld.param.b64 %rd29, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd30, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r230, %tid.x;
ld.param.b64 %rd31, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r231, %r230, 2, 5;
ld.param.b64 %rd32, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
shl.b32 %r232, %r230, 4;
ld.param.b64 %rd33, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
and.b32 %r233, %r232, 16;
ld.param.b64 %rd5, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
or.b32 %r234, %r231, %r229;
or.b32 %r235, %r233, %r229;
and.b32 %r236, %r230, 3;
shl.b32 %r237, %r236, 4;
bfe.u32 %r238, %r230, 1, 6;
@EXTENT_12@
shl.b32 %r239, %r234, 1;
mul.hi.u32 %r240, %r227, 1431655766;
add.s32 %r241, %r239, %r240;
mul.lo.s32 %r242, %r240, 3;
sub.s32 %r243, %r227, %r242;
shl.b32 %r244, %r227, 6;
and.b32 %r245, %r244, 64;
shl.b32 %r246, %r234, 7;
mul.wide.u32 %rd34, %r246, 2;
add.s64 %rd35, %rd30, %rd34;
mul.wide.u32 %rd36, %r245, 2;
add.s64 %rd37, %rd35, %rd36;
mul.wide.u32 %rd38, %r237, 2;
add.s64 %rd1, %rd37, %rd38;
"# else r#"mov.u32 %r219, %ctaid.x;
ld.param.b64 %rd42, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r220, %ctaid.y;
shl.b32 %r221, %r220, 5;
ld.param.b64 %rd43, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd44, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r222, %tid.x;
ld.param.b64 %rd45, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r223, %r222, 2, 5;
ld.param.b64 %rd46, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
and.b32 %r224, %r222, 31;
ld.param.b64 %rd47, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
shl.b32 %r225, %r222, 4;
ld.param.b64 %rd19, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
and.b32 %r226, %r225, 16;
or.b32 %r227, %r223, %r221;
or.b32 %r228, %r221, %r224;
or.b32 %r229, %r226, %r221;
and.b32 %r230, %r222, 96;
bfe.u32 %r231, %r222, 5, 2;
and.b32 %r232, %r222, 3;
shl.b32 %r233, %r232, 4;
bfe.u32 %r234, %r222, 1, 6;
@EXTENT_5@
xor.b32 %r235, %r231, 1;
shl.b32 %r236, %r227, 1;
mul.hi.u32 %r237, %r219, 1431655766;
add.s32 %r238, %r236, %r237;
mul.lo.s32 %r239, %r237, 3;
sub.s32 %r240, %r219, %r239;
shl.b32 %r241, %r219, 6;
and.b32 %r242, %r241, 64;
shl.b32 %r243, %r227, 7;
shl.b32 %r244, %r228, 7;
mad.wide.u32 %rd48, %r243, 2, %rd44;
mul.wide.u32 %rd49, %r242, 2;
add.s64 %rd50, %rd48, %rd49;
mul.wide.u32 %rd51, %r233, 2;
add.s64 %rd1, %rd50, %rd51;
"#) ++
  r#"add.s64 %rd2, %rd1, 16;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
"# ++
  (if vector then r#"mov.b32 {%rs65, %rs66}, %r1;
mov.b32 {%rs67, %rs68}, %r2;
mov.b32 {%rs69, %rs70}, %r3;
mov.b32 {%rs71, %rs72}, %r4;
"# else r#"mov.b32 {%rs82, %rs83}, %r1;
mov.b32 {%rs84, %rs85}, %r2;
mov.b32 {%rs86, %rs87}, %r3;
mov.b32 {%rs88, %rs89}, %r4;
"#) ++
  r#"mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
"# ++
  (if vector then r#"mov.b32 {%rs73, %rs74}, %r6;
mov.b32 {%rs75, %rs76}, %r7;
mov.b32 {%rs77, %rs78}, %r8;
mov.b32 {%rs79, %rs80}, %r9;
cvt.f32.bf16 %r247, %rs65;
cvt.f32.bf16 %r248, %rs66;
cvt.f32.bf16 %r249, %rs67;
cvt.f32.bf16 %r250, %rs68;
cvt.f32.bf16 %r251, %rs69;
cvt.f32.bf16 %r252, %rs70;
cvt.f32.bf16 %r253, %rs71;
cvt.f32.bf16 %r254, %rs72;
cvt.f32.bf16 %r255, %rs73;
cvt.f32.bf16 %r256, %rs74;
cvt.f32.bf16 %r257, %rs75;
cvt.f32.bf16 %r258, %rs76;
cvt.f32.bf16 %r259, %rs77;
cvt.f32.bf16 %r260, %rs78;
cvt.f32.bf16 %r261, %rs79;
cvt.f32.bf16 %r262, %rs80;
add.s64 %rd39, %rd31, %rd34;
add.s64 %rd40, %rd39, %rd36;
add.s64 %rd3, %rd40, %rd38;
add.s64 %rd4, %rd3, 16;
mov.u32 %r10, %r5;
"# else r#"mov.b32 {%rs90, %rs91}, %r6;
mov.b32 {%rs92, %rs93}, %r7;
mov.b32 {%rs94, %rs95}, %r8;
mov.b32 {%rs96, %rs97}, %r9;
cvt.f32.bf16 %r245, %rs82;
cvt.f32.bf16 %r246, %rs83;
cvt.f32.bf16 %r247, %rs84;
cvt.f32.bf16 %r248, %rs85;
cvt.f32.bf16 %r249, %rs86;
cvt.f32.bf16 %r250, %rs87;
cvt.f32.bf16 %r251, %rs88;
cvt.f32.bf16 %r252, %rs89;
cvt.f32.bf16 %r253, %rs90;
cvt.f32.bf16 %r254, %rs91;
cvt.f32.bf16 %r255, %rs92;
cvt.f32.bf16 %r256, %rs93;
cvt.f32.bf16 %r257, %rs94;
cvt.f32.bf16 %r258, %rs95;
cvt.f32.bf16 %r259, %rs96;
cvt.f32.bf16 %r260, %rs97;
mad.wide.u32 %rd52, %r244, 2, %rd45;
add.s64 %rd53, %rd52, %rd49;
mad.wide.u32 %rd3, %r235, 2, %rd53;
cvt.u64.u32 %rd54, %r231;
xor.b64 %rd55, %rd54, 1;
shl.b64 %rd56, %rd55, 1;
add.s64 %rd57, %rd53, %rd56;
add.s64 %rd4, %rd57, 8;
add.s64 %rd5, %rd57, 16;
add.s64 %rd6, %rd57, 24;
add.s64 %rd7, %rd57, 32;
add.s64 %rd8, %rd57, 40;
add.s64 %rd9, %rd57, 48;
add.s64 %rd10, %rd57, 56;
add.s64 %rd11, %rd57, 64;
add.s64 %rd12, %rd57, 72;
add.s64 %rd13, %rd57, 80;
add.s64 %rd14, %rd57, 88;
add.s64 %rd15, %rd57, 96;
add.s64 %rd16, %rd57, 104;
add.s64 %rd17, %rd57, 112;
add.s64 %rd18, %rd57, 120;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd3 + 0 ];
mov.u16 %rs3, %rs2;
@%p1 ld.global.b16 { %rs3 }, [ %rd4 + 0 ];
mov.u16 %rs4, %rs2;
@%p1 ld.global.b16 { %rs4 }, [ %rd5 + 0 ];
mov.u16 %rs5, %rs2;
@%p1 ld.global.b16 { %rs5 }, [ %rd6 + 0 ];
mov.u16 %rs6, %rs2;
@%p1 ld.global.b16 { %rs6 }, [ %rd7 + 0 ];
mov.u16 %rs7, %rs2;
@%p1 ld.global.b16 { %rs7 }, [ %rd8 + 0 ];
mov.u16 %rs8, %rs2;
@%p1 ld.global.b16 { %rs8 }, [ %rd9 + 0 ];
mov.u16 %rs9, %rs2;
@%p1 ld.global.b16 { %rs9 }, [ %rd10 + 0 ];
mov.u16 %rs10, %rs2;
@%p1 ld.global.b16 { %rs10 }, [ %rd11 + 0 ];
mov.u16 %rs11, %rs2;
@%p1 ld.global.b16 { %rs11 }, [ %rd12 + 0 ];
mov.u16 %rs12, %rs2;
@%p1 ld.global.b16 { %rs12 }, [ %rd13 + 0 ];
mov.u16 %rs13, %rs2;
@%p1 ld.global.b16 { %rs13 }, [ %rd14 + 0 ];
mov.u16 %rs14, %rs2;
@%p1 ld.global.b16 { %rs14 }, [ %rd15 + 0 ];
mov.u16 %rs15, %rs2;
@%p1 ld.global.b16 { %rs15 }, [ %rd16 + 0 ];
mov.u16 %rs16, %rs2;
@%p1 ld.global.b16 { %rs16 }, [ %rd17 + 0 ];
mov.u16 %rs17, %rs2;
@%p1 ld.global.b16 { %rs17 }, [ %rd18 + 0 ];
and.b32 %r261, %r225, 2032;
mov.b32 %r262, global_smem;
add.s32 %r263, %r262, %r261;
mov.b32 %r264, {%rs1, %rs3};
mov.b32 %r265, {%rs6, %rs7};
mov.b32 %r266, {%rs10, %rs11};
mov.b32 %r267, {%rs14, %rs15};
st.shared.v4.b32 [%r263], {%r264, %r265, %r266, %r267};
bar.sync 0;
shl.b32 %r268, %r222, 6;
and.b32 %r269, %r268, 1536;
and.b32 %r270, %r222, 7;
shl.b32 %r271, %r270, 4;
shl.b32 %r272, %r230, 2;
add.s32 %r273, %r262, %r269;
add.s32 %r274, %r273, %r272;
add.s32 %r275, %r274, %r271;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r276, %r277, %r278, %r279}, [%r275];
mov.b32 {%rs98, %rs99}, %r276;
mov.b32 {%rs100, %rs101}, %r277;
mov.b32 {%rs102, %rs103}, %r278;
mov.b32 {%rs104, %rs105}, %r279;
bar.sync 0;
mov.b32 %r280, {%rs4, %rs5};
mov.b32 %r281, {%rs8, %rs9};
mov.b32 %r282, {%rs12, %rs13};
mov.b32 %r283, {%rs16, %rs17};
st.shared.v4.b32 [%r263], {%r280, %r281, %r282, %r283};
bar.sync 0;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r284, %r285, %r286, %r287}, [%r275];
mov.b32 {%rs106, %rs107}, %r284;
mov.b32 {%rs108, %rs109}, %r285;
mov.b32 {%rs110, %rs111}, %r286;
mov.b32 {%rs112, %rs113}, %r287;
cvt.f32.bf16 %r288, %rs98;
cvt.f32.bf16 %r289, %rs100;
cvt.f32.bf16 %r290, %rs102;
cvt.f32.bf16 %r291, %rs104;
cvt.f32.bf16 %r292, %rs99;
cvt.f32.bf16 %r293, %rs101;
cvt.f32.bf16 %r294, %rs103;
cvt.f32.bf16 %r295, %rs105;
cvt.f32.bf16 %r296, %rs106;
cvt.f32.bf16 %r297, %rs108;
cvt.f32.bf16 %r298, %rs110;
cvt.f32.bf16 %r299, %rs112;
cvt.f32.bf16 %r300, %rs107;
cvt.f32.bf16 %r301, %rs109;
cvt.f32.bf16 %r302, %rs111;
cvt.f32.bf16 %r303, %rs113;
mov.u32 %r10, 0x0;
ld.global.b32 { %r10 }, [ %rd19 + 0 ];
mul.lo.s32 %r304, %r227, 768;
mad.wide.u32 %rd58, %r304, 2, %rd43;
mad.wide.s32 %rd59, %r241, 2, %rd58;
add.s64 %rd20, %rd59, %rd51;
add.s64 %rd21, %rd20, 16;
"#) ++
  r#"mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs81, %rs82}, %r10;
mov.b32 {%rs83, %rs84}, %r11;
mov.b32 {%rs85, %rs86}, %r12;
mov.b32 {%rs87, %rs88}, %r13;
"# else "") ++
  r#"mov.u32 %r14, %r5;
"# ++
  (if vector then "" else r#"@%p1 ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd20 + 0 ];
mov.b32 {%rs114, %rs115}, %r11;
mov.b32 {%rs116, %rs117}, %r12;
mov.b32 {%rs118, %rs119}, %r13;
mov.b32 {%rs120, %rs121}, %r14;
"#) ++
  r#"mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs89, %rs90}, %r14;
mov.b32 {%rs91, %rs92}, %r15;
mov.b32 {%rs93, %rs94}, %r16;
mov.b32 {%rs95, %rs96}, %r17;
cvt.f32.bf16 %r263, %rs82;
cvt.f32.bf16 %r264, %rs81;
cvt.f32.bf16 %r265, %rs84;
cvt.f32.bf16 %r266, %rs83;
cvt.f32.bf16 %r267, %rs86;
cvt.f32.bf16 %r268, %rs85;
cvt.f32.bf16 %r269, %rs88;
cvt.f32.bf16 %r270, %rs87;
cvt.f32.bf16 %r271, %rs90;
cvt.f32.bf16 %r272, %rs89;
cvt.f32.bf16 %r273, %rs92;
cvt.f32.bf16 %r274, %rs91;
cvt.f32.bf16 %r275, %rs94;
cvt.f32.bf16 %r276, %rs93;
cvt.f32.bf16 %r277, %rs96;
cvt.f32.bf16 %r278, %rs95;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd5 + 0 ];
mul.lo.s32 %r279, %r234, 768;
mad.wide.u32 %rd41, %r279, 2, %rd29;
mad.wide.s32 %rd42, %r244, 2, %rd41;
add.s64 %rd6, %rd42, %rd38;
add.s64 %rd7, %rd6, 16;
"# else r#"mov.u32 %r18, %r5;
@%p1 ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd21 + 0 ];
mov.b32 {%rs122, %rs123}, %r15;
mov.b32 {%rs124, %rs125}, %r16;
mov.b32 {%rs126, %rs127}, %r17;
mov.b32 {%rs128, %rs129}, %r18;
cvt.f32.bf16 %r305, %rs114;
cvt.f32.bf16 %r306, %rs115;
cvt.f32.bf16 %r307, %rs116;
cvt.f32.bf16 %r308, %rs117;
cvt.f32.bf16 %r309, %rs118;
cvt.f32.bf16 %r310, %rs119;
cvt.f32.bf16 %r311, %rs120;
cvt.f32.bf16 %r312, %rs121;
cvt.f32.bf16 %r313, %rs122;
cvt.f32.bf16 %r314, %rs123;
cvt.f32.bf16 %r315, %rs124;
cvt.f32.bf16 %r316, %rs125;
cvt.f32.bf16 %r317, %rs126;
cvt.f32.bf16 %r318, %rs127;
cvt.f32.bf16 %r319, %rs128;
cvt.f32.bf16 %r320, %rs129;
mul.lo.s32 %r321, %r238, 192;
mul.wide.s32 %rd60, %r321, 2;
add.s64 %rd61, %rd40, %rd60;
shl.b32 %r322, %r240, 6;
mul.wide.u32 %rd62, %r322, 2;
add.s64 %rd63, %rd61, %rd62;
add.s64 %rd22, %rd63, %rd51;
add.s64 %rd23, %rd22, 16;
"#) ++
  r#"mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
mov.u32 %r22, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd6 + 0 ];
mov.b32 {%rs97, %rs98}, %r19;
mov.b32 {%rs99, %rs100}, %r20;
mov.b32 {%rs101, %rs102}, %r21;
mov.b32 {%rs103, %rs104}, %r22;
"# else r#"@%p1 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd22 + 0 ];
mov.b32 {%rs130, %rs131}, %r19;
mov.b32 {%rs132, %rs133}, %r20;
mov.b32 {%rs134, %rs135}, %r21;
mov.b32 {%rs136, %rs137}, %r22;
"#) ++
  r#"mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
mov.u32 %r26, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd7 + 0 ];
mov.b32 {%rs105, %rs106}, %r23;
mov.b32 {%rs107, %rs108}, %r24;
mov.b32 {%rs109, %rs110}, %r25;
mov.b32 {%rs111, %rs112}, %r26;
cvt.f32.bf16 %r280, %rs97;
cvt.f32.bf16 %r281, %rs98;
cvt.f32.bf16 %r282, %rs99;
cvt.f32.bf16 %r283, %rs100;
cvt.f32.bf16 %r284, %rs101;
cvt.f32.bf16 %r285, %rs102;
cvt.f32.bf16 %r286, %rs103;
cvt.f32.bf16 %r287, %rs104;
cvt.f32.bf16 %r288, %rs105;
cvt.f32.bf16 %r289, %rs106;
cvt.f32.bf16 %r290, %rs107;
cvt.f32.bf16 %r291, %rs108;
cvt.f32.bf16 %r292, %rs109;
cvt.f32.bf16 %r293, %rs110;
cvt.f32.bf16 %r294, %rs111;
cvt.f32.bf16 %r295, %rs112;
mul.lo.s32 %r296, %r241, 192;
mul.wide.s32 %rd43, %r296, 2;
add.s64 %rd44, %rd26, %rd43;
shl.b32 %r297, %r243, 6;
mul.wide.u32 %rd45, %r297, 2;
add.s64 %rd46, %rd44, %rd45;
add.s64 %rd8, %rd46, %rd38;
add.s64 %rd9, %rd8, 16;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
mov.u32 %r30, %r5;
@%p1 ld.global.v4.b32 { %r27, %r28, %r29, %r30 }, [ %rd8 + 0 ];
mov.b32 {%rs113, %rs114}, %r27;
mov.b32 {%rs115, %rs116}, %r28;
mov.b32 {%rs117, %rs118}, %r29;
mov.b32 {%rs119, %rs120}, %r30;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
mov.u32 %r34, %r5;
@%p1 ld.global.v4.b32 { %r31, %r32, %r33, %r34 }, [ %rd9 + 0 ];
mov.b32 {%rs121, %rs122}, %r31;
mov.b32 {%rs123, %rs124}, %r32;
mov.b32 {%rs125, %rs126}, %r33;
mov.b32 {%rs127, %rs128}, %r34;
cvt.f32.bf16 %r298, %rs113;
cvt.f32.bf16 %r299, %rs114;
cvt.f32.bf16 %r300, %rs115;
cvt.f32.bf16 %r301, %rs116;
cvt.f32.bf16 %r302, %rs117;
cvt.f32.bf16 %r303, %rs118;
cvt.f32.bf16 %r304, %rs119;
cvt.f32.bf16 %r305, %rs120;
cvt.f32.bf16 %r306, %rs121;
cvt.f32.bf16 %r307, %rs122;
cvt.f32.bf16 %r308, %rs123;
cvt.f32.bf16 %r309, %rs124;
cvt.f32.bf16 %r310, %rs125;
cvt.f32.bf16 %r311, %rs126;
cvt.f32.bf16 %r312, %rs127;
cvt.f32.bf16 %r313, %rs128;
mul.f32 %r314, %r281, %r281;
"# else r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd23 + 0 ];
mov.b32 {%rs138, %rs139}, %r23;
mov.b32 {%rs140, %rs141}, %r24;
mov.b32 {%rs142, %rs143}, %r25;
mov.b32 {%rs144, %rs145}, %r26;
cvt.f32.bf16 %r323, %rs130;
cvt.f32.bf16 %r324, %rs131;
cvt.f32.bf16 %r325, %rs132;
cvt.f32.bf16 %r326, %rs133;
cvt.f32.bf16 %r327, %rs134;
cvt.f32.bf16 %r328, %rs135;
cvt.f32.bf16 %r329, %rs136;
cvt.f32.bf16 %r330, %rs137;
cvt.f32.bf16 %r331, %rs138;
cvt.f32.bf16 %r332, %rs139;
cvt.f32.bf16 %r333, %rs140;
cvt.f32.bf16 %r334, %rs141;
cvt.f32.bf16 %r335, %rs142;
cvt.f32.bf16 %r336, %rs143;
cvt.f32.bf16 %r337, %rs144;
cvt.f32.bf16 %r338, %rs145;
mul.f32 %r339, %r306, %r306;
"#) ++
  r#"$L__tmp1:
"# ++
  (if vector then r#"fma.rn.f32 %r315, %r280, %r280, %r314;
fma.rn.f32 %r316, %r282, %r282, %r315;
fma.rn.f32 %r317, %r283, %r283, %r316;
fma.rn.f32 %r318, %r284, %r284, %r317;
fma.rn.f32 %r319, %r285, %r285, %r318;
fma.rn.f32 %r320, %r286, %r286, %r319;
fma.rn.f32 %r321, %r287, %r287, %r320;
fma.rn.f32 %r322, %r288, %r288, %r321;
fma.rn.f32 %r323, %r289, %r289, %r322;
fma.rn.f32 %r324, %r290, %r290, %r323;
fma.rn.f32 %r325, %r291, %r291, %r324;
fma.rn.f32 %r326, %r292, %r292, %r325;
fma.rn.f32 %r327, %r293, %r293, %r326;
fma.rn.f32 %r328, %r294, %r294, %r327;
fma.rn.f32 %r329, %r295, %r295, %r328;
"# else r#"fma.rn.f32 %r340, %r305, %r305, %r339;
fma.rn.f32 %r341, %r307, %r307, %r340;
fma.rn.f32 %r342, %r308, %r308, %r341;
fma.rn.f32 %r343, %r309, %r309, %r342;
fma.rn.f32 %r344, %r310, %r310, %r343;
fma.rn.f32 %r345, %r311, %r311, %r344;
fma.rn.f32 %r346, %r312, %r312, %r345;
fma.rn.f32 %r347, %r313, %r313, %r346;
fma.rn.f32 %r348, %r314, %r314, %r347;
fma.rn.f32 %r349, %r315, %r315, %r348;
fma.rn.f32 %r350, %r316, %r316, %r349;
fma.rn.f32 %r351, %r317, %r317, %r350;
fma.rn.f32 %r352, %r318, %r318, %r351;
fma.rn.f32 %r353, %r319, %r319, %r352;
fma.rn.f32 %r354, %r320, %r320, %r353;
"#) ++
  r#"$L__tmp2:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r330, %r329, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r355, %r354, 2, 31, -1;
"#) ++
  r#"$L__tmp3:
"# ++
  (if vector then r#"add.f32 %r331, %r329, %r330;
"# else r#"add.f32 %r356, %r354, %r355;
"#) ++
  r#"$L__tmp4:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r332, %r331, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r357, %r356, 1, 31, -1;
"#) ++
  r#"$L__tmp5:
"# ++
  (if vector then r#"add.f32 %r333, %r331, %r332;
mov.b32 %r334, 0f42800000;
"# else r#"add.f32 %r358, %r356, %r357;
mov.b32 %r359, 0f42800000;
"#) ++
  r#"$L__tmp6:
"# ++
  (if vector then r#"div.full.f32 %r335, %r333, %r334;
add.f32 %r336, %r335, 0f34000000;
rsqrt.approx.ftz.f32 %r337, %r336;
mul.f32 %r338, %r337, %r280;
mul.f32 %r339, %r337, %r281;
mul.f32 %r340, %r337, %r282;
mul.f32 %r341, %r337, %r283;
mul.f32 %r342, %r337, %r284;
mul.f32 %r343, %r337, %r285;
mul.f32 %r344, %r337, %r286;
mul.f32 %r345, %r337, %r287;
mul.f32 %r346, %r337, %r288;
mul.f32 %r347, %r337, %r289;
mul.f32 %r348, %r337, %r290;
mul.f32 %r349, %r337, %r291;
mul.f32 %r350, %r337, %r292;
mul.f32 %r351, %r337, %r293;
mul.f32 %r352, %r337, %r294;
mul.f32 %r353, %r337, %r295;
mul.f32 %r354, %r263, %r299;
mul.f32 %r355, %r264, %r298;
mul.f32 %r356, %r265, %r301;
mul.f32 %r357, %r266, %r300;
mul.f32 %r358, %r267, %r303;
mul.f32 %r359, %r268, %r302;
mul.f32 %r360, %r269, %r305;
mul.f32 %r361, %r270, %r304;
mul.f32 %r362, %r271, %r307;
mul.f32 %r363, %r272, %r306;
mul.f32 %r364, %r273, %r309;
mul.f32 %r365, %r274, %r308;
mul.f32 %r366, %r275, %r311;
mul.f32 %r367, %r276, %r310;
mul.f32 %r368, %r277, %r313;
mul.f32 %r369, %r278, %r312;
fma.rn.f32 %r370, %r247, %r298, %r354;
fma.rn.f32 %r371, %r248, %r299, %r355;
fma.rn.f32 %r372, %r249, %r300, %r356;
fma.rn.f32 %r373, %r250, %r301, %r357;
fma.rn.f32 %r374, %r251, %r302, %r358;
fma.rn.f32 %r375, %r252, %r303, %r359;
fma.rn.f32 %r376, %r253, %r304, %r360;
fma.rn.f32 %r377, %r254, %r305, %r361;
fma.rn.f32 %r378, %r255, %r306, %r362;
fma.rn.f32 %r379, %r256, %r307, %r363;
fma.rn.f32 %r380, %r257, %r308, %r364;
fma.rn.f32 %r381, %r258, %r309, %r365;
fma.rn.f32 %r382, %r259, %r310, %r366;
fma.rn.f32 %r383, %r260, %r311, %r367;
fma.rn.f32 %r384, %r261, %r312, %r368;
fma.rn.f32 %r385, %r262, %r313, %r369;
mul.f32 %r386, %r371, %r339;
"# else r#"div.full.f32 %r360, %r358, %r359;
add.f32 %r361, %r360, 0f34000000;
rsqrt.approx.ftz.f32 %r362, %r361;
mul.f32 %r363, %r362, %r305;
mul.f32 %r364, %r362, %r306;
mul.f32 %r365, %r362, %r307;
mul.f32 %r366, %r362, %r308;
mul.f32 %r367, %r362, %r309;
mul.f32 %r368, %r362, %r310;
mul.f32 %r369, %r362, %r311;
mul.f32 %r370, %r362, %r312;
mul.f32 %r371, %r362, %r313;
mul.f32 %r372, %r362, %r314;
mul.f32 %r373, %r362, %r315;
mul.f32 %r374, %r362, %r316;
mul.f32 %r375, %r362, %r317;
mul.f32 %r376, %r362, %r318;
mul.f32 %r377, %r362, %r319;
mul.f32 %r378, %r362, %r320;
mul.f32 %r379, %r288, %r324;
mul.f32 %r380, %r289, %r323;
mul.f32 %r381, %r290, %r326;
mul.f32 %r382, %r291, %r325;
mul.f32 %r383, %r292, %r328;
mul.f32 %r384, %r293, %r327;
mul.f32 %r385, %r294, %r330;
mul.f32 %r386, %r295, %r329;
mul.f32 %r387, %r296, %r332;
mul.f32 %r388, %r297, %r331;
mul.f32 %r389, %r298, %r334;
mul.f32 %r390, %r299, %r333;
mul.f32 %r391, %r300, %r336;
mul.f32 %r392, %r301, %r335;
mul.f32 %r393, %r302, %r338;
mul.f32 %r394, %r303, %r337;
fma.rn.f32 %r395, %r245, %r323, %r379;
fma.rn.f32 %r396, %r246, %r324, %r380;
fma.rn.f32 %r397, %r247, %r325, %r381;
fma.rn.f32 %r398, %r248, %r326, %r382;
fma.rn.f32 %r399, %r249, %r327, %r383;
fma.rn.f32 %r400, %r250, %r328, %r384;
fma.rn.f32 %r401, %r251, %r329, %r385;
fma.rn.f32 %r402, %r252, %r330, %r386;
fma.rn.f32 %r403, %r253, %r331, %r387;
fma.rn.f32 %r404, %r254, %r332, %r388;
fma.rn.f32 %r405, %r255, %r333, %r389;
fma.rn.f32 %r406, %r256, %r334, %r390;
fma.rn.f32 %r407, %r257, %r335, %r391;
fma.rn.f32 %r408, %r258, %r336, %r392;
fma.rn.f32 %r409, %r259, %r337, %r393;
fma.rn.f32 %r410, %r260, %r338, %r394;
mul.f32 %r411, %r396, %r364;
"#) ++
  r#"$L__tmp7:
"# ++
  (if vector then r#"fma.rn.f32 %r387, %r370, %r338, %r386;
fma.rn.f32 %r388, %r372, %r340, %r387;
fma.rn.f32 %r389, %r373, %r341, %r388;
fma.rn.f32 %r390, %r374, %r342, %r389;
fma.rn.f32 %r391, %r375, %r343, %r390;
fma.rn.f32 %r392, %r376, %r344, %r391;
fma.rn.f32 %r393, %r377, %r345, %r392;
fma.rn.f32 %r394, %r378, %r346, %r393;
fma.rn.f32 %r395, %r379, %r347, %r394;
fma.rn.f32 %r396, %r380, %r348, %r395;
fma.rn.f32 %r397, %r381, %r349, %r396;
fma.rn.f32 %r398, %r382, %r350, %r397;
fma.rn.f32 %r399, %r383, %r351, %r398;
fma.rn.f32 %r400, %r384, %r352, %r399;
fma.rn.f32 %r401, %r385, %r353, %r400;
"# else r#"fma.rn.f32 %r412, %r395, %r363, %r411;
fma.rn.f32 %r413, %r397, %r365, %r412;
fma.rn.f32 %r414, %r398, %r366, %r413;
fma.rn.f32 %r415, %r399, %r367, %r414;
fma.rn.f32 %r416, %r400, %r368, %r415;
fma.rn.f32 %r417, %r401, %r369, %r416;
fma.rn.f32 %r418, %r402, %r370, %r417;
fma.rn.f32 %r419, %r403, %r371, %r418;
fma.rn.f32 %r420, %r404, %r372, %r419;
fma.rn.f32 %r421, %r405, %r373, %r420;
fma.rn.f32 %r422, %r406, %r374, %r421;
fma.rn.f32 %r423, %r407, %r375, %r422;
fma.rn.f32 %r424, %r408, %r376, %r423;
fma.rn.f32 %r425, %r409, %r377, %r424;
fma.rn.f32 %r426, %r410, %r378, %r425;
"#) ++
  r#"$L__tmp8:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r402, %r401, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r427, %r426, 2, 31, -1;
"#) ++
  r#"$L__tmp9:
"# ++
  (if vector then r#"add.f32 %r403, %r401, %r402;
"# else r#"add.f32 %r428, %r426, %r427;
"#) ++
  r#"$L__tmp10:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r404, %r403, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r429, %r428, 1, 31, -1;
"#) ++
  r#"$L__tmp11:
"# ++
  (if vector then r#"add.f32 %r405, %r403, %r404;
"# else r#"add.f32 %r430, %r428, %r429;
"#) ++
  r#"$L__tmp12:
"# ++
  (if vector then r#"div.full.f32 %r406, %r405, %r334;
neg.f32 %r407, %r338;
fma.rn.f32 %r408, %r407, %r406, %r370;
neg.f32 %r409, %r339;
fma.rn.f32 %r410, %r409, %r406, %r371;
neg.f32 %r411, %r340;
fma.rn.f32 %r412, %r411, %r406, %r372;
neg.f32 %r413, %r341;
fma.rn.f32 %r414, %r413, %r406, %r373;
neg.f32 %r415, %r342;
fma.rn.f32 %r416, %r415, %r406, %r374;
neg.f32 %r417, %r343;
fma.rn.f32 %r418, %r417, %r406, %r375;
neg.f32 %r419, %r344;
fma.rn.f32 %r420, %r419, %r406, %r376;
neg.f32 %r421, %r345;
fma.rn.f32 %r422, %r421, %r406, %r377;
neg.f32 %r423, %r346;
fma.rn.f32 %r424, %r423, %r406, %r378;
neg.f32 %r425, %r347;
fma.rn.f32 %r426, %r425, %r406, %r379;
neg.f32 %r427, %r348;
fma.rn.f32 %r428, %r427, %r406, %r380;
neg.f32 %r429, %r349;
fma.rn.f32 %r430, %r429, %r406, %r381;
neg.f32 %r431, %r350;
fma.rn.f32 %r432, %r431, %r406, %r382;
neg.f32 %r433, %r351;
fma.rn.f32 %r434, %r433, %r406, %r383;
neg.f32 %r435, %r352;
fma.rn.f32 %r436, %r435, %r406, %r384;
neg.f32 %r437, %r353;
fma.rn.f32 %r438, %r437, %r406, %r385;
mul.f32 %r439, %r337, %r408;
mul.f32 %r440, %r337, %r410;
mul.f32 %r441, %r337, %r412;
mul.f32 %r442, %r337, %r414;
mul.f32 %r443, %r337, %r416;
mul.f32 %r444, %r337, %r418;
mul.f32 %r445, %r337, %r420;
mul.f32 %r446, %r337, %r422;
mul.f32 %r447, %r337, %r424;
mul.f32 %r448, %r337, %r426;
mul.f32 %r449, %r337, %r428;
mul.f32 %r450, %r337, %r430;
mul.f32 %r451, %r337, %r432;
mul.f32 %r452, %r337, %r434;
mul.f32 %r453, %r337, %r436;
mul.f32 %r454, %r337, %r438;
cvt.rn.bf16.f32 %rs129, %r439;
cvt.rn.bf16.f32 %rs130, %r440;
cvt.rn.bf16.f32 %rs131, %r441;
cvt.rn.bf16.f32 %rs132, %r442;
cvt.rn.bf16.f32 %rs133, %r443;
cvt.rn.bf16.f32 %rs134, %r444;
cvt.rn.bf16.f32 %rs135, %r445;
cvt.rn.bf16.f32 %rs136, %r446;
cvt.rn.bf16.f32 %rs137, %r447;
cvt.rn.bf16.f32 %rs138, %r448;
cvt.rn.bf16.f32 %rs139, %r449;
cvt.rn.bf16.f32 %rs140, %r450;
cvt.rn.bf16.f32 %rs141, %r451;
cvt.rn.bf16.f32 %rs142, %r452;
cvt.rn.bf16.f32 %rs143, %r453;
cvt.rn.bf16.f32 %rs144, %r454;
shl.b32 %r455, %r236, 5;
and.b32 %r456, %r232, 896;
shl.b32 %r457, %r230, 2;
and.b32 %r458, %r457, 16;
bfe.s32 %r459, %r230, 6, 1;
and.b32 %r460, %r459, 1056;
or.b32 %r461, %r455, %r456;
xor.b32 %r462, %r461, %r460;
or.b32 %r463, %r462, %r458;
mov.b32 %r464, global_smem;
add.s32 %r465, %r464, %r463;
mov.b32 %r466, {%rs141, %rs143};
mov.b32 %r467, {%rs137, %rs139};
mov.b32 %r468, {%rs133, %rs135};
mov.b32 %r469, {%rs129, %rs131};
st.shared.v4.b32 [%r465], {%r469, %r468, %r467, %r466};
xor.b32 %r470, %r463, 64;
add.s32 %r471, %r464, %r470;
mov.b32 %r472, {%rs142, %rs144};
mov.b32 %r473, {%rs138, %rs140};
mov.b32 %r474, {%rs134, %rs136};
mov.b32 %r475, {%rs130, %rs132};
st.shared.v4.b32 [%r471+2048], {%r475, %r474, %r473, %r472};
"# else r#"div.full.f32 %r431, %r430, %r359;
neg.f32 %r432, %r363;
fma.rn.f32 %r433, %r432, %r431, %r395;
neg.f32 %r434, %r364;
fma.rn.f32 %r435, %r434, %r431, %r396;
neg.f32 %r436, %r365;
fma.rn.f32 %r437, %r436, %r431, %r397;
neg.f32 %r438, %r366;
fma.rn.f32 %r439, %r438, %r431, %r398;
neg.f32 %r440, %r367;
fma.rn.f32 %r441, %r440, %r431, %r399;
neg.f32 %r442, %r368;
fma.rn.f32 %r443, %r442, %r431, %r400;
neg.f32 %r444, %r369;
fma.rn.f32 %r445, %r444, %r431, %r401;
neg.f32 %r446, %r370;
fma.rn.f32 %r447, %r446, %r431, %r402;
neg.f32 %r448, %r371;
fma.rn.f32 %r449, %r448, %r431, %r403;
neg.f32 %r450, %r372;
fma.rn.f32 %r451, %r450, %r431, %r404;
neg.f32 %r452, %r373;
fma.rn.f32 %r453, %r452, %r431, %r405;
neg.f32 %r454, %r374;
fma.rn.f32 %r455, %r454, %r431, %r406;
neg.f32 %r456, %r375;
fma.rn.f32 %r457, %r456, %r431, %r407;
neg.f32 %r458, %r376;
fma.rn.f32 %r459, %r458, %r431, %r408;
neg.f32 %r460, %r377;
fma.rn.f32 %r461, %r460, %r431, %r409;
neg.f32 %r462, %r378;
fma.rn.f32 %r463, %r462, %r431, %r410;
mul.f32 %r464, %r362, %r433;
mul.f32 %r465, %r362, %r435;
mul.f32 %r466, %r362, %r437;
mul.f32 %r467, %r362, %r439;
mul.f32 %r468, %r362, %r441;
mul.f32 %r469, %r362, %r443;
mul.f32 %r470, %r362, %r445;
mul.f32 %r471, %r362, %r447;
mul.f32 %r472, %r362, %r449;
mul.f32 %r473, %r362, %r451;
mul.f32 %r474, %r362, %r453;
mul.f32 %r475, %r362, %r455;
mul.f32 %r476, %r362, %r457;
mul.f32 %r477, %r362, %r459;
mul.f32 %r478, %r362, %r461;
mul.f32 %r479, %r362, %r463;
cvt.rn.bf16.f32 %rs146, %r464;
cvt.rn.bf16.f32 %rs147, %r465;
cvt.rn.bf16.f32 %rs148, %r466;
cvt.rn.bf16.f32 %rs149, %r467;
cvt.rn.bf16.f32 %rs150, %r468;
cvt.rn.bf16.f32 %rs151, %r469;
cvt.rn.bf16.f32 %rs152, %r470;
cvt.rn.bf16.f32 %rs153, %r471;
cvt.rn.bf16.f32 %rs154, %r472;
cvt.rn.bf16.f32 %rs155, %r473;
cvt.rn.bf16.f32 %rs156, %r474;
cvt.rn.bf16.f32 %rs157, %r475;
cvt.rn.bf16.f32 %rs158, %r476;
cvt.rn.bf16.f32 %rs159, %r477;
cvt.rn.bf16.f32 %rs160, %r478;
cvt.rn.bf16.f32 %rs161, %r479;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"shl.b32 %r476, %r230, 9;
and.b32 %r477, %r476, 3072;
and.b32 %r478, %r232, 496;
and.b32 %r479, %r230, 96;
or.b32 %r480, %r477, %r478;
xor.b32 %r481, %r480, %r479;
add.s32 %r482, %r464, %r481;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r483, %r484, %r485, %r486}, [%r482];
mov.b32 {%rs145, %rs146}, %r483;
mov.b32 {%rs147, %rs148}, %r484;
mov.b32 {%rs149, %rs150}, %r485;
mov.b32 {%rs151, %rs152}, %r486;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r487, %r488, %r489, %r490}, [%r482+512];
mov.b32 {%rs153, %rs154}, %r487;
mov.b32 {%rs155, %rs156}, %r488;
mov.b32 {%rs157, %rs158}, %r489;
mov.b32 {%rs159, %rs160}, %r490;
cvt.f32.bf16 %r491, %rs145;
cvt.f32.bf16 %r492, %rs146;
cvt.f32.bf16 %r493, %rs147;
cvt.f32.bf16 %r494, %rs148;
cvt.f32.bf16 %r495, %rs149;
cvt.f32.bf16 %r496, %rs150;
cvt.f32.bf16 %r497, %rs151;
cvt.f32.bf16 %r498, %rs152;
cvt.f32.bf16 %r499, %rs153;
cvt.f32.bf16 %r500, %rs154;
cvt.f32.bf16 %r501, %rs155;
cvt.f32.bf16 %r502, %rs156;
cvt.f32.bf16 %r503, %rs157;
cvt.f32.bf16 %r504, %rs158;
cvt.f32.bf16 %r505, %rs159;
cvt.f32.bf16 %r506, %rs160;
cvt.f32.bf16 %r507, %rs129;
cvt.f32.bf16 %r508, %rs130;
cvt.f32.bf16 %r509, %rs131;
cvt.f32.bf16 %r510, %rs132;
cvt.f32.bf16 %r511, %rs133;
cvt.f32.bf16 %r512, %rs134;
cvt.f32.bf16 %r513, %rs135;
cvt.f32.bf16 %r514, %rs136;
cvt.f32.bf16 %r515, %rs137;
cvt.f32.bf16 %r516, %rs138;
cvt.f32.bf16 %r517, %rs139;
cvt.f32.bf16 %r518, %rs140;
cvt.f32.bf16 %r519, %rs141;
cvt.f32.bf16 %r520, %rs142;
cvt.f32.bf16 %r521, %rs143;
cvt.f32.bf16 %r522, %rs144;
div.full.f32 %r35, %r491, %r18;
div.full.f32 %r36, %r492, %r18;
div.full.f32 %r37, %r493, %r18;
div.full.f32 %r38, %r494, %r18;
div.full.f32 %r39, %r495, %r18;
div.full.f32 %r40, %r496, %r18;
div.full.f32 %r41, %r497, %r18;
div.full.f32 %r42, %r498, %r18;
div.full.f32 %r43, %r499, %r18;
div.full.f32 %r44, %r500, %r18;
div.full.f32 %r45, %r501, %r18;
div.full.f32 %r46, %r502, %r18;
div.full.f32 %r47, %r503, %r18;
div.full.f32 %r48, %r504, %r18;
div.full.f32 %r49, %r505, %r18;
div.full.f32 %r50, %r506, %r18;
div.full.f32 %r51, %r507, %r18;
div.full.f32 %r52, %r508, %r18;
div.full.f32 %r53, %r509, %r18;
div.full.f32 %r54, %r510, %r18;
div.full.f32 %r55, %r511, %r18;
div.full.f32 %r56, %r512, %r18;
div.full.f32 %r57, %r513, %r18;
div.full.f32 %r58, %r514, %r18;
div.full.f32 %r59, %r515, %r18;
div.full.f32 %r60, %r516, %r18;
div.full.f32 %r61, %r517, %r18;
div.full.f32 %r62, %r518, %r18;
div.full.f32 %r63, %r519, %r18;
div.full.f32 %r64, %r520, %r18;
div.full.f32 %r65, %r521, %r18;
div.full.f32 %r66, %r522, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r50, %r49;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r52, %r51;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r54, %r53;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r56, %r55;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r58, %r57;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r60, %r59;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r62, %r61;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r64, %r63;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r66, %r65;
or.b32 %r523, %r237, %r244;
or.b32 %r524, %r238, %r244;
mul.lo.s32 %r525, %r234, 1536;
cvt.s64.s32 %rd47, %r525;
add.s64 %rd48, %rd32, %rd47;
cvt.s64.s32 %rd49, %r523;
add.s64 %rd10, %rd48, %rd49;
mov.b32 %r67, {%rs9, %rs10};
mov.b32 %r68, {%rs11, %rs12};
mov.b32 %r69, {%rs13, %rs14};
mov.b32 %r70, {%rs15, %rs16};
@%p1 st.global.v4.b32 [ %rd10 + 0 ], { %r67, %r68, %r69, %r70 };
@EXTENT_13@
cvt.s64.s32 %rd50, %r526;
add.s64 %rd51, %rd33, %rd50;
cvt.u64.u32 %rd52, %r235;
add.s64 %rd11, %rd51, %rd52;
mov.b32 %r71, {%rs1, %rs2};
mov.b32 %r72, {%rs3, %rs4};
mov.b32 %r73, {%rs5, %rs6};
mov.b32 %r74, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd11 + 0 ], { %r71, %r72, %r73, %r74 };
add.s32 %r527, %r244, 384;
mad.wide.s32 %rd53, %r527, 2, %rd41;
add.s64 %rd12, %rd53, %rd38;
add.s64 %rd13, %rd12, 16;
"# else r#"shl.b32 %r480, %r232, 5;
and.b32 %r481, %r225, 896;
shl.b32 %r482, %r222, 2;
and.b32 %r483, %r482, 16;
bfe.s32 %r484, %r222, 6, 1;
and.b32 %r485, %r484, 1056;
or.b32 %r486, %r480, %r481;
xor.b32 %r487, %r486, %r485;
or.b32 %r488, %r487, %r483;
add.s32 %r489, %r262, %r488;
mov.b32 %r490, {%rs158, %rs160};
mov.b32 %r491, {%rs154, %rs156};
mov.b32 %r492, {%rs150, %rs152};
mov.b32 %r493, {%rs146, %rs148};
st.shared.v4.b32 [%r489], {%r493, %r492, %r491, %r490};
xor.b32 %r494, %r488, 64;
add.s32 %r495, %r262, %r494;
mov.b32 %r496, {%rs159, %rs161};
mov.b32 %r497, {%rs155, %rs157};
mov.b32 %r498, {%rs151, %rs153};
mov.b32 %r499, {%rs147, %rs149};
st.shared.v4.b32 [%r495+2048], {%r499, %r498, %r497, %r496};
bar.sync 0;
shl.b32 %r500, %r222, 9;
and.b32 %r501, %r500, 3072;
shl.b32 %r502, %r224, 4;
or.b32 %r503, %r501, %r502;
xor.b32 %r504, %r503, %r230;
add.s32 %r505, %r262, %r504;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r506, %r507, %r508, %r509}, [%r505];
mov.b32 {%rs162, %rs163}, %r506;
mov.b32 {%rs164, %rs165}, %r507;
mov.b32 {%rs166, %rs167}, %r508;
mov.b32 {%rs168, %rs169}, %r509;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r510, %r511, %r512, %r513}, [%r505+512];
mov.b32 {%rs170, %rs171}, %r510;
mov.b32 {%rs172, %rs173}, %r511;
mov.b32 {%rs174, %rs175}, %r512;
mov.b32 {%rs176, %rs177}, %r513;
cvt.f32.bf16 %r514, %rs162;
cvt.f32.bf16 %r515, %rs163;
cvt.f32.bf16 %r516, %rs164;
cvt.f32.bf16 %r517, %rs165;
cvt.f32.bf16 %r518, %rs166;
cvt.f32.bf16 %r519, %rs167;
cvt.f32.bf16 %r520, %rs168;
cvt.f32.bf16 %r521, %rs169;
cvt.f32.bf16 %r522, %rs170;
cvt.f32.bf16 %r523, %rs171;
cvt.f32.bf16 %r524, %rs172;
cvt.f32.bf16 %r525, %rs173;
cvt.f32.bf16 %r526, %rs174;
cvt.f32.bf16 %r527, %rs175;
cvt.f32.bf16 %r528, %rs176;
cvt.f32.bf16 %r529, %rs177;
cvt.f32.bf16 %r530, %rs146;
cvt.f32.bf16 %r531, %rs147;
cvt.f32.bf16 %r532, %rs148;
cvt.f32.bf16 %r533, %rs149;
cvt.f32.bf16 %r534, %rs150;
cvt.f32.bf16 %r535, %rs151;
cvt.f32.bf16 %r536, %rs152;
cvt.f32.bf16 %r537, %rs153;
cvt.f32.bf16 %r538, %rs154;
cvt.f32.bf16 %r539, %rs155;
cvt.f32.bf16 %r540, %rs156;
cvt.f32.bf16 %r541, %rs157;
cvt.f32.bf16 %r542, %rs158;
cvt.f32.bf16 %r543, %rs159;
cvt.f32.bf16 %r544, %rs160;
cvt.f32.bf16 %r545, %rs161;
div.full.f32 %r27, %r514, %r10;
div.full.f32 %r28, %r515, %r10;
div.full.f32 %r29, %r516, %r10;
div.full.f32 %r30, %r517, %r10;
div.full.f32 %r31, %r518, %r10;
div.full.f32 %r32, %r519, %r10;
div.full.f32 %r33, %r520, %r10;
div.full.f32 %r34, %r521, %r10;
div.full.f32 %r35, %r522, %r10;
div.full.f32 %r36, %r523, %r10;
div.full.f32 %r37, %r524, %r10;
div.full.f32 %r38, %r525, %r10;
div.full.f32 %r39, %r526, %r10;
div.full.f32 %r40, %r527, %r10;
div.full.f32 %r41, %r528, %r10;
div.full.f32 %r42, %r529, %r10;
div.full.f32 %r43, %r530, %r10;
div.full.f32 %r44, %r531, %r10;
div.full.f32 %r45, %r532, %r10;
div.full.f32 %r46, %r533, %r10;
div.full.f32 %r47, %r534, %r10;
div.full.f32 %r48, %r535, %r10;
div.full.f32 %r49, %r536, %r10;
div.full.f32 %r50, %r537, %r10;
div.full.f32 %r51, %r538, %r10;
div.full.f32 %r52, %r539, %r10;
div.full.f32 %r53, %r540, %r10;
div.full.f32 %r54, %r541, %r10;
div.full.f32 %r55, %r542, %r10;
div.full.f32 %r56, %r543, %r10;
div.full.f32 %r57, %r544, %r10;
div.full.f32 %r58, %r545, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r28, %r27;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r30, %r29;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r32, %r31;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r34, %r33;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r50, %r49;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r52, %r51;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r54, %r53;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r56, %r55;
cvt.rn.satfinite.e4m3x2.f32 %rs33, %r58, %r57;
or.b32 %r546, %r233, %r241;
or.b32 %r547, %r234, %r241;
mul.lo.s32 %r548, %r227, 1536;
cvt.s64.s32 %rd64, %r548;
add.s64 %rd65, %rd46, %rd64;
cvt.s64.s32 %rd66, %r546;
add.s64 %rd24, %rd65, %rd66;
mov.b32 %r59, {%rs26, %rs27};
mov.b32 %r60, {%rs28, %rs29};
mov.b32 %r61, {%rs30, %rs31};
mov.b32 %r62, {%rs32, %rs33};
@%p1 st.global.v4.b32 [ %rd24 + 0 ], { %r59, %r60, %r61, %r62 };
@EXTENT_6@
cvt.s64.s32 %rd67, %r549;
add.s64 %rd68, %rd47, %rd67;
cvt.u64.u32 %rd69, %r229;
add.s64 %rd25, %rd68, %rd69;
mov.b32 %r63, {%rs18, %rs19};
mov.b32 %r64, {%rs20, %rs21};
mov.b32 %r65, {%rs22, %rs23};
mov.b32 %r66, {%rs24, %rs25};
@%p1 st.global.v4.b32 [ %rd25 + 0 ], { %r63, %r64, %r65, %r66 };
add.s32 %r550, %r241, 384;
mad.wide.s32 %rd70, %r550, 2, %rd58;
add.s64 %rd26, %rd70, %rd51;
add.s64 %rd27, %rd26, 16;
mov.u32 %r67, %r5;
mov.u32 %r68, %r5;
mov.u32 %r69, %r5;
mov.u32 %r70, %r5;
@%p1 ld.global.v4.b32 { %r67, %r68, %r69, %r70 }, [ %rd26 + 0 ];
mov.b32 {%rs178, %rs179}, %r67;
mov.b32 {%rs180, %rs181}, %r68;
mov.b32 {%rs182, %rs183}, %r69;
mov.b32 {%rs184, %rs185}, %r70;
mov.u32 %r71, %r5;
mov.u32 %r72, %r5;
mov.u32 %r73, %r5;
mov.u32 %r74, %r5;
@%p1 ld.global.v4.b32 { %r71, %r72, %r73, %r74 }, [ %rd27 + 0 ];
mov.b32 {%rs186, %rs187}, %r71;
mov.b32 {%rs188, %rs189}, %r72;
mov.b32 {%rs190, %rs191}, %r73;
mov.b32 {%rs192, %rs193}, %r74;
cvt.f32.bf16 %r551, %rs178;
cvt.f32.bf16 %r552, %rs179;
cvt.f32.bf16 %r553, %rs180;
cvt.f32.bf16 %r554, %rs181;
cvt.f32.bf16 %r555, %rs182;
cvt.f32.bf16 %r556, %rs183;
cvt.f32.bf16 %r557, %rs184;
cvt.f32.bf16 %r558, %rs185;
cvt.f32.bf16 %r559, %rs186;
cvt.f32.bf16 %r560, %rs187;
cvt.f32.bf16 %r561, %rs188;
cvt.f32.bf16 %r562, %rs189;
cvt.f32.bf16 %r563, %rs190;
cvt.f32.bf16 %r564, %rs191;
cvt.f32.bf16 %r565, %rs192;
cvt.f32.bf16 %r566, %rs193;
add.s64 %rd71, %rd41, %rd60;
add.s64 %rd72, %rd71, %rd62;
add.s64 %rd28, %rd72, %rd51;
add.s64 %rd29, %rd28, 16;
"#) ++
  r#"mov.u32 %r75, %r5;
mov.u32 %r76, %r5;
mov.u32 %r77, %r5;
mov.u32 %r78, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd12 + 0 ];
mov.b32 {%rs161, %rs162}, %r75;
mov.b32 {%rs163, %rs164}, %r76;
mov.b32 {%rs165, %rs166}, %r77;
mov.b32 {%rs167, %rs168}, %r78;
"# else r#"@%p1 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd28 + 0 ];
"#) ++
  r#"mov.u32 %r79, %r5;
mov.u32 %r80, %r5;
mov.u32 %r81, %r5;
mov.u32 %r82, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd13 + 0 ];
mov.b32 {%rs169, %rs170}, %r79;
mov.b32 {%rs171, %rs172}, %r80;
mov.b32 {%rs173, %rs174}, %r81;
mov.b32 {%rs175, %rs176}, %r82;
cvt.f32.bf16 %r528, %rs161;
cvt.f32.bf16 %r529, %rs162;
cvt.f32.bf16 %r530, %rs163;
cvt.f32.bf16 %r531, %rs164;
cvt.f32.bf16 %r532, %rs165;
cvt.f32.bf16 %r533, %rs166;
cvt.f32.bf16 %r534, %rs167;
cvt.f32.bf16 %r535, %rs168;
cvt.f32.bf16 %r536, %rs169;
cvt.f32.bf16 %r537, %rs170;
cvt.f32.bf16 %r538, %rs171;
cvt.f32.bf16 %r539, %rs172;
cvt.f32.bf16 %r540, %rs173;
cvt.f32.bf16 %r541, %rs174;
cvt.f32.bf16 %r542, %rs175;
cvt.f32.bf16 %r543, %rs176;
add.s64 %rd54, %rd27, %rd43;
add.s64 %rd55, %rd54, %rd45;
add.s64 %rd14, %rd55, %rd38;
add.s64 %rd15, %rd14, 16;
mov.u32 %r83, %r5;
mov.u32 %r84, %r5;
mov.u32 %r85, %r5;
mov.u32 %r86, %r5;
@%p1 ld.global.v4.b32 { %r83, %r84, %r85, %r86 }, [ %rd14 + 0 ];
mov.u32 %r87, %r5;
mov.u32 %r88, %r5;
mov.u32 %r89, %r5;
mov.u32 %r90, %r5;
@%p1 ld.global.v4.b32 { %r87, %r88, %r89, %r90 }, [ %rd15 + 0 ];
mul.f32 %r544, %r529, %r529;
"# else r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd29 + 0 ];
mul.f32 %r567, %r552, %r552;
"#) ++
  r#"$L__tmp13:
"# ++
  (if vector then r#"fma.rn.f32 %r545, %r528, %r528, %r544;
fma.rn.f32 %r546, %r530, %r530, %r545;
fma.rn.f32 %r547, %r531, %r531, %r546;
fma.rn.f32 %r548, %r532, %r532, %r547;
fma.rn.f32 %r549, %r533, %r533, %r548;
fma.rn.f32 %r550, %r534, %r534, %r549;
fma.rn.f32 %r551, %r535, %r535, %r550;
fma.rn.f32 %r552, %r536, %r536, %r551;
fma.rn.f32 %r553, %r537, %r537, %r552;
fma.rn.f32 %r554, %r538, %r538, %r553;
fma.rn.f32 %r555, %r539, %r539, %r554;
fma.rn.f32 %r556, %r540, %r540, %r555;
fma.rn.f32 %r557, %r541, %r541, %r556;
fma.rn.f32 %r558, %r542, %r542, %r557;
fma.rn.f32 %r559, %r543, %r543, %r558;
"# else r#"fma.rn.f32 %r568, %r551, %r551, %r567;
fma.rn.f32 %r569, %r553, %r553, %r568;
fma.rn.f32 %r570, %r554, %r554, %r569;
fma.rn.f32 %r571, %r555, %r555, %r570;
fma.rn.f32 %r572, %r556, %r556, %r571;
fma.rn.f32 %r573, %r557, %r557, %r572;
fma.rn.f32 %r574, %r558, %r558, %r573;
fma.rn.f32 %r575, %r559, %r559, %r574;
fma.rn.f32 %r576, %r560, %r560, %r575;
fma.rn.f32 %r577, %r561, %r561, %r576;
fma.rn.f32 %r578, %r562, %r562, %r577;
fma.rn.f32 %r579, %r563, %r563, %r578;
fma.rn.f32 %r580, %r564, %r564, %r579;
fma.rn.f32 %r581, %r565, %r565, %r580;
fma.rn.f32 %r582, %r566, %r566, %r581;
"#) ++
  r#"$L__tmp14:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r560, %r559, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r583, %r582, 2, 31, -1;
"#) ++
  r#"$L__tmp15:
"# ++
  (if vector then r#"add.f32 %r561, %r559, %r560;
"# else r#"add.f32 %r584, %r582, %r583;
"#) ++
  r#"$L__tmp16:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r562, %r561, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r585, %r584, 1, 31, -1;
"#) ++
  r#"$L__tmp17:
"# ++
  (if vector then r#"add.f32 %r563, %r561, %r562;
"# else r#"add.f32 %r586, %r584, %r585;
"#) ++
  r#"$L__tmp18:
"# ++
  (if vector then r#"div.full.f32 %r564, %r563, %r334;
add.f32 %r565, %r564, 0f34000000;
rsqrt.approx.ftz.f32 %r566, %r565;
mov.b32 {%rs177, %rs178}, %r90;
cvt.f32.bf16 %r567, %rs178;
cvt.f32.bf16 %r568, %rs177;
mov.b32 {%rs179, %rs180}, %r89;
cvt.f32.bf16 %r569, %rs180;
cvt.f32.bf16 %r570, %rs179;
mov.b32 {%rs181, %rs182}, %r88;
cvt.f32.bf16 %r571, %rs182;
cvt.f32.bf16 %r572, %rs181;
mov.b32 {%rs183, %rs184}, %r87;
cvt.f32.bf16 %r573, %rs184;
cvt.f32.bf16 %r574, %rs183;
mov.b32 {%rs185, %rs186}, %r86;
cvt.f32.bf16 %r575, %rs186;
cvt.f32.bf16 %r576, %rs185;
mov.b32 {%rs187, %rs188}, %r85;
cvt.f32.bf16 %r577, %rs188;
cvt.f32.bf16 %r578, %rs187;
mov.b32 {%rs189, %rs190}, %r84;
cvt.f32.bf16 %r579, %rs190;
cvt.f32.bf16 %r580, %rs189;
mov.b32 {%rs191, %rs192}, %r83;
cvt.f32.bf16 %r581, %rs192;
cvt.f32.bf16 %r582, %rs191;
bfe.u32 %r583, %r230, 3, 4;
or.b32 %r584, %r229, %r583;
or.b32 %r585, %r584, 16;
mul.lo.s32 %r586, %r585, 1536;
cvt.s64.s32 %rd56, %r586;
add.s64 %rd57, %rd32, %rd56;
mul.lo.s32 %r587, %r584, 1536;
cvt.s64.s32 %rd58, %r587;
add.s64 %rd59, %rd32, %rd58;
mul.lo.s32 %r588, %r585, 768;
mul.lo.s32 %r589, %r584, 768;
mul.f32 %r590, %r566, %r528;
mul.f32 %r591, %r566, %r529;
mul.f32 %r592, %r566, %r530;
mul.f32 %r593, %r566, %r531;
mul.f32 %r594, %r566, %r532;
mul.f32 %r595, %r566, %r533;
mul.f32 %r596, %r566, %r534;
mul.f32 %r597, %r566, %r535;
mul.f32 %r598, %r566, %r536;
mul.f32 %r599, %r566, %r537;
mul.f32 %r600, %r566, %r538;
mul.f32 %r601, %r566, %r539;
mul.f32 %r602, %r566, %r540;
mul.f32 %r603, %r566, %r541;
mul.f32 %r604, %r566, %r542;
mul.f32 %r605, %r566, %r543;
mul.f32 %r606, %r263, %r581;
mul.f32 %r607, %r264, %r582;
mul.f32 %r608, %r265, %r579;
mul.f32 %r609, %r266, %r580;
mul.f32 %r610, %r267, %r577;
mul.f32 %r611, %r268, %r578;
mul.f32 %r612, %r269, %r575;
mul.f32 %r613, %r270, %r576;
mul.f32 %r614, %r271, %r573;
mul.f32 %r615, %r272, %r574;
mul.f32 %r616, %r273, %r571;
mul.f32 %r617, %r274, %r572;
mul.f32 %r618, %r275, %r569;
mul.f32 %r619, %r276, %r570;
mul.f32 %r620, %r277, %r567;
mul.f32 %r621, %r278, %r568;
fma.rn.f32 %r622, %r247, %r582, %r606;
fma.rn.f32 %r623, %r248, %r581, %r607;
fma.rn.f32 %r624, %r249, %r580, %r608;
fma.rn.f32 %r625, %r250, %r579, %r609;
fma.rn.f32 %r626, %r251, %r578, %r610;
fma.rn.f32 %r627, %r252, %r577, %r611;
fma.rn.f32 %r628, %r253, %r576, %r612;
fma.rn.f32 %r629, %r254, %r575, %r613;
fma.rn.f32 %r630, %r255, %r574, %r614;
fma.rn.f32 %r631, %r256, %r573, %r615;
fma.rn.f32 %r632, %r257, %r572, %r616;
fma.rn.f32 %r633, %r258, %r571, %r617;
fma.rn.f32 %r634, %r259, %r570, %r618;
fma.rn.f32 %r635, %r260, %r569, %r619;
fma.rn.f32 %r636, %r261, %r568, %r620;
fma.rn.f32 %r637, %r262, %r567, %r621;
mul.f32 %r638, %r623, %r591;
"# else r#"div.full.f32 %r587, %r586, %r359;
add.f32 %r588, %r587, 0f34000000;
rsqrt.approx.ftz.f32 %r589, %r588;
mov.b32 {%rs194, %rs195}, %r82;
cvt.f32.bf16 %r590, %rs195;
cvt.f32.bf16 %r591, %rs194;
mov.b32 {%rs196, %rs197}, %r81;
cvt.f32.bf16 %r592, %rs197;
cvt.f32.bf16 %r593, %rs196;
mov.b32 {%rs198, %rs199}, %r80;
cvt.f32.bf16 %r594, %rs199;
cvt.f32.bf16 %r595, %rs198;
mov.b32 {%rs200, %rs201}, %r79;
cvt.f32.bf16 %r596, %rs201;
cvt.f32.bf16 %r597, %rs200;
mov.b32 {%rs202, %rs203}, %r78;
cvt.f32.bf16 %r598, %rs203;
cvt.f32.bf16 %r599, %rs202;
mov.b32 {%rs204, %rs205}, %r77;
cvt.f32.bf16 %r600, %rs205;
cvt.f32.bf16 %r601, %rs204;
mov.b32 {%rs206, %rs207}, %r76;
cvt.f32.bf16 %r602, %rs207;
cvt.f32.bf16 %r603, %rs206;
mov.b32 {%rs208, %rs209}, %r75;
cvt.f32.bf16 %r604, %rs209;
cvt.f32.bf16 %r605, %rs208;
bfe.u32 %r606, %r222, 3, 4;
or.b32 %r607, %r221, %r606;
or.b32 %r608, %r607, 16;
mul.lo.s32 %r609, %r608, 1536;
cvt.s64.s32 %rd73, %r609;
add.s64 %rd74, %rd46, %rd73;
mul.lo.s32 %r610, %r607, 1536;
cvt.s64.s32 %rd75, %r610;
add.s64 %rd76, %rd46, %rd75;
mul.lo.s32 %r611, %r608, 768;
mul.lo.s32 %r612, %r607, 768;
mul.f32 %r613, %r589, %r551;
mul.f32 %r614, %r589, %r552;
mul.f32 %r615, %r589, %r553;
mul.f32 %r616, %r589, %r554;
mul.f32 %r617, %r589, %r555;
mul.f32 %r618, %r589, %r556;
mul.f32 %r619, %r589, %r557;
mul.f32 %r620, %r589, %r558;
mul.f32 %r621, %r589, %r559;
mul.f32 %r622, %r589, %r560;
mul.f32 %r623, %r589, %r561;
mul.f32 %r624, %r589, %r562;
mul.f32 %r625, %r589, %r563;
mul.f32 %r626, %r589, %r564;
mul.f32 %r627, %r589, %r565;
mul.f32 %r628, %r589, %r566;
mul.f32 %r629, %r288, %r604;
mul.f32 %r630, %r289, %r605;
mul.f32 %r631, %r290, %r602;
mul.f32 %r632, %r291, %r603;
mul.f32 %r633, %r292, %r600;
mul.f32 %r634, %r293, %r601;
mul.f32 %r635, %r294, %r598;
mul.f32 %r636, %r295, %r599;
mul.f32 %r637, %r296, %r596;
mul.f32 %r638, %r297, %r597;
mul.f32 %r639, %r298, %r594;
mul.f32 %r640, %r299, %r595;
mul.f32 %r641, %r300, %r592;
mul.f32 %r642, %r301, %r593;
mul.f32 %r643, %r302, %r590;
mul.f32 %r644, %r303, %r591;
fma.rn.f32 %r645, %r245, %r605, %r629;
fma.rn.f32 %r646, %r246, %r604, %r630;
fma.rn.f32 %r647, %r247, %r603, %r631;
fma.rn.f32 %r648, %r248, %r602, %r632;
fma.rn.f32 %r649, %r249, %r601, %r633;
fma.rn.f32 %r650, %r250, %r600, %r634;
fma.rn.f32 %r651, %r251, %r599, %r635;
fma.rn.f32 %r652, %r252, %r598, %r636;
fma.rn.f32 %r653, %r253, %r597, %r637;
fma.rn.f32 %r654, %r254, %r596, %r638;
fma.rn.f32 %r655, %r255, %r595, %r639;
fma.rn.f32 %r656, %r256, %r594, %r640;
fma.rn.f32 %r657, %r257, %r593, %r641;
fma.rn.f32 %r658, %r258, %r592, %r642;
fma.rn.f32 %r659, %r259, %r591, %r643;
fma.rn.f32 %r660, %r260, %r590, %r644;
mul.f32 %r661, %r646, %r614;
"#) ++
  r#"$L__tmp19:
"# ++
  (if vector then r#"fma.rn.f32 %r639, %r622, %r590, %r638;
fma.rn.f32 %r640, %r624, %r592, %r639;
fma.rn.f32 %r641, %r625, %r593, %r640;
fma.rn.f32 %r642, %r626, %r594, %r641;
fma.rn.f32 %r643, %r627, %r595, %r642;
fma.rn.f32 %r644, %r628, %r596, %r643;
fma.rn.f32 %r645, %r629, %r597, %r644;
fma.rn.f32 %r646, %r630, %r598, %r645;
fma.rn.f32 %r647, %r631, %r599, %r646;
fma.rn.f32 %r648, %r632, %r600, %r647;
fma.rn.f32 %r649, %r633, %r601, %r648;
fma.rn.f32 %r650, %r634, %r602, %r649;
fma.rn.f32 %r651, %r635, %r603, %r650;
fma.rn.f32 %r652, %r636, %r604, %r651;
fma.rn.f32 %r653, %r637, %r605, %r652;
"# else r#"fma.rn.f32 %r662, %r645, %r613, %r661;
fma.rn.f32 %r663, %r647, %r615, %r662;
fma.rn.f32 %r664, %r648, %r616, %r663;
fma.rn.f32 %r665, %r649, %r617, %r664;
fma.rn.f32 %r666, %r650, %r618, %r665;
fma.rn.f32 %r667, %r651, %r619, %r666;
fma.rn.f32 %r668, %r652, %r620, %r667;
fma.rn.f32 %r669, %r653, %r621, %r668;
fma.rn.f32 %r670, %r654, %r622, %r669;
fma.rn.f32 %r671, %r655, %r623, %r670;
fma.rn.f32 %r672, %r656, %r624, %r671;
fma.rn.f32 %r673, %r657, %r625, %r672;
fma.rn.f32 %r674, %r658, %r626, %r673;
fma.rn.f32 %r675, %r659, %r627, %r674;
fma.rn.f32 %r676, %r660, %r628, %r675;
"#) ++
  r#"$L__tmp20:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r654, %r653, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r677, %r676, 2, 31, -1;
"#) ++
  r#"$L__tmp21:
"# ++
  (if vector then r#"add.f32 %r655, %r653, %r654;
"# else r#"add.f32 %r678, %r676, %r677;
"#) ++
  r#"$L__tmp22:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r656, %r655, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r679, %r678, 1, 31, -1;
"#) ++
  r#"$L__tmp23:
"# ++
  (if vector then r#"add.f32 %r657, %r655, %r656;
"# else r#"add.f32 %r680, %r678, %r679;
"#) ++
  r#"$L__tmp24:
"# ++
  (if vector then r#"div.full.f32 %r658, %r657, %r334;
neg.f32 %r659, %r590;
fma.rn.f32 %r660, %r659, %r658, %r622;
neg.f32 %r661, %r591;
fma.rn.f32 %r662, %r661, %r658, %r623;
neg.f32 %r663, %r592;
fma.rn.f32 %r664, %r663, %r658, %r624;
neg.f32 %r665, %r593;
fma.rn.f32 %r666, %r665, %r658, %r625;
neg.f32 %r667, %r594;
fma.rn.f32 %r668, %r667, %r658, %r626;
neg.f32 %r669, %r595;
fma.rn.f32 %r670, %r669, %r658, %r627;
neg.f32 %r671, %r596;
fma.rn.f32 %r672, %r671, %r658, %r628;
neg.f32 %r673, %r597;
fma.rn.f32 %r674, %r673, %r658, %r629;
neg.f32 %r675, %r598;
fma.rn.f32 %r676, %r675, %r658, %r630;
neg.f32 %r677, %r599;
fma.rn.f32 %r678, %r677, %r658, %r631;
neg.f32 %r679, %r600;
fma.rn.f32 %r680, %r679, %r658, %r632;
neg.f32 %r681, %r601;
fma.rn.f32 %r682, %r681, %r658, %r633;
neg.f32 %r683, %r602;
fma.rn.f32 %r684, %r683, %r658, %r634;
neg.f32 %r685, %r603;
fma.rn.f32 %r686, %r685, %r658, %r635;
neg.f32 %r687, %r604;
fma.rn.f32 %r688, %r687, %r658, %r636;
neg.f32 %r689, %r605;
fma.rn.f32 %r690, %r689, %r658, %r637;
mul.f32 %r691, %r566, %r660;
mul.f32 %r692, %r566, %r662;
mul.f32 %r693, %r566, %r664;
mul.f32 %r694, %r566, %r666;
mul.f32 %r695, %r566, %r668;
mul.f32 %r696, %r566, %r670;
mul.f32 %r697, %r566, %r672;
mul.f32 %r698, %r566, %r674;
mul.f32 %r699, %r566, %r676;
mul.f32 %r700, %r566, %r678;
mul.f32 %r701, %r566, %r680;
mul.f32 %r702, %r566, %r682;
mul.f32 %r703, %r566, %r684;
mul.f32 %r704, %r566, %r686;
mul.f32 %r705, %r566, %r688;
mul.f32 %r706, %r566, %r690;
cvt.rn.bf16.f32 %rs193, %r691;
cvt.rn.bf16.f32 %rs194, %r692;
cvt.rn.bf16.f32 %rs195, %r693;
cvt.rn.bf16.f32 %rs196, %r694;
cvt.rn.bf16.f32 %rs197, %r695;
cvt.rn.bf16.f32 %rs198, %r696;
cvt.rn.bf16.f32 %rs199, %r697;
cvt.rn.bf16.f32 %rs200, %r698;
cvt.rn.bf16.f32 %rs201, %r699;
cvt.rn.bf16.f32 %rs202, %r700;
cvt.rn.bf16.f32 %rs203, %r701;
cvt.rn.bf16.f32 %rs204, %r702;
cvt.rn.bf16.f32 %rs205, %r703;
cvt.rn.bf16.f32 %rs206, %r704;
cvt.rn.bf16.f32 %rs207, %r705;
cvt.rn.bf16.f32 %rs208, %r706;
"# else r#"div.full.f32 %r681, %r680, %r359;
neg.f32 %r682, %r613;
fma.rn.f32 %r683, %r682, %r681, %r645;
neg.f32 %r684, %r614;
fma.rn.f32 %r685, %r684, %r681, %r646;
neg.f32 %r686, %r615;
fma.rn.f32 %r687, %r686, %r681, %r647;
neg.f32 %r688, %r616;
fma.rn.f32 %r689, %r688, %r681, %r648;
neg.f32 %r690, %r617;
fma.rn.f32 %r691, %r690, %r681, %r649;
neg.f32 %r692, %r618;
fma.rn.f32 %r693, %r692, %r681, %r650;
neg.f32 %r694, %r619;
fma.rn.f32 %r695, %r694, %r681, %r651;
neg.f32 %r696, %r620;
fma.rn.f32 %r697, %r696, %r681, %r652;
neg.f32 %r698, %r621;
fma.rn.f32 %r699, %r698, %r681, %r653;
neg.f32 %r700, %r622;
fma.rn.f32 %r701, %r700, %r681, %r654;
neg.f32 %r702, %r623;
fma.rn.f32 %r703, %r702, %r681, %r655;
neg.f32 %r704, %r624;
fma.rn.f32 %r705, %r704, %r681, %r656;
neg.f32 %r706, %r625;
fma.rn.f32 %r707, %r706, %r681, %r657;
neg.f32 %r708, %r626;
fma.rn.f32 %r709, %r708, %r681, %r658;
neg.f32 %r710, %r627;
fma.rn.f32 %r711, %r710, %r681, %r659;
neg.f32 %r712, %r628;
fma.rn.f32 %r713, %r712, %r681, %r660;
mul.f32 %r714, %r589, %r683;
mul.f32 %r715, %r589, %r685;
mul.f32 %r716, %r589, %r687;
mul.f32 %r717, %r589, %r689;
mul.f32 %r718, %r589, %r691;
mul.f32 %r719, %r589, %r693;
mul.f32 %r720, %r589, %r695;
mul.f32 %r721, %r589, %r697;
mul.f32 %r722, %r589, %r699;
mul.f32 %r723, %r589, %r701;
mul.f32 %r724, %r589, %r703;
mul.f32 %r725, %r589, %r705;
mul.f32 %r726, %r589, %r707;
mul.f32 %r727, %r589, %r709;
mul.f32 %r728, %r589, %r711;
mul.f32 %r729, %r589, %r713;
cvt.rn.bf16.f32 %rs210, %r714;
cvt.rn.bf16.f32 %rs211, %r715;
cvt.rn.bf16.f32 %rs212, %r716;
cvt.rn.bf16.f32 %rs213, %r717;
cvt.rn.bf16.f32 %rs214, %r718;
cvt.rn.bf16.f32 %rs215, %r719;
cvt.rn.bf16.f32 %rs216, %r720;
cvt.rn.bf16.f32 %rs217, %r721;
cvt.rn.bf16.f32 %rs218, %r722;
cvt.rn.bf16.f32 %rs219, %r723;
cvt.rn.bf16.f32 %rs220, %r724;
cvt.rn.bf16.f32 %rs221, %r725;
cvt.rn.bf16.f32 %rs222, %r726;
cvt.rn.bf16.f32 %rs223, %r727;
cvt.rn.bf16.f32 %rs224, %r728;
cvt.rn.bf16.f32 %rs225, %r729;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"mov.b32 %r707, {%rs205, %rs207};
mov.b32 %r708, {%rs201, %rs203};
mov.b32 %r709, {%rs197, %rs199};
mov.b32 %r710, {%rs193, %rs195};
st.shared.v4.b32 [%r465], {%r710, %r709, %r708, %r707};
mov.b32 %r711, {%rs206, %rs208};
mov.b32 %r712, {%rs202, %rs204};
mov.b32 %r713, {%rs198, %rs200};
mov.b32 %r714, {%rs194, %rs196};
st.shared.v4.b32 [%r471+2048], {%r714, %r713, %r712, %r711};
"# else r#"mov.b32 %r730, {%rs222, %rs224};
mov.b32 %r731, {%rs218, %rs220};
mov.b32 %r732, {%rs214, %rs216};
mov.b32 %r733, {%rs210, %rs212};
st.shared.v4.b32 [%r489], {%r733, %r732, %r731, %r730};
mov.b32 %r734, {%rs223, %rs225};
mov.b32 %r735, {%rs219, %rs221};
mov.b32 %r736, {%rs215, %rs217};
mov.b32 %r737, {%rs211, %rs213};
st.shared.v4.b32 [%r495+2048], {%r737, %r736, %r735, %r734};
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r715, %r716, %r717, %r718}, [%r482];
mov.b32 {%rs209, %rs210}, %r715;
mov.b32 {%rs211, %rs212}, %r716;
mov.b32 {%rs213, %rs214}, %r717;
mov.b32 {%rs215, %rs216}, %r718;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r719, %r720, %r721, %r722}, [%r482+512];
mov.b32 {%rs217, %rs218}, %r719;
mov.b32 {%rs219, %rs220}, %r720;
mov.b32 {%rs221, %rs222}, %r721;
mov.b32 {%rs223, %rs224}, %r722;
cvt.f32.bf16 %r723, %rs209;
cvt.f32.bf16 %r724, %rs210;
cvt.f32.bf16 %r725, %rs211;
cvt.f32.bf16 %r726, %rs212;
cvt.f32.bf16 %r727, %rs213;
cvt.f32.bf16 %r728, %rs214;
cvt.f32.bf16 %r729, %rs215;
cvt.f32.bf16 %r730, %rs216;
cvt.f32.bf16 %r731, %rs217;
cvt.f32.bf16 %r732, %rs218;
cvt.f32.bf16 %r733, %rs219;
cvt.f32.bf16 %r734, %rs220;
cvt.f32.bf16 %r735, %rs221;
cvt.f32.bf16 %r736, %rs222;
cvt.f32.bf16 %r737, %rs223;
cvt.f32.bf16 %r738, %rs224;
cvt.f32.bf16 %r739, %rs193;
cvt.f32.bf16 %r740, %rs194;
cvt.f32.bf16 %r741, %rs195;
cvt.f32.bf16 %r742, %rs196;
cvt.f32.bf16 %r743, %rs197;
cvt.f32.bf16 %r744, %rs198;
cvt.f32.bf16 %r745, %rs199;
cvt.f32.bf16 %r746, %rs200;
cvt.f32.bf16 %r747, %rs201;
cvt.f32.bf16 %r748, %rs202;
cvt.f32.bf16 %r749, %rs203;
cvt.f32.bf16 %r750, %rs204;
cvt.f32.bf16 %r751, %rs205;
cvt.f32.bf16 %r752, %rs206;
cvt.f32.bf16 %r753, %rs207;
cvt.f32.bf16 %r754, %rs208;
div.full.f32 %r91, %r723, %r18;
div.full.f32 %r92, %r724, %r18;
div.full.f32 %r93, %r725, %r18;
div.full.f32 %r94, %r726, %r18;
div.full.f32 %r95, %r727, %r18;
div.full.f32 %r96, %r728, %r18;
div.full.f32 %r97, %r729, %r18;
div.full.f32 %r98, %r730, %r18;
div.full.f32 %r99, %r731, %r18;
div.full.f32 %r100, %r732, %r18;
div.full.f32 %r101, %r733, %r18;
div.full.f32 %r102, %r734, %r18;
div.full.f32 %r103, %r735, %r18;
div.full.f32 %r104, %r736, %r18;
div.full.f32 %r105, %r737, %r18;
div.full.f32 %r106, %r738, %r18;
div.full.f32 %r107, %r739, %r18;
div.full.f32 %r108, %r740, %r18;
div.full.f32 %r109, %r741, %r18;
div.full.f32 %r110, %r742, %r18;
div.full.f32 %r111, %r743, %r18;
div.full.f32 %r112, %r744, %r18;
div.full.f32 %r113, %r745, %r18;
div.full.f32 %r114, %r746, %r18;
div.full.f32 %r115, %r747, %r18;
div.full.f32 %r116, %r748, %r18;
div.full.f32 %r117, %r749, %r18;
div.full.f32 %r118, %r750, %r18;
div.full.f32 %r119, %r751, %r18;
div.full.f32 %r120, %r752, %r18;
div.full.f32 %r121, %r753, %r18;
div.full.f32 %r122, %r754, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs17, %r92, %r91;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r94, %r93;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r96, %r95;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r98, %r97;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r114, %r113;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r116, %r115;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r118, %r117;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r120, %r119;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r122, %r121;
or.b32 %r755, %r527, %r237;
or.b32 %r756, %r527, %r238;
cvt.s64.s32 %rd60, %r755;
add.s64 %rd16, %rd48, %rd60;
mov.b32 %r123, {%rs25, %rs26};
mov.b32 %r124, {%rs27, %rs28};
mov.b32 %r125, {%rs29, %rs30};
mov.b32 %r126, {%rs31, %rs32};
@%p1 st.global.v4.b32 [ %rd16 + 0 ], { %r123, %r124, %r125, %r126 };
@EXTENT_14@
cvt.s64.s32 %rd61, %r757;
add.s64 %rd62, %rd33, %rd61;
add.s64 %rd17, %rd62, %rd52;
mov.b32 %r127, {%rs17, %rs18};
mov.b32 %r128, {%rs19, %rs20};
mov.b32 %r129, {%rs21, %rs22};
mov.b32 %r130, {%rs23, %rs24};
@%p1 st.global.v4.b32 [ %rd17 + 0 ], { %r127, %r128, %r129, %r130 };
and.b32 %r758, %r230, 7;
shl.b32 %r759, %r758, 4;
mad.wide.u32 %rd63, %r589, 2, %rd28;
mad.wide.u32 %rd64, %r588, 2, %rd28;
shl.b32 %r760, %r227, 7;
mul.wide.s32 %rd65, %r760, 2;
add.s64 %rd66, %rd63, %rd65;
add.s64 %rd67, %rd64, %rd65;
mul.wide.u32 %rd68, %r759, 2;
add.s64 %rd18, %rd66, %rd68;
add.s64 %rd19, %rd18, 16;
add.s64 %rd20, %rd67, %rd68;
add.s64 %rd21, %rd20, 16;
"# else r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r738, %r739, %r740, %r741}, [%r505];
mov.b32 {%rs226, %rs227}, %r738;
mov.b32 {%rs228, %rs229}, %r739;
mov.b32 {%rs230, %rs231}, %r740;
mov.b32 {%rs232, %rs233}, %r741;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r742, %r743, %r744, %r745}, [%r505+512];
mov.b32 {%rs234, %rs235}, %r742;
mov.b32 {%rs236, %rs237}, %r743;
mov.b32 {%rs238, %rs239}, %r744;
mov.b32 {%rs240, %rs241}, %r745;
cvt.f32.bf16 %r746, %rs226;
cvt.f32.bf16 %r747, %rs227;
cvt.f32.bf16 %r748, %rs228;
cvt.f32.bf16 %r749, %rs229;
cvt.f32.bf16 %r750, %rs230;
cvt.f32.bf16 %r751, %rs231;
cvt.f32.bf16 %r752, %rs232;
cvt.f32.bf16 %r753, %rs233;
cvt.f32.bf16 %r754, %rs234;
cvt.f32.bf16 %r755, %rs235;
cvt.f32.bf16 %r756, %rs236;
cvt.f32.bf16 %r757, %rs237;
cvt.f32.bf16 %r758, %rs238;
cvt.f32.bf16 %r759, %rs239;
cvt.f32.bf16 %r760, %rs240;
cvt.f32.bf16 %r761, %rs241;
cvt.f32.bf16 %r762, %rs210;
cvt.f32.bf16 %r763, %rs211;
cvt.f32.bf16 %r764, %rs212;
cvt.f32.bf16 %r765, %rs213;
cvt.f32.bf16 %r766, %rs214;
cvt.f32.bf16 %r767, %rs215;
cvt.f32.bf16 %r768, %rs216;
cvt.f32.bf16 %r769, %rs217;
cvt.f32.bf16 %r770, %rs218;
cvt.f32.bf16 %r771, %rs219;
cvt.f32.bf16 %r772, %rs220;
cvt.f32.bf16 %r773, %rs221;
cvt.f32.bf16 %r774, %rs222;
cvt.f32.bf16 %r775, %rs223;
cvt.f32.bf16 %r776, %rs224;
cvt.f32.bf16 %r777, %rs225;
div.full.f32 %r83, %r746, %r10;
div.full.f32 %r84, %r747, %r10;
div.full.f32 %r85, %r748, %r10;
div.full.f32 %r86, %r749, %r10;
div.full.f32 %r87, %r750, %r10;
div.full.f32 %r88, %r751, %r10;
div.full.f32 %r89, %r752, %r10;
div.full.f32 %r90, %r753, %r10;
div.full.f32 %r91, %r754, %r10;
div.full.f32 %r92, %r755, %r10;
div.full.f32 %r93, %r756, %r10;
div.full.f32 %r94, %r757, %r10;
div.full.f32 %r95, %r758, %r10;
div.full.f32 %r96, %r759, %r10;
div.full.f32 %r97, %r760, %r10;
div.full.f32 %r98, %r761, %r10;
div.full.f32 %r99, %r762, %r10;
div.full.f32 %r100, %r763, %r10;
div.full.f32 %r101, %r764, %r10;
div.full.f32 %r102, %r765, %r10;
div.full.f32 %r103, %r766, %r10;
div.full.f32 %r104, %r767, %r10;
div.full.f32 %r105, %r768, %r10;
div.full.f32 %r106, %r769, %r10;
div.full.f32 %r107, %r770, %r10;
div.full.f32 %r108, %r771, %r10;
div.full.f32 %r109, %r772, %r10;
div.full.f32 %r110, %r773, %r10;
div.full.f32 %r111, %r774, %r10;
div.full.f32 %r112, %r775, %r10;
div.full.f32 %r113, %r776, %r10;
div.full.f32 %r114, %r777, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs34, %r84, %r83;
cvt.rn.satfinite.e4m3x2.f32 %rs35, %r86, %r85;
cvt.rn.satfinite.e4m3x2.f32 %rs36, %r88, %r87;
cvt.rn.satfinite.e4m3x2.f32 %rs37, %r90, %r89;
cvt.rn.satfinite.e4m3x2.f32 %rs38, %r92, %r91;
cvt.rn.satfinite.e4m3x2.f32 %rs39, %r94, %r93;
cvt.rn.satfinite.e4m3x2.f32 %rs40, %r96, %r95;
cvt.rn.satfinite.e4m3x2.f32 %rs41, %r98, %r97;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs49, %r114, %r113;
or.b32 %r778, %r550, %r233;
or.b32 %r779, %r550, %r234;
cvt.s64.s32 %rd77, %r778;
add.s64 %rd30, %rd65, %rd77;
mov.b32 %r115, {%rs42, %rs43};
mov.b32 %r116, {%rs44, %rs45};
mov.b32 %r117, {%rs46, %rs47};
mov.b32 %r118, {%rs48, %rs49};
@%p1 st.global.v4.b32 [ %rd30 + 0 ], { %r115, %r116, %r117, %r118 };
@EXTENT_7@
cvt.s64.s32 %rd78, %r780;
add.s64 %rd79, %rd47, %rd78;
add.s64 %rd31, %rd79, %rd69;
mov.b32 %r119, {%rs34, %rs35};
mov.b32 %r120, {%rs36, %rs37};
mov.b32 %r121, {%rs38, %rs39};
mov.b32 %r122, {%rs40, %rs41};
@%p1 st.global.v4.b32 [ %rd31 + 0 ], { %r119, %r120, %r121, %r122 };
mad.wide.u32 %rd80, %r612, 2, %rd42;
mad.wide.u32 %rd81, %r611, 2, %rd42;
shl.b32 %r781, %r219, 7;
mul.wide.s32 %rd82, %r781, 2;
add.s64 %rd83, %rd80, %rd82;
add.s64 %rd84, %rd81, %rd82;
mul.wide.u32 %rd85, %r271, 2;
add.s64 %rd32, %rd83, %rd85;
add.s64 %rd33, %rd32, 16;
add.s64 %rd34, %rd84, %rd85;
add.s64 %rd35, %rd34, 16;
mov.u32 %r123, %r5;
mov.u32 %r124, %r5;
mov.u32 %r125, %r5;
mov.u32 %r126, %r5;
@%p1 ld.global.v4.b32 { %r123, %r124, %r125, %r126 }, [ %rd32 + 0 ];
mov.b32 {%rs242, %rs243}, %r123;
mov.b32 {%rs244, %rs245}, %r124;
mov.b32 {%rs246, %rs247}, %r125;
mov.b32 {%rs248, %rs249}, %r126;
mov.u32 %r127, %r5;
mov.u32 %r128, %r5;
mov.u32 %r129, %r5;
mov.u32 %r130, %r5;
@%p1 ld.global.v4.b32 { %r127, %r128, %r129, %r130 }, [ %rd33 + 0 ];
mov.b32 {%rs250, %rs251}, %r127;
mov.b32 {%rs252, %rs253}, %r128;
mov.b32 {%rs254, %rs255}, %r129;
mov.b32 {%rs256, %rs257}, %r130;
"#) ++
  r#"mov.u32 %r131, %r5;
mov.u32 %r132, %r5;
mov.u32 %r133, %r5;
mov.u32 %r134, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd18 + 0 ];
mov.b32 {%rs225, %rs226}, %r131;
mov.b32 {%rs227, %rs228}, %r132;
mov.b32 {%rs229, %rs230}, %r133;
mov.b32 {%rs231, %rs232}, %r134;
"# else r#"@%p1 ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd34 + 0 ];
mov.b32 {%rs258, %rs259}, %r131;
mov.b32 {%rs260, %rs261}, %r132;
mov.b32 {%rs262, %rs263}, %r133;
mov.b32 {%rs264, %rs265}, %r134;
"#) ++
  r#"mov.u32 %r135, %r5;
mov.u32 %r136, %r5;
mov.u32 %r137, %r5;
mov.u32 %r138, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd19 + 0 ];
mov.b32 {%rs233, %rs234}, %r135;
mov.b32 {%rs235, %rs236}, %r136;
mov.b32 {%rs237, %rs238}, %r137;
mov.b32 {%rs239, %rs240}, %r138;
mov.u32 %r139, %r5;
mov.u32 %r140, %r5;
mov.u32 %r141, %r5;
mov.u32 %r142, %r5;
@%p1 ld.global.v4.b32 { %r139, %r140, %r141, %r142 }, [ %rd20 + 0 ];
mov.b32 {%rs241, %rs242}, %r139;
mov.b32 {%rs243, %rs244}, %r140;
mov.b32 {%rs245, %rs246}, %r141;
mov.b32 {%rs247, %rs248}, %r142;
mov.u32 %r143, %r5;
mov.u32 %r144, %r5;
mov.u32 %r145, %r5;
mov.u32 %r146, %r5;
@%p1 ld.global.v4.b32 { %r143, %r144, %r145, %r146 }, [ %rd21 + 0 ];
mov.b32 {%rs249, %rs250}, %r143;
mov.b32 {%rs251, %rs252}, %r144;
mov.b32 {%rs253, %rs254}, %r145;
mov.b32 {%rs255, %rs256}, %r146;
"# else r#"@%p1 ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd35 + 0 ];
mov.b32 {%rs266, %rs267}, %r135;
mov.b32 {%rs268, %rs269}, %r136;
mov.b32 {%rs270, %rs271}, %r137;
mov.b32 {%rs272, %rs273}, %r138;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"shl.b32 %r761, %r230, 3;
and.b32 %r762, %r761, 896;
bfe.s32 %r763, %r230, 3, 1;
and.b32 %r764, %r763, 1040;
or.b32 %r765, %r759, %r762;
xor.b32 %r766, %r765, %r764;
add.s32 %r767, %r464, %r766;
mov.b32 %r768, {%rs225, %rs227};
mov.b32 %r769, {%rs229, %rs231};
mov.b32 %r770, {%rs233, %rs235};
mov.b32 %r771, {%rs237, %rs239};
st.shared.v4.b32 [%r767], {%r768, %r769, %r770, %r771};
xor.b32 %r772, %r766, 64;
add.s32 %r773, %r464, %r772;
mov.b32 {_, %rs257}, %r135;
mov.b32 {_, %rs258}, %r137;
mov.b32 {_, %rs259}, %r131;
mov.b32 {_, %rs260}, %r133;
mov.b32 %r774, {%rs260, %rs232};
mov.b32 %r775, {%rs259, %rs228};
mov.b32 %r776, {%rs258, %rs240};
mov.b32 %r777, {%rs257, %rs236};
st.shared.v4.b32 [%r773+4096], {%r775, %r774, %r777, %r776};
xor.b32 %r778, %r766, 32;
add.s32 %r779, %r464, %r778;
mov.b32 %r780, {%rs241, %rs243};
mov.b32 %r781, {%rs245, %rs247};
mov.b32 %r782, {%rs249, %rs251};
mov.b32 %r783, {%rs253, %rs255};
st.shared.v4.b32 [%r779+2048], {%r780, %r781, %r782, %r783};
xor.b32 %r784, %r766, 96;
add.s32 %r785, %r464, %r784;
mov.b32 {_, %rs261}, %r143;
mov.b32 {_, %rs262}, %r145;
mov.b32 {_, %rs263}, %r139;
mov.b32 {_, %rs264}, %r141;
mov.b32 %r786, {%rs264, %rs248};
mov.b32 %r787, {%rs263, %rs244};
mov.b32 %r788, {%rs262, %rs256};
mov.b32 %r789, {%rs261, %rs252};
st.shared.v4.b32 [%r785+6144], {%r787, %r786, %r789, %r788};
"# else r#"shl.b32 %r782, %r222, 3;
and.b32 %r783, %r782, 896;
bfe.s32 %r784, %r222, 3, 1;
and.b32 %r785, %r784, 1040;
or.b32 %r786, %r271, %r783;
xor.b32 %r787, %r786, %r785;
add.s32 %r788, %r262, %r787;
mov.b32 %r789, {%rs242, %rs244};
mov.b32 %r790, {%rs246, %rs248};
mov.b32 %r791, {%rs250, %rs252};
mov.b32 %r792, {%rs254, %rs256};
st.shared.v4.b32 [%r788], {%r789, %r790, %r791, %r792};
xor.b32 %r793, %r787, 64;
add.s32 %r794, %r262, %r793;
mov.b32 {_, %rs274}, %r127;
mov.b32 {_, %rs275}, %r129;
mov.b32 {_, %rs276}, %r123;
mov.b32 {_, %rs277}, %r125;
mov.b32 %r795, {%rs277, %rs249};
mov.b32 %r796, {%rs276, %rs245};
mov.b32 %r797, {%rs275, %rs257};
mov.b32 %r798, {%rs274, %rs253};
st.shared.v4.b32 [%r794+4096], {%r796, %r795, %r798, %r797};
xor.b32 %r799, %r787, 32;
add.s32 %r800, %r262, %r799;
mov.b32 %r801, {%rs258, %rs260};
mov.b32 %r802, {%rs262, %rs264};
mov.b32 %r803, {%rs266, %rs268};
mov.b32 %r804, {%rs270, %rs272};
st.shared.v4.b32 [%r800+2048], {%r801, %r802, %r803, %r804};
xor.b32 %r805, %r787, 96;
add.s32 %r806, %r262, %r805;
mov.b32 {_, %rs278}, %r135;
mov.b32 {_, %rs279}, %r137;
mov.b32 {_, %rs280}, %r131;
mov.b32 {_, %rs281}, %r133;
mov.b32 %r807, {%rs281, %rs265};
mov.b32 %r808, {%rs280, %rs261};
mov.b32 %r809, {%rs279, %rs273};
mov.b32 %r810, {%rs278, %rs269};
st.shared.v4.b32 [%r806+6144], {%r808, %r807, %r810, %r809};
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"shl.b32 %r790, %r758, 10;
shr.u32 %r791, %r479, 1;
xor.b32 %r792, %r478, %r791;
or.b32 %r793, %r792, %r790;
add.s32 %r794, %r464, %r793;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r795, %r796, %r797, %r798}, [%r794];
mov.b32 {%rs265, %rs266}, %r795;
mov.b32 {%rs267, %rs268}, %r796;
mov.b32 {%rs269, %rs270}, %r797;
mov.b32 {%rs271, %rs272}, %r798;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r799, %r800, %r801, %r802}, [%r794+512];
mov.b32 {%rs273, %rs274}, %r799;
mov.b32 {%rs275, %rs276}, %r800;
mov.b32 {%rs277, %rs278}, %r801;
mov.b32 {%rs279, %rs280}, %r802;
xor.b32 %r803, %r793, 64;
add.s32 %r804, %r464, %r803;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r805, %r806, %r807, %r808}, [%r804];
mov.b32 {%rs281, %rs282}, %r805;
mov.b32 {%rs283, %rs284}, %r806;
mov.b32 {%rs285, %rs286}, %r807;
mov.b32 {%rs287, %rs288}, %r808;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r809, %r810, %r811, %r812}, [%r804+512];
mov.b32 {%rs289, %rs290}, %r809;
mov.b32 {%rs291, %rs292}, %r810;
mov.b32 {%rs293, %rs294}, %r811;
mov.b32 {%rs295, %rs296}, %r812;
cvt.f32.bf16 %r813, %rs265;
cvt.f32.bf16 %r814, %rs266;
cvt.f32.bf16 %r815, %rs267;
cvt.f32.bf16 %r816, %rs268;
cvt.f32.bf16 %r817, %rs269;
cvt.f32.bf16 %r818, %rs270;
cvt.f32.bf16 %r819, %rs271;
cvt.f32.bf16 %r820, %rs272;
cvt.f32.bf16 %r821, %rs273;
cvt.f32.bf16 %r822, %rs274;
cvt.f32.bf16 %r823, %rs275;
cvt.f32.bf16 %r824, %rs276;
cvt.f32.bf16 %r825, %rs277;
cvt.f32.bf16 %r826, %rs278;
cvt.f32.bf16 %r827, %rs279;
cvt.f32.bf16 %r828, %rs280;
cvt.f32.bf16 %r829, %rs281;
cvt.f32.bf16 %r830, %rs282;
cvt.f32.bf16 %r831, %rs283;
cvt.f32.bf16 %r832, %rs284;
cvt.f32.bf16 %r833, %rs285;
cvt.f32.bf16 %r834, %rs286;
cvt.f32.bf16 %r835, %rs287;
cvt.f32.bf16 %r836, %rs288;
cvt.f32.bf16 %r837, %rs289;
cvt.f32.bf16 %r838, %rs290;
cvt.f32.bf16 %r839, %rs291;
cvt.f32.bf16 %r840, %rs292;
cvt.f32.bf16 %r841, %rs293;
cvt.f32.bf16 %r842, %rs294;
cvt.f32.bf16 %r843, %rs295;
cvt.f32.bf16 %r844, %rs296;
cvt.f32.bf16 %r845, %rs225;
cvt.f32.bf16 %r846, %rs226;
cvt.f32.bf16 %r847, %rs227;
cvt.f32.bf16 %r848, %rs228;
cvt.f32.bf16 %r849, %rs229;
cvt.f32.bf16 %r850, %rs230;
cvt.f32.bf16 %r851, %rs231;
cvt.f32.bf16 %r852, %rs232;
cvt.f32.bf16 %r853, %rs233;
cvt.f32.bf16 %r854, %rs234;
cvt.f32.bf16 %r855, %rs235;
cvt.f32.bf16 %r856, %rs236;
cvt.f32.bf16 %r857, %rs237;
cvt.f32.bf16 %r858, %rs238;
cvt.f32.bf16 %r859, %rs239;
cvt.f32.bf16 %r860, %rs240;
cvt.f32.bf16 %r861, %rs241;
cvt.f32.bf16 %r862, %rs242;
cvt.f32.bf16 %r863, %rs243;
cvt.f32.bf16 %r864, %rs244;
cvt.f32.bf16 %r865, %rs245;
cvt.f32.bf16 %r866, %rs246;
cvt.f32.bf16 %r867, %rs247;
cvt.f32.bf16 %r868, %rs248;
cvt.f32.bf16 %r869, %rs249;
cvt.f32.bf16 %r870, %rs250;
cvt.f32.bf16 %r871, %rs251;
cvt.f32.bf16 %r872, %rs252;
cvt.f32.bf16 %r873, %rs253;
cvt.f32.bf16 %r874, %rs254;
cvt.f32.bf16 %r875, %rs255;
cvt.f32.bf16 %r876, %rs256;
div.full.f32 %r147, %r813, %r18;
div.full.f32 %r148, %r814, %r18;
div.full.f32 %r149, %r815, %r18;
div.full.f32 %r150, %r816, %r18;
div.full.f32 %r151, %r817, %r18;
div.full.f32 %r152, %r818, %r18;
div.full.f32 %r153, %r819, %r18;
div.full.f32 %r154, %r820, %r18;
div.full.f32 %r155, %r821, %r18;
div.full.f32 %r156, %r822, %r18;
div.full.f32 %r157, %r823, %r18;
div.full.f32 %r158, %r824, %r18;
div.full.f32 %r159, %r825, %r18;
div.full.f32 %r160, %r826, %r18;
div.full.f32 %r161, %r827, %r18;
div.full.f32 %r162, %r828, %r18;
div.full.f32 %r163, %r829, %r18;
div.full.f32 %r164, %r830, %r18;
div.full.f32 %r165, %r831, %r18;
div.full.f32 %r166, %r832, %r18;
div.full.f32 %r167, %r833, %r18;
div.full.f32 %r168, %r834, %r18;
div.full.f32 %r169, %r835, %r18;
div.full.f32 %r170, %r836, %r18;
div.full.f32 %r171, %r837, %r18;
div.full.f32 %r172, %r838, %r18;
div.full.f32 %r173, %r839, %r18;
div.full.f32 %r174, %r840, %r18;
div.full.f32 %r175, %r841, %r18;
div.full.f32 %r176, %r842, %r18;
div.full.f32 %r177, %r843, %r18;
div.full.f32 %r178, %r844, %r18;
div.full.f32 %r179, %r845, %r18;
div.full.f32 %r180, %r846, %r18;
div.full.f32 %r181, %r847, %r18;
div.full.f32 %r182, %r848, %r18;
div.full.f32 %r183, %r849, %r18;
div.full.f32 %r184, %r850, %r18;
div.full.f32 %r185, %r851, %r18;
div.full.f32 %r186, %r852, %r18;
div.full.f32 %r187, %r853, %r18;
div.full.f32 %r188, %r854, %r18;
div.full.f32 %r189, %r855, %r18;
div.full.f32 %r190, %r856, %r18;
div.full.f32 %r191, %r857, %r18;
div.full.f32 %r192, %r858, %r18;
div.full.f32 %r193, %r859, %r18;
div.full.f32 %r194, %r860, %r18;
div.full.f32 %r195, %r861, %r18;
div.full.f32 %r196, %r862, %r18;
div.full.f32 %r197, %r863, %r18;
div.full.f32 %r198, %r864, %r18;
div.full.f32 %r199, %r865, %r18;
div.full.f32 %r200, %r866, %r18;
div.full.f32 %r201, %r867, %r18;
div.full.f32 %r202, %r868, %r18;
div.full.f32 %r203, %r869, %r18;
div.full.f32 %r204, %r870, %r18;
div.full.f32 %r205, %r871, %r18;
div.full.f32 %r206, %r872, %r18;
div.full.f32 %r207, %r873, %r18;
div.full.f32 %r208, %r874, %r18;
div.full.f32 %r209, %r875, %r18;
div.full.f32 %r210, %r876, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs33, %r148, %r147;
cvt.rn.satfinite.e4m3x2.f32 %rs34, %r150, %r149;
cvt.rn.satfinite.e4m3x2.f32 %rs35, %r152, %r151;
cvt.rn.satfinite.e4m3x2.f32 %rs36, %r154, %r153;
cvt.rn.satfinite.e4m3x2.f32 %rs37, %r156, %r155;
cvt.rn.satfinite.e4m3x2.f32 %rs38, %r158, %r157;
cvt.rn.satfinite.e4m3x2.f32 %rs39, %r160, %r159;
cvt.rn.satfinite.e4m3x2.f32 %rs40, %r162, %r161;
cvt.rn.satfinite.e4m3x2.f32 %rs41, %r164, %r163;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r166, %r165;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r168, %r167;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r170, %r169;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r172, %r171;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r174, %r173;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r176, %r175;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r178, %r177;
cvt.rn.satfinite.e4m3x2.f32 %rs49, %r180, %r179;
cvt.rn.satfinite.e4m3x2.f32 %rs50, %r182, %r181;
cvt.rn.satfinite.e4m3x2.f32 %rs51, %r184, %r183;
cvt.rn.satfinite.e4m3x2.f32 %rs52, %r186, %r185;
cvt.rn.satfinite.e4m3x2.f32 %rs53, %r188, %r187;
cvt.rn.satfinite.e4m3x2.f32 %rs54, %r190, %r189;
cvt.rn.satfinite.e4m3x2.f32 %rs55, %r192, %r191;
cvt.rn.satfinite.e4m3x2.f32 %rs56, %r194, %r193;
cvt.rn.satfinite.e4m3x2.f32 %rs57, %r196, %r195;
cvt.rn.satfinite.e4m3x2.f32 %rs58, %r198, %r197;
cvt.rn.satfinite.e4m3x2.f32 %rs59, %r200, %r199;
cvt.rn.satfinite.e4m3x2.f32 %rs60, %r202, %r201;
cvt.rn.satfinite.e4m3x2.f32 %rs61, %r204, %r203;
cvt.rn.satfinite.e4m3x2.f32 %rs62, %r206, %r205;
cvt.rn.satfinite.e4m3x2.f32 %rs63, %r208, %r207;
cvt.rn.satfinite.e4m3x2.f32 %rs64, %r210, %r209;
add.s32 %r877, %r760, 768;
or.b32 %r878, %r877, %r759;
or.b32 %r879, %r877, %r238;
cvt.s64.s32 %rd69, %r878;
add.s64 %rd22, %rd59, %rd69;
add.s64 %rd23, %rd57, %rd69;
mov.b32 %r211, {%rs49, %rs50};
mov.b32 %r212, {%rs51, %rs52};
mov.b32 %r213, {%rs53, %rs54};
mov.b32 %r214, {%rs55, %rs56};
@%p1 st.global.v4.b32 [ %rd22 + 0 ], { %r211, %r212, %r213, %r214 };
mov.b32 %r215, {%rs57, %rs58};
mov.b32 %r216, {%rs59, %rs60};
mov.b32 %r217, {%rs61, %rs62};
mov.b32 %r218, {%rs63, %rs64};
@%p1 st.global.v4.b32 [ %rd23 + 0 ], { %r215, %r216, %r217, %r218 };
@EXTENT_15@
cvt.s64.s32 %rd70, %r880;
add.s64 %rd71, %rd33, %rd70;
@EXTENT_16@
add.s64 %rd24, %rd71, %rd52;
@EXTENT_17@
mov.b32 %r219, {%rs33, %rs34};
mov.b32 %r220, {%rs35, %rs36};
mov.b32 %r221, {%rs37, %rs38};
mov.b32 %r222, {%rs39, %rs40};
@%p1 st.global.v4.b32 [ %rd24 + 0 ], { %r219, %r220, %r221, %r222 };
mov.b32 %r223, {%rs41, %rs42};
mov.b32 %r224, {%rs43, %rs44};
mov.b32 %r225, {%rs45, %rs46};
mov.b32 %r226, {%rs47, %rs48};
@%p1 st.global.v4.b32 [ %rd25 + 0 ], { %r223, %r224, %r225, %r226 };
"# else r#"shl.b32 %r811, %r270, 10;
shr.u32 %r812, %r230, 1;
xor.b32 %r813, %r502, %r812;
or.b32 %r814, %r813, %r811;
add.s32 %r815, %r262, %r814;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r816, %r817, %r818, %r819}, [%r815];
mov.b32 {%rs282, %rs283}, %r816;
mov.b32 {%rs284, %rs285}, %r817;
mov.b32 {%rs286, %rs287}, %r818;
mov.b32 {%rs288, %rs289}, %r819;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r820, %r821, %r822, %r823}, [%r815+512];
mov.b32 {%rs290, %rs291}, %r820;
mov.b32 {%rs292, %rs293}, %r821;
mov.b32 {%rs294, %rs295}, %r822;
mov.b32 {%rs296, %rs297}, %r823;
xor.b32 %r824, %r814, 64;
add.s32 %r825, %r262, %r824;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r826, %r827, %r828, %r829}, [%r825];
mov.b32 {%rs298, %rs299}, %r826;
mov.b32 {%rs300, %rs301}, %r827;
mov.b32 {%rs302, %rs303}, %r828;
mov.b32 {%rs304, %rs305}, %r829;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r830, %r831, %r832, %r833}, [%r825+512];
mov.b32 {%rs306, %rs307}, %r830;
mov.b32 {%rs308, %rs309}, %r831;
mov.b32 {%rs310, %rs311}, %r832;
mov.b32 {%rs312, %rs313}, %r833;
cvt.f32.bf16 %r834, %rs282;
cvt.f32.bf16 %r835, %rs283;
cvt.f32.bf16 %r836, %rs284;
cvt.f32.bf16 %r837, %rs285;
cvt.f32.bf16 %r838, %rs286;
cvt.f32.bf16 %r839, %rs287;
cvt.f32.bf16 %r840, %rs288;
cvt.f32.bf16 %r841, %rs289;
cvt.f32.bf16 %r842, %rs290;
cvt.f32.bf16 %r843, %rs291;
cvt.f32.bf16 %r844, %rs292;
cvt.f32.bf16 %r845, %rs293;
cvt.f32.bf16 %r846, %rs294;
cvt.f32.bf16 %r847, %rs295;
cvt.f32.bf16 %r848, %rs296;
cvt.f32.bf16 %r849, %rs297;
cvt.f32.bf16 %r850, %rs298;
cvt.f32.bf16 %r851, %rs299;
cvt.f32.bf16 %r852, %rs300;
cvt.f32.bf16 %r853, %rs301;
cvt.f32.bf16 %r854, %rs302;
cvt.f32.bf16 %r855, %rs303;
cvt.f32.bf16 %r856, %rs304;
cvt.f32.bf16 %r857, %rs305;
cvt.f32.bf16 %r858, %rs306;
cvt.f32.bf16 %r859, %rs307;
cvt.f32.bf16 %r860, %rs308;
cvt.f32.bf16 %r861, %rs309;
cvt.f32.bf16 %r862, %rs310;
cvt.f32.bf16 %r863, %rs311;
cvt.f32.bf16 %r864, %rs312;
cvt.f32.bf16 %r865, %rs313;
cvt.f32.bf16 %r866, %rs242;
cvt.f32.bf16 %r867, %rs243;
cvt.f32.bf16 %r868, %rs244;
cvt.f32.bf16 %r869, %rs245;
cvt.f32.bf16 %r870, %rs246;
cvt.f32.bf16 %r871, %rs247;
cvt.f32.bf16 %r872, %rs248;
cvt.f32.bf16 %r873, %rs249;
cvt.f32.bf16 %r874, %rs250;
cvt.f32.bf16 %r875, %rs251;
cvt.f32.bf16 %r876, %rs252;
cvt.f32.bf16 %r877, %rs253;
cvt.f32.bf16 %r878, %rs254;
cvt.f32.bf16 %r879, %rs255;
cvt.f32.bf16 %r880, %rs256;
cvt.f32.bf16 %r881, %rs257;
cvt.f32.bf16 %r882, %rs258;
cvt.f32.bf16 %r883, %rs259;
cvt.f32.bf16 %r884, %rs260;
cvt.f32.bf16 %r885, %rs261;
cvt.f32.bf16 %r886, %rs262;
cvt.f32.bf16 %r887, %rs263;
cvt.f32.bf16 %r888, %rs264;
cvt.f32.bf16 %r889, %rs265;
cvt.f32.bf16 %r890, %rs266;
cvt.f32.bf16 %r891, %rs267;
cvt.f32.bf16 %r892, %rs268;
cvt.f32.bf16 %r893, %rs269;
cvt.f32.bf16 %r894, %rs270;
cvt.f32.bf16 %r895, %rs271;
cvt.f32.bf16 %r896, %rs272;
cvt.f32.bf16 %r897, %rs273;
div.full.f32 %r139, %r834, %r10;
div.full.f32 %r140, %r835, %r10;
div.full.f32 %r141, %r836, %r10;
div.full.f32 %r142, %r837, %r10;
div.full.f32 %r143, %r838, %r10;
div.full.f32 %r144, %r839, %r10;
div.full.f32 %r145, %r840, %r10;
div.full.f32 %r146, %r841, %r10;
div.full.f32 %r147, %r842, %r10;
div.full.f32 %r148, %r843, %r10;
div.full.f32 %r149, %r844, %r10;
div.full.f32 %r150, %r845, %r10;
div.full.f32 %r151, %r846, %r10;
div.full.f32 %r152, %r847, %r10;
div.full.f32 %r153, %r848, %r10;
div.full.f32 %r154, %r849, %r10;
div.full.f32 %r155, %r850, %r10;
div.full.f32 %r156, %r851, %r10;
div.full.f32 %r157, %r852, %r10;
div.full.f32 %r158, %r853, %r10;
div.full.f32 %r159, %r854, %r10;
div.full.f32 %r160, %r855, %r10;
div.full.f32 %r161, %r856, %r10;
div.full.f32 %r162, %r857, %r10;
div.full.f32 %r163, %r858, %r10;
div.full.f32 %r164, %r859, %r10;
div.full.f32 %r165, %r860, %r10;
div.full.f32 %r166, %r861, %r10;
div.full.f32 %r167, %r862, %r10;
div.full.f32 %r168, %r863, %r10;
div.full.f32 %r169, %r864, %r10;
div.full.f32 %r170, %r865, %r10;
div.full.f32 %r171, %r866, %r10;
div.full.f32 %r172, %r867, %r10;
div.full.f32 %r173, %r868, %r10;
div.full.f32 %r174, %r869, %r10;
div.full.f32 %r175, %r870, %r10;
div.full.f32 %r176, %r871, %r10;
div.full.f32 %r177, %r872, %r10;
div.full.f32 %r178, %r873, %r10;
div.full.f32 %r179, %r874, %r10;
div.full.f32 %r180, %r875, %r10;
div.full.f32 %r181, %r876, %r10;
div.full.f32 %r182, %r877, %r10;
div.full.f32 %r183, %r878, %r10;
div.full.f32 %r184, %r879, %r10;
div.full.f32 %r185, %r880, %r10;
div.full.f32 %r186, %r881, %r10;
div.full.f32 %r187, %r882, %r10;
div.full.f32 %r188, %r883, %r10;
div.full.f32 %r189, %r884, %r10;
div.full.f32 %r190, %r885, %r10;
div.full.f32 %r191, %r886, %r10;
div.full.f32 %r192, %r887, %r10;
div.full.f32 %r193, %r888, %r10;
div.full.f32 %r194, %r889, %r10;
div.full.f32 %r195, %r890, %r10;
div.full.f32 %r196, %r891, %r10;
div.full.f32 %r197, %r892, %r10;
div.full.f32 %r198, %r893, %r10;
div.full.f32 %r199, %r894, %r10;
div.full.f32 %r200, %r895, %r10;
div.full.f32 %r201, %r896, %r10;
div.full.f32 %r202, %r897, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs50, %r140, %r139;
cvt.rn.satfinite.e4m3x2.f32 %rs51, %r142, %r141;
cvt.rn.satfinite.e4m3x2.f32 %rs52, %r144, %r143;
cvt.rn.satfinite.e4m3x2.f32 %rs53, %r146, %r145;
cvt.rn.satfinite.e4m3x2.f32 %rs54, %r148, %r147;
cvt.rn.satfinite.e4m3x2.f32 %rs55, %r150, %r149;
cvt.rn.satfinite.e4m3x2.f32 %rs56, %r152, %r151;
cvt.rn.satfinite.e4m3x2.f32 %rs57, %r154, %r153;
cvt.rn.satfinite.e4m3x2.f32 %rs58, %r156, %r155;
cvt.rn.satfinite.e4m3x2.f32 %rs59, %r158, %r157;
cvt.rn.satfinite.e4m3x2.f32 %rs60, %r160, %r159;
cvt.rn.satfinite.e4m3x2.f32 %rs61, %r162, %r161;
cvt.rn.satfinite.e4m3x2.f32 %rs62, %r164, %r163;
cvt.rn.satfinite.e4m3x2.f32 %rs63, %r166, %r165;
cvt.rn.satfinite.e4m3x2.f32 %rs64, %r168, %r167;
cvt.rn.satfinite.e4m3x2.f32 %rs65, %r170, %r169;
cvt.rn.satfinite.e4m3x2.f32 %rs66, %r172, %r171;
cvt.rn.satfinite.e4m3x2.f32 %rs67, %r174, %r173;
cvt.rn.satfinite.e4m3x2.f32 %rs68, %r176, %r175;
cvt.rn.satfinite.e4m3x2.f32 %rs69, %r178, %r177;
cvt.rn.satfinite.e4m3x2.f32 %rs70, %r180, %r179;
cvt.rn.satfinite.e4m3x2.f32 %rs71, %r182, %r181;
cvt.rn.satfinite.e4m3x2.f32 %rs72, %r184, %r183;
cvt.rn.satfinite.e4m3x2.f32 %rs73, %r186, %r185;
cvt.rn.satfinite.e4m3x2.f32 %rs74, %r188, %r187;
cvt.rn.satfinite.e4m3x2.f32 %rs75, %r190, %r189;
cvt.rn.satfinite.e4m3x2.f32 %rs76, %r192, %r191;
cvt.rn.satfinite.e4m3x2.f32 %rs77, %r194, %r193;
cvt.rn.satfinite.e4m3x2.f32 %rs78, %r196, %r195;
cvt.rn.satfinite.e4m3x2.f32 %rs79, %r198, %r197;
cvt.rn.satfinite.e4m3x2.f32 %rs80, %r200, %r199;
cvt.rn.satfinite.e4m3x2.f32 %rs81, %r202, %r201;
add.s32 %r898, %r781, 768;
or.b32 %r899, %r898, %r271;
or.b32 %r900, %r898, %r234;
cvt.s64.s32 %rd86, %r899;
add.s64 %rd36, %rd76, %rd86;
add.s64 %rd37, %rd74, %rd86;
mov.b32 %r203, {%rs66, %rs67};
mov.b32 %r204, {%rs68, %rs69};
mov.b32 %r205, {%rs70, %rs71};
mov.b32 %r206, {%rs72, %rs73};
@%p1 st.global.v4.b32 [ %rd36 + 0 ], { %r203, %r204, %r205, %r206 };
mov.b32 %r207, {%rs74, %rs75};
mov.b32 %r208, {%rs76, %rs77};
mov.b32 %r209, {%rs78, %rs79};
mov.b32 %r210, {%rs80, %rs81};
@%p1 st.global.v4.b32 [ %rd37 + 0 ], { %r207, %r208, %r209, %r210 };
@EXTENT_8@
cvt.s64.s32 %rd87, %r901;
add.s64 %rd88, %rd47, %rd87;
@EXTENT_9@
add.s64 %rd38, %rd88, %rd69;
@EXTENT_10@
mov.b32 %r211, {%rs50, %rs51};
mov.b32 %r212, {%rs52, %rs53};
mov.b32 %r213, {%rs54, %rs55};
mov.b32 %r214, {%rs56, %rs57};
@%p1 st.global.v4.b32 [ %rd38 + 0 ], { %r211, %r212, %r213, %r214 };
mov.b32 %r215, {%rs58, %rs59};
mov.b32 %r216, {%rs60, %rs61};
mov.b32 %r217, {%rs62, %rs63};
mov.b32 %r218, {%rs64, %rs65};
@%p1 st.global.v4.b32 [ %rd39 + 0 ], { %r215, %r216, %r217, %r218 };
"#) ++
  r#"ret;
$L__tmp25:
$L__func_end0:
}
"#

private def halfValueTTGIR (vector : Bool) : String :=
  r#"#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
"# ++
  (if vector then r#"#blocked2 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
"# else r#"#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
"#) ++
  r#"module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_qkv_norm_rope_pack_fp8_backward_kernel(%grad_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_v: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %qk: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<64> : tensor<64x1xi32, #blocked>
@EXTENT_18@
%cst_1 = arith.constant dense<384> : tensor<32x1xi32, #blocked1>
%cst_2 = arith.constant dense<1152> : tensor<32x1xi32, #blocked1>
"# ++
  (if vector then r#"%cst_3 = arith.constant dense<64> : tensor<32x1xi32, #blocked1>
%cst_4 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
@EXTENT_20@
%cst_7 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked1>
%cst_8 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_9 = arith.constant dense<6.400000e+01> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# else r#"%cst_3 = arith.constant dense<64> : tensor<32x1xi32, #blocked2>
%cst_4 = arith.constant dense<64> : tensor<32x1xi32, #blocked1>
%cst_5 = arith.constant dense<1> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_6 = arith.constant dense<64> : tensor<1x64xi32, #blocked2>
%cst_7 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>
@EXTENT_19@
%cst_11 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked1>
%cst_12 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked2>
%cst_13 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_14 = arith.constant dense<6.400000e+01> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#) ++
  r#"%logical_head = tt.get_program_id x : i32
%token = tt.get_program_id y : i32
"# ++
  (if vector then r#"%token_10 = arith.muli %token, 32 : i32
%token_11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_12 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_13 = tt.splat %token_10 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_14 = tt.splat %token_10 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_15 = arith.addi %token_13, %token_11 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_16 = arith.addi %token_14, %token_12 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_d = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask = arith.cmpi slt, %token_15, %cst_6 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_18 = arith.cmpi slt, %token_16, %cst_5 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"# else r#"%token_15 = arith.muli %token, 32 : i32
%token_16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_19 = tt.splat %token_15 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_20 = tt.splat %token_15 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_21 = tt.splat %token_15 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_22 = arith.addi %token_19, %token_16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_23 = arith.addi %token_20, %token_17 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_24 = arith.addi %token_21, %token_18 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_d = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offs_d_25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_26 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask = arith.cmpi slt, %token_22, %cst_10 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_27 = arith.cmpi slt, %token_23, %cst_9 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_mask_28 = arith.cmpi slt, %token_24, %cst_8 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"#) ++
  r#"%qk_mask = tt.expand_dims %token_mask {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi1, #blocked1>
"# ++
  (if vector then r#"%qk_mask_19 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%qk_mask_20 = arith.cmpi slt, %qk_mask_19, %cst_4 : tensor<1x64xi32, #blocked1>
%qk_mask_21 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_22 = tt.broadcast %qk_mask_20 : tensor<1x64xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_23 = arith.andi %qk_mask_21, %qk_mask_22 : tensor<32x64xi1, #blocked1>
%f1 = tt.expand_dims %token_15 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_24 = arith.muli %f1, %cst_3 : tensor<32x1xi32, #blocked1>
%f1_25 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_26 = tt.addptr %f1_25, %f1_24 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_27 = tt.broadcast %f1_26 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_28 = tt.broadcast %qk_mask_19 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%f1_29 = tt.addptr %f1_27, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f1_30 = tt.load %f1_29, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_31 = arith.extf %f1_30 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%f2 = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f2_32 = tt.addptr %f2, %f1_24 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f2_33 = tt.broadcast %f2_32 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f2_34 = tt.addptr %f2_33, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f2_35 = tt.load %f2_34, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f2_flip = tt.reshape %f2_35 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked2>
%y = tt.bitcast %f2_flip : tensor<32x32x2xbf16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_36 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_158: i16, %y_159: i16):
%y_160 = arith.xori %y_158, %y_159 : i16
tt.reduce.return %y_160 : i16
}) : (tensor<32x32x2xi16, #blocked2>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_37 = tt.expand_dims %y_36 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x32x1xi16, #blocked2>
%y_38 = tt.broadcast %y_37 : tensor<32x32x1xi16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_39 = arith.xori %y, %y_38 : tensor<32x32x2xi16, #blocked2>
%x = tt.bitcast %y_39 : tensor<32x32x2xi16, #blocked2> -> tensor<32x32x2xbf16, #blocked2>
%f2_flip_40 = tt.reshape %x : tensor<32x32x2xbf16, #blocked2> -> tensor<32x64xbf16, #blocked1>
%f2_flip_41 = arith.extf %f2_flip_40 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"# else r#"%qk_mask_29 = tt.expand_dims %token_mask_27 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi1, #blocked2>
%qk_mask_30 = tt.expand_dims %offs_d_25 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%qk_mask_31 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
%qk_mask_32 = arith.cmpi slt, %qk_mask_30, %cst_7 : tensor<1x64xi32, #blocked1>
%qk_mask_33 = arith.cmpi slt, %qk_mask_31, %cst_6 : tensor<1x64xi32, #blocked2>
%qk_mask_34 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_35 = tt.broadcast %qk_mask_29 : tensor<32x1xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
%qk_mask_36 = tt.broadcast %qk_mask_32 : tensor<1x64xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
%qk_mask_37 = tt.broadcast %qk_mask_33 : tensor<1x64xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
%qk_mask_38 = arith.andi %qk_mask_34, %qk_mask_36 : tensor<32x64xi1, #blocked1>
%qk_mask_39 = arith.andi %qk_mask_35, %qk_mask_37 : tensor<32x64xi1, #blocked2>
%offs_flip = arith.xori %offs_d, %cst_5 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%f1 = tt.expand_dims %token_22 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_40 = tt.expand_dims %token_23 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
%f1_41 = arith.muli %f1, %cst_4 : tensor<32x1xi32, #blocked1>
%f1_42 = arith.muli %f1_40, %cst_3 : tensor<32x1xi32, #blocked2>
%f1_43 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_44 = tt.addptr %f1_43, %f1_41 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_45 = tt.broadcast %f1_44 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_46 = tt.broadcast %qk_mask_30 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%f1_47 = tt.addptr %f1_45, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%f1_48 = tt.load %f1_47, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%f1_49 = arith.extf %f1_48 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%f2_flip = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked2>
%f2_flip_50 = tt.addptr %f2_flip, %f1_42 : tensor<32x1x!tt.ptr<bf16>, #blocked2>, tensor<32x1xi32, #blocked2>
%f2_flip_51 = tt.expand_dims %offs_flip {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
%f2_flip_52 = tt.broadcast %f2_flip_50 : tensor<32x1x!tt.ptr<bf16>, #blocked2> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
%f2_flip_53 = tt.broadcast %f2_flip_51 : tensor<1x64xi32, #blocked2> -> tensor<32x64xi32, #blocked2>
%f2_flip_54 = tt.addptr %f2_flip_52, %f2_flip_53 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi32, #blocked2>
%f2_flip_55 = tt.load %f2_flip_54, %qk_mask_39, %cst_12 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
%f2_flip_56 = ttg.convert_layout %f2_flip_55 : tensor<32x64xbf16, #blocked2> -> tensor<32x64xbf16, #blocked1>
%f2_flip_57 = arith.extf %f2_flip_56 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"#) ++
  r#"%scale = tt.load %grad_scale_ptr : !tt.ptr<f32>
"# ++
  (if vector then r#"%x_42 = arith.muli %f1, %cst_2 : tensor<32x1xi32, #blocked1>
%x_43 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_44 = tt.addptr %x_43, %x_42 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_45 = arith.muli %logical_head, 64 : i32
%x_46 = tt.splat %x_45 : i32 -> tensor<32x1xi32, #blocked1>
%x_47 = tt.addptr %x_44, %x_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_48 = tt.broadcast %x_47 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_49 = tt.addptr %x_48, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_50 = tt.load %x_49, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_51 = arith.extf %x_50 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"# else r#"%x = arith.muli %f1, %cst_2 : tensor<32x1xi32, #blocked1>
%x_58 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_59 = tt.addptr %x_58, %x : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_60 = arith.muli %logical_head, 64 : i32
%x_61 = tt.splat %x_60 : i32 -> tensor<32x1xi32, #blocked1>
%x_62 = tt.addptr %x_59, %x_61 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_63 = tt.broadcast %x_62 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_64 = tt.addptr %x_63, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_65 = tt.load %x_64, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_66 = arith.extf %x_65 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
"#) ++
  r#"%grad_ptr = arith.muli %f1, %cst_1 : tensor<32x1xi32, #blocked1>
"# ++
  (if vector then r#"%grad_ptr_52 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_53 = tt.addptr %grad_ptr_52, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_54 = tt.addptr %grad_ptr_53, %x_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_55 = tt.broadcast %grad_ptr_54 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_56 = tt.addptr %grad_ptr_55, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad = tt.load %grad_ptr_56, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked2>
%y_57 = tt.bitcast %grad_flip : tensor<32x32x2xbf16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_58 = "tt.reduce"(%y_57) <{axis = 2 : i32}> ({
^bb0(%y_158: i16, %y_159: i16):
%y_160 = arith.xori %y_158, %y_159 : i16
tt.reduce.return %y_160 : i16
}) : (tensor<32x32x2xi16, #blocked2>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_59 = tt.expand_dims %y_58 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x32x1xi16, #blocked2>
%y_60 = tt.broadcast %y_59 : tensor<32x32x1xi16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_61 = arith.xori %y_57, %y_60 : tensor<32x32x2xi16, #blocked2>
%x_62 = tt.bitcast %y_61 : tensor<32x32x2xi16, #blocked2> -> tensor<32x32x2xbf16, #blocked2>
%grad_flip_63 = tt.reshape %x_62 : tensor<32x32x2xbf16, #blocked2> -> tensor<32x64xbf16, #blocked1>
%grad_64 = arith.extf %grad : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_65 = arith.extf %grad_flip_63 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd = arith.mulf %x_51, %x_51 : tensor<32x64xf32, #blocked1>
%rstd_66 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_158: f32, %rstd_159: f32):
%rstd_160 = arith.addf %rstd_158, %rstd_159 : f32
tt.reduce.return %rstd_160 : f32
"# else r#"%grad_ptr_67 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_68 = tt.addptr %grad_ptr_67, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_69 = tt.addptr %grad_ptr_68, %x_61 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_70 = tt.broadcast %grad_ptr_69 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_71 = tt.addptr %grad_ptr_70, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad = tt.load %grad_ptr_71, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%y = tt.bitcast %grad_flip : tensor<32x32x2xbf16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_72 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_172: i16, %y_173: i16):
%y_174 = arith.xori %y_172, %y_173 : i16
tt.reduce.return %y_174 : i16
}) : (tensor<32x32x2xi16, #blocked3>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_73 = tt.expand_dims %y_72 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x32x1xi16, #blocked3>
%y_74 = tt.broadcast %y_73 : tensor<32x32x1xi16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_75 = arith.xori %y, %y_74 : tensor<32x32x2xi16, #blocked3>
%x_76 = tt.bitcast %y_75 : tensor<32x32x2xi16, #blocked3> -> tensor<32x32x2xbf16, #blocked3>
%grad_flip_77 = tt.reshape %x_76 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%grad_78 = arith.extf %grad : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_79 = arith.extf %grad_flip_77 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd = arith.mulf %x_66, %x_66 : tensor<32x64xf32, #blocked1>
%rstd_80 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_172: f32, %rstd_173: f32):
%rstd_174 = arith.addf %rstd_172, %rstd_173 : f32
tt.reduce.return %rstd_174 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%rstd_67 = arith.divf %rstd_66, %cst_9 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_68 = arith.addf %rstd_67, %cst_8 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_69 = math.rsqrt %rstd_68 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_69 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_70 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_71 = arith.mulf %x_51, %normalized_70 : tensor<32x64xf32, #blocked1>
%grad_normalized = arith.mulf %f1_31, %grad_64 : tensor<32x64xf32, #blocked1>
%grad_normalized_72 = arith.mulf %f2_flip_41, %grad_flip_65 : tensor<32x64xf32, #blocked1>
%grad_normalized_73 = arith.addf %grad_normalized, %grad_normalized_72 : tensor<32x64xf32, #blocked1>
%correction = arith.mulf %grad_normalized_73, %normalized_71 : tensor<32x64xf32, #blocked1>
%correction_74 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_158: f32, %correction_159: f32):
%correction_160 = arith.addf %correction_158, %correction_159 : f32
tt.reduce.return %correction_160 : f32
"# else r#"%rstd_81 = arith.divf %rstd_80, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_82 = arith.addf %rstd_81, %cst_13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_83 = math.rsqrt %rstd_82 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_83 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_84 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_85 = arith.mulf %x_66, %normalized_84 : tensor<32x64xf32, #blocked1>
%grad_normalized = arith.mulf %f1_49, %grad_78 : tensor<32x64xf32, #blocked1>
%grad_normalized_86 = arith.mulf %f2_flip_57, %grad_flip_79 : tensor<32x64xf32, #blocked1>
%grad_normalized_87 = arith.addf %grad_normalized, %grad_normalized_86 : tensor<32x64xf32, #blocked1>
%correction = arith.mulf %grad_normalized_87, %normalized_85 : tensor<32x64xf32, #blocked1>
%correction_88 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_172: f32, %correction_173: f32):
%correction_174 = arith.addf %correction_172, %correction_173 : f32
tt.reduce.return %correction_174 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%correction_75 = arith.divf %correction_74, %cst_9 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_75 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_76 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_77 = arith.mulf %normalized_71, %dx_76 : tensor<32x64xf32, #blocked1>
%dx_78 = arith.subf %grad_normalized_73, %dx_77 : tensor<32x64xf32, #blocked1>
%dx_79 = arith.mulf %normalized_70, %dx_78 : tensor<32x64xf32, #blocked1>
%q = arith.truncf %dx_79 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked3>
%q_80 = arith.extf %0 : tensor<32x64xbf16, #blocked3> to tensor<32x64xf32, #blocked3>
%q_81 = arith.extf %q : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_82 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked3>
%q_83 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked1>
%q_84 = arith.divf %q_80, %q_82 : tensor<32x64xf32, #blocked3>
%q_85 = arith.divf %q_81, %q_83 : tensor<32x64xf32, #blocked1>
%q_86 = tt.fp_to_fp %q_84, rounding = rtne : tensor<32x64xf32, #blocked3> -> tensor<32x64xf8E4M3FN, #blocked3>
%1 = tt.trans %q_86 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked3> -> tensor<64x32xf8E4M3FN, #blocked>
%q_87 = tt.fp_to_fp %q_85, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature = tt.splat %x_45 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_88 = tt.splat %x_45 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_89 = arith.addi %feature, %offs_d : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_90 = arith.addi %feature_88, %offs_d_17 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"# else r#"%correction_89 = arith.divf %correction_88, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_89 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_90 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_91 = arith.mulf %normalized_85, %dx_90 : tensor<32x64xf32, #blocked1>
%dx_92 = arith.subf %grad_normalized_87, %dx_91 : tensor<32x64xf32, #blocked1>
%dx_93 = arith.mulf %normalized_84, %dx_92 : tensor<32x64xf32, #blocked1>
%q = arith.truncf %dx_93 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked4>
%q_94 = arith.extf %0 : tensor<32x64xbf16, #blocked4> to tensor<32x64xf32, #blocked4>
%q_95 = arith.extf %q : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_96 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked4>
%q_97 = tt.splat %scale : f32 -> tensor<32x64xf32, #blocked1>
%q_98 = arith.divf %q_94, %q_96 : tensor<32x64xf32, #blocked4>
%q_99 = arith.divf %q_95, %q_97 : tensor<32x64xf32, #blocked1>
%q_100 = tt.fp_to_fp %q_98, rounding = rtne : tensor<32x64xf32, #blocked4> -> tensor<32x64xf8E4M3FN, #blocked4>
%1 = tt.trans %q_100 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked4> -> tensor<64x32xf8E4M3FN, #blocked>
%q_101 = tt.fp_to_fp %q_99, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature = tt.splat %x_60 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_102 = tt.splat %x_60 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_103 = arith.addi %feature, %offs_d_25 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_104 = arith.addi %feature_102, %offs_d_26 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"#) ++
  r#"%2 = tt.splat %grad_row : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>
"# ++
  (if vector then r#"%3 = tt.addptr %2, %x_42 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x1xi32, #blocked1>
%4 = tt.expand_dims %feature_89 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"# else r#"%3 = tt.addptr %2, %x : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x1xi32, #blocked1>
%4 = tt.expand_dims %feature_103 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"#) ++
  r#"%5 = tt.broadcast %3 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%6 = tt.broadcast %4 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%7 = tt.addptr %5, %6 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x64xi32, #blocked1>
"# ++
  (if vector then r#"tt.store %7, %q_87, %qk_mask_23 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%8 = tt.expand_dims %offs_d_17 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"# else r#"tt.store %7, %q_101, %qk_mask_38 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%8 = tt.expand_dims %offs_d_26 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"#) ++
  r#"%9 = arith.cmpi slt, %8, %cst : tensor<64x1xi32, #blocked>
"# ++
  (if vector then r#"%10 = tt.expand_dims %token_mask_18 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
"# else r#"%10 = tt.expand_dims %token_mask_28 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
"#) ++
  r#"%11 = tt.broadcast %9 : tensor<64x1xi1, #blocked> -> tensor<64x32xi1, #blocked>
%12 = tt.broadcast %10 : tensor<1x32xi1, #blocked> -> tensor<64x32xi1, #blocked>
%13 = arith.andi %11, %12 : tensor<64x32xi1, #blocked>
"# ++
  (if vector then r#"%14 = tt.expand_dims %feature_90 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"# else r#"%14 = tt.expand_dims %feature_104 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"#) ++
  r#"%15 = arith.muli %14, %cst_0 : tensor<64x1xi32, #blocked>
%16 = tt.splat %grad_transposed : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>
%17 = tt.addptr %16, %15 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
"# ++
  (if vector then r#"%18 = tt.expand_dims %token_16 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
"# else r#"%18 = tt.expand_dims %token_24 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
"#) ++
  r#"%19 = tt.broadcast %17 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
%20 = tt.broadcast %18 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
%21 = tt.addptr %19, %20 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x32xi32, #blocked>
tt.store %21, %1, %13 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
%input_head = arith.addi %logical_head, 6 : i32
"# ++
  (if vector then r#"%x_91 = arith.muli %input_head, 64 : i32
%x_92 = tt.splat %x_91 : i32 -> tensor<32x1xi32, #blocked1>
%x_93 = tt.addptr %x_44, %x_92 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_94 = tt.broadcast %x_93 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_95 = tt.addptr %x_94, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_96 = tt.load %x_95, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_97 = arith.extf %x_96 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr_98 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_99 = tt.addptr %grad_ptr_98, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_100 = tt.addptr %grad_ptr_99, %x_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_101 = tt.broadcast %grad_ptr_100 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_102 = tt.addptr %grad_ptr_101, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_103 = tt.load %grad_ptr_102, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip_104 = tt.reshape %grad_103 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked2>
%y_105 = tt.bitcast %grad_flip_104 : tensor<32x32x2xbf16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_106 = "tt.reduce"(%y_105) <{axis = 2 : i32}> ({
^bb0(%y_158: i16, %y_159: i16):
%y_160 = arith.xori %y_158, %y_159 : i16
tt.reduce.return %y_160 : i16
}) : (tensor<32x32x2xi16, #blocked2>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_107 = tt.expand_dims %y_106 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x32x1xi16, #blocked2>
%y_108 = tt.broadcast %y_107 : tensor<32x32x1xi16, #blocked2> -> tensor<32x32x2xi16, #blocked2>
%y_109 = arith.xori %y_105, %y_108 : tensor<32x32x2xi16, #blocked2>
%x_110 = tt.bitcast %y_109 : tensor<32x32x2xi16, #blocked2> -> tensor<32x32x2xbf16, #blocked2>
%grad_flip_111 = tt.reshape %x_110 : tensor<32x32x2xbf16, #blocked2> -> tensor<32x64xbf16, #blocked1>
%grad_112 = arith.extf %grad_103 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_113 = arith.extf %grad_flip_111 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd_114 = arith.mulf %x_97, %x_97 : tensor<32x64xf32, #blocked1>
%rstd_115 = "tt.reduce"(%rstd_114) <{axis = 1 : i32}> ({
^bb0(%rstd_158: f32, %rstd_159: f32):
%rstd_160 = arith.addf %rstd_158, %rstd_159 : f32
tt.reduce.return %rstd_160 : f32
"# else r#"%x_105 = arith.muli %input_head, 64 : i32
%x_106 = tt.splat %x_105 : i32 -> tensor<32x1xi32, #blocked1>
%x_107 = tt.addptr %x_59, %x_106 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_108 = tt.broadcast %x_107 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_109 = tt.addptr %x_108, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%x_110 = tt.load %x_109, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%x_111 = arith.extf %x_110 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_ptr_112 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_113 = tt.addptr %grad_ptr_112, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_114 = tt.addptr %grad_ptr_113, %x_61 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_115 = tt.broadcast %grad_ptr_114 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_ptr_116 = tt.addptr %grad_ptr_115, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_117 = tt.load %grad_ptr_116, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_flip_118 = tt.reshape %grad_117 : tensor<32x64xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%y_119 = tt.bitcast %grad_flip_118 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_120 = "tt.reduce"(%y_119) <{axis = 2 : i32}> ({
^bb0(%y_172: i16, %y_173: i16):
%y_174 = arith.xori %y_172, %y_173 : i16
tt.reduce.return %y_174 : i16
}) : (tensor<32x32x2xi16, #blocked3>) -> tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_121 = tt.expand_dims %y_120 {axis = 2 : i32} : tensor<32x32xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x32x1xi16, #blocked3>
%y_122 = tt.broadcast %y_121 : tensor<32x32x1xi16, #blocked3> -> tensor<32x32x2xi16, #blocked3>
%y_123 = arith.xori %y_119, %y_122 : tensor<32x32x2xi16, #blocked3>
%x_124 = tt.bitcast %y_123 : tensor<32x32x2xi16, #blocked3> -> tensor<32x32x2xbf16, #blocked3>
%grad_flip_125 = tt.reshape %x_124 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked1>
%grad_126 = arith.extf %grad_117 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%grad_flip_127 = arith.extf %grad_flip_125 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%rstd_128 = arith.mulf %x_111, %x_111 : tensor<32x64xf32, #blocked1>
%rstd_129 = "tt.reduce"(%rstd_128) <{axis = 1 : i32}> ({
^bb0(%rstd_172: f32, %rstd_173: f32):
%rstd_174 = arith.addf %rstd_172, %rstd_173 : f32
tt.reduce.return %rstd_174 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%rstd_116 = arith.divf %rstd_115, %cst_9 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_117 = arith.addf %rstd_116, %cst_8 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_118 = math.rsqrt %rstd_117 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_119 = tt.expand_dims %rstd_118 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_120 = tt.broadcast %normalized_119 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_121 = arith.mulf %x_97, %normalized_120 : tensor<32x64xf32, #blocked1>
%grad_normalized_122 = arith.mulf %f1_31, %grad_112 : tensor<32x64xf32, #blocked1>
%grad_normalized_123 = arith.mulf %f2_flip_41, %grad_flip_113 : tensor<32x64xf32, #blocked1>
%grad_normalized_124 = arith.addf %grad_normalized_122, %grad_normalized_123 : tensor<32x64xf32, #blocked1>
%correction_125 = arith.mulf %grad_normalized_124, %normalized_121 : tensor<32x64xf32, #blocked1>
%correction_126 = "tt.reduce"(%correction_125) <{axis = 1 : i32}> ({
^bb0(%correction_158: f32, %correction_159: f32):
%correction_160 = arith.addf %correction_158, %correction_159 : f32
tt.reduce.return %correction_160 : f32
"# else r#"%rstd_130 = arith.divf %rstd_129, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_131 = arith.addf %rstd_130, %cst_13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_132 = math.rsqrt %rstd_131 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_133 = tt.expand_dims %rstd_132 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_134 = tt.broadcast %normalized_133 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%normalized_135 = arith.mulf %x_111, %normalized_134 : tensor<32x64xf32, #blocked1>
%grad_normalized_136 = arith.mulf %f1_49, %grad_126 : tensor<32x64xf32, #blocked1>
%grad_normalized_137 = arith.mulf %f2_flip_57, %grad_flip_127 : tensor<32x64xf32, #blocked1>
%grad_normalized_138 = arith.addf %grad_normalized_136, %grad_normalized_137 : tensor<32x64xf32, #blocked1>
%correction_139 = arith.mulf %grad_normalized_138, %normalized_135 : tensor<32x64xf32, #blocked1>
%correction_140 = "tt.reduce"(%correction_139) <{axis = 1 : i32}> ({
^bb0(%correction_172: f32, %correction_173: f32):
%correction_174 = arith.addf %correction_172, %correction_173 : f32
tt.reduce.return %correction_174 : f32
"#) ++
  r#"}) : (tensor<32x64xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if vector then r#"%correction_127 = arith.divf %correction_126, %cst_9 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_128 = tt.expand_dims %correction_127 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_129 = tt.broadcast %dx_128 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_130 = arith.mulf %normalized_121, %dx_129 : tensor<32x64xf32, #blocked1>
%dx_131 = arith.subf %grad_normalized_124, %dx_130 : tensor<32x64xf32, #blocked1>
%dx_132 = arith.mulf %normalized_120, %dx_131 : tensor<32x64xf32, #blocked1>
%q_133 = arith.truncf %dx_132 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%22 = ttg.convert_layout %q_133 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked3>
%q_134 = arith.extf %22 : tensor<32x64xbf16, #blocked3> to tensor<32x64xf32, #blocked3>
%q_135 = arith.extf %q_133 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_136 = arith.divf %q_134, %q_82 : tensor<32x64xf32, #blocked3>
%q_137 = arith.divf %q_135, %q_83 : tensor<32x64xf32, #blocked1>
%q_138 = tt.fp_to_fp %q_136, rounding = rtne : tensor<32x64xf32, #blocked3> -> tensor<32x64xf8E4M3FN, #blocked3>
%23 = tt.trans %q_138 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked3> -> tensor<64x32xf8E4M3FN, #blocked>
%q_139 = tt.fp_to_fp %q_137, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature_140 = tt.splat %x_91 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_141 = tt.splat %x_91 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_142 = arith.addi %feature_140, %offs_d : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_143 = arith.addi %feature_141, %offs_d_17 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%24 = tt.expand_dims %feature_142 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"# else r#"%correction_141 = arith.divf %correction_140, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_142 = tt.expand_dims %correction_141 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_143 = tt.broadcast %dx_142 : tensor<32x1xf32, #blocked1> -> tensor<32x64xf32, #blocked1>
%dx_144 = arith.mulf %normalized_135, %dx_143 : tensor<32x64xf32, #blocked1>
%dx_145 = arith.subf %grad_normalized_138, %dx_144 : tensor<32x64xf32, #blocked1>
%dx_146 = arith.mulf %normalized_134, %dx_145 : tensor<32x64xf32, #blocked1>
%q_147 = arith.truncf %dx_146 : tensor<32x64xf32, #blocked1> to tensor<32x64xbf16, #blocked1>
%22 = ttg.convert_layout %q_147 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked4>
%q_148 = arith.extf %22 : tensor<32x64xbf16, #blocked4> to tensor<32x64xf32, #blocked4>
%q_149 = arith.extf %q_147 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%q_150 = arith.divf %q_148, %q_96 : tensor<32x64xf32, #blocked4>
%q_151 = arith.divf %q_149, %q_97 : tensor<32x64xf32, #blocked1>
%q_152 = tt.fp_to_fp %q_150, rounding = rtne : tensor<32x64xf32, #blocked4> -> tensor<32x64xf8E4M3FN, #blocked4>
%23 = tt.trans %q_152 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked4> -> tensor<64x32xf8E4M3FN, #blocked>
%q_153 = tt.fp_to_fp %q_151, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%feature_154 = tt.splat %x_105 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_155 = tt.splat %x_105 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_156 = arith.addi %feature_154, %offs_d_25 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_157 = arith.addi %feature_155, %offs_d_26 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%24 = tt.expand_dims %feature_156 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"#) ++
  r#"%25 = tt.broadcast %24 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%26 = tt.addptr %5, %25 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x64xi32, #blocked1>
"# ++
  (if vector then r#"tt.store %26, %q_139, %qk_mask_23 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%27 = tt.expand_dims %feature_143 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"# else r#"tt.store %26, %q_153, %qk_mask_38 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%27 = tt.expand_dims %feature_157 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"#) ++
  r#"%28 = arith.muli %27, %cst_0 : tensor<64x1xi32, #blocked>
%29 = tt.addptr %16, %28 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
%30 = tt.broadcast %29 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
%31 = tt.addptr %30, %20 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x32xi32, #blocked>
tt.store %31, %23, %13 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
%grad_v_value = tt.splat %grad_v : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
"# ++
  (if vector then r#"%grad_v_value_144 = tt.addptr %grad_v_value, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_145 = tt.addptr %grad_v_value_144, %x_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_146 = tt.broadcast %grad_v_value_145 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_v_value_147 = tt.addptr %grad_v_value_146, %f1_28 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_v_value_148 = tt.load %grad_v_value_147, %qk_mask_23, %cst_7 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%32 = ttg.convert_layout %grad_v_value_148 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked3>
%grad_v_value_149 = arith.extf %32 : tensor<32x64xbf16, #blocked3> to tensor<32x64xf32, #blocked3>
%grad_v_value_150 = arith.extf %grad_v_value_148 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%qv = arith.divf %grad_v_value_149, %q_82 : tensor<32x64xf32, #blocked3>
%qv_151 = arith.divf %grad_v_value_150, %q_83 : tensor<32x64xf32, #blocked1>
%qv_152 = tt.fp_to_fp %qv, rounding = rtne : tensor<32x64xf32, #blocked3> -> tensor<32x64xf8E4M3FN, #blocked3>
%33 = tt.trans %qv_152 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked3> -> tensor<64x32xf8E4M3FN, #blocked>
%qv_153 = tt.fp_to_fp %qv_151, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%v_feature = arith.addi %x_45, 768 : i32
%v_feature_154 = tt.splat %v_feature : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_155 = tt.splat %v_feature : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%v_feature_156 = arith.addi %v_feature_154, %offs_d : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_157 = arith.addi %v_feature_155, %offs_d_17 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%34 = tt.expand_dims %v_feature_156 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"# else r#"%grad_v_value_158 = tt.addptr %grad_v_value, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_159 = tt.addptr %grad_v_value_158, %x_61 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_160 = tt.broadcast %grad_v_value_159 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x64x!tt.ptr<bf16>, #blocked1>
%grad_v_value_161 = tt.addptr %grad_v_value_160, %f1_46 : tensor<32x64x!tt.ptr<bf16>, #blocked1>, tensor<32x64xi32, #blocked1>
%grad_v_value_162 = tt.load %grad_v_value_161, %qk_mask_38, %cst_11 : tensor<32x64x!tt.ptr<bf16>, #blocked1>
%32 = ttg.convert_layout %grad_v_value_162 : tensor<32x64xbf16, #blocked1> -> tensor<32x64xbf16, #blocked4>
%grad_v_value_163 = arith.extf %32 : tensor<32x64xbf16, #blocked4> to tensor<32x64xf32, #blocked4>
%grad_v_value_164 = arith.extf %grad_v_value_162 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%qv = arith.divf %grad_v_value_163, %q_96 : tensor<32x64xf32, #blocked4>
%qv_165 = arith.divf %grad_v_value_164, %q_97 : tensor<32x64xf32, #blocked1>
%qv_166 = tt.fp_to_fp %qv, rounding = rtne : tensor<32x64xf32, #blocked4> -> tensor<32x64xf8E4M3FN, #blocked4>
%33 = tt.trans %qv_166 {order = array<i32: 1, 0>} : tensor<32x64xf8E4M3FN, #blocked4> -> tensor<64x32xf8E4M3FN, #blocked>
%qv_167 = tt.fp_to_fp %qv_165, rounding = rtne : tensor<32x64xf32, #blocked1> -> tensor<32x64xf8E4M3FN, #blocked1>
%v_feature = arith.addi %x_60, 768 : i32
%v_feature_168 = tt.splat %v_feature : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_169 = tt.splat %v_feature : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%v_feature_170 = arith.addi %v_feature_168, %offs_d_25 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_171 = arith.addi %v_feature_169, %offs_d_26 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%34 = tt.expand_dims %v_feature_170 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
"#) ++
  r#"%35 = tt.broadcast %34 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
%36 = tt.addptr %5, %35 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x64xi32, #blocked1>
"# ++
  (if vector then r#"tt.store %36, %qv_153, %qk_mask_23 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%37 = tt.expand_dims %v_feature_157 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"# else r#"tt.store %36, %qv_167, %qk_mask_38 : tensor<32x64x!tt.ptr<f8E4M3FN>, #blocked1>
%37 = tt.expand_dims %v_feature_171 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
"#) ++
  r#"%38 = arith.muli %37, %cst_0 : tensor<64x1xi32, #blocked>
%39 = tt.addptr %16, %38 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x1xi32, #blocked>
%40 = tt.broadcast %39 : tensor<64x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
%41 = tt.addptr %40, %20 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x32xi32, #blocked>
tt.store %41, %33, %13 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
tt.return
}
}
"#

private def halfValuePTX (vector : Bool) : String :=
  r#".version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry _qkv_norm_rope_pack_fp8_backward_kernel(
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_0,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_1,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_2,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_3,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_4,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_5,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_6,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_7,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_8,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_9,
.param .u64 .ptr .global .align 1 _qkv_norm_rope_pack_fp8_backward_kernel_param_10
)
.reqntid 128
{
.reg .pred %p<2>;
"# ++
  (if vector then r#".reg .b16 %rs<245>;
.reg .b32 %r<747>;
.reg .b64 %rd<57>;
"# else r#".reg .b16 %rs<262>;
.reg .b32 %r<769>;
.reg .b64 %rd<74>;
"#) ++
  r#"$L__func_begin0:
"# ++
  (if vector then r#"ld.param.b64 %rd22, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd23, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"# else r#"ld.param.b64 %rd36, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd37, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"#) ++
  r#"$L__tmp0:
"# ++
  (if vector then r#"mov.u32 %r179, %ctaid.x;
ld.param.b64 %rd24, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r180, %ctaid.y;
shl.b32 %r181, %r180, 5;
ld.param.b64 %rd25, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd26, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r182, %tid.x;
ld.param.b64 %rd27, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r183, %r182, 2, 5;
ld.param.b64 %rd28, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
shl.b32 %r184, %r182, 4;
ld.param.b64 %rd29, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
and.b32 %r185, %r184, 16;
ld.param.b64 %rd5, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
or.b32 %r186, %r183, %r181;
or.b32 %r187, %r185, %r181;
and.b32 %r188, %r182, 3;
shl.b32 %r189, %r188, 4;
bfe.u32 %r190, %r182, 1, 6;
@EXTENT_25@
shl.b32 %r191, %r186, 6;
mul.wide.u32 %rd30, %r191, 2;
add.s64 %rd31, %rd26, %rd30;
mul.wide.u32 %rd32, %r189, 2;
add.s64 %rd1, %rd31, %rd32;
"# else r#"mov.u32 %r171, %ctaid.x;
ld.param.b64 %rd38, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r172, %ctaid.y;
shl.b32 %r173, %r172, 5;
ld.param.b64 %rd39, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd40, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r174, %tid.x;
ld.param.b64 %rd41, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r175, %r174, 2, 5;
ld.param.b64 %rd42, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
and.b32 %r176, %r174, 31;
ld.param.b64 %rd43, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
shl.b32 %r177, %r174, 4;
ld.param.b64 %rd19, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
and.b32 %r178, %r177, 16;
or.b32 %r179, %r175, %r173;
or.b32 %r180, %r173, %r176;
or.b32 %r181, %r178, %r173;
and.b32 %r182, %r174, 96;
bfe.u32 %r183, %r174, 5, 2;
and.b32 %r184, %r174, 3;
shl.b32 %r185, %r184, 4;
bfe.u32 %r186, %r174, 1, 6;
@EXTENT_21@
xor.b32 %r187, %r183, 1;
shl.b32 %r188, %r179, 6;
shl.b32 %r189, %r180, 6;
mad.wide.u32 %rd44, %r188, 2, %rd40;
mul.wide.u32 %rd45, %r185, 2;
add.s64 %rd1, %rd44, %rd45;
"#) ++
  r#"add.s64 %rd2, %rd1, 16;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
"# ++
  (if vector then r#"mov.b32 {%rs49, %rs50}, %r1;
mov.b32 {%rs51, %rs52}, %r2;
mov.b32 {%rs53, %rs54}, %r3;
mov.b32 {%rs55, %rs56}, %r4;
"# else r#"mov.b32 {%rs66, %rs67}, %r1;
mov.b32 {%rs68, %rs69}, %r2;
mov.b32 {%rs70, %rs71}, %r3;
mov.b32 {%rs72, %rs73}, %r4;
"#) ++
  r#"mov.u32 %r6, %r5;
mov.u32 %r7, %r5;
mov.u32 %r8, %r5;
mov.u32 %r9, %r5;
@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
"# ++
  (if vector then r#"mov.b32 {%rs57, %rs58}, %r6;
mov.b32 {%rs59, %rs60}, %r7;
mov.b32 {%rs61, %rs62}, %r8;
mov.b32 {%rs63, %rs64}, %r9;
cvt.f32.bf16 %r192, %rs49;
cvt.f32.bf16 %r193, %rs50;
cvt.f32.bf16 %r194, %rs51;
cvt.f32.bf16 %r195, %rs52;
cvt.f32.bf16 %r196, %rs53;
cvt.f32.bf16 %r197, %rs54;
cvt.f32.bf16 %r198, %rs55;
cvt.f32.bf16 %r199, %rs56;
cvt.f32.bf16 %r200, %rs57;
cvt.f32.bf16 %r201, %rs58;
cvt.f32.bf16 %r202, %rs59;
cvt.f32.bf16 %r203, %rs60;
cvt.f32.bf16 %r204, %rs61;
cvt.f32.bf16 %r205, %rs62;
cvt.f32.bf16 %r206, %rs63;
cvt.f32.bf16 %r207, %rs64;
add.s64 %rd33, %rd27, %rd30;
add.s64 %rd3, %rd33, %rd32;
add.s64 %rd4, %rd3, 16;
mov.u32 %r10, %r5;
"# else r#"mov.b32 {%rs74, %rs75}, %r6;
mov.b32 {%rs76, %rs77}, %r7;
mov.b32 {%rs78, %rs79}, %r8;
mov.b32 {%rs80, %rs81}, %r9;
cvt.f32.bf16 %r190, %rs66;
cvt.f32.bf16 %r191, %rs67;
cvt.f32.bf16 %r192, %rs68;
cvt.f32.bf16 %r193, %rs69;
cvt.f32.bf16 %r194, %rs70;
cvt.f32.bf16 %r195, %rs71;
cvt.f32.bf16 %r196, %rs72;
cvt.f32.bf16 %r197, %rs73;
cvt.f32.bf16 %r198, %rs74;
cvt.f32.bf16 %r199, %rs75;
cvt.f32.bf16 %r200, %rs76;
cvt.f32.bf16 %r201, %rs77;
cvt.f32.bf16 %r202, %rs78;
cvt.f32.bf16 %r203, %rs79;
cvt.f32.bf16 %r204, %rs80;
cvt.f32.bf16 %r205, %rs81;
mad.wide.u32 %rd46, %r189, 2, %rd41;
mad.wide.u32 %rd3, %r187, 2, %rd46;
cvt.u64.u32 %rd47, %r183;
xor.b64 %rd48, %rd47, 1;
shl.b64 %rd49, %rd48, 1;
add.s64 %rd50, %rd46, %rd49;
add.s64 %rd4, %rd50, 8;
add.s64 %rd5, %rd50, 16;
add.s64 %rd6, %rd50, 24;
add.s64 %rd7, %rd50, 32;
add.s64 %rd8, %rd50, 40;
add.s64 %rd9, %rd50, 48;
add.s64 %rd10, %rd50, 56;
add.s64 %rd11, %rd50, 64;
add.s64 %rd12, %rd50, 72;
add.s64 %rd13, %rd50, 80;
add.s64 %rd14, %rd50, 88;
add.s64 %rd15, %rd50, 96;
add.s64 %rd16, %rd50, 104;
add.s64 %rd17, %rd50, 112;
add.s64 %rd18, %rd50, 120;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd3 + 0 ];
mov.u16 %rs3, %rs2;
@%p1 ld.global.b16 { %rs3 }, [ %rd4 + 0 ];
mov.u16 %rs4, %rs2;
@%p1 ld.global.b16 { %rs4 }, [ %rd5 + 0 ];
mov.u16 %rs5, %rs2;
@%p1 ld.global.b16 { %rs5 }, [ %rd6 + 0 ];
mov.u16 %rs6, %rs2;
@%p1 ld.global.b16 { %rs6 }, [ %rd7 + 0 ];
mov.u16 %rs7, %rs2;
@%p1 ld.global.b16 { %rs7 }, [ %rd8 + 0 ];
mov.u16 %rs8, %rs2;
@%p1 ld.global.b16 { %rs8 }, [ %rd9 + 0 ];
mov.u16 %rs9, %rs2;
@%p1 ld.global.b16 { %rs9 }, [ %rd10 + 0 ];
mov.u16 %rs10, %rs2;
@%p1 ld.global.b16 { %rs10 }, [ %rd11 + 0 ];
mov.u16 %rs11, %rs2;
@%p1 ld.global.b16 { %rs11 }, [ %rd12 + 0 ];
mov.u16 %rs12, %rs2;
@%p1 ld.global.b16 { %rs12 }, [ %rd13 + 0 ];
mov.u16 %rs13, %rs2;
@%p1 ld.global.b16 { %rs13 }, [ %rd14 + 0 ];
mov.u16 %rs14, %rs2;
@%p1 ld.global.b16 { %rs14 }, [ %rd15 + 0 ];
mov.u16 %rs15, %rs2;
@%p1 ld.global.b16 { %rs15 }, [ %rd16 + 0 ];
mov.u16 %rs16, %rs2;
@%p1 ld.global.b16 { %rs16 }, [ %rd17 + 0 ];
mov.u16 %rs17, %rs2;
@%p1 ld.global.b16 { %rs17 }, [ %rd18 + 0 ];
and.b32 %r206, %r177, 2032;
mov.b32 %r207, global_smem;
add.s32 %r208, %r207, %r206;
mov.b32 %r209, {%rs1, %rs3};
mov.b32 %r210, {%rs6, %rs7};
mov.b32 %r211, {%rs10, %rs11};
mov.b32 %r212, {%rs14, %rs15};
st.shared.v4.b32 [%r208], {%r209, %r210, %r211, %r212};
bar.sync 0;
shl.b32 %r213, %r174, 6;
and.b32 %r214, %r213, 1536;
and.b32 %r215, %r177, 112;
shl.b32 %r216, %r182, 2;
add.s32 %r217, %r207, %r214;
add.s32 %r218, %r217, %r215;
add.s32 %r219, %r218, %r216;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r222, %r223}, [%r219];
mov.b32 {%rs82, %rs83}, %r220;
mov.b32 {%rs84, %rs85}, %r221;
mov.b32 {%rs86, %rs87}, %r222;
mov.b32 {%rs88, %rs89}, %r223;
bar.sync 0;
mov.b32 %r224, {%rs4, %rs5};
mov.b32 %r225, {%rs8, %rs9};
mov.b32 %r226, {%rs12, %rs13};
mov.b32 %r227, {%rs16, %rs17};
st.shared.v4.b32 [%r208], {%r224, %r225, %r226, %r227};
bar.sync 0;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r230, %r231}, [%r219];
mov.b32 {%rs90, %rs91}, %r228;
mov.b32 {%rs92, %rs93}, %r229;
mov.b32 {%rs94, %rs95}, %r230;
mov.b32 {%rs96, %rs97}, %r231;
cvt.f32.bf16 %r232, %rs82;
cvt.f32.bf16 %r233, %rs84;
cvt.f32.bf16 %r234, %rs86;
cvt.f32.bf16 %r235, %rs88;
cvt.f32.bf16 %r236, %rs83;
cvt.f32.bf16 %r237, %rs85;
cvt.f32.bf16 %r238, %rs87;
cvt.f32.bf16 %r239, %rs89;
cvt.f32.bf16 %r240, %rs90;
cvt.f32.bf16 %r241, %rs92;
cvt.f32.bf16 %r242, %rs94;
cvt.f32.bf16 %r243, %rs96;
cvt.f32.bf16 %r244, %rs91;
cvt.f32.bf16 %r245, %rs93;
cvt.f32.bf16 %r246, %rs95;
cvt.f32.bf16 %r247, %rs97;
mov.u32 %r10, 0x0;
ld.global.b32 { %r10 }, [ %rd19 + 0 ];
mul.lo.s32 %r248, %r179, 1152;
cvt.s64.s32 %rd51, %r248;
mad.wide.s32 %rd52, %r248, 2, %rd39;
shl.b32 %r249, %r171, 6;
mul.wide.s32 %rd53, %r249, 2;
add.s64 %rd54, %rd52, %rd53;
add.s64 %rd20, %rd54, %rd45;
add.s64 %rd21, %rd20, 16;
"#) ++
  r#"mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs65, %rs66}, %r10;
mov.b32 {%rs67, %rs68}, %r11;
mov.b32 {%rs69, %rs70}, %r12;
mov.b32 {%rs71, %rs72}, %r13;
"# else "") ++
  r#"mov.u32 %r14, %r5;
"# ++
  (if vector then "" else r#"@%p1 ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd20 + 0 ];
mov.b32 {%rs98, %rs99}, %r11;
mov.b32 {%rs100, %rs101}, %r12;
mov.b32 {%rs102, %rs103}, %r13;
mov.b32 {%rs104, %rs105}, %r14;
"#) ++
  r#"mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs73, %rs74}, %r14;
mov.b32 {%rs75, %rs76}, %r15;
mov.b32 {%rs77, %rs78}, %r16;
mov.b32 {%rs79, %rs80}, %r17;
cvt.f32.bf16 %r208, %rs66;
cvt.f32.bf16 %r209, %rs65;
cvt.f32.bf16 %r210, %rs68;
cvt.f32.bf16 %r211, %rs67;
cvt.f32.bf16 %r212, %rs70;
cvt.f32.bf16 %r213, %rs69;
cvt.f32.bf16 %r214, %rs72;
cvt.f32.bf16 %r215, %rs71;
cvt.f32.bf16 %r216, %rs74;
cvt.f32.bf16 %r217, %rs73;
cvt.f32.bf16 %r218, %rs76;
cvt.f32.bf16 %r219, %rs75;
cvt.f32.bf16 %r220, %rs78;
cvt.f32.bf16 %r221, %rs77;
cvt.f32.bf16 %r222, %rs80;
cvt.f32.bf16 %r223, %rs79;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd5 + 0 ];
mul.lo.s32 %r224, %r186, 1152;
cvt.s64.s32 %rd34, %r224;
mad.wide.s32 %rd35, %r224, 2, %rd25;
shl.b32 %r225, %r179, 6;
mul.wide.s32 %rd36, %r225, 2;
add.s64 %rd37, %rd35, %rd36;
add.s64 %rd6, %rd37, %rd32;
add.s64 %rd7, %rd6, 16;
"# else r#"mov.u32 %r18, %r5;
@%p1 ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd21 + 0 ];
mov.b32 {%rs106, %rs107}, %r15;
mov.b32 {%rs108, %rs109}, %r16;
mov.b32 {%rs110, %rs111}, %r17;
mov.b32 {%rs112, %rs113}, %r18;
cvt.f32.bf16 %r250, %rs98;
cvt.f32.bf16 %r251, %rs99;
cvt.f32.bf16 %r252, %rs100;
cvt.f32.bf16 %r253, %rs101;
cvt.f32.bf16 %r254, %rs102;
cvt.f32.bf16 %r255, %rs103;
cvt.f32.bf16 %r256, %rs104;
cvt.f32.bf16 %r257, %rs105;
cvt.f32.bf16 %r258, %rs106;
cvt.f32.bf16 %r259, %rs107;
cvt.f32.bf16 %r260, %rs108;
cvt.f32.bf16 %r261, %rs109;
cvt.f32.bf16 %r262, %rs110;
cvt.f32.bf16 %r263, %rs111;
cvt.f32.bf16 %r264, %rs112;
cvt.f32.bf16 %r265, %rs113;
mul.lo.s32 %r266, %r179, 384;
mul.wide.u32 %rd55, %r266, 2;
add.s64 %rd56, %rd36, %rd55;
add.s64 %rd57, %rd56, %rd53;
add.s64 %rd22, %rd57, %rd45;
add.s64 %rd23, %rd22, 16;
"#) ++
  r#"mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
mov.u32 %r22, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd6 + 0 ];
mov.b32 {%rs81, %rs82}, %r19;
mov.b32 {%rs83, %rs84}, %r20;
mov.b32 {%rs85, %rs86}, %r21;
mov.b32 {%rs87, %rs88}, %r22;
"# else r#"@%p1 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd22 + 0 ];
mov.b32 {%rs114, %rs115}, %r19;
mov.b32 {%rs116, %rs117}, %r20;
mov.b32 {%rs118, %rs119}, %r21;
mov.b32 {%rs120, %rs121}, %r22;
"#) ++
  r#"mov.u32 %r23, %r5;
mov.u32 %r24, %r5;
mov.u32 %r25, %r5;
mov.u32 %r26, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd7 + 0 ];
mov.b32 {%rs89, %rs90}, %r23;
mov.b32 {%rs91, %rs92}, %r24;
mov.b32 {%rs93, %rs94}, %r25;
mov.b32 {%rs95, %rs96}, %r26;
cvt.f32.bf16 %r226, %rs81;
cvt.f32.bf16 %r227, %rs82;
cvt.f32.bf16 %r228, %rs83;
cvt.f32.bf16 %r229, %rs84;
cvt.f32.bf16 %r230, %rs85;
cvt.f32.bf16 %r231, %rs86;
cvt.f32.bf16 %r232, %rs87;
cvt.f32.bf16 %r233, %rs88;
cvt.f32.bf16 %r234, %rs89;
cvt.f32.bf16 %r235, %rs90;
cvt.f32.bf16 %r236, %rs91;
cvt.f32.bf16 %r237, %rs92;
cvt.f32.bf16 %r238, %rs93;
cvt.f32.bf16 %r239, %rs94;
cvt.f32.bf16 %r240, %rs95;
cvt.f32.bf16 %r241, %rs96;
mul.lo.s32 %r242, %r186, 384;
mul.wide.u32 %rd38, %r242, 2;
add.s64 %rd39, %rd22, %rd38;
add.s64 %rd40, %rd39, %rd36;
add.s64 %rd8, %rd40, %rd32;
add.s64 %rd9, %rd8, 16;
mov.u32 %r27, %r5;
mov.u32 %r28, %r5;
mov.u32 %r29, %r5;
mov.u32 %r30, %r5;
@%p1 ld.global.v4.b32 { %r27, %r28, %r29, %r30 }, [ %rd8 + 0 ];
mov.b32 {%rs97, %rs98}, %r27;
mov.b32 {%rs99, %rs100}, %r28;
mov.b32 {%rs101, %rs102}, %r29;
mov.b32 {%rs103, %rs104}, %r30;
mov.u32 %r31, %r5;
mov.u32 %r32, %r5;
mov.u32 %r33, %r5;
mov.u32 %r34, %r5;
@%p1 ld.global.v4.b32 { %r31, %r32, %r33, %r34 }, [ %rd9 + 0 ];
mov.b32 {%rs105, %rs106}, %r31;
mov.b32 {%rs107, %rs108}, %r32;
mov.b32 {%rs109, %rs110}, %r33;
mov.b32 {%rs111, %rs112}, %r34;
cvt.f32.bf16 %r243, %rs97;
cvt.f32.bf16 %r244, %rs98;
cvt.f32.bf16 %r245, %rs99;
cvt.f32.bf16 %r246, %rs100;
cvt.f32.bf16 %r247, %rs101;
cvt.f32.bf16 %r248, %rs102;
cvt.f32.bf16 %r249, %rs103;
cvt.f32.bf16 %r250, %rs104;
cvt.f32.bf16 %r251, %rs105;
cvt.f32.bf16 %r252, %rs106;
cvt.f32.bf16 %r253, %rs107;
cvt.f32.bf16 %r254, %rs108;
cvt.f32.bf16 %r255, %rs109;
cvt.f32.bf16 %r256, %rs110;
cvt.f32.bf16 %r257, %rs111;
cvt.f32.bf16 %r258, %rs112;
mul.f32 %r259, %r227, %r227;
"# else r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd23 + 0 ];
mov.b32 {%rs122, %rs123}, %r23;
mov.b32 {%rs124, %rs125}, %r24;
mov.b32 {%rs126, %rs127}, %r25;
mov.b32 {%rs128, %rs129}, %r26;
cvt.f32.bf16 %r267, %rs114;
cvt.f32.bf16 %r268, %rs115;
cvt.f32.bf16 %r269, %rs116;
cvt.f32.bf16 %r270, %rs117;
cvt.f32.bf16 %r271, %rs118;
cvt.f32.bf16 %r272, %rs119;
cvt.f32.bf16 %r273, %rs120;
cvt.f32.bf16 %r274, %rs121;
cvt.f32.bf16 %r275, %rs122;
cvt.f32.bf16 %r276, %rs123;
cvt.f32.bf16 %r277, %rs124;
cvt.f32.bf16 %r278, %rs125;
cvt.f32.bf16 %r279, %rs126;
cvt.f32.bf16 %r280, %rs127;
cvt.f32.bf16 %r281, %rs128;
cvt.f32.bf16 %r282, %rs129;
mul.f32 %r283, %r251, %r251;
"#) ++
  r#"$L__tmp1:
"# ++
  (if vector then r#"fma.rn.f32 %r260, %r226, %r226, %r259;
fma.rn.f32 %r261, %r228, %r228, %r260;
fma.rn.f32 %r262, %r229, %r229, %r261;
fma.rn.f32 %r263, %r230, %r230, %r262;
fma.rn.f32 %r264, %r231, %r231, %r263;
fma.rn.f32 %r265, %r232, %r232, %r264;
fma.rn.f32 %r266, %r233, %r233, %r265;
fma.rn.f32 %r267, %r234, %r234, %r266;
fma.rn.f32 %r268, %r235, %r235, %r267;
fma.rn.f32 %r269, %r236, %r236, %r268;
fma.rn.f32 %r270, %r237, %r237, %r269;
fma.rn.f32 %r271, %r238, %r238, %r270;
fma.rn.f32 %r272, %r239, %r239, %r271;
fma.rn.f32 %r273, %r240, %r240, %r272;
fma.rn.f32 %r274, %r241, %r241, %r273;
"# else r#"fma.rn.f32 %r284, %r250, %r250, %r283;
fma.rn.f32 %r285, %r252, %r252, %r284;
fma.rn.f32 %r286, %r253, %r253, %r285;
fma.rn.f32 %r287, %r254, %r254, %r286;
fma.rn.f32 %r288, %r255, %r255, %r287;
fma.rn.f32 %r289, %r256, %r256, %r288;
fma.rn.f32 %r290, %r257, %r257, %r289;
fma.rn.f32 %r291, %r258, %r258, %r290;
fma.rn.f32 %r292, %r259, %r259, %r291;
fma.rn.f32 %r293, %r260, %r260, %r292;
fma.rn.f32 %r294, %r261, %r261, %r293;
fma.rn.f32 %r295, %r262, %r262, %r294;
fma.rn.f32 %r296, %r263, %r263, %r295;
fma.rn.f32 %r297, %r264, %r264, %r296;
fma.rn.f32 %r298, %r265, %r265, %r297;
"#) ++
  r#"$L__tmp2:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r275, %r274, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r299, %r298, 2, 31, -1;
"#) ++
  r#"$L__tmp3:
"# ++
  (if vector then r#"add.f32 %r276, %r274, %r275;
"# else r#"add.f32 %r300, %r298, %r299;
"#) ++
  r#"$L__tmp4:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r277, %r276, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r301, %r300, 1, 31, -1;
"#) ++
  r#"$L__tmp5:
"# ++
  (if vector then r#"add.f32 %r278, %r276, %r277;
mov.b32 %r279, 0f42800000;
"# else r#"add.f32 %r302, %r300, %r301;
mov.b32 %r303, 0f42800000;
"#) ++
  r#"$L__tmp6:
"# ++
  (if vector then r#"div.full.f32 %r280, %r278, %r279;
add.f32 %r281, %r280, 0f34000000;
rsqrt.approx.ftz.f32 %r282, %r281;
mul.f32 %r283, %r282, %r226;
mul.f32 %r284, %r282, %r227;
mul.f32 %r285, %r282, %r228;
mul.f32 %r286, %r282, %r229;
mul.f32 %r287, %r282, %r230;
mul.f32 %r288, %r282, %r231;
mul.f32 %r289, %r282, %r232;
mul.f32 %r290, %r282, %r233;
mul.f32 %r291, %r282, %r234;
mul.f32 %r292, %r282, %r235;
mul.f32 %r293, %r282, %r236;
mul.f32 %r294, %r282, %r237;
mul.f32 %r295, %r282, %r238;
mul.f32 %r296, %r282, %r239;
mul.f32 %r297, %r282, %r240;
mul.f32 %r298, %r282, %r241;
mul.f32 %r299, %r208, %r244;
mul.f32 %r300, %r209, %r243;
mul.f32 %r301, %r210, %r246;
mul.f32 %r302, %r211, %r245;
mul.f32 %r303, %r212, %r248;
mul.f32 %r304, %r213, %r247;
mul.f32 %r305, %r214, %r250;
mul.f32 %r306, %r215, %r249;
mul.f32 %r307, %r216, %r252;
mul.f32 %r308, %r217, %r251;
mul.f32 %r309, %r218, %r254;
mul.f32 %r310, %r219, %r253;
mul.f32 %r311, %r220, %r256;
mul.f32 %r312, %r221, %r255;
mul.f32 %r313, %r222, %r258;
mul.f32 %r314, %r223, %r257;
fma.rn.f32 %r315, %r192, %r243, %r299;
fma.rn.f32 %r316, %r193, %r244, %r300;
fma.rn.f32 %r317, %r194, %r245, %r301;
fma.rn.f32 %r318, %r195, %r246, %r302;
fma.rn.f32 %r319, %r196, %r247, %r303;
fma.rn.f32 %r320, %r197, %r248, %r304;
fma.rn.f32 %r321, %r198, %r249, %r305;
fma.rn.f32 %r322, %r199, %r250, %r306;
fma.rn.f32 %r323, %r200, %r251, %r307;
fma.rn.f32 %r324, %r201, %r252, %r308;
fma.rn.f32 %r325, %r202, %r253, %r309;
fma.rn.f32 %r326, %r203, %r254, %r310;
fma.rn.f32 %r327, %r204, %r255, %r311;
fma.rn.f32 %r328, %r205, %r256, %r312;
fma.rn.f32 %r329, %r206, %r257, %r313;
fma.rn.f32 %r330, %r207, %r258, %r314;
mul.f32 %r331, %r316, %r284;
"# else r#"div.full.f32 %r304, %r302, %r303;
add.f32 %r305, %r304, 0f34000000;
rsqrt.approx.ftz.f32 %r306, %r305;
mul.f32 %r307, %r306, %r250;
mul.f32 %r308, %r306, %r251;
mul.f32 %r309, %r306, %r252;
mul.f32 %r310, %r306, %r253;
mul.f32 %r311, %r306, %r254;
mul.f32 %r312, %r306, %r255;
mul.f32 %r313, %r306, %r256;
mul.f32 %r314, %r306, %r257;
mul.f32 %r315, %r306, %r258;
mul.f32 %r316, %r306, %r259;
mul.f32 %r317, %r306, %r260;
mul.f32 %r318, %r306, %r261;
mul.f32 %r319, %r306, %r262;
mul.f32 %r320, %r306, %r263;
mul.f32 %r321, %r306, %r264;
mul.f32 %r322, %r306, %r265;
mul.f32 %r323, %r232, %r268;
mul.f32 %r324, %r233, %r267;
mul.f32 %r325, %r234, %r270;
mul.f32 %r326, %r235, %r269;
mul.f32 %r327, %r236, %r272;
mul.f32 %r328, %r237, %r271;
mul.f32 %r329, %r238, %r274;
mul.f32 %r330, %r239, %r273;
mul.f32 %r331, %r240, %r276;
mul.f32 %r332, %r241, %r275;
mul.f32 %r333, %r242, %r278;
mul.f32 %r334, %r243, %r277;
mul.f32 %r335, %r244, %r280;
mul.f32 %r336, %r245, %r279;
mul.f32 %r337, %r246, %r282;
mul.f32 %r338, %r247, %r281;
fma.rn.f32 %r339, %r190, %r267, %r323;
fma.rn.f32 %r340, %r191, %r268, %r324;
fma.rn.f32 %r341, %r192, %r269, %r325;
fma.rn.f32 %r342, %r193, %r270, %r326;
fma.rn.f32 %r343, %r194, %r271, %r327;
fma.rn.f32 %r344, %r195, %r272, %r328;
fma.rn.f32 %r345, %r196, %r273, %r329;
fma.rn.f32 %r346, %r197, %r274, %r330;
fma.rn.f32 %r347, %r198, %r275, %r331;
fma.rn.f32 %r348, %r199, %r276, %r332;
fma.rn.f32 %r349, %r200, %r277, %r333;
fma.rn.f32 %r350, %r201, %r278, %r334;
fma.rn.f32 %r351, %r202, %r279, %r335;
fma.rn.f32 %r352, %r203, %r280, %r336;
fma.rn.f32 %r353, %r204, %r281, %r337;
fma.rn.f32 %r354, %r205, %r282, %r338;
mul.f32 %r355, %r340, %r308;
"#) ++
  r#"$L__tmp7:
"# ++
  (if vector then r#"fma.rn.f32 %r332, %r315, %r283, %r331;
fma.rn.f32 %r333, %r317, %r285, %r332;
fma.rn.f32 %r334, %r318, %r286, %r333;
fma.rn.f32 %r335, %r319, %r287, %r334;
fma.rn.f32 %r336, %r320, %r288, %r335;
fma.rn.f32 %r337, %r321, %r289, %r336;
fma.rn.f32 %r338, %r322, %r290, %r337;
fma.rn.f32 %r339, %r323, %r291, %r338;
fma.rn.f32 %r340, %r324, %r292, %r339;
fma.rn.f32 %r341, %r325, %r293, %r340;
fma.rn.f32 %r342, %r326, %r294, %r341;
fma.rn.f32 %r343, %r327, %r295, %r342;
fma.rn.f32 %r344, %r328, %r296, %r343;
fma.rn.f32 %r345, %r329, %r297, %r344;
fma.rn.f32 %r346, %r330, %r298, %r345;
"# else r#"fma.rn.f32 %r356, %r339, %r307, %r355;
fma.rn.f32 %r357, %r341, %r309, %r356;
fma.rn.f32 %r358, %r342, %r310, %r357;
fma.rn.f32 %r359, %r343, %r311, %r358;
fma.rn.f32 %r360, %r344, %r312, %r359;
fma.rn.f32 %r361, %r345, %r313, %r360;
fma.rn.f32 %r362, %r346, %r314, %r361;
fma.rn.f32 %r363, %r347, %r315, %r362;
fma.rn.f32 %r364, %r348, %r316, %r363;
fma.rn.f32 %r365, %r349, %r317, %r364;
fma.rn.f32 %r366, %r350, %r318, %r365;
fma.rn.f32 %r367, %r351, %r319, %r366;
fma.rn.f32 %r368, %r352, %r320, %r367;
fma.rn.f32 %r369, %r353, %r321, %r368;
fma.rn.f32 %r370, %r354, %r322, %r369;
"#) ++
  r#"$L__tmp8:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r347, %r346, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r371, %r370, 2, 31, -1;
"#) ++
  r#"$L__tmp9:
"# ++
  (if vector then r#"add.f32 %r348, %r346, %r347;
"# else r#"add.f32 %r372, %r370, %r371;
"#) ++
  r#"$L__tmp10:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r349, %r348, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r373, %r372, 1, 31, -1;
"#) ++
  r#"$L__tmp11:
"# ++
  (if vector then r#"add.f32 %r350, %r348, %r349;
"# else r#"add.f32 %r374, %r372, %r373;
"#) ++
  r#"$L__tmp12:
"# ++
  (if vector then r#"div.full.f32 %r351, %r350, %r279;
neg.f32 %r352, %r283;
fma.rn.f32 %r353, %r352, %r351, %r315;
neg.f32 %r354, %r284;
fma.rn.f32 %r355, %r354, %r351, %r316;
neg.f32 %r356, %r285;
fma.rn.f32 %r357, %r356, %r351, %r317;
neg.f32 %r358, %r286;
fma.rn.f32 %r359, %r358, %r351, %r318;
neg.f32 %r360, %r287;
fma.rn.f32 %r361, %r360, %r351, %r319;
neg.f32 %r362, %r288;
fma.rn.f32 %r363, %r362, %r351, %r320;
neg.f32 %r364, %r289;
fma.rn.f32 %r365, %r364, %r351, %r321;
neg.f32 %r366, %r290;
fma.rn.f32 %r367, %r366, %r351, %r322;
neg.f32 %r368, %r291;
fma.rn.f32 %r369, %r368, %r351, %r323;
neg.f32 %r370, %r292;
fma.rn.f32 %r371, %r370, %r351, %r324;
neg.f32 %r372, %r293;
fma.rn.f32 %r373, %r372, %r351, %r325;
neg.f32 %r374, %r294;
fma.rn.f32 %r375, %r374, %r351, %r326;
neg.f32 %r376, %r295;
fma.rn.f32 %r377, %r376, %r351, %r327;
neg.f32 %r378, %r296;
fma.rn.f32 %r379, %r378, %r351, %r328;
neg.f32 %r380, %r297;
fma.rn.f32 %r381, %r380, %r351, %r329;
neg.f32 %r382, %r298;
fma.rn.f32 %r383, %r382, %r351, %r330;
mul.f32 %r384, %r282, %r353;
mul.f32 %r385, %r282, %r355;
mul.f32 %r386, %r282, %r357;
mul.f32 %r387, %r282, %r359;
mul.f32 %r388, %r282, %r361;
mul.f32 %r389, %r282, %r363;
mul.f32 %r390, %r282, %r365;
mul.f32 %r391, %r282, %r367;
mul.f32 %r392, %r282, %r369;
mul.f32 %r393, %r282, %r371;
mul.f32 %r394, %r282, %r373;
mul.f32 %r395, %r282, %r375;
mul.f32 %r396, %r282, %r377;
mul.f32 %r397, %r282, %r379;
mul.f32 %r398, %r282, %r381;
mul.f32 %r399, %r282, %r383;
cvt.rn.bf16.f32 %rs113, %r384;
cvt.rn.bf16.f32 %rs114, %r385;
cvt.rn.bf16.f32 %rs115, %r386;
cvt.rn.bf16.f32 %rs116, %r387;
cvt.rn.bf16.f32 %rs117, %r388;
cvt.rn.bf16.f32 %rs118, %r389;
cvt.rn.bf16.f32 %rs119, %r390;
cvt.rn.bf16.f32 %rs120, %r391;
cvt.rn.bf16.f32 %rs121, %r392;
cvt.rn.bf16.f32 %rs122, %r393;
cvt.rn.bf16.f32 %rs123, %r394;
cvt.rn.bf16.f32 %rs124, %r395;
cvt.rn.bf16.f32 %rs125, %r396;
cvt.rn.bf16.f32 %rs126, %r397;
cvt.rn.bf16.f32 %rs127, %r398;
cvt.rn.bf16.f32 %rs128, %r399;
shl.b32 %r400, %r188, 5;
and.b32 %r401, %r184, 896;
shl.b32 %r402, %r182, 2;
and.b32 %r403, %r402, 16;
bfe.s32 %r404, %r182, 6, 1;
and.b32 %r405, %r404, 1056;
or.b32 %r406, %r400, %r401;
xor.b32 %r407, %r406, %r405;
or.b32 %r408, %r407, %r403;
mov.b32 %r409, global_smem;
add.s32 %r410, %r409, %r408;
mov.b32 %r411, {%rs125, %rs127};
mov.b32 %r412, {%rs121, %rs123};
mov.b32 %r413, {%rs117, %rs119};
mov.b32 %r414, {%rs113, %rs115};
st.shared.v4.b32 [%r410], {%r414, %r413, %r412, %r411};
xor.b32 %r415, %r408, 64;
add.s32 %r416, %r409, %r415;
mov.b32 %r417, {%rs126, %rs128};
mov.b32 %r418, {%rs122, %rs124};
mov.b32 %r419, {%rs118, %rs120};
mov.b32 %r420, {%rs114, %rs116};
st.shared.v4.b32 [%r416+2048], {%r420, %r419, %r418, %r417};
"# else r#"div.full.f32 %r375, %r374, %r303;
neg.f32 %r376, %r307;
fma.rn.f32 %r377, %r376, %r375, %r339;
neg.f32 %r378, %r308;
fma.rn.f32 %r379, %r378, %r375, %r340;
neg.f32 %r380, %r309;
fma.rn.f32 %r381, %r380, %r375, %r341;
neg.f32 %r382, %r310;
fma.rn.f32 %r383, %r382, %r375, %r342;
neg.f32 %r384, %r311;
fma.rn.f32 %r385, %r384, %r375, %r343;
neg.f32 %r386, %r312;
fma.rn.f32 %r387, %r386, %r375, %r344;
neg.f32 %r388, %r313;
fma.rn.f32 %r389, %r388, %r375, %r345;
neg.f32 %r390, %r314;
fma.rn.f32 %r391, %r390, %r375, %r346;
neg.f32 %r392, %r315;
fma.rn.f32 %r393, %r392, %r375, %r347;
neg.f32 %r394, %r316;
fma.rn.f32 %r395, %r394, %r375, %r348;
neg.f32 %r396, %r317;
fma.rn.f32 %r397, %r396, %r375, %r349;
neg.f32 %r398, %r318;
fma.rn.f32 %r399, %r398, %r375, %r350;
neg.f32 %r400, %r319;
fma.rn.f32 %r401, %r400, %r375, %r351;
neg.f32 %r402, %r320;
fma.rn.f32 %r403, %r402, %r375, %r352;
neg.f32 %r404, %r321;
fma.rn.f32 %r405, %r404, %r375, %r353;
neg.f32 %r406, %r322;
fma.rn.f32 %r407, %r406, %r375, %r354;
mul.f32 %r408, %r306, %r377;
mul.f32 %r409, %r306, %r379;
mul.f32 %r410, %r306, %r381;
mul.f32 %r411, %r306, %r383;
mul.f32 %r412, %r306, %r385;
mul.f32 %r413, %r306, %r387;
mul.f32 %r414, %r306, %r389;
mul.f32 %r415, %r306, %r391;
mul.f32 %r416, %r306, %r393;
mul.f32 %r417, %r306, %r395;
mul.f32 %r418, %r306, %r397;
mul.f32 %r419, %r306, %r399;
mul.f32 %r420, %r306, %r401;
mul.f32 %r421, %r306, %r403;
mul.f32 %r422, %r306, %r405;
mul.f32 %r423, %r306, %r407;
cvt.rn.bf16.f32 %rs130, %r408;
cvt.rn.bf16.f32 %rs131, %r409;
cvt.rn.bf16.f32 %rs132, %r410;
cvt.rn.bf16.f32 %rs133, %r411;
cvt.rn.bf16.f32 %rs134, %r412;
cvt.rn.bf16.f32 %rs135, %r413;
cvt.rn.bf16.f32 %rs136, %r414;
cvt.rn.bf16.f32 %rs137, %r415;
cvt.rn.bf16.f32 %rs138, %r416;
cvt.rn.bf16.f32 %rs139, %r417;
cvt.rn.bf16.f32 %rs140, %r418;
cvt.rn.bf16.f32 %rs141, %r419;
cvt.rn.bf16.f32 %rs142, %r420;
cvt.rn.bf16.f32 %rs143, %r421;
cvt.rn.bf16.f32 %rs144, %r422;
cvt.rn.bf16.f32 %rs145, %r423;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"shl.b32 %r421, %r182, 9;
and.b32 %r422, %r421, 3072;
and.b32 %r423, %r184, 496;
and.b32 %r424, %r182, 96;
or.b32 %r425, %r422, %r423;
xor.b32 %r426, %r425, %r424;
add.s32 %r427, %r409, %r426;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r428, %r429, %r430, %r431}, [%r427];
mov.b32 {%rs129, %rs130}, %r428;
mov.b32 {%rs131, %rs132}, %r429;
mov.b32 {%rs133, %rs134}, %r430;
mov.b32 {%rs135, %rs136}, %r431;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r432, %r433, %r434, %r435}, [%r427+512];
mov.b32 {%rs137, %rs138}, %r432;
mov.b32 {%rs139, %rs140}, %r433;
mov.b32 {%rs141, %rs142}, %r434;
mov.b32 {%rs143, %rs144}, %r435;
cvt.f32.bf16 %r436, %rs129;
cvt.f32.bf16 %r437, %rs130;
cvt.f32.bf16 %r438, %rs131;
cvt.f32.bf16 %r439, %rs132;
cvt.f32.bf16 %r440, %rs133;
cvt.f32.bf16 %r441, %rs134;
cvt.f32.bf16 %r442, %rs135;
cvt.f32.bf16 %r443, %rs136;
cvt.f32.bf16 %r444, %rs137;
cvt.f32.bf16 %r445, %rs138;
cvt.f32.bf16 %r446, %rs139;
cvt.f32.bf16 %r447, %rs140;
cvt.f32.bf16 %r448, %rs141;
cvt.f32.bf16 %r449, %rs142;
cvt.f32.bf16 %r450, %rs143;
cvt.f32.bf16 %r451, %rs144;
cvt.f32.bf16 %r452, %rs113;
cvt.f32.bf16 %r453, %rs114;
cvt.f32.bf16 %r454, %rs115;
cvt.f32.bf16 %r455, %rs116;
cvt.f32.bf16 %r456, %rs117;
cvt.f32.bf16 %r457, %rs118;
cvt.f32.bf16 %r458, %rs119;
cvt.f32.bf16 %r459, %rs120;
cvt.f32.bf16 %r460, %rs121;
cvt.f32.bf16 %r461, %rs122;
cvt.f32.bf16 %r462, %rs123;
cvt.f32.bf16 %r463, %rs124;
cvt.f32.bf16 %r464, %rs125;
cvt.f32.bf16 %r465, %rs126;
cvt.f32.bf16 %r466, %rs127;
cvt.f32.bf16 %r467, %rs128;
div.full.f32 %r35, %r436, %r18;
div.full.f32 %r36, %r437, %r18;
div.full.f32 %r37, %r438, %r18;
div.full.f32 %r38, %r439, %r18;
div.full.f32 %r39, %r440, %r18;
div.full.f32 %r40, %r441, %r18;
div.full.f32 %r41, %r442, %r18;
div.full.f32 %r42, %r443, %r18;
div.full.f32 %r43, %r444, %r18;
div.full.f32 %r44, %r445, %r18;
div.full.f32 %r45, %r446, %r18;
div.full.f32 %r46, %r447, %r18;
div.full.f32 %r47, %r448, %r18;
div.full.f32 %r48, %r449, %r18;
div.full.f32 %r49, %r450, %r18;
div.full.f32 %r50, %r451, %r18;
div.full.f32 %r51, %r452, %r18;
div.full.f32 %r52, %r453, %r18;
div.full.f32 %r53, %r454, %r18;
div.full.f32 %r54, %r455, %r18;
div.full.f32 %r55, %r456, %r18;
div.full.f32 %r56, %r457, %r18;
div.full.f32 %r57, %r458, %r18;
div.full.f32 %r58, %r459, %r18;
div.full.f32 %r59, %r460, %r18;
div.full.f32 %r60, %r461, %r18;
div.full.f32 %r61, %r462, %r18;
div.full.f32 %r62, %r463, %r18;
div.full.f32 %r63, %r464, %r18;
div.full.f32 %r64, %r465, %r18;
div.full.f32 %r65, %r466, %r18;
div.full.f32 %r66, %r467, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r50, %r49;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r52, %r51;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r54, %r53;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r56, %r55;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r58, %r57;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r60, %r59;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r62, %r61;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r64, %r63;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r66, %r65;
or.b32 %r468, %r189, %r225;
or.b32 %r469, %r190, %r225;
add.s64 %rd41, %rd28, %rd34;
cvt.s64.s32 %rd42, %r468;
add.s64 %rd10, %rd41, %rd42;
mov.b32 %r67, {%rs9, %rs10};
mov.b32 %r68, {%rs11, %rs12};
mov.b32 %r69, {%rs13, %rs14};
mov.b32 %r70, {%rs15, %rs16};
@%p1 st.global.v4.b32 [ %rd10 + 0 ], { %r67, %r68, %r69, %r70 };
@EXTENT_26@
cvt.s64.s32 %rd43, %r470;
add.s64 %rd44, %rd29, %rd43;
cvt.u64.u32 %rd45, %r187;
add.s64 %rd11, %rd44, %rd45;
mov.b32 %r71, {%rs1, %rs2};
mov.b32 %r72, {%rs3, %rs4};
mov.b32 %r73, {%rs5, %rs6};
mov.b32 %r74, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd11 + 0 ], { %r71, %r72, %r73, %r74 };
add.s32 %r471, %r225, 384;
mad.wide.s32 %rd46, %r471, 2, %rd35;
add.s64 %rd12, %rd46, %rd32;
add.s64 %rd13, %rd12, 16;
"# else r#"shl.b32 %r424, %r184, 5;
and.b32 %r425, %r177, 896;
shl.b32 %r426, %r174, 2;
and.b32 %r427, %r426, 16;
bfe.s32 %r428, %r174, 6, 1;
and.b32 %r429, %r428, 1056;
or.b32 %r430, %r424, %r425;
xor.b32 %r431, %r430, %r429;
or.b32 %r432, %r431, %r427;
add.s32 %r433, %r207, %r432;
mov.b32 %r434, {%rs142, %rs144};
mov.b32 %r435, {%rs138, %rs140};
mov.b32 %r436, {%rs134, %rs136};
mov.b32 %r437, {%rs130, %rs132};
st.shared.v4.b32 [%r433], {%r437, %r436, %r435, %r434};
xor.b32 %r438, %r432, 64;
add.s32 %r439, %r207, %r438;
mov.b32 %r440, {%rs143, %rs145};
mov.b32 %r441, {%rs139, %rs141};
mov.b32 %r442, {%rs135, %rs137};
mov.b32 %r443, {%rs131, %rs133};
st.shared.v4.b32 [%r439+2048], {%r443, %r442, %r441, %r440};
bar.sync 0;
shl.b32 %r444, %r174, 9;
and.b32 %r445, %r444, 3072;
shl.b32 %r446, %r176, 4;
or.b32 %r447, %r445, %r446;
xor.b32 %r448, %r447, %r182;
add.s32 %r449, %r207, %r448;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r450, %r451, %r452, %r453}, [%r449];
mov.b32 {%rs146, %rs147}, %r450;
mov.b32 {%rs148, %rs149}, %r451;
mov.b32 {%rs150, %rs151}, %r452;
mov.b32 {%rs152, %rs153}, %r453;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r454, %r455, %r456, %r457}, [%r449+512];
mov.b32 {%rs154, %rs155}, %r454;
mov.b32 {%rs156, %rs157}, %r455;
mov.b32 {%rs158, %rs159}, %r456;
mov.b32 {%rs160, %rs161}, %r457;
cvt.f32.bf16 %r458, %rs146;
cvt.f32.bf16 %r459, %rs147;
cvt.f32.bf16 %r460, %rs148;
cvt.f32.bf16 %r461, %rs149;
cvt.f32.bf16 %r462, %rs150;
cvt.f32.bf16 %r463, %rs151;
cvt.f32.bf16 %r464, %rs152;
cvt.f32.bf16 %r465, %rs153;
cvt.f32.bf16 %r466, %rs154;
cvt.f32.bf16 %r467, %rs155;
cvt.f32.bf16 %r468, %rs156;
cvt.f32.bf16 %r469, %rs157;
cvt.f32.bf16 %r470, %rs158;
cvt.f32.bf16 %r471, %rs159;
cvt.f32.bf16 %r472, %rs160;
cvt.f32.bf16 %r473, %rs161;
cvt.f32.bf16 %r474, %rs130;
cvt.f32.bf16 %r475, %rs131;
cvt.f32.bf16 %r476, %rs132;
cvt.f32.bf16 %r477, %rs133;
cvt.f32.bf16 %r478, %rs134;
cvt.f32.bf16 %r479, %rs135;
cvt.f32.bf16 %r480, %rs136;
cvt.f32.bf16 %r481, %rs137;
cvt.f32.bf16 %r482, %rs138;
cvt.f32.bf16 %r483, %rs139;
cvt.f32.bf16 %r484, %rs140;
cvt.f32.bf16 %r485, %rs141;
cvt.f32.bf16 %r486, %rs142;
cvt.f32.bf16 %r487, %rs143;
cvt.f32.bf16 %r488, %rs144;
cvt.f32.bf16 %r489, %rs145;
div.full.f32 %r27, %r458, %r10;
div.full.f32 %r28, %r459, %r10;
div.full.f32 %r29, %r460, %r10;
div.full.f32 %r30, %r461, %r10;
div.full.f32 %r31, %r462, %r10;
div.full.f32 %r32, %r463, %r10;
div.full.f32 %r33, %r464, %r10;
div.full.f32 %r34, %r465, %r10;
div.full.f32 %r35, %r466, %r10;
div.full.f32 %r36, %r467, %r10;
div.full.f32 %r37, %r468, %r10;
div.full.f32 %r38, %r469, %r10;
div.full.f32 %r39, %r470, %r10;
div.full.f32 %r40, %r471, %r10;
div.full.f32 %r41, %r472, %r10;
div.full.f32 %r42, %r473, %r10;
div.full.f32 %r43, %r474, %r10;
div.full.f32 %r44, %r475, %r10;
div.full.f32 %r45, %r476, %r10;
div.full.f32 %r46, %r477, %r10;
div.full.f32 %r47, %r478, %r10;
div.full.f32 %r48, %r479, %r10;
div.full.f32 %r49, %r480, %r10;
div.full.f32 %r50, %r481, %r10;
div.full.f32 %r51, %r482, %r10;
div.full.f32 %r52, %r483, %r10;
div.full.f32 %r53, %r484, %r10;
div.full.f32 %r54, %r485, %r10;
div.full.f32 %r55, %r486, %r10;
div.full.f32 %r56, %r487, %r10;
div.full.f32 %r57, %r488, %r10;
div.full.f32 %r58, %r489, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r28, %r27;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r30, %r29;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r32, %r31;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r34, %r33;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r50, %r49;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r52, %r51;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r54, %r53;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r56, %r55;
cvt.rn.satfinite.e4m3x2.f32 %rs33, %r58, %r57;
or.b32 %r490, %r185, %r249;
or.b32 %r491, %r186, %r249;
add.s64 %rd58, %rd42, %rd51;
cvt.s64.s32 %rd59, %r490;
add.s64 %rd24, %rd58, %rd59;
mov.b32 %r59, {%rs26, %rs27};
mov.b32 %r60, {%rs28, %rs29};
mov.b32 %r61, {%rs30, %rs31};
mov.b32 %r62, {%rs32, %rs33};
@%p1 st.global.v4.b32 [ %rd24 + 0 ], { %r59, %r60, %r61, %r62 };
@EXTENT_22@
cvt.s64.s32 %rd60, %r492;
add.s64 %rd61, %rd43, %rd60;
cvt.u64.u32 %rd62, %r181;
add.s64 %rd25, %rd61, %rd62;
mov.b32 %r63, {%rs18, %rs19};
mov.b32 %r64, {%rs20, %rs21};
mov.b32 %r65, {%rs22, %rs23};
mov.b32 %r66, {%rs24, %rs25};
@%p1 st.global.v4.b32 [ %rd25 + 0 ], { %r63, %r64, %r65, %r66 };
add.s32 %r493, %r249, 384;
mad.wide.s32 %rd63, %r493, 2, %rd52;
add.s64 %rd26, %rd63, %rd45;
add.s64 %rd27, %rd26, 16;
mov.u32 %r67, %r5;
mov.u32 %r68, %r5;
mov.u32 %r69, %r5;
mov.u32 %r70, %r5;
@%p1 ld.global.v4.b32 { %r67, %r68, %r69, %r70 }, [ %rd26 + 0 ];
mov.b32 {%rs162, %rs163}, %r67;
mov.b32 {%rs164, %rs165}, %r68;
mov.b32 {%rs166, %rs167}, %r69;
mov.b32 {%rs168, %rs169}, %r70;
mov.u32 %r71, %r5;
mov.u32 %r72, %r5;
mov.u32 %r73, %r5;
mov.u32 %r74, %r5;
@%p1 ld.global.v4.b32 { %r71, %r72, %r73, %r74 }, [ %rd27 + 0 ];
mov.b32 {%rs170, %rs171}, %r71;
mov.b32 {%rs172, %rs173}, %r72;
mov.b32 {%rs174, %rs175}, %r73;
mov.b32 {%rs176, %rs177}, %r74;
cvt.f32.bf16 %r494, %rs162;
cvt.f32.bf16 %r495, %rs163;
cvt.f32.bf16 %r496, %rs164;
cvt.f32.bf16 %r497, %rs165;
cvt.f32.bf16 %r498, %rs166;
cvt.f32.bf16 %r499, %rs167;
cvt.f32.bf16 %r500, %rs168;
cvt.f32.bf16 %r501, %rs169;
cvt.f32.bf16 %r502, %rs170;
cvt.f32.bf16 %r503, %rs171;
cvt.f32.bf16 %r504, %rs172;
cvt.f32.bf16 %r505, %rs173;
cvt.f32.bf16 %r506, %rs174;
cvt.f32.bf16 %r507, %rs175;
cvt.f32.bf16 %r508, %rs176;
cvt.f32.bf16 %r509, %rs177;
add.s64 %rd64, %rd37, %rd55;
add.s64 %rd65, %rd64, %rd53;
add.s64 %rd28, %rd65, %rd45;
add.s64 %rd29, %rd28, 16;
"#) ++
  r#"mov.u32 %r75, %r5;
mov.u32 %r76, %r5;
mov.u32 %r77, %r5;
mov.u32 %r78, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd12 + 0 ];
mov.b32 {%rs145, %rs146}, %r75;
mov.b32 {%rs147, %rs148}, %r76;
mov.b32 {%rs149, %rs150}, %r77;
mov.b32 {%rs151, %rs152}, %r78;
"# else r#"@%p1 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd28 + 0 ];
"#) ++
  r#"mov.u32 %r79, %r5;
mov.u32 %r80, %r5;
mov.u32 %r81, %r5;
mov.u32 %r82, %r5;
"# ++
  (if vector then r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd13 + 0 ];
mov.b32 {%rs153, %rs154}, %r79;
mov.b32 {%rs155, %rs156}, %r80;
mov.b32 {%rs157, %rs158}, %r81;
mov.b32 {%rs159, %rs160}, %r82;
cvt.f32.bf16 %r472, %rs145;
cvt.f32.bf16 %r473, %rs146;
cvt.f32.bf16 %r474, %rs147;
cvt.f32.bf16 %r475, %rs148;
cvt.f32.bf16 %r476, %rs149;
cvt.f32.bf16 %r477, %rs150;
cvt.f32.bf16 %r478, %rs151;
cvt.f32.bf16 %r479, %rs152;
cvt.f32.bf16 %r480, %rs153;
cvt.f32.bf16 %r481, %rs154;
cvt.f32.bf16 %r482, %rs155;
cvt.f32.bf16 %r483, %rs156;
cvt.f32.bf16 %r484, %rs157;
cvt.f32.bf16 %r485, %rs158;
cvt.f32.bf16 %r486, %rs159;
cvt.f32.bf16 %r487, %rs160;
add.s64 %rd47, %rd23, %rd38;
add.s64 %rd48, %rd47, %rd36;
add.s64 %rd14, %rd48, %rd32;
add.s64 %rd15, %rd14, 16;
mov.u32 %r83, %r5;
mov.u32 %r84, %r5;
mov.u32 %r85, %r5;
mov.u32 %r86, %r5;
@%p1 ld.global.v4.b32 { %r83, %r84, %r85, %r86 }, [ %rd14 + 0 ];
mov.u32 %r87, %r5;
mov.u32 %r88, %r5;
mov.u32 %r89, %r5;
mov.u32 %r90, %r5;
@%p1 ld.global.v4.b32 { %r87, %r88, %r89, %r90 }, [ %rd15 + 0 ];
mul.f32 %r488, %r473, %r473;
"# else r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd29 + 0 ];
mul.f32 %r510, %r495, %r495;
"#) ++
  r#"$L__tmp13:
"# ++
  (if vector then r#"fma.rn.f32 %r489, %r472, %r472, %r488;
fma.rn.f32 %r490, %r474, %r474, %r489;
fma.rn.f32 %r491, %r475, %r475, %r490;
fma.rn.f32 %r492, %r476, %r476, %r491;
fma.rn.f32 %r493, %r477, %r477, %r492;
fma.rn.f32 %r494, %r478, %r478, %r493;
fma.rn.f32 %r495, %r479, %r479, %r494;
fma.rn.f32 %r496, %r480, %r480, %r495;
fma.rn.f32 %r497, %r481, %r481, %r496;
fma.rn.f32 %r498, %r482, %r482, %r497;
fma.rn.f32 %r499, %r483, %r483, %r498;
fma.rn.f32 %r500, %r484, %r484, %r499;
fma.rn.f32 %r501, %r485, %r485, %r500;
fma.rn.f32 %r502, %r486, %r486, %r501;
fma.rn.f32 %r503, %r487, %r487, %r502;
"# else r#"fma.rn.f32 %r511, %r494, %r494, %r510;
fma.rn.f32 %r512, %r496, %r496, %r511;
fma.rn.f32 %r513, %r497, %r497, %r512;
fma.rn.f32 %r514, %r498, %r498, %r513;
fma.rn.f32 %r515, %r499, %r499, %r514;
fma.rn.f32 %r516, %r500, %r500, %r515;
fma.rn.f32 %r517, %r501, %r501, %r516;
fma.rn.f32 %r518, %r502, %r502, %r517;
fma.rn.f32 %r519, %r503, %r503, %r518;
fma.rn.f32 %r520, %r504, %r504, %r519;
fma.rn.f32 %r521, %r505, %r505, %r520;
fma.rn.f32 %r522, %r506, %r506, %r521;
fma.rn.f32 %r523, %r507, %r507, %r522;
fma.rn.f32 %r524, %r508, %r508, %r523;
fma.rn.f32 %r525, %r509, %r509, %r524;
"#) ++
  r#"$L__tmp14:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r504, %r503, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r526, %r525, 2, 31, -1;
"#) ++
  r#"$L__tmp15:
"# ++
  (if vector then r#"add.f32 %r505, %r503, %r504;
"# else r#"add.f32 %r527, %r525, %r526;
"#) ++
  r#"$L__tmp16:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r506, %r505, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r528, %r527, 1, 31, -1;
"#) ++
  r#"$L__tmp17:
"# ++
  (if vector then r#"add.f32 %r507, %r505, %r506;
"# else r#"add.f32 %r529, %r527, %r528;
"#) ++
  r#"$L__tmp18:
"# ++
  (if vector then r#"div.full.f32 %r508, %r507, %r279;
add.f32 %r509, %r508, 0f34000000;
rsqrt.approx.ftz.f32 %r510, %r509;
mov.b32 {%rs161, %rs162}, %r90;
cvt.f32.bf16 %r511, %rs162;
cvt.f32.bf16 %r512, %rs161;
mov.b32 {%rs163, %rs164}, %r89;
cvt.f32.bf16 %r513, %rs164;
cvt.f32.bf16 %r514, %rs163;
mov.b32 {%rs165, %rs166}, %r88;
cvt.f32.bf16 %r515, %rs166;
cvt.f32.bf16 %r516, %rs165;
mov.b32 {%rs167, %rs168}, %r87;
cvt.f32.bf16 %r517, %rs168;
cvt.f32.bf16 %r518, %rs167;
mov.b32 {%rs169, %rs170}, %r86;
cvt.f32.bf16 %r519, %rs170;
cvt.f32.bf16 %r520, %rs169;
mov.b32 {%rs171, %rs172}, %r85;
cvt.f32.bf16 %r521, %rs172;
cvt.f32.bf16 %r522, %rs171;
mov.b32 {%rs173, %rs174}, %r84;
cvt.f32.bf16 %r523, %rs174;
cvt.f32.bf16 %r524, %rs173;
mov.b32 {%rs175, %rs176}, %r83;
cvt.f32.bf16 %r525, %rs176;
cvt.f32.bf16 %r526, %rs175;
mul.f32 %r527, %r510, %r472;
mul.f32 %r528, %r510, %r473;
mul.f32 %r529, %r510, %r474;
mul.f32 %r530, %r510, %r475;
mul.f32 %r531, %r510, %r476;
mul.f32 %r532, %r510, %r477;
mul.f32 %r533, %r510, %r478;
mul.f32 %r534, %r510, %r479;
mul.f32 %r535, %r510, %r480;
mul.f32 %r536, %r510, %r481;
mul.f32 %r537, %r510, %r482;
mul.f32 %r538, %r510, %r483;
mul.f32 %r539, %r510, %r484;
mul.f32 %r540, %r510, %r485;
mul.f32 %r541, %r510, %r486;
mul.f32 %r542, %r510, %r487;
mul.f32 %r543, %r208, %r525;
mul.f32 %r544, %r209, %r526;
mul.f32 %r545, %r210, %r523;
mul.f32 %r546, %r211, %r524;
mul.f32 %r547, %r212, %r521;
mul.f32 %r548, %r213, %r522;
mul.f32 %r549, %r214, %r519;
mul.f32 %r550, %r215, %r520;
mul.f32 %r551, %r216, %r517;
mul.f32 %r552, %r217, %r518;
mul.f32 %r553, %r218, %r515;
mul.f32 %r554, %r219, %r516;
mul.f32 %r555, %r220, %r513;
mul.f32 %r556, %r221, %r514;
mul.f32 %r557, %r222, %r511;
mul.f32 %r558, %r223, %r512;
fma.rn.f32 %r559, %r192, %r526, %r543;
fma.rn.f32 %r560, %r193, %r525, %r544;
fma.rn.f32 %r561, %r194, %r524, %r545;
fma.rn.f32 %r562, %r195, %r523, %r546;
fma.rn.f32 %r563, %r196, %r522, %r547;
fma.rn.f32 %r564, %r197, %r521, %r548;
fma.rn.f32 %r565, %r198, %r520, %r549;
fma.rn.f32 %r566, %r199, %r519, %r550;
fma.rn.f32 %r567, %r200, %r518, %r551;
fma.rn.f32 %r568, %r201, %r517, %r552;
fma.rn.f32 %r569, %r202, %r516, %r553;
fma.rn.f32 %r570, %r203, %r515, %r554;
fma.rn.f32 %r571, %r204, %r514, %r555;
fma.rn.f32 %r572, %r205, %r513, %r556;
fma.rn.f32 %r573, %r206, %r512, %r557;
fma.rn.f32 %r574, %r207, %r511, %r558;
mul.f32 %r575, %r560, %r528;
"# else r#"div.full.f32 %r530, %r529, %r303;
add.f32 %r531, %r530, 0f34000000;
rsqrt.approx.ftz.f32 %r532, %r531;
mov.b32 {%rs178, %rs179}, %r82;
cvt.f32.bf16 %r533, %rs179;
cvt.f32.bf16 %r534, %rs178;
mov.b32 {%rs180, %rs181}, %r81;
cvt.f32.bf16 %r535, %rs181;
cvt.f32.bf16 %r536, %rs180;
mov.b32 {%rs182, %rs183}, %r80;
cvt.f32.bf16 %r537, %rs183;
cvt.f32.bf16 %r538, %rs182;
mov.b32 {%rs184, %rs185}, %r79;
cvt.f32.bf16 %r539, %rs185;
cvt.f32.bf16 %r540, %rs184;
mov.b32 {%rs186, %rs187}, %r78;
cvt.f32.bf16 %r541, %rs187;
cvt.f32.bf16 %r542, %rs186;
mov.b32 {%rs188, %rs189}, %r77;
cvt.f32.bf16 %r543, %rs189;
cvt.f32.bf16 %r544, %rs188;
mov.b32 {%rs190, %rs191}, %r76;
cvt.f32.bf16 %r545, %rs191;
cvt.f32.bf16 %r546, %rs190;
mov.b32 {%rs192, %rs193}, %r75;
cvt.f32.bf16 %r547, %rs193;
cvt.f32.bf16 %r548, %rs192;
mul.f32 %r549, %r532, %r494;
mul.f32 %r550, %r532, %r495;
mul.f32 %r551, %r532, %r496;
mul.f32 %r552, %r532, %r497;
mul.f32 %r553, %r532, %r498;
mul.f32 %r554, %r532, %r499;
mul.f32 %r555, %r532, %r500;
mul.f32 %r556, %r532, %r501;
mul.f32 %r557, %r532, %r502;
mul.f32 %r558, %r532, %r503;
mul.f32 %r559, %r532, %r504;
mul.f32 %r560, %r532, %r505;
mul.f32 %r561, %r532, %r506;
mul.f32 %r562, %r532, %r507;
mul.f32 %r563, %r532, %r508;
mul.f32 %r564, %r532, %r509;
mul.f32 %r565, %r232, %r547;
mul.f32 %r566, %r233, %r548;
mul.f32 %r567, %r234, %r545;
mul.f32 %r568, %r235, %r546;
mul.f32 %r569, %r236, %r543;
mul.f32 %r570, %r237, %r544;
mul.f32 %r571, %r238, %r541;
mul.f32 %r572, %r239, %r542;
mul.f32 %r573, %r240, %r539;
mul.f32 %r574, %r241, %r540;
mul.f32 %r575, %r242, %r537;
mul.f32 %r576, %r243, %r538;
mul.f32 %r577, %r244, %r535;
mul.f32 %r578, %r245, %r536;
mul.f32 %r579, %r246, %r533;
mul.f32 %r580, %r247, %r534;
fma.rn.f32 %r581, %r190, %r548, %r565;
fma.rn.f32 %r582, %r191, %r547, %r566;
fma.rn.f32 %r583, %r192, %r546, %r567;
fma.rn.f32 %r584, %r193, %r545, %r568;
fma.rn.f32 %r585, %r194, %r544, %r569;
fma.rn.f32 %r586, %r195, %r543, %r570;
fma.rn.f32 %r587, %r196, %r542, %r571;
fma.rn.f32 %r588, %r197, %r541, %r572;
fma.rn.f32 %r589, %r198, %r540, %r573;
fma.rn.f32 %r590, %r199, %r539, %r574;
fma.rn.f32 %r591, %r200, %r538, %r575;
fma.rn.f32 %r592, %r201, %r537, %r576;
fma.rn.f32 %r593, %r202, %r536, %r577;
fma.rn.f32 %r594, %r203, %r535, %r578;
fma.rn.f32 %r595, %r204, %r534, %r579;
fma.rn.f32 %r596, %r205, %r533, %r580;
mul.f32 %r597, %r582, %r550;
"#) ++
  r#"$L__tmp19:
"# ++
  (if vector then r#"fma.rn.f32 %r576, %r559, %r527, %r575;
fma.rn.f32 %r577, %r561, %r529, %r576;
fma.rn.f32 %r578, %r562, %r530, %r577;
fma.rn.f32 %r579, %r563, %r531, %r578;
fma.rn.f32 %r580, %r564, %r532, %r579;
fma.rn.f32 %r581, %r565, %r533, %r580;
fma.rn.f32 %r582, %r566, %r534, %r581;
fma.rn.f32 %r583, %r567, %r535, %r582;
fma.rn.f32 %r584, %r568, %r536, %r583;
fma.rn.f32 %r585, %r569, %r537, %r584;
fma.rn.f32 %r586, %r570, %r538, %r585;
fma.rn.f32 %r587, %r571, %r539, %r586;
fma.rn.f32 %r588, %r572, %r540, %r587;
fma.rn.f32 %r589, %r573, %r541, %r588;
fma.rn.f32 %r590, %r574, %r542, %r589;
"# else r#"fma.rn.f32 %r598, %r581, %r549, %r597;
fma.rn.f32 %r599, %r583, %r551, %r598;
fma.rn.f32 %r600, %r584, %r552, %r599;
fma.rn.f32 %r601, %r585, %r553, %r600;
fma.rn.f32 %r602, %r586, %r554, %r601;
fma.rn.f32 %r603, %r587, %r555, %r602;
fma.rn.f32 %r604, %r588, %r556, %r603;
fma.rn.f32 %r605, %r589, %r557, %r604;
fma.rn.f32 %r606, %r590, %r558, %r605;
fma.rn.f32 %r607, %r591, %r559, %r606;
fma.rn.f32 %r608, %r592, %r560, %r607;
fma.rn.f32 %r609, %r593, %r561, %r608;
fma.rn.f32 %r610, %r594, %r562, %r609;
fma.rn.f32 %r611, %r595, %r563, %r610;
fma.rn.f32 %r612, %r596, %r564, %r611;
"#) ++
  r#"$L__tmp20:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r591, %r590, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r613, %r612, 2, 31, -1;
"#) ++
  r#"$L__tmp21:
"# ++
  (if vector then r#"add.f32 %r592, %r590, %r591;
"# else r#"add.f32 %r614, %r612, %r613;
"#) ++
  r#"$L__tmp22:
"# ++
  (if vector then r#"shfl.sync.bfly.b32 %r593, %r592, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r615, %r614, 1, 31, -1;
"#) ++
  r#"$L__tmp23:
"# ++
  (if vector then r#"add.f32 %r594, %r592, %r593;
"# else r#"add.f32 %r616, %r614, %r615;
"#) ++
  r#"$L__tmp24:
"# ++
  (if vector then r#"div.full.f32 %r595, %r594, %r279;
neg.f32 %r596, %r527;
fma.rn.f32 %r597, %r596, %r595, %r559;
neg.f32 %r598, %r528;
fma.rn.f32 %r599, %r598, %r595, %r560;
neg.f32 %r600, %r529;
fma.rn.f32 %r601, %r600, %r595, %r561;
neg.f32 %r602, %r530;
fma.rn.f32 %r603, %r602, %r595, %r562;
neg.f32 %r604, %r531;
fma.rn.f32 %r605, %r604, %r595, %r563;
neg.f32 %r606, %r532;
fma.rn.f32 %r607, %r606, %r595, %r564;
neg.f32 %r608, %r533;
fma.rn.f32 %r609, %r608, %r595, %r565;
neg.f32 %r610, %r534;
fma.rn.f32 %r611, %r610, %r595, %r566;
neg.f32 %r612, %r535;
fma.rn.f32 %r613, %r612, %r595, %r567;
neg.f32 %r614, %r536;
fma.rn.f32 %r615, %r614, %r595, %r568;
neg.f32 %r616, %r537;
fma.rn.f32 %r617, %r616, %r595, %r569;
neg.f32 %r618, %r538;
fma.rn.f32 %r619, %r618, %r595, %r570;
neg.f32 %r620, %r539;
fma.rn.f32 %r621, %r620, %r595, %r571;
neg.f32 %r622, %r540;
fma.rn.f32 %r623, %r622, %r595, %r572;
neg.f32 %r624, %r541;
fma.rn.f32 %r625, %r624, %r595, %r573;
neg.f32 %r626, %r542;
fma.rn.f32 %r627, %r626, %r595, %r574;
mul.f32 %r628, %r510, %r597;
mul.f32 %r629, %r510, %r599;
mul.f32 %r630, %r510, %r601;
mul.f32 %r631, %r510, %r603;
mul.f32 %r632, %r510, %r605;
mul.f32 %r633, %r510, %r607;
mul.f32 %r634, %r510, %r609;
mul.f32 %r635, %r510, %r611;
mul.f32 %r636, %r510, %r613;
mul.f32 %r637, %r510, %r615;
mul.f32 %r638, %r510, %r617;
mul.f32 %r639, %r510, %r619;
mul.f32 %r640, %r510, %r621;
mul.f32 %r641, %r510, %r623;
mul.f32 %r642, %r510, %r625;
mul.f32 %r643, %r510, %r627;
cvt.rn.bf16.f32 %rs177, %r628;
cvt.rn.bf16.f32 %rs178, %r629;
cvt.rn.bf16.f32 %rs179, %r630;
cvt.rn.bf16.f32 %rs180, %r631;
cvt.rn.bf16.f32 %rs181, %r632;
cvt.rn.bf16.f32 %rs182, %r633;
cvt.rn.bf16.f32 %rs183, %r634;
cvt.rn.bf16.f32 %rs184, %r635;
cvt.rn.bf16.f32 %rs185, %r636;
cvt.rn.bf16.f32 %rs186, %r637;
cvt.rn.bf16.f32 %rs187, %r638;
cvt.rn.bf16.f32 %rs188, %r639;
cvt.rn.bf16.f32 %rs189, %r640;
cvt.rn.bf16.f32 %rs190, %r641;
cvt.rn.bf16.f32 %rs191, %r642;
cvt.rn.bf16.f32 %rs192, %r643;
"# else r#"div.full.f32 %r617, %r616, %r303;
neg.f32 %r618, %r549;
fma.rn.f32 %r619, %r618, %r617, %r581;
neg.f32 %r620, %r550;
fma.rn.f32 %r621, %r620, %r617, %r582;
neg.f32 %r622, %r551;
fma.rn.f32 %r623, %r622, %r617, %r583;
neg.f32 %r624, %r552;
fma.rn.f32 %r625, %r624, %r617, %r584;
neg.f32 %r626, %r553;
fma.rn.f32 %r627, %r626, %r617, %r585;
neg.f32 %r628, %r554;
fma.rn.f32 %r629, %r628, %r617, %r586;
neg.f32 %r630, %r555;
fma.rn.f32 %r631, %r630, %r617, %r587;
neg.f32 %r632, %r556;
fma.rn.f32 %r633, %r632, %r617, %r588;
neg.f32 %r634, %r557;
fma.rn.f32 %r635, %r634, %r617, %r589;
neg.f32 %r636, %r558;
fma.rn.f32 %r637, %r636, %r617, %r590;
neg.f32 %r638, %r559;
fma.rn.f32 %r639, %r638, %r617, %r591;
neg.f32 %r640, %r560;
fma.rn.f32 %r641, %r640, %r617, %r592;
neg.f32 %r642, %r561;
fma.rn.f32 %r643, %r642, %r617, %r593;
neg.f32 %r644, %r562;
fma.rn.f32 %r645, %r644, %r617, %r594;
neg.f32 %r646, %r563;
fma.rn.f32 %r647, %r646, %r617, %r595;
neg.f32 %r648, %r564;
fma.rn.f32 %r649, %r648, %r617, %r596;
mul.f32 %r650, %r532, %r619;
mul.f32 %r651, %r532, %r621;
mul.f32 %r652, %r532, %r623;
mul.f32 %r653, %r532, %r625;
mul.f32 %r654, %r532, %r627;
mul.f32 %r655, %r532, %r629;
mul.f32 %r656, %r532, %r631;
mul.f32 %r657, %r532, %r633;
mul.f32 %r658, %r532, %r635;
mul.f32 %r659, %r532, %r637;
mul.f32 %r660, %r532, %r639;
mul.f32 %r661, %r532, %r641;
mul.f32 %r662, %r532, %r643;
mul.f32 %r663, %r532, %r645;
mul.f32 %r664, %r532, %r647;
mul.f32 %r665, %r532, %r649;
cvt.rn.bf16.f32 %rs194, %r650;
cvt.rn.bf16.f32 %rs195, %r651;
cvt.rn.bf16.f32 %rs196, %r652;
cvt.rn.bf16.f32 %rs197, %r653;
cvt.rn.bf16.f32 %rs198, %r654;
cvt.rn.bf16.f32 %rs199, %r655;
cvt.rn.bf16.f32 %rs200, %r656;
cvt.rn.bf16.f32 %rs201, %r657;
cvt.rn.bf16.f32 %rs202, %r658;
cvt.rn.bf16.f32 %rs203, %r659;
cvt.rn.bf16.f32 %rs204, %r660;
cvt.rn.bf16.f32 %rs205, %r661;
cvt.rn.bf16.f32 %rs206, %r662;
cvt.rn.bf16.f32 %rs207, %r663;
cvt.rn.bf16.f32 %rs208, %r664;
cvt.rn.bf16.f32 %rs209, %r665;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"mov.b32 %r644, {%rs189, %rs191};
mov.b32 %r645, {%rs185, %rs187};
mov.b32 %r646, {%rs181, %rs183};
mov.b32 %r647, {%rs177, %rs179};
st.shared.v4.b32 [%r410], {%r647, %r646, %r645, %r644};
mov.b32 %r648, {%rs190, %rs192};
mov.b32 %r649, {%rs186, %rs188};
mov.b32 %r650, {%rs182, %rs184};
mov.b32 %r651, {%rs178, %rs180};
st.shared.v4.b32 [%r416+2048], {%r651, %r650, %r649, %r648};
"# else r#"mov.b32 %r666, {%rs206, %rs208};
mov.b32 %r667, {%rs202, %rs204};
mov.b32 %r668, {%rs198, %rs200};
mov.b32 %r669, {%rs194, %rs196};
st.shared.v4.b32 [%r433], {%r669, %r668, %r667, %r666};
mov.b32 %r670, {%rs207, %rs209};
mov.b32 %r671, {%rs203, %rs205};
mov.b32 %r672, {%rs199, %rs201};
mov.b32 %r673, {%rs195, %rs197};
st.shared.v4.b32 [%r439+2048], {%r673, %r672, %r671, %r670};
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r652, %r653, %r654, %r655}, [%r427];
mov.b32 {%rs193, %rs194}, %r652;
mov.b32 {%rs195, %rs196}, %r653;
mov.b32 {%rs197, %rs198}, %r654;
mov.b32 {%rs199, %rs200}, %r655;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r656, %r657, %r658, %r659}, [%r427+512];
mov.b32 {%rs201, %rs202}, %r656;
mov.b32 {%rs203, %rs204}, %r657;
mov.b32 {%rs205, %rs206}, %r658;
mov.b32 {%rs207, %rs208}, %r659;
cvt.f32.bf16 %r660, %rs193;
cvt.f32.bf16 %r661, %rs194;
cvt.f32.bf16 %r662, %rs195;
cvt.f32.bf16 %r663, %rs196;
cvt.f32.bf16 %r664, %rs197;
cvt.f32.bf16 %r665, %rs198;
cvt.f32.bf16 %r666, %rs199;
cvt.f32.bf16 %r667, %rs200;
cvt.f32.bf16 %r668, %rs201;
cvt.f32.bf16 %r669, %rs202;
cvt.f32.bf16 %r670, %rs203;
cvt.f32.bf16 %r671, %rs204;
cvt.f32.bf16 %r672, %rs205;
cvt.f32.bf16 %r673, %rs206;
cvt.f32.bf16 %r674, %rs207;
cvt.f32.bf16 %r675, %rs208;
cvt.f32.bf16 %r676, %rs177;
cvt.f32.bf16 %r677, %rs178;
cvt.f32.bf16 %r678, %rs179;
cvt.f32.bf16 %r679, %rs180;
cvt.f32.bf16 %r680, %rs181;
cvt.f32.bf16 %r681, %rs182;
cvt.f32.bf16 %r682, %rs183;
cvt.f32.bf16 %r683, %rs184;
cvt.f32.bf16 %r684, %rs185;
cvt.f32.bf16 %r685, %rs186;
cvt.f32.bf16 %r686, %rs187;
cvt.f32.bf16 %r687, %rs188;
cvt.f32.bf16 %r688, %rs189;
cvt.f32.bf16 %r689, %rs190;
cvt.f32.bf16 %r690, %rs191;
cvt.f32.bf16 %r691, %rs192;
div.full.f32 %r91, %r660, %r18;
div.full.f32 %r92, %r661, %r18;
div.full.f32 %r93, %r662, %r18;
div.full.f32 %r94, %r663, %r18;
div.full.f32 %r95, %r664, %r18;
div.full.f32 %r96, %r665, %r18;
div.full.f32 %r97, %r666, %r18;
div.full.f32 %r98, %r667, %r18;
div.full.f32 %r99, %r668, %r18;
div.full.f32 %r100, %r669, %r18;
div.full.f32 %r101, %r670, %r18;
div.full.f32 %r102, %r671, %r18;
div.full.f32 %r103, %r672, %r18;
div.full.f32 %r104, %r673, %r18;
div.full.f32 %r105, %r674, %r18;
div.full.f32 %r106, %r675, %r18;
div.full.f32 %r107, %r676, %r18;
div.full.f32 %r108, %r677, %r18;
div.full.f32 %r109, %r678, %r18;
div.full.f32 %r110, %r679, %r18;
div.full.f32 %r111, %r680, %r18;
div.full.f32 %r112, %r681, %r18;
div.full.f32 %r113, %r682, %r18;
div.full.f32 %r114, %r683, %r18;
div.full.f32 %r115, %r684, %r18;
div.full.f32 %r116, %r685, %r18;
div.full.f32 %r117, %r686, %r18;
div.full.f32 %r118, %r687, %r18;
div.full.f32 %r119, %r688, %r18;
div.full.f32 %r120, %r689, %r18;
div.full.f32 %r121, %r690, %r18;
div.full.f32 %r122, %r691, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs17, %r92, %r91;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r94, %r93;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r96, %r95;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r98, %r97;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r114, %r113;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r116, %r115;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r118, %r117;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r120, %r119;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r122, %r121;
or.b32 %r692, %r471, %r189;
or.b32 %r693, %r471, %r190;
cvt.s64.s32 %rd49, %r692;
add.s64 %rd16, %rd41, %rd49;
mov.b32 %r123, {%rs25, %rs26};
mov.b32 %r124, {%rs27, %rs28};
mov.b32 %r125, {%rs29, %rs30};
mov.b32 %r126, {%rs31, %rs32};
@%p1 st.global.v4.b32 [ %rd16 + 0 ], { %r123, %r124, %r125, %r126 };
@EXTENT_27@
cvt.s64.s32 %rd50, %r694;
add.s64 %rd51, %rd29, %rd50;
add.s64 %rd17, %rd51, %rd45;
mov.b32 %r127, {%rs17, %rs18};
mov.b32 %r128, {%rs19, %rs20};
mov.b32 %r129, {%rs21, %rs22};
mov.b32 %r130, {%rs23, %rs24};
@%p1 st.global.v4.b32 [ %rd17 + 0 ], { %r127, %r128, %r129, %r130 };
add.s64 %rd52, %rd24, %rd38;
add.s64 %rd53, %rd52, %rd36;
add.s64 %rd18, %rd53, %rd32;
add.s64 %rd19, %rd18, 16;
mov.u32 %r131, %r5;
mov.u32 %r132, %r5;
mov.u32 %r133, %r5;
mov.u32 %r134, %r5;
@%p1 ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd18 + 0 ];
mov.b32 {%rs209, %rs210}, %r131;
mov.b32 {%rs211, %rs212}, %r132;
mov.b32 {%rs213, %rs214}, %r133;
mov.b32 {%rs215, %rs216}, %r134;
mov.u32 %r135, %r5;
mov.u32 %r136, %r5;
mov.u32 %r137, %r5;
mov.u32 %r138, %r5;
@%p1 ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd19 + 0 ];
mov.b32 {%rs217, %rs218}, %r135;
mov.b32 {%rs219, %rs220}, %r136;
mov.b32 {%rs221, %rs222}, %r137;
mov.b32 {%rs223, %rs224}, %r138;
"# else r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r674, %r675, %r676, %r677}, [%r449];
mov.b32 {%rs210, %rs211}, %r674;
mov.b32 {%rs212, %rs213}, %r675;
mov.b32 {%rs214, %rs215}, %r676;
mov.b32 {%rs216, %rs217}, %r677;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r678, %r679, %r680, %r681}, [%r449+512];
mov.b32 {%rs218, %rs219}, %r678;
mov.b32 {%rs220, %rs221}, %r679;
mov.b32 {%rs222, %rs223}, %r680;
mov.b32 {%rs224, %rs225}, %r681;
cvt.f32.bf16 %r682, %rs210;
cvt.f32.bf16 %r683, %rs211;
cvt.f32.bf16 %r684, %rs212;
cvt.f32.bf16 %r685, %rs213;
cvt.f32.bf16 %r686, %rs214;
cvt.f32.bf16 %r687, %rs215;
cvt.f32.bf16 %r688, %rs216;
cvt.f32.bf16 %r689, %rs217;
cvt.f32.bf16 %r690, %rs218;
cvt.f32.bf16 %r691, %rs219;
cvt.f32.bf16 %r692, %rs220;
cvt.f32.bf16 %r693, %rs221;
cvt.f32.bf16 %r694, %rs222;
cvt.f32.bf16 %r695, %rs223;
cvt.f32.bf16 %r696, %rs224;
cvt.f32.bf16 %r697, %rs225;
cvt.f32.bf16 %r698, %rs194;
cvt.f32.bf16 %r699, %rs195;
cvt.f32.bf16 %r700, %rs196;
cvt.f32.bf16 %r701, %rs197;
cvt.f32.bf16 %r702, %rs198;
cvt.f32.bf16 %r703, %rs199;
cvt.f32.bf16 %r704, %rs200;
cvt.f32.bf16 %r705, %rs201;
cvt.f32.bf16 %r706, %rs202;
cvt.f32.bf16 %r707, %rs203;
cvt.f32.bf16 %r708, %rs204;
cvt.f32.bf16 %r709, %rs205;
cvt.f32.bf16 %r710, %rs206;
cvt.f32.bf16 %r711, %rs207;
cvt.f32.bf16 %r712, %rs208;
cvt.f32.bf16 %r713, %rs209;
div.full.f32 %r83, %r682, %r10;
div.full.f32 %r84, %r683, %r10;
div.full.f32 %r85, %r684, %r10;
div.full.f32 %r86, %r685, %r10;
div.full.f32 %r87, %r686, %r10;
div.full.f32 %r88, %r687, %r10;
div.full.f32 %r89, %r688, %r10;
div.full.f32 %r90, %r689, %r10;
div.full.f32 %r91, %r690, %r10;
div.full.f32 %r92, %r691, %r10;
div.full.f32 %r93, %r692, %r10;
div.full.f32 %r94, %r693, %r10;
div.full.f32 %r95, %r694, %r10;
div.full.f32 %r96, %r695, %r10;
div.full.f32 %r97, %r696, %r10;
div.full.f32 %r98, %r697, %r10;
div.full.f32 %r99, %r698, %r10;
div.full.f32 %r100, %r699, %r10;
div.full.f32 %r101, %r700, %r10;
div.full.f32 %r102, %r701, %r10;
div.full.f32 %r103, %r702, %r10;
div.full.f32 %r104, %r703, %r10;
div.full.f32 %r105, %r704, %r10;
div.full.f32 %r106, %r705, %r10;
div.full.f32 %r107, %r706, %r10;
div.full.f32 %r108, %r707, %r10;
div.full.f32 %r109, %r708, %r10;
div.full.f32 %r110, %r709, %r10;
div.full.f32 %r111, %r710, %r10;
div.full.f32 %r112, %r711, %r10;
div.full.f32 %r113, %r712, %r10;
div.full.f32 %r114, %r713, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs34, %r84, %r83;
cvt.rn.satfinite.e4m3x2.f32 %rs35, %r86, %r85;
cvt.rn.satfinite.e4m3x2.f32 %rs36, %r88, %r87;
cvt.rn.satfinite.e4m3x2.f32 %rs37, %r90, %r89;
cvt.rn.satfinite.e4m3x2.f32 %rs38, %r92, %r91;
cvt.rn.satfinite.e4m3x2.f32 %rs39, %r94, %r93;
cvt.rn.satfinite.e4m3x2.f32 %rs40, %r96, %r95;
cvt.rn.satfinite.e4m3x2.f32 %rs41, %r98, %r97;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs49, %r114, %r113;
or.b32 %r714, %r493, %r185;
or.b32 %r715, %r493, %r186;
cvt.s64.s32 %rd66, %r714;
add.s64 %rd30, %rd58, %rd66;
mov.b32 %r115, {%rs42, %rs43};
mov.b32 %r116, {%rs44, %rs45};
mov.b32 %r117, {%rs46, %rs47};
mov.b32 %r118, {%rs48, %rs49};
@%p1 st.global.v4.b32 [ %rd30 + 0 ], { %r115, %r116, %r117, %r118 };
@EXTENT_23@
cvt.s64.s32 %rd67, %r716;
add.s64 %rd68, %rd43, %rd67;
add.s64 %rd31, %rd68, %rd62;
mov.b32 %r119, {%rs34, %rs35};
mov.b32 %r120, {%rs36, %rs37};
mov.b32 %r121, {%rs38, %rs39};
mov.b32 %r122, {%rs40, %rs41};
@%p1 st.global.v4.b32 [ %rd31 + 0 ], { %r119, %r120, %r121, %r122 };
add.s64 %rd69, %rd38, %rd55;
add.s64 %rd70, %rd69, %rd53;
add.s64 %rd32, %rd70, %rd45;
add.s64 %rd33, %rd32, 16;
mov.u32 %r123, %r5;
mov.u32 %r124, %r5;
mov.u32 %r125, %r5;
mov.u32 %r126, %r5;
@%p1 ld.global.v4.b32 { %r123, %r124, %r125, %r126 }, [ %rd32 + 0 ];
mov.b32 {%rs226, %rs227}, %r123;
mov.b32 {%rs228, %rs229}, %r124;
mov.b32 {%rs230, %rs231}, %r125;
mov.b32 {%rs232, %rs233}, %r126;
mov.u32 %r127, %r5;
mov.u32 %r128, %r5;
mov.u32 %r129, %r5;
mov.u32 %r130, %r5;
@%p1 ld.global.v4.b32 { %r127, %r128, %r129, %r130 }, [ %rd33 + 0 ];
mov.b32 {%rs234, %rs235}, %r127;
mov.b32 {%rs236, %rs237}, %r128;
mov.b32 {%rs238, %rs239}, %r129;
mov.b32 {%rs240, %rs241}, %r130;
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"mov.b32 %r695, {%rs209, %rs211};
mov.b32 %r696, {%rs213, %rs215};
mov.b32 %r697, {%rs217, %rs219};
mov.b32 %r698, {%rs221, %rs223};
st.shared.v4.b32 [%r410], {%r695, %r696, %r697, %r698};
mov.b32 {_, %rs225}, %r135;
mov.b32 {_, %rs226}, %r137;
mov.b32 {_, %rs227}, %r131;
mov.b32 {_, %rs228}, %r133;
mov.b32 %r699, {%rs228, %rs216};
mov.b32 %r700, {%rs227, %rs212};
mov.b32 %r701, {%rs226, %rs224};
mov.b32 %r702, {%rs225, %rs220};
st.shared.v4.b32 [%r416+2048], {%r700, %r699, %r702, %r701};
"# else r#"mov.b32 %r717, {%rs226, %rs228};
mov.b32 %r718, {%rs230, %rs232};
mov.b32 %r719, {%rs234, %rs236};
mov.b32 %r720, {%rs238, %rs240};
st.shared.v4.b32 [%r433], {%r717, %r718, %r719, %r720};
mov.b32 {_, %rs242}, %r127;
mov.b32 {_, %rs243}, %r129;
mov.b32 {_, %rs244}, %r123;
mov.b32 {_, %rs245}, %r125;
mov.b32 %r721, {%rs245, %rs233};
mov.b32 %r722, {%rs244, %rs229};
mov.b32 %r723, {%rs243, %rs241};
mov.b32 %r724, {%rs242, %rs237};
st.shared.v4.b32 [%r439+2048], {%r722, %r721, %r724, %r723};
"#) ++
  r#"bar.sync 0;
"# ++
  (if vector then r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r703, %r704, %r705, %r706}, [%r427];
mov.b32 {%rs229, %rs230}, %r703;
mov.b32 {%rs231, %rs232}, %r704;
mov.b32 {%rs233, %rs234}, %r705;
mov.b32 {%rs235, %rs236}, %r706;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r707, %r708, %r709, %r710}, [%r427+512];
mov.b32 {%rs237, %rs238}, %r707;
mov.b32 {%rs239, %rs240}, %r708;
mov.b32 {%rs241, %rs242}, %r709;
mov.b32 {%rs243, %rs244}, %r710;
cvt.f32.bf16 %r711, %rs229;
cvt.f32.bf16 %r712, %rs230;
cvt.f32.bf16 %r713, %rs231;
cvt.f32.bf16 %r714, %rs232;
cvt.f32.bf16 %r715, %rs233;
cvt.f32.bf16 %r716, %rs234;
cvt.f32.bf16 %r717, %rs235;
cvt.f32.bf16 %r718, %rs236;
cvt.f32.bf16 %r719, %rs237;
cvt.f32.bf16 %r720, %rs238;
cvt.f32.bf16 %r721, %rs239;
cvt.f32.bf16 %r722, %rs240;
cvt.f32.bf16 %r723, %rs241;
cvt.f32.bf16 %r724, %rs242;
cvt.f32.bf16 %r725, %rs243;
cvt.f32.bf16 %r726, %rs244;
cvt.f32.bf16 %r727, %rs209;
cvt.f32.bf16 %r728, %rs210;
cvt.f32.bf16 %r729, %rs211;
cvt.f32.bf16 %r730, %rs212;
cvt.f32.bf16 %r731, %rs213;
cvt.f32.bf16 %r732, %rs214;
cvt.f32.bf16 %r733, %rs215;
cvt.f32.bf16 %r734, %rs216;
cvt.f32.bf16 %r735, %rs217;
cvt.f32.bf16 %r736, %rs218;
cvt.f32.bf16 %r737, %rs219;
cvt.f32.bf16 %r738, %rs220;
cvt.f32.bf16 %r739, %rs221;
cvt.f32.bf16 %r740, %rs222;
cvt.f32.bf16 %r741, %rs223;
cvt.f32.bf16 %r742, %rs224;
div.full.f32 %r139, %r711, %r18;
div.full.f32 %r140, %r712, %r18;
div.full.f32 %r141, %r713, %r18;
div.full.f32 %r142, %r714, %r18;
div.full.f32 %r143, %r715, %r18;
div.full.f32 %r144, %r716, %r18;
div.full.f32 %r145, %r717, %r18;
div.full.f32 %r146, %r718, %r18;
div.full.f32 %r147, %r719, %r18;
div.full.f32 %r148, %r720, %r18;
div.full.f32 %r149, %r721, %r18;
div.full.f32 %r150, %r722, %r18;
div.full.f32 %r151, %r723, %r18;
div.full.f32 %r152, %r724, %r18;
div.full.f32 %r153, %r725, %r18;
div.full.f32 %r154, %r726, %r18;
div.full.f32 %r155, %r727, %r18;
div.full.f32 %r156, %r728, %r18;
div.full.f32 %r157, %r729, %r18;
div.full.f32 %r158, %r730, %r18;
div.full.f32 %r159, %r731, %r18;
div.full.f32 %r160, %r732, %r18;
div.full.f32 %r161, %r733, %r18;
div.full.f32 %r162, %r734, %r18;
div.full.f32 %r163, %r735, %r18;
div.full.f32 %r164, %r736, %r18;
div.full.f32 %r165, %r737, %r18;
div.full.f32 %r166, %r738, %r18;
div.full.f32 %r167, %r739, %r18;
div.full.f32 %r168, %r740, %r18;
div.full.f32 %r169, %r741, %r18;
div.full.f32 %r170, %r742, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs33, %r140, %r139;
cvt.rn.satfinite.e4m3x2.f32 %rs34, %r142, %r141;
cvt.rn.satfinite.e4m3x2.f32 %rs35, %r144, %r143;
cvt.rn.satfinite.e4m3x2.f32 %rs36, %r146, %r145;
cvt.rn.satfinite.e4m3x2.f32 %rs37, %r148, %r147;
cvt.rn.satfinite.e4m3x2.f32 %rs38, %r150, %r149;
cvt.rn.satfinite.e4m3x2.f32 %rs39, %r152, %r151;
cvt.rn.satfinite.e4m3x2.f32 %rs40, %r154, %r153;
cvt.rn.satfinite.e4m3x2.f32 %rs41, %r156, %r155;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r158, %r157;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r160, %r159;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r162, %r161;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r164, %r163;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r166, %r165;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r168, %r167;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r170, %r169;
add.s32 %r743, %r225, 768;
or.b32 %r744, %r743, %r189;
or.b32 %r745, %r743, %r190;
cvt.s64.s32 %rd54, %r744;
add.s64 %rd20, %rd41, %rd54;
mov.b32 %r171, {%rs41, %rs42};
mov.b32 %r172, {%rs43, %rs44};
mov.b32 %r173, {%rs45, %rs46};
mov.b32 %r174, {%rs47, %rs48};
@%p1 st.global.v4.b32 [ %rd20 + 0 ], { %r171, %r172, %r173, %r174 };
@EXTENT_28@
cvt.s64.s32 %rd55, %r746;
add.s64 %rd56, %rd29, %rd55;
add.s64 %rd21, %rd56, %rd45;
mov.b32 %r175, {%rs33, %rs34};
mov.b32 %r176, {%rs35, %rs36};
mov.b32 %r177, {%rs37, %rs38};
mov.b32 %r178, {%rs39, %rs40};
@%p1 st.global.v4.b32 [ %rd21 + 0 ], { %r175, %r176, %r177, %r178 };
"# else r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r725, %r726, %r727, %r728}, [%r449];
mov.b32 {%rs246, %rs247}, %r725;
mov.b32 {%rs248, %rs249}, %r726;
mov.b32 {%rs250, %rs251}, %r727;
mov.b32 {%rs252, %rs253}, %r728;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r729, %r730, %r731, %r732}, [%r449+512];
mov.b32 {%rs254, %rs255}, %r729;
mov.b32 {%rs256, %rs257}, %r730;
mov.b32 {%rs258, %rs259}, %r731;
mov.b32 {%rs260, %rs261}, %r732;
cvt.f32.bf16 %r733, %rs246;
cvt.f32.bf16 %r734, %rs247;
cvt.f32.bf16 %r735, %rs248;
cvt.f32.bf16 %r736, %rs249;
cvt.f32.bf16 %r737, %rs250;
cvt.f32.bf16 %r738, %rs251;
cvt.f32.bf16 %r739, %rs252;
cvt.f32.bf16 %r740, %rs253;
cvt.f32.bf16 %r741, %rs254;
cvt.f32.bf16 %r742, %rs255;
cvt.f32.bf16 %r743, %rs256;
cvt.f32.bf16 %r744, %rs257;
cvt.f32.bf16 %r745, %rs258;
cvt.f32.bf16 %r746, %rs259;
cvt.f32.bf16 %r747, %rs260;
cvt.f32.bf16 %r748, %rs261;
cvt.f32.bf16 %r749, %rs226;
cvt.f32.bf16 %r750, %rs227;
cvt.f32.bf16 %r751, %rs228;
cvt.f32.bf16 %r752, %rs229;
cvt.f32.bf16 %r753, %rs230;
cvt.f32.bf16 %r754, %rs231;
cvt.f32.bf16 %r755, %rs232;
cvt.f32.bf16 %r756, %rs233;
cvt.f32.bf16 %r757, %rs234;
cvt.f32.bf16 %r758, %rs235;
cvt.f32.bf16 %r759, %rs236;
cvt.f32.bf16 %r760, %rs237;
cvt.f32.bf16 %r761, %rs238;
cvt.f32.bf16 %r762, %rs239;
cvt.f32.bf16 %r763, %rs240;
cvt.f32.bf16 %r764, %rs241;
div.full.f32 %r131, %r733, %r10;
div.full.f32 %r132, %r734, %r10;
div.full.f32 %r133, %r735, %r10;
div.full.f32 %r134, %r736, %r10;
div.full.f32 %r135, %r737, %r10;
div.full.f32 %r136, %r738, %r10;
div.full.f32 %r137, %r739, %r10;
div.full.f32 %r138, %r740, %r10;
div.full.f32 %r139, %r741, %r10;
div.full.f32 %r140, %r742, %r10;
div.full.f32 %r141, %r743, %r10;
div.full.f32 %r142, %r744, %r10;
div.full.f32 %r143, %r745, %r10;
div.full.f32 %r144, %r746, %r10;
div.full.f32 %r145, %r747, %r10;
div.full.f32 %r146, %r748, %r10;
div.full.f32 %r147, %r749, %r10;
div.full.f32 %r148, %r750, %r10;
div.full.f32 %r149, %r751, %r10;
div.full.f32 %r150, %r752, %r10;
div.full.f32 %r151, %r753, %r10;
div.full.f32 %r152, %r754, %r10;
div.full.f32 %r153, %r755, %r10;
div.full.f32 %r154, %r756, %r10;
div.full.f32 %r155, %r757, %r10;
div.full.f32 %r156, %r758, %r10;
div.full.f32 %r157, %r759, %r10;
div.full.f32 %r158, %r760, %r10;
div.full.f32 %r159, %r761, %r10;
div.full.f32 %r160, %r762, %r10;
div.full.f32 %r161, %r763, %r10;
div.full.f32 %r162, %r764, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs50, %r132, %r131;
cvt.rn.satfinite.e4m3x2.f32 %rs51, %r134, %r133;
cvt.rn.satfinite.e4m3x2.f32 %rs52, %r136, %r135;
cvt.rn.satfinite.e4m3x2.f32 %rs53, %r138, %r137;
cvt.rn.satfinite.e4m3x2.f32 %rs54, %r140, %r139;
cvt.rn.satfinite.e4m3x2.f32 %rs55, %r142, %r141;
cvt.rn.satfinite.e4m3x2.f32 %rs56, %r144, %r143;
cvt.rn.satfinite.e4m3x2.f32 %rs57, %r146, %r145;
cvt.rn.satfinite.e4m3x2.f32 %rs58, %r148, %r147;
cvt.rn.satfinite.e4m3x2.f32 %rs59, %r150, %r149;
cvt.rn.satfinite.e4m3x2.f32 %rs60, %r152, %r151;
cvt.rn.satfinite.e4m3x2.f32 %rs61, %r154, %r153;
cvt.rn.satfinite.e4m3x2.f32 %rs62, %r156, %r155;
cvt.rn.satfinite.e4m3x2.f32 %rs63, %r158, %r157;
cvt.rn.satfinite.e4m3x2.f32 %rs64, %r160, %r159;
cvt.rn.satfinite.e4m3x2.f32 %rs65, %r162, %r161;
add.s32 %r765, %r249, 768;
or.b32 %r766, %r765, %r185;
or.b32 %r767, %r765, %r186;
cvt.s64.s32 %rd71, %r766;
add.s64 %rd34, %rd58, %rd71;
mov.b32 %r163, {%rs58, %rs59};
mov.b32 %r164, {%rs60, %rs61};
mov.b32 %r165, {%rs62, %rs63};
mov.b32 %r166, {%rs64, %rs65};
@%p1 st.global.v4.b32 [ %rd34 + 0 ], { %r163, %r164, %r165, %r166 };
@EXTENT_24@
cvt.s64.s32 %rd72, %r768;
add.s64 %rd73, %rd43, %rd72;
add.s64 %rd35, %rd73, %rd62;
mov.b32 %r167, {%rs50, %rs51};
mov.b32 %r168, {%rs52, %rs53};
mov.b32 %r169, {%rs54, %rs55};
mov.b32 %r170, {%rs56, %rs57};
@%p1 st.global.v4.b32 [ %rd35 + 0 ], { %r167, %r168, %r169, %r170 };
"#) ++
  r#"ret;
$L__tmp25:
$L__func_end0:
}
"#

private def forLength (n : Length) (a b c d : String) : String :=
  match n with
  | .t16k => a
  | .t32k => b
  | .t40k => c
  | .t48k => d
  | .validation => ""

private def lines (n : Length) (text : String) : List String :=
  let extent : List (String × String) := [
    ("@EXTENT_0@", forLength n r#"%cst_2 = arith.constant dense<16384> : tensor<64x1xi32, #blocked2>
"# r#"%cst_2 = arith.constant dense<32768> : tensor<64x1xi32, #blocked2>
"# r#"%cst_2 = arith.constant dense<40960> : tensor<64x1xi32, #blocked2>
"# r#"%cst_2 = arith.constant dense<49152> : tensor<64x1xi32, #blocked2>
"#),
    ("@EXTENT_1@", forLength n r#"%cst_5 = arith.constant dense<16384> : tensor<128x1xi32, #blocked2>
"# r#"%cst_5 = arith.constant dense<32768> : tensor<128x1xi32, #blocked2>
"# r#"%cst_5 = arith.constant dense<40960> : tensor<128x1xi32, #blocked2>
"# r#"%cst_5 = arith.constant dense<49152> : tensor<128x1xi32, #blocked2>
"#),
    ("@EXTENT_2@", forLength n r#"%cst_15 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_16 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_17 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%cst_18 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_15 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_16 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_17 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%cst_18 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_15 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_16 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_17 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%cst_18 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_15 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_16 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_17 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
%cst_18 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#),
    ("@EXTENT_3@", forLength n r#"%cst_12 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_13 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_14 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_12 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_13 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_14 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_12 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_13 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_14 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_12 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_13 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%cst_14 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#),
    ("@EXTENT_4@", forLength n r#".reg .b32 %r<902>;
.reg .b64 %rd<89>;
"# r#".reg .b32 %r<902>;
.reg .b64 %rd<89>;
"# r#".reg .b32 %r<903>;
.reg .b64 %rd<91>;
"# r#".reg .b32 %r<903>;
.reg .b64 %rd<91>;
"#),
    ("@EXTENT_5@", forLength n r#"setp.lt.u32 %p1, %r220, 512;
"# r#"setp.lt.u32 %p1, %r220, 1024;
"# r#"setp.lt.u32 %p1, %r220, 1280;
"# r#"setp.lt.u32 %p1, %r220, 1536;
"#),
    ("@EXTENT_6@", forLength n r#"shl.b32 %r549, %r547, 14;
"# r#"shl.b32 %r549, %r547, 15;
"# r#"mul.lo.s32 %r549, %r547, 40960;
"# r#"mul.lo.s32 %r549, %r547, 49152;
"#),
    ("@EXTENT_7@", forLength n r#"shl.b32 %r780, %r779, 14;
"# r#"shl.b32 %r780, %r779, 15;
"# r#"mul.lo.s32 %r780, %r779, 40960;
"# r#"mul.lo.s32 %r780, %r779, 49152;
"#),
    ("@EXTENT_8@", forLength n r#"shl.b32 %r901, %r900, 14;
"# r#"shl.b32 %r901, %r900, 15;
"# r#"mul.lo.s32 %r901, %r900, 40960;
add.s32 %r902, %r901, 2621440;
"# r#"mul.lo.s32 %r901, %r900, 49152;
add.s32 %r902, %r901, 3145728;
"#),
    ("@EXTENT_9@", forLength n "" "" r#"cvt.s64.s32 %rd89, %r902;
add.s64 %rd90, %rd47, %rd89;
"# r#"cvt.s64.s32 %rd89, %r902;
add.s64 %rd90, %rd47, %rd89;
"#),
    ("@EXTENT_10@", forLength n r#"add.s64 %rd39, %rd38, 1048576;
"# r#"add.s64 %rd39, %rd38, 2097152;
"# r#"add.s64 %rd39, %rd90, %rd69;
"# r#"add.s64 %rd39, %rd90, %rd69;
"#),
    ("@EXTENT_11@", forLength n r#".reg .b32 %r<881>;
.reg .b64 %rd<72>;
"# r#".reg .b32 %r<881>;
.reg .b64 %rd<72>;
"# r#".reg .b32 %r<882>;
.reg .b64 %rd<74>;
"# r#".reg .b32 %r<882>;
.reg .b64 %rd<74>;
"#),
    ("@EXTENT_12@", forLength n r#"setp.lt.u32 %p1, %r228, 512;
"# r#"setp.lt.u32 %p1, %r228, 1024;
"# r#"setp.lt.u32 %p1, %r228, 1280;
"# r#"setp.lt.u32 %p1, %r228, 1536;
"#),
    ("@EXTENT_13@", forLength n r#"shl.b32 %r526, %r524, 14;
"# r#"shl.b32 %r526, %r524, 15;
"# r#"mul.lo.s32 %r526, %r524, 40960;
"# r#"mul.lo.s32 %r526, %r524, 49152;
"#),
    ("@EXTENT_14@", forLength n r#"shl.b32 %r757, %r756, 14;
"# r#"shl.b32 %r757, %r756, 15;
"# r#"mul.lo.s32 %r757, %r756, 40960;
"# r#"mul.lo.s32 %r757, %r756, 49152;
"#),
    ("@EXTENT_15@", forLength n r#"shl.b32 %r880, %r879, 14;
"# r#"shl.b32 %r880, %r879, 15;
"# r#"mul.lo.s32 %r880, %r879, 40960;
add.s32 %r881, %r880, 2621440;
"# r#"mul.lo.s32 %r880, %r879, 49152;
add.s32 %r881, %r880, 3145728;
"#),
    ("@EXTENT_16@", forLength n "" "" r#"cvt.s64.s32 %rd72, %r881;
add.s64 %rd73, %rd33, %rd72;
"# r#"cvt.s64.s32 %rd72, %r881;
add.s64 %rd73, %rd33, %rd72;
"#),
    ("@EXTENT_17@", forLength n r#"add.s64 %rd25, %rd24, 1048576;
"# r#"add.s64 %rd25, %rd24, 2097152;
"# r#"add.s64 %rd25, %rd73, %rd52;
"# r#"add.s64 %rd25, %rd73, %rd52;
"#),
    ("@EXTENT_18@", forLength n r#"%cst_0 = arith.constant dense<16384> : tensor<64x1xi32, #blocked>
"# r#"%cst_0 = arith.constant dense<32768> : tensor<64x1xi32, #blocked>
"# r#"%cst_0 = arith.constant dense<40960> : tensor<64x1xi32, #blocked>
"# r#"%cst_0 = arith.constant dense<49152> : tensor<64x1xi32, #blocked>
"#),
    ("@EXTENT_19@", forLength n r#"%cst_8 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_9 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%cst_10 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_8 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_9 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%cst_10 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_8 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_9 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%cst_10 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_8 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_9 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%cst_10 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#),
    ("@EXTENT_20@", forLength n r#"%cst_5 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_6 = arith.constant dense<16384> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_5 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_6 = arith.constant dense<32768> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_5 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_6 = arith.constant dense<40960> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# r#"%cst_5 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_6 = arith.constant dense<49152> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#),
    ("@EXTENT_21@", forLength n r#"setp.lt.u32 %p1, %r172, 512;
"# r#"setp.lt.u32 %p1, %r172, 1024;
"# r#"setp.lt.u32 %p1, %r172, 1280;
"# r#"setp.lt.u32 %p1, %r172, 1536;
"#),
    ("@EXTENT_22@", forLength n r#"shl.b32 %r492, %r491, 14;
"# r#"shl.b32 %r492, %r491, 15;
"# r#"mul.lo.s32 %r492, %r491, 40960;
"# r#"mul.lo.s32 %r492, %r491, 49152;
"#),
    ("@EXTENT_23@", forLength n r#"shl.b32 %r716, %r715, 14;
"# r#"shl.b32 %r716, %r715, 15;
"# r#"mul.lo.s32 %r716, %r715, 40960;
"# r#"mul.lo.s32 %r716, %r715, 49152;
"#),
    ("@EXTENT_24@", forLength n r#"shl.b32 %r768, %r767, 14;
"# r#"shl.b32 %r768, %r767, 15;
"# r#"mul.lo.s32 %r768, %r767, 40960;
"# r#"mul.lo.s32 %r768, %r767, 49152;
"#),
    ("@EXTENT_25@", forLength n r#"setp.lt.u32 %p1, %r180, 512;
"# r#"setp.lt.u32 %p1, %r180, 1024;
"# r#"setp.lt.u32 %p1, %r180, 1280;
"# r#"setp.lt.u32 %p1, %r180, 1536;
"#),
    ("@EXTENT_26@", forLength n r#"shl.b32 %r470, %r469, 14;
"# r#"shl.b32 %r470, %r469, 15;
"# r#"mul.lo.s32 %r470, %r469, 40960;
"# r#"mul.lo.s32 %r470, %r469, 49152;
"#),
    ("@EXTENT_27@", forLength n r#"shl.b32 %r694, %r693, 14;
"# r#"shl.b32 %r694, %r693, 15;
"# r#"mul.lo.s32 %r694, %r693, 40960;
"# r#"mul.lo.s32 %r694, %r693, 49152;
"#),
    ("@EXTENT_28@", forLength n r#"shl.b32 %r746, %r745, 14;
"# r#"shl.b32 %r746, %r745, 15;
"# r#"mul.lo.s32 %r746, %r745, 40960;
"# r#"mul.lo.s32 %r746, %r745, 49152;
"#)
  ]
  let rendered := extent.foldl (fun current (marker, fragment) => current.replace marker fragment) text
  (rendered.splitOn "\n").filter (!·.isEmpty)

structure Artifacts where
  referenceIR : String
  referencePTX : String
  vectorIR : String
  vectorPTX : String
  deriving BEq

def matchesArtifacts (s : Shape) (n : Length) (a : Artifacts) : Bool :=
  let ir := match s with | .paired => pairedTTGIR | .halfValue => halfValueTTGIR
  let ptx := match s with | .paired => pairedPTX | .halfValue => halfValuePTX
  n != .validation &&
    QKVTTIR.normalizeTTIR a.referenceIR == some (lines n (ir false)) &&
    QKVTTIR.normalizeTTIR a.vectorIR == some (lines n (ir true)) &&
    QKVTTIR.normalizePTX a.referencePTX == lines n (ptx false) &&
    QKVTTIR.normalizePTX a.vectorPTX == lines n (ptx true)

structure Certificate (s : Shape) (n : Length) (a : Artifacts) : Type where
  matched : matchesArtifacts s n a = true
  factorFlip : ∀ lo hi : BitVec 16, (lo ^^^ (lo ^^^ hi), hi ^^^ (lo ^^^ hi)) = (hi, lo)
  headDecode : ∀ h : Head, h.val * 1431655766 / 2 ^ 32 = h.val / 3 ∧
    h.val - (h.val * 1431655766 / 2 ^ 32) * 3 = h.val % 3 ∧
    (h.val <<< 6) &&& 64 = (h.val % 2) * 64
  sourceRange : ∀ (t : Fin n.tokens) p h d, qkAddress s t p h d < n.tokens * s.qkPitch ∧
    gradAddress s t h d < n.tokens * 384 ∧ factorAddress s t h d < n.tokens * s.factorPitch
  valueRange : ∀ (t : Fin n.tokens) h d, gradVAddress s t h d < n.tokens * 6 * s.valueWidth
  range : ∀ p : Point s n, rowAddress p < n.tokens * s.features ∧
    transposedAddress p < n.tokens * s.features ∧ n.tokens * s.features < 2 ^ 31
  rowUnique : ∀ p q : Point s n, rowAddress p = rowAddress q → p = q
  transposeUnique : ∀ p q : Point s n, transposedAddress p = transposedAddress q → p = q
  stores : ∀ op read (p : Point s n), writes true op read p = writes false op read p

def certify (s : Shape) (n : Length) (a : Artifacts) : Option (Certificate s n a) :=
  if h : matchesArtifacts s n a = true then
    some ⟨h, reversed_pair_bits, paired_head_lowering, read_addresses_bounded s, value_read_bounded s,
      addresses_bounded, row_address_unique, transpose_address_unique, both_layouts_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward.Narrow
