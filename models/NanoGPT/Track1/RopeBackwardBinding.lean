import models.NanoGPT.Track1.RopeBackward
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Normalized compiler output for the wide QKV backward. The differences
cover predicated K loads and the vector load and register permutation of
the rotary factor. The four training lengths have separate bounds. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward
open RopeShift (Length Head)

private def irTemplate (trimmed : Bool) : String :=
  r#"#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
"# ++
  (if trimmed then r#"#blocked2 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
"# else r#"#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 8], order = [0, 1]}>
"#) ++
  r#"module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_qkv_norm_rope_pack_fp8_backward_kernel(%grad_q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_k: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_v: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %qk: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %factor2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %grad_row: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_transposed: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %grad_scale_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<128> : tensor<128x1xi32, #blocked>
%cst_0 = arith.constant dense<@TOKENS@> : tensor<128x1xi32, #blocked>
"# ++
  (if trimmed then r#"%cst_1 = arith.constant dense<64> : tensor<1x128xi32, #blocked1>
%cst_2 = arith.constant dense<1> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_3 = arith.constant dense<@LAST@> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_4 = arith.constant dense<@LAST@> : tensor<32x1xi32, #blocked1>
"# else r#"%cst_1 = arith.constant dense<1> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_2 = arith.constant dense<@LAST@> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_3 = arith.constant dense<@LAST@> : tensor<32x1xi32, #blocked1>
%cst_4 = arith.constant dense<64> : tensor<1x128xi32, #blocked1>
"#) ++
  r#"%cst_5 = arith.constant dense<768> : tensor<32x1xi32, #blocked1>
%cst_6 = arith.constant dense<2304> : tensor<32x1xi32, #blocked1>
%cst_7 = arith.constant dense<0> : tensor<32x1xi32, #blocked1>
"# ++
  (if trimmed then r#"%cst_8 = arith.constant dense<128> : tensor<32x1xi32, #blocked1>
%cst_9 = arith.constant dense<128> : tensor<1x128xi32, #blocked1>
%cst_10 = arith.constant dense<@TOKENS@> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_11 = arith.constant dense<@TOKENS@> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_12 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #blocked1>
%cst_13 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_14 = arith.constant dense<1.280000e+02> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# else r#"%cst_8 = arith.constant dense<128> : tensor<32x1xi32, #blocked2>
%cst_9 = arith.constant dense<128> : tensor<32x1xi32, #blocked1>
%cst_10 = arith.constant dense<1> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%cst_11 = arith.constant dense<128> : tensor<1x128xi32, #blocked2>
%cst_12 = arith.constant dense<128> : tensor<1x128xi32, #blocked1>
%cst_13 = arith.constant dense<@TOKENS@> : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%cst_14 = arith.constant dense<@TOKENS@> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%cst_15 = arith.constant dense<@TOKENS@> : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_16 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #blocked1>
%cst_17 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #blocked2>
%cst_18 = arith.constant dense<1.1920929E-7> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%cst_19 = arith.constant dense<1.280000e+02> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"#) ++
  r#"%logical_head = tt.get_program_id x : i32
%token = tt.get_program_id y : i32
"# ++
  (if trimmed then r#"%token_15 = arith.muli %token, 32 : i32
%token_16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_17 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_18 = tt.splat %token_15 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_19 = tt.splat %token_15 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_20 = arith.addi %token_18, %token_16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_21 = arith.addi %token_19, %token_17 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_d = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask = arith.cmpi slt, %token_20, %cst_11 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_23 = arith.cmpi slt, %token_21, %cst_10 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"# else r#"%token_20 = arith.muli %token, 32 : i32
%token_21 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_23 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_24 = tt.splat %token_20 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_25 = tt.splat %token_20 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_26 = tt.splat %token_20 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%token_27 = arith.addi %token_24, %token_21 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_28 = arith.addi %token_25, %token_22 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_29 = arith.addi %token_26, %token_23 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_d = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offs_d_30 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_d_31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%token_mask = arith.cmpi slt, %token_27, %cst_15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%token_mask_32 = arith.cmpi slt, %token_28, %cst_14 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
%token_mask_33 = arith.cmpi slt, %token_29, %cst_13 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
"#) ++
  r#"%qk_mask = tt.expand_dims %token_mask {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi1, #blocked1>
"# ++
  (if trimmed then r#"%qk_mask_24 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%qk_mask_25 = arith.cmpi slt, %qk_mask_24, %cst_9 : tensor<1x128xi32, #blocked1>
%qk_mask_26 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%qk_mask_27 = tt.broadcast %qk_mask_25 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%qk_mask_28 = arith.andi %qk_mask_26, %qk_mask_27 : tensor<32x128xi1, #blocked1>
%f1 = tt.expand_dims %token_20 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_29 = arith.muli %f1, %cst_8 : tensor<32x1xi32, #blocked1>
%f1_30 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_31 = tt.addptr %f1_30, %f1_29 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_32 = tt.broadcast %f1_31 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f1_33 = tt.broadcast %qk_mask_24 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
%f1_34 = tt.addptr %f1_32, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%f1_35 = tt.load %f1_34, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f1_36 = arith.extf %f1_35 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%f2 = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f2_37 = tt.addptr %f2, %f1_29 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f2_38 = tt.broadcast %f2_37 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f2_39 = tt.addptr %f2_38, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%f2_40 = tt.load %f2_39, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f2_flip = tt.reshape %f2_40 : tensor<32x128xbf16, #blocked1> -> tensor<32x64x2xbf16, #blocked2>
%y = tt.bitcast %f2_flip : tensor<32x64x2xbf16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_41 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_183: i16, %y_184: i16):
%y_185 = arith.xori %y_183, %y_184 : i16
tt.reduce.return %y_185 : i16
}) : (tensor<32x64x2xi16, #blocked2>) -> tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_42 = tt.expand_dims %y_41 {axis = 2 : i32} : tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x64x1xi16, #blocked2>
%y_43 = tt.broadcast %y_42 : tensor<32x64x1xi16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_44 = arith.xori %y, %y_43 : tensor<32x64x2xi16, #blocked2>
%x = tt.bitcast %y_44 : tensor<32x64x2xi16, #blocked2> -> tensor<32x64x2xbf16, #blocked2>
%f2_flip_45 = tt.reshape %x : tensor<32x64x2xbf16, #blocked2> -> tensor<32x128xbf16, #blocked1>
%f2_flip_46 = arith.extf %f2_flip_45 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
"# else r#"%qk_mask_34 = tt.expand_dims %token_mask_32 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi1, #blocked2>
%qk_mask_35 = tt.expand_dims %offs_d_30 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%qk_mask_36 = tt.expand_dims %offs_d {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
%qk_mask_37 = arith.cmpi slt, %qk_mask_35, %cst_12 : tensor<1x128xi32, #blocked1>
%qk_mask_38 = arith.cmpi slt, %qk_mask_36, %cst_11 : tensor<1x128xi32, #blocked2>
%qk_mask_39 = tt.broadcast %qk_mask : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%qk_mask_40 = tt.broadcast %qk_mask_34 : tensor<32x1xi1, #blocked2> -> tensor<32x128xi1, #blocked2>
%qk_mask_41 = tt.broadcast %qk_mask_37 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%qk_mask_42 = tt.broadcast %qk_mask_38 : tensor<1x128xi1, #blocked2> -> tensor<32x128xi1, #blocked2>
%qk_mask_43 = arith.andi %qk_mask_39, %qk_mask_41 : tensor<32x128xi1, #blocked1>
%qk_mask_44 = arith.andi %qk_mask_40, %qk_mask_42 : tensor<32x128xi1, #blocked2>
%offs_flip = arith.xori %offs_d, %cst_10 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%f1 = tt.expand_dims %token_27 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%f1_45 = tt.expand_dims %token_28 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<32x1xi32, #blocked2>
%f1_46 = arith.muli %f1, %cst_9 : tensor<32x1xi32, #blocked1>
%f1_47 = arith.muli %f1_45, %cst_8 : tensor<32x1xi32, #blocked2>
%f1_48 = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%f1_49 = tt.addptr %f1_48, %f1_46 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%f1_50 = tt.broadcast %f1_49 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f1_51 = tt.broadcast %qk_mask_35 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
%f1_52 = tt.addptr %f1_50, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%f1_53 = tt.load %f1_52, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%f1_54 = arith.extf %f1_53 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%f2_flip = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked2>
%f2_flip_55 = tt.addptr %f2_flip, %f1_47 : tensor<32x1x!tt.ptr<bf16>, #blocked2>, tensor<32x1xi32, #blocked2>
%f2_flip_56 = tt.expand_dims %offs_flip {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
%f2_flip_57 = tt.broadcast %f2_flip_55 : tensor<32x1x!tt.ptr<bf16>, #blocked2> -> tensor<32x128x!tt.ptr<bf16>, #blocked2>
%f2_flip_58 = tt.broadcast %f2_flip_56 : tensor<1x128xi32, #blocked2> -> tensor<32x128xi32, #blocked2>
%f2_flip_59 = tt.addptr %f2_flip_57, %f2_flip_58 : tensor<32x128x!tt.ptr<bf16>, #blocked2>, tensor<32x128xi32, #blocked2>
%f2_flip_60 = tt.load %f2_flip_59, %qk_mask_44, %cst_17 : tensor<32x128x!tt.ptr<bf16>, #blocked2>
%f2_flip_61 = ttg.convert_layout %f2_flip_60 : tensor<32x128xbf16, #blocked2> -> tensor<32x128xbf16, #blocked1>
%f2_flip_62 = arith.extf %f2_flip_61 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
"#) ++
  r#"%scale = tt.load %grad_scale_ptr : !tt.ptr<f32>
"# ++
  (if trimmed then r#"%x_47 = arith.muli %f1, %cst_6 : tensor<32x1xi32, #blocked1>
%x_48 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_49 = tt.addptr %x_48, %x_47 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_50 = arith.muli %logical_head, 128 : i32
%x_51 = tt.splat %x_50 : i32 -> tensor<32x1xi32, #blocked1>
%x_52 = tt.addptr %x_49, %x_51 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_53 = tt.broadcast %x_52 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_54 = tt.addptr %x_53, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%x_55 = tt.load %x_54, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_56 = arith.extf %x_55 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
"# else r#"%x = arith.muli %f1, %cst_6 : tensor<32x1xi32, #blocked1>
%x_63 = tt.splat %qk : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%x_64 = tt.addptr %x_63, %x : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_65 = arith.muli %logical_head, 128 : i32
%x_66 = tt.splat %x_65 : i32 -> tensor<32x1xi32, #blocked1>
%x_67 = tt.addptr %x_64, %x_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_68 = tt.broadcast %x_67 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_69 = tt.addptr %x_68, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%x_70 = tt.load %x_69, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_71 = arith.extf %x_70 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
"#) ++
  r#"%grad_ptr = arith.muli %f1, %cst_5 : tensor<32x1xi32, #blocked1>
"# ++
  (if trimmed then r#"%grad_ptr_57 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_58 = tt.addptr %grad_ptr_57, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_59 = tt.addptr %grad_ptr_58, %x_51 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_60 = tt.broadcast %grad_ptr_59 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_ptr_61 = tt.addptr %grad_ptr_60, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad = tt.load %grad_ptr_61, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x128xbf16, #blocked1> -> tensor<32x64x2xbf16, #blocked2>
%y_62 = tt.bitcast %grad_flip : tensor<32x64x2xbf16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_63 = "tt.reduce"(%y_62) <{axis = 2 : i32}> ({
^bb0(%y_183: i16, %y_184: i16):
%y_185 = arith.xori %y_183, %y_184 : i16
tt.reduce.return %y_185 : i16
}) : (tensor<32x64x2xi16, #blocked2>) -> tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_64 = tt.expand_dims %y_63 {axis = 2 : i32} : tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x64x1xi16, #blocked2>
%y_65 = tt.broadcast %y_64 : tensor<32x64x1xi16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_66 = arith.xori %y_62, %y_65 : tensor<32x64x2xi16, #blocked2>
%x_67 = tt.bitcast %y_66 : tensor<32x64x2xi16, #blocked2> -> tensor<32x64x2xbf16, #blocked2>
%grad_flip_68 = tt.reshape %x_67 : tensor<32x64x2xbf16, #blocked2> -> tensor<32x128xbf16, #blocked1>
%grad_69 = arith.extf %grad : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_flip_70 = arith.extf %grad_flip_68 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%rstd = arith.mulf %x_56, %x_56 : tensor<32x128xf32, #blocked1>
%rstd_71 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_183: f32, %rstd_184: f32):
%rstd_185 = arith.addf %rstd_183, %rstd_184 : f32
tt.reduce.return %rstd_185 : f32
"# else r#"%grad_ptr_72 = tt.splat %grad_q : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_73 = tt.addptr %grad_ptr_72, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_74 = tt.addptr %grad_ptr_73, %x_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_75 = tt.broadcast %grad_ptr_74 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_ptr_76 = tt.addptr %grad_ptr_75, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad = tt.load %grad_ptr_76, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_flip = tt.reshape %grad : tensor<32x128xbf16, #blocked1> -> tensor<32x64x2xbf16, #blocked3>
%y = tt.bitcast %grad_flip : tensor<32x64x2xbf16, #blocked3> -> tensor<32x64x2xi16, #blocked3>
%y_77 = "tt.reduce"(%y) <{axis = 2 : i32}> ({
^bb0(%y_194: i16, %y_195: i16):
%y_196 = arith.xori %y_194, %y_195 : i16
tt.reduce.return %y_196 : i16
}) : (tensor<32x64x2xi16, #blocked3>) -> tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_78 = tt.expand_dims %y_77 {axis = 2 : i32} : tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x64x1xi16, #blocked3>
%y_79 = tt.broadcast %y_78 : tensor<32x64x1xi16, #blocked3> -> tensor<32x64x2xi16, #blocked3>
%y_80 = arith.xori %y, %y_79 : tensor<32x64x2xi16, #blocked3>
%x_81 = tt.bitcast %y_80 : tensor<32x64x2xi16, #blocked3> -> tensor<32x64x2xbf16, #blocked3>
%grad_flip_82 = tt.reshape %x_81 : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%grad_83 = arith.extf %grad : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_flip_84 = arith.extf %grad_flip_82 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%rstd = arith.mulf %x_71, %x_71 : tensor<32x128xf32, #blocked1>
%rstd_85 = "tt.reduce"(%rstd) <{axis = 1 : i32}> ({
^bb0(%rstd_194: f32, %rstd_195: f32):
%rstd_196 = arith.addf %rstd_194, %rstd_195 : f32
tt.reduce.return %rstd_196 : f32
"#) ++
  r#"}) : (tensor<32x128xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if trimmed then r#"%rstd_72 = arith.divf %rstd_71, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_73 = arith.addf %rstd_72, %cst_13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_74 = math.rsqrt %rstd_73 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_74 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_75 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%normalized_76 = arith.mulf %x_56, %normalized_75 : tensor<32x128xf32, #blocked1>
%grad_normalized = arith.mulf %f1_36, %grad_69 : tensor<32x128xf32, #blocked1>
%grad_normalized_77 = arith.mulf %f2_flip_46, %grad_flip_70 : tensor<32x128xf32, #blocked1>
%grad_normalized_78 = arith.addf %grad_normalized, %grad_normalized_77 : tensor<32x128xf32, #blocked1>
%correction = arith.mulf %grad_normalized_78, %normalized_76 : tensor<32x128xf32, #blocked1>
%correction_79 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_183: f32, %correction_184: f32):
%correction_185 = arith.addf %correction_183, %correction_184 : f32
tt.reduce.return %correction_185 : f32
"# else r#"%rstd_86 = arith.divf %rstd_85, %cst_19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_87 = arith.addf %rstd_86, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_88 = math.rsqrt %rstd_87 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized = tt.expand_dims %rstd_88 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_89 = tt.broadcast %normalized : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%normalized_90 = arith.mulf %x_71, %normalized_89 : tensor<32x128xf32, #blocked1>
%grad_normalized = arith.mulf %f1_54, %grad_83 : tensor<32x128xf32, #blocked1>
%grad_normalized_91 = arith.mulf %f2_flip_62, %grad_flip_84 : tensor<32x128xf32, #blocked1>
%grad_normalized_92 = arith.addf %grad_normalized, %grad_normalized_91 : tensor<32x128xf32, #blocked1>
%correction = arith.mulf %grad_normalized_92, %normalized_90 : tensor<32x128xf32, #blocked1>
%correction_93 = "tt.reduce"(%correction) <{axis = 1 : i32}> ({
^bb0(%correction_194: f32, %correction_195: f32):
%correction_196 = arith.addf %correction_194, %correction_195 : f32
tt.reduce.return %correction_196 : f32
"#) ++
  r#"}) : (tensor<32x128xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if trimmed then r#"%correction_80 = arith.divf %correction_79, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_80 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_81 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%dx_82 = arith.mulf %normalized_76, %dx_81 : tensor<32x128xf32, #blocked1>
%dx_83 = arith.subf %grad_normalized_78, %dx_82 : tensor<32x128xf32, #blocked1>
%dx_84 = arith.mulf %normalized_75, %dx_83 : tensor<32x128xf32, #blocked1>
%q = arith.truncf %dx_84 : tensor<32x128xf32, #blocked1> to tensor<32x128xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked3>
%q_85 = arith.extf %0 : tensor<32x128xbf16, #blocked3> to tensor<32x128xf32, #blocked3>
%q_86 = arith.extf %q : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%q_87 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked3>
%q_88 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked1>
%q_89 = arith.divf %q_85, %q_87 : tensor<32x128xf32, #blocked3>
%q_90 = arith.divf %q_86, %q_88 : tensor<32x128xf32, #blocked1>
%q_91 = tt.fp_to_fp %q_89, rounding = rtne : tensor<32x128xf32, #blocked3> -> tensor<32x128xf8E4M3FN, #blocked3>
%1 = tt.trans %q_91 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked3> -> tensor<128x32xf8E4M3FN, #blocked>
%q_92 = tt.fp_to_fp %q_90, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%feature = tt.splat %x_50 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_93 = tt.splat %x_50 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_94 = arith.addi %feature, %offs_d : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_95 = arith.addi %feature_93, %offs_d_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"# else r#"%correction_94 = arith.divf %correction_93, %cst_19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx = tt.expand_dims %correction_94 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_95 = tt.broadcast %dx : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%dx_96 = arith.mulf %normalized_90, %dx_95 : tensor<32x128xf32, #blocked1>
%dx_97 = arith.subf %grad_normalized_92, %dx_96 : tensor<32x128xf32, #blocked1>
%dx_98 = arith.mulf %normalized_89, %dx_97 : tensor<32x128xf32, #blocked1>
%q = arith.truncf %dx_98 : tensor<32x128xf32, #blocked1> to tensor<32x128xbf16, #blocked1>
%0 = ttg.convert_layout %q : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked4>
%q_99 = arith.extf %0 : tensor<32x128xbf16, #blocked4> to tensor<32x128xf32, #blocked4>
%q_100 = arith.extf %q : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%q_101 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked4>
%q_102 = tt.splat %scale : f32 -> tensor<32x128xf32, #blocked1>
%q_103 = arith.divf %q_99, %q_101 : tensor<32x128xf32, #blocked4>
%q_104 = arith.divf %q_100, %q_102 : tensor<32x128xf32, #blocked1>
%q_105 = tt.fp_to_fp %q_103, rounding = rtne : tensor<32x128xf32, #blocked4> -> tensor<32x128xf8E4M3FN, #blocked4>
%1 = tt.trans %q_105 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked4> -> tensor<128x32xf8E4M3FN, #blocked>
%q_106 = tt.fp_to_fp %q_104, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%feature = tt.splat %x_65 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_107 = tt.splat %x_65 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_108 = arith.addi %feature, %offs_d_30 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_109 = arith.addi %feature_107, %offs_d_31 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
"#) ++
  r#"%2 = tt.splat %grad_row : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>
"# ++
  (if trimmed then r#"%3 = tt.addptr %2, %x_47 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x1xi32, #blocked1>
%4 = tt.expand_dims %feature_94 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"# else r#"%3 = tt.addptr %2, %x : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x1xi32, #blocked1>
%4 = tt.expand_dims %feature_108 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"#) ++
  r#"%5 = tt.broadcast %3 : tensor<32x1x!tt.ptr<f8E4M3FN>, #blocked1> -> tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%6 = tt.broadcast %4 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
%7 = tt.addptr %5, %6 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x128xi32, #blocked1>
"# ++
  (if trimmed then r#"tt.store %7, %q_92, %qk_mask_28 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%8 = tt.expand_dims %offs_d_22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"# else r#"tt.store %7, %q_106, %qk_mask_43 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%8 = tt.expand_dims %offs_d_31 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"#) ++
  r#"%9 = arith.cmpi slt, %8, %cst : tensor<128x1xi32, #blocked>
"# ++
  (if trimmed then r#"%10 = tt.expand_dims %token_mask_23 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
"# else r#"%10 = tt.expand_dims %token_mask_33 {axis = 0 : i32} : tensor<32xi1, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi1, #blocked>
"#) ++
  r#"%11 = tt.broadcast %9 : tensor<128x1xi1, #blocked> -> tensor<128x32xi1, #blocked>
%12 = tt.broadcast %10 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
%13 = arith.andi %11, %12 : tensor<128x32xi1, #blocked>
"# ++
  (if trimmed then r#"%14 = tt.expand_dims %feature_95 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"# else r#"%14 = tt.expand_dims %feature_109 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"#) ++
  r#"%15 = arith.muli %14, %cst_0 : tensor<128x1xi32, #blocked>
%16 = tt.splat %grad_transposed : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked>
%17 = tt.addptr %16, %15 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x1xi32, #blocked>
"# ++
  (if trimmed then r#"%18 = tt.expand_dims %token_21 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
"# else r#"%18 = tt.expand_dims %token_29 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
"#) ++
  r#"%19 = tt.broadcast %17 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
%20 = tt.broadcast %18 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
%21 = tt.addptr %19, %20 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x32xi32, #blocked>
tt.store %21, %1, %13 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
%input_head = arith.addi %logical_head, 6 : i32
"# ++
  (if trimmed then r#"%x_96 = arith.muli %input_head, 128 : i32
%x_97 = tt.splat %x_96 : i32 -> tensor<32x1xi32, #blocked1>
%x_98 = tt.addptr %x_49, %x_97 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_99 = tt.broadcast %x_98 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_100 = tt.addptr %x_99, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%x_101 = tt.load %x_100, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_102 = arith.extf %x_101 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_ptr_103 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_104 = tt.addptr %grad_ptr_103, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_105 = tt.addptr %grad_ptr_104, %x_51 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_106 = tt.broadcast %grad_ptr_105 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_ptr_107 = tt.addptr %grad_ptr_106, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_mask = arith.cmpi slt, %qk_mask_24, %cst_1 : tensor<1x128xi32, #blocked1>
%grad_mask_108 = arith.cmpi eq, %f1, %cst_7 : tensor<32x1xi32, #blocked1>
%grad_mask_109 = tt.broadcast %grad_mask : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%grad_mask_110 = tt.broadcast %grad_mask_108 : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%grad_mask_111 = arith.ori %grad_mask_109, %grad_mask_110 : tensor<32x128xi1, #blocked1>
%grad_mask_112 = arith.andi %qk_mask_28, %grad_mask_111 : tensor<32x128xi1, #blocked1>
%grad_113 = tt.load %grad_ptr_107, %grad_mask_112, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_flip_114 = tt.reshape %grad_113 : tensor<32x128xbf16, #blocked1> -> tensor<32x64x2xbf16, #blocked2>
%y_115 = tt.bitcast %grad_flip_114 : tensor<32x64x2xbf16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_116 = "tt.reduce"(%y_115) <{axis = 2 : i32}> ({
^bb0(%y_183: i16, %y_184: i16):
%y_185 = arith.xori %y_183, %y_184 : i16
tt.reduce.return %y_185 : i16
}) : (tensor<32x64x2xi16, #blocked2>) -> tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>>
%y_117 = tt.expand_dims %y_116 {axis = 2 : i32} : tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<32x64x1xi16, #blocked2>
%y_118 = tt.broadcast %y_117 : tensor<32x64x1xi16, #blocked2> -> tensor<32x64x2xi16, #blocked2>
%y_119 = arith.xori %y_115, %y_118 : tensor<32x64x2xi16, #blocked2>
%x_120 = tt.bitcast %y_119 : tensor<32x64x2xi16, #blocked2> -> tensor<32x64x2xbf16, #blocked2>
%grad_flip_121 = tt.reshape %x_120 : tensor<32x64x2xbf16, #blocked2> -> tensor<32x128xbf16, #blocked1>
%grad_122 = arith.extf %grad_113 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_flip_123 = arith.extf %grad_flip_121 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%rstd_124 = arith.mulf %x_102, %x_102 : tensor<32x128xf32, #blocked1>
%rstd_125 = "tt.reduce"(%rstd_124) <{axis = 1 : i32}> ({
^bb0(%rstd_183: f32, %rstd_184: f32):
%rstd_185 = arith.addf %rstd_183, %rstd_184 : f32
tt.reduce.return %rstd_185 : f32
"# else r#"%x_110 = arith.muli %input_head, 128 : i32
%x_111 = tt.splat %x_110 : i32 -> tensor<32x1xi32, #blocked1>
%x_112 = tt.addptr %x_64, %x_111 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%x_113 = tt.broadcast %x_112 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_114 = tt.addptr %x_113, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%x_115 = tt.load %x_114, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%x_116 = arith.extf %x_115 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_ptr_117 = tt.splat %grad_k : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
%grad_ptr_118 = tt.addptr %grad_ptr_117, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_119 = tt.addptr %grad_ptr_118, %x_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_ptr_120 = tt.broadcast %grad_ptr_119 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_ptr_121 = tt.addptr %grad_ptr_120, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_122 = tt.load %grad_ptr_121, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_flip_123 = tt.reshape %grad_122 : tensor<32x128xbf16, #blocked1> -> tensor<32x64x2xbf16, #blocked3>
%y_124 = tt.bitcast %grad_flip_123 : tensor<32x64x2xbf16, #blocked3> -> tensor<32x64x2xi16, #blocked3>
%y_125 = "tt.reduce"(%y_124) <{axis = 2 : i32}> ({
^bb0(%y_194: i16, %y_195: i16):
%y_196 = arith.xori %y_194, %y_195 : i16
tt.reduce.return %y_196 : i16
}) : (tensor<32x64x2xi16, #blocked3>) -> tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked3}>>
%y_126 = tt.expand_dims %y_125 {axis = 2 : i32} : tensor<32x64xi16, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<32x64x1xi16, #blocked3>
%y_127 = tt.broadcast %y_126 : tensor<32x64x1xi16, #blocked3> -> tensor<32x64x2xi16, #blocked3>
%y_128 = arith.xori %y_124, %y_127 : tensor<32x64x2xi16, #blocked3>
%x_129 = tt.bitcast %y_128 : tensor<32x64x2xi16, #blocked3> -> tensor<32x64x2xbf16, #blocked3>
%grad_flip_130 = tt.reshape %x_129 : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%grad_131 = arith.extf %grad_122 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%grad_flip_132 = arith.extf %grad_flip_130 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%rstd_133 = arith.mulf %x_116, %x_116 : tensor<32x128xf32, #blocked1>
%rstd_134 = "tt.reduce"(%rstd_133) <{axis = 1 : i32}> ({
^bb0(%rstd_194: f32, %rstd_195: f32):
%rstd_196 = arith.addf %rstd_194, %rstd_195 : f32
tt.reduce.return %rstd_196 : f32
"#) ++
  r#"}) : (tensor<32x128xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if trimmed then r#"%rstd_126 = arith.divf %rstd_125, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_127 = arith.addf %rstd_126, %cst_13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_128 = math.rsqrt %rstd_127 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_129 = tt.expand_dims %rstd_128 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_130 = tt.broadcast %normalized_129 : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%normalized_131 = arith.mulf %x_102, %normalized_130 : tensor<32x128xf32, #blocked1>
%grad_normalized_132 = arith.mulf %f1_36, %grad_122 : tensor<32x128xf32, #blocked1>
%grad_normalized_133 = arith.mulf %f2_flip_46, %grad_flip_123 : tensor<32x128xf32, #blocked1>
%grad_normalized_134 = arith.addf %grad_normalized_132, %grad_normalized_133 : tensor<32x128xf32, #blocked1>
%next_token = arith.addi %token_20, %cst_2 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%next_token_135 = arith.minsi %next_token, %cst_3 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%next_mask = arith.cmpi slt, %f1, %cst_4 : tensor<32x1xi32, #blocked1>
%next_mask_136 = tt.broadcast %next_mask : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%next_mask_137 = arith.andi %qk_mask_28, %next_mask_136 : tensor<32x128xi1, #blocked1>
%next_mask_138 = arith.cmpi sge, %qk_mask_24, %cst_1 : tensor<1x128xi32, #blocked1>
%next_mask_139 = tt.broadcast %next_mask_138 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%next_mask_140 = arith.andi %next_mask_137, %next_mask_139 : tensor<32x128xi1, #blocked1>
%grad_next = tt.expand_dims %next_token_135 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%grad_next_141 = arith.muli %grad_next, %cst_5 : tensor<32x1xi32, #blocked1>
%grad_next_142 = tt.addptr %grad_ptr_103, %grad_next_141 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_next_143 = tt.addptr %grad_next_142, %x_51 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_next_144 = tt.broadcast %grad_next_143 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_next_145 = tt.addptr %grad_next_144, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_next_146 = tt.load %grad_next_145, %next_mask_140, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_next_147 = arith.extf %grad_next_146 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%stationary_grad = arith.addf %grad_122, %grad_next_147 : tensor<32x128xf32, #blocked1>
%stationary_grad_148 = arith.select %grad_mask_110, %stationary_grad, %grad_next_147 : tensor<32x128xi1, #blocked1>, tensor<32x128xf32, #blocked1>
%grad_normalized_149 = arith.select %next_mask_139, %stationary_grad_148, %grad_normalized_134 : tensor<32x128xi1, #blocked1>, tensor<32x128xf32, #blocked1>
%correction_150 = arith.mulf %grad_normalized_149, %normalized_131 : tensor<32x128xf32, #blocked1>
%correction_151 = "tt.reduce"(%correction_150) <{axis = 1 : i32}> ({
^bb0(%correction_183: f32, %correction_184: f32):
%correction_185 = arith.addf %correction_183, %correction_184 : f32
tt.reduce.return %correction_185 : f32
"# else r#"%rstd_135 = arith.divf %rstd_134, %cst_19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_136 = arith.addf %rstd_135, %cst_18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%rstd_137 = math.rsqrt %rstd_136 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%normalized_138 = tt.expand_dims %rstd_137 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%normalized_139 = tt.broadcast %normalized_138 : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%normalized_140 = arith.mulf %x_116, %normalized_139 : tensor<32x128xf32, #blocked1>
%grad_normalized_141 = arith.mulf %f1_54, %grad_131 : tensor<32x128xf32, #blocked1>
%grad_normalized_142 = arith.mulf %f2_flip_62, %grad_flip_132 : tensor<32x128xf32, #blocked1>
%grad_normalized_143 = arith.addf %grad_normalized_141, %grad_normalized_142 : tensor<32x128xf32, #blocked1>
%next_token = arith.addi %token_27, %cst_1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%next_token_144 = arith.minsi %next_token, %cst_2 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%grad_next = arith.cmpi slt, %f1, %cst_3 : tensor<32x1xi32, #blocked1>
%grad_next_145 = tt.broadcast %grad_next : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%grad_next_146 = arith.andi %qk_mask_43, %grad_next_145 : tensor<32x128xi1, #blocked1>
%grad_next_147 = tt.expand_dims %next_token_144 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
%grad_next_148 = arith.muli %grad_next_147, %cst_5 : tensor<32x1xi32, #blocked1>
%grad_next_149 = tt.addptr %grad_ptr_117, %grad_next_148 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_next_150 = tt.addptr %grad_next_149, %x_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_next_151 = tt.broadcast %grad_next_150 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_next_152 = tt.addptr %grad_next_151, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_next_153 = tt.load %grad_next_152, %grad_next_146, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_next_154 = arith.extf %grad_next_153 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%stationary_grad = arith.cmpi eq, %f1, %cst_7 : tensor<32x1xi32, #blocked1>
%stationary_grad_155 = arith.addf %grad_131, %grad_next_154 : tensor<32x128xf32, #blocked1>
%stationary_grad_156 = tt.broadcast %stationary_grad : tensor<32x1xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%stationary_grad_157 = arith.select %stationary_grad_156, %stationary_grad_155, %grad_next_154 : tensor<32x128xi1, #blocked1>, tensor<32x128xf32, #blocked1>
%grad_normalized_158 = arith.cmpi sge, %qk_mask_35, %cst_4 : tensor<1x128xi32, #blocked1>
%grad_normalized_159 = tt.broadcast %grad_normalized_158 : tensor<1x128xi1, #blocked1> -> tensor<32x128xi1, #blocked1>
%grad_normalized_160 = arith.select %grad_normalized_159, %stationary_grad_157, %grad_normalized_143 : tensor<32x128xi1, #blocked1>, tensor<32x128xf32, #blocked1>
%correction_161 = arith.mulf %grad_normalized_160, %normalized_140 : tensor<32x128xf32, #blocked1>
%correction_162 = "tt.reduce"(%correction_161) <{axis = 1 : i32}> ({
^bb0(%correction_194: f32, %correction_195: f32):
%correction_196 = arith.addf %correction_194, %correction_195 : f32
tt.reduce.return %correction_196 : f32
"#) ++
  r#"}) : (tensor<32x128xf32, #blocked1>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
"# ++
  (if trimmed then r#"%correction_152 = arith.divf %correction_151, %cst_14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_153 = tt.expand_dims %correction_152 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_154 = tt.broadcast %dx_153 : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%dx_155 = arith.mulf %normalized_131, %dx_154 : tensor<32x128xf32, #blocked1>
%dx_156 = arith.subf %grad_normalized_149, %dx_155 : tensor<32x128xf32, #blocked1>
%dx_157 = arith.mulf %normalized_130, %dx_156 : tensor<32x128xf32, #blocked1>
%q_158 = arith.truncf %dx_157 : tensor<32x128xf32, #blocked1> to tensor<32x128xbf16, #blocked1>
%22 = ttg.convert_layout %q_158 : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked3>
%q_159 = arith.extf %22 : tensor<32x128xbf16, #blocked3> to tensor<32x128xf32, #blocked3>
%q_160 = arith.extf %q_158 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%q_161 = arith.divf %q_159, %q_87 : tensor<32x128xf32, #blocked3>
%q_162 = arith.divf %q_160, %q_88 : tensor<32x128xf32, #blocked1>
%q_163 = tt.fp_to_fp %q_161, rounding = rtne : tensor<32x128xf32, #blocked3> -> tensor<32x128xf8E4M3FN, #blocked3>
%23 = tt.trans %q_163 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked3> -> tensor<128x32xf8E4M3FN, #blocked>
%q_164 = tt.fp_to_fp %q_162, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%feature_165 = tt.splat %x_96 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_166 = tt.splat %x_96 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_167 = arith.addi %feature_165, %offs_d : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_168 = arith.addi %feature_166, %offs_d_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%24 = tt.expand_dims %feature_167 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"# else r#"%correction_163 = arith.divf %correction_162, %cst_19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%dx_164 = tt.expand_dims %correction_163 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xf32, #blocked1>
%dx_165 = tt.broadcast %dx_164 : tensor<32x1xf32, #blocked1> -> tensor<32x128xf32, #blocked1>
%dx_166 = arith.mulf %normalized_140, %dx_165 : tensor<32x128xf32, #blocked1>
%dx_167 = arith.subf %grad_normalized_160, %dx_166 : tensor<32x128xf32, #blocked1>
%dx_168 = arith.mulf %normalized_139, %dx_167 : tensor<32x128xf32, #blocked1>
%q_169 = arith.truncf %dx_168 : tensor<32x128xf32, #blocked1> to tensor<32x128xbf16, #blocked1>
%22 = ttg.convert_layout %q_169 : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked4>
%q_170 = arith.extf %22 : tensor<32x128xbf16, #blocked4> to tensor<32x128xf32, #blocked4>
%q_171 = arith.extf %q_169 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%q_172 = arith.divf %q_170, %q_101 : tensor<32x128xf32, #blocked4>
%q_173 = arith.divf %q_171, %q_102 : tensor<32x128xf32, #blocked1>
%q_174 = tt.fp_to_fp %q_172, rounding = rtne : tensor<32x128xf32, #blocked4> -> tensor<32x128xf8E4M3FN, #blocked4>
%23 = tt.trans %q_174 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked4> -> tensor<128x32xf8E4M3FN, #blocked>
%q_175 = tt.fp_to_fp %q_173, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%feature_176 = tt.splat %x_110 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_177 = tt.splat %x_110 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%feature_178 = arith.addi %feature_176, %offs_d_30 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%feature_179 = arith.addi %feature_177, %offs_d_31 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%24 = tt.expand_dims %feature_178 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"#) ++
  r#"%25 = tt.broadcast %24 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
%26 = tt.addptr %5, %25 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x128xi32, #blocked1>
"# ++
  (if trimmed then r#"tt.store %26, %q_164, %qk_mask_28 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%27 = tt.expand_dims %feature_168 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"# else r#"tt.store %26, %q_175, %qk_mask_43 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%27 = tt.expand_dims %feature_179 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"#) ++
  r#"%28 = arith.muli %27, %cst_0 : tensor<128x1xi32, #blocked>
%29 = tt.addptr %16, %28 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x1xi32, #blocked>
%30 = tt.broadcast %29 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
%31 = tt.addptr %30, %20 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x32xi32, #blocked>
tt.store %31, %23, %13 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
%grad_v_value = tt.splat %grad_v : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>, #blocked1>
"# ++
  (if trimmed then r#"%grad_v_value_169 = tt.addptr %grad_v_value, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_170 = tt.addptr %grad_v_value_169, %x_51 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_171 = tt.broadcast %grad_v_value_170 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_v_value_172 = tt.addptr %grad_v_value_171, %f1_33 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_v_value_173 = tt.load %grad_v_value_172, %qk_mask_28, %cst_12 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%32 = ttg.convert_layout %grad_v_value_173 : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked3>
%grad_v_value_174 = arith.extf %32 : tensor<32x128xbf16, #blocked3> to tensor<32x128xf32, #blocked3>
%grad_v_value_175 = arith.extf %grad_v_value_173 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%qv = arith.divf %grad_v_value_174, %q_87 : tensor<32x128xf32, #blocked3>
%qv_176 = arith.divf %grad_v_value_175, %q_88 : tensor<32x128xf32, #blocked1>
%qv_177 = tt.fp_to_fp %qv, rounding = rtne : tensor<32x128xf32, #blocked3> -> tensor<32x128xf8E4M3FN, #blocked3>
%33 = tt.trans %qv_177 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked3> -> tensor<128x32xf8E4M3FN, #blocked>
%qv_178 = tt.fp_to_fp %qv_176, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%v_feature = arith.addi %x_50, 1536 : i32
%v_feature_179 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_180 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%v_feature_181 = arith.addi %v_feature_179, %offs_d : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_182 = arith.addi %v_feature_180, %offs_d_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%34 = tt.expand_dims %v_feature_181 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"# else r#"%grad_v_value_180 = tt.addptr %grad_v_value, %grad_ptr : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_181 = tt.addptr %grad_v_value_180, %x_66 : tensor<32x1x!tt.ptr<bf16>, #blocked1>, tensor<32x1xi32, #blocked1>
%grad_v_value_182 = tt.broadcast %grad_v_value_181 : tensor<32x1x!tt.ptr<bf16>, #blocked1> -> tensor<32x128x!tt.ptr<bf16>, #blocked1>
%grad_v_value_183 = tt.addptr %grad_v_value_182, %f1_51 : tensor<32x128x!tt.ptr<bf16>, #blocked1>, tensor<32x128xi32, #blocked1>
%grad_v_value_184 = tt.load %grad_v_value_183, %qk_mask_43, %cst_16 : tensor<32x128x!tt.ptr<bf16>, #blocked1>
%32 = ttg.convert_layout %grad_v_value_184 : tensor<32x128xbf16, #blocked1> -> tensor<32x128xbf16, #blocked4>
%grad_v_value_185 = arith.extf %32 : tensor<32x128xbf16, #blocked4> to tensor<32x128xf32, #blocked4>
%grad_v_value_186 = arith.extf %grad_v_value_184 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%qv = arith.divf %grad_v_value_185, %q_101 : tensor<32x128xf32, #blocked4>
%qv_187 = arith.divf %grad_v_value_186, %q_102 : tensor<32x128xf32, #blocked1>
%qv_188 = tt.fp_to_fp %qv, rounding = rtne : tensor<32x128xf32, #blocked4> -> tensor<32x128xf8E4M3FN, #blocked4>
%33 = tt.trans %qv_188 {order = array<i32: 1, 0>} : tensor<32x128xf8E4M3FN, #blocked4> -> tensor<128x32xf8E4M3FN, #blocked>
%qv_189 = tt.fp_to_fp %qv_187, rounding = rtne : tensor<32x128xf32, #blocked1> -> tensor<32x128xf8E4M3FN, #blocked1>
%v_feature = arith.addi %x_65, 1536 : i32
%v_feature_190 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_191 = tt.splat %v_feature : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%v_feature_192 = arith.addi %v_feature_190, %offs_d_30 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%v_feature_193 = arith.addi %v_feature_191, %offs_d_31 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%34 = tt.expand_dims %v_feature_192 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
"#) ++
  r#"%35 = tt.broadcast %34 : tensor<1x128xi32, #blocked1> -> tensor<32x128xi32, #blocked1>
%36 = tt.addptr %5, %35 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<32x128xi32, #blocked1>
"# ++
  (if trimmed then r#"tt.store %36, %qv_178, %qk_mask_28 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%37 = tt.expand_dims %v_feature_182 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"# else r#"tt.store %36, %qv_189, %qk_mask_43 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked1>
%37 = tt.expand_dims %v_feature_193 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
"#) ++
  r#"%38 = arith.muli %37, %cst_0 : tensor<128x1xi32, #blocked>
%39 = tt.addptr %16, %38 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x1xi32, #blocked>
%40 = tt.broadcast %39 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
%41 = tt.addptr %40, %20 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>, tensor<128x32xi32, #blocked>
tt.store %41, %33, %13 : tensor<128x32x!tt.ptr<f8E4M3FN>, #blocked>
tt.return
}
}
"#

private def ptxTemplate (trimmed : Bool) : String :=
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
.reqntid 256
{
"# ++
  (if trimmed then r#".reg .pred %p<10>;
.reg .b16 %rs<261>;
.reg .b32 %r<829>;
.reg .b64 %rd<62>;
"# else r#".reg .pred %p<6>;
.reg .b16 %rs<278>;
.reg .b32 %r<860>;
.reg .b64 %rd<79>;
"#) ++
  r#"$L__func_begin0:
"# ++
  (if trimmed then r#"ld.param.b64 %rd24, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd25, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"# else r#"ld.param.b64 %rd38, [_qkv_norm_rope_pack_fp8_backward_kernel_param_0];
ld.param.b64 %rd39, [_qkv_norm_rope_pack_fp8_backward_kernel_param_1];
"#) ++
  r#"$L__tmp0:
"# ++
  (if trimmed then r#"mov.u32 %r187, %ctaid.x;
ld.param.b64 %rd26, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r188, %ctaid.y;
shl.b32 %r189, %r188, 5;
ld.param.b64 %rd27, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd28, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r190, %tid.x;
ld.param.b64 %rd29, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r191, %r190, 3, 5;
ld.param.b64 %rd30, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
shl.b32 %r192, %r190, 4;
ld.param.b64 %rd31, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
and.b32 %r193, %r192, 16;
ld.param.b64 %rd5, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
or.b32 %r194, %r191, %r189;
or.b32 %r195, %r193, %r189;
and.b32 %r196, %r190, 7;
shl.b32 %r197, %r196, 4;
shr.u32 %r198, %r190, 1;
bfe.u32 %r199, %r190, 1, 7;
setp.lt.u32 %p1, %r188, @BLOCKS@;
shl.b32 %r200, %r194, 7;
mul.wide.u32 %rd32, %r200, 2;
add.s64 %rd33, %rd28, %rd32;
mul.wide.u32 %rd34, %r197, 2;
add.s64 %rd1, %rd33, %rd34;
"# else r#"mov.u32 %r179, %ctaid.x;
ld.param.b64 %rd40, [_qkv_norm_rope_pack_fp8_backward_kernel_param_2];
mov.u32 %r180, %ctaid.y;
shl.b32 %r181, %r180, 5;
ld.param.b64 %rd41, [_qkv_norm_rope_pack_fp8_backward_kernel_param_3];
ld.param.b64 %rd42, [_qkv_norm_rope_pack_fp8_backward_kernel_param_4];
mov.u32 %r182, %tid.x;
ld.param.b64 %rd43, [_qkv_norm_rope_pack_fp8_backward_kernel_param_5];
bfe.u32 %r183, %r182, 3, 5;
ld.param.b64 %rd44, [_qkv_norm_rope_pack_fp8_backward_kernel_param_6];
and.b32 %r184, %r182, 31;
ld.param.b64 %rd45, [_qkv_norm_rope_pack_fp8_backward_kernel_param_7];
shl.b32 %r185, %r182, 4;
ld.param.b64 %rd19, [_qkv_norm_rope_pack_fp8_backward_kernel_param_8];
and.b32 %r186, %r185, 16;
or.b32 %r187, %r183, %r181;
or.b32 %r188, %r181, %r184;
or.b32 %r189, %r186, %r181;
and.b32 %r190, %r182, 224;
bfe.u32 %r191, %r182, 5, 3;
and.b32 %r192, %r182, 7;
shl.b32 %r193, %r192, 4;
bfe.u32 %r194, %r182, 1, 7;
setp.lt.u32 %p1, %r180, @BLOCKS@;
xor.b32 %r195, %r191, 1;
shl.b32 %r196, %r187, 7;
shl.b32 %r197, %r188, 7;
mad.wide.u32 %rd46, %r196, 2, %rd42;
mul.wide.u32 %rd47, %r193, 2;
add.s64 %rd1, %rd46, %rd47;
"#) ++
  r#"add.s64 %rd2, %rd1, 16;
mov.b32 %r5, 0;
mov.u32 %r1, %r5;
mov.u32 %r2, %r5;
mov.u32 %r3, %r5;
mov.u32 %r4, %r5;
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
"# ++
  (if trimmed then r#"mov.b32 {%rs49, %rs50}, %r1;
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
  (if trimmed then r#"mov.b32 {%rs57, %rs58}, %r6;
mov.b32 {%rs59, %rs60}, %r7;
mov.b32 {%rs61, %rs62}, %r8;
mov.b32 {%rs63, %rs64}, %r9;
cvt.f32.bf16 %r201, %rs49;
cvt.f32.bf16 %r202, %rs50;
cvt.f32.bf16 %r203, %rs51;
cvt.f32.bf16 %r204, %rs52;
cvt.f32.bf16 %r205, %rs53;
cvt.f32.bf16 %r206, %rs54;
cvt.f32.bf16 %r207, %rs55;
cvt.f32.bf16 %r208, %rs56;
cvt.f32.bf16 %r209, %rs57;
cvt.f32.bf16 %r210, %rs58;
cvt.f32.bf16 %r211, %rs59;
cvt.f32.bf16 %r212, %rs60;
cvt.f32.bf16 %r213, %rs61;
cvt.f32.bf16 %r214, %rs62;
cvt.f32.bf16 %r215, %rs63;
cvt.f32.bf16 %r216, %rs64;
add.s64 %rd35, %rd29, %rd32;
add.s64 %rd3, %rd35, %rd34;
add.s64 %rd4, %rd3, 16;
mov.u32 %r10, %r5;
"# else r#"mov.b32 {%rs74, %rs75}, %r6;
mov.b32 {%rs76, %rs77}, %r7;
mov.b32 {%rs78, %rs79}, %r8;
mov.b32 {%rs80, %rs81}, %r9;
cvt.f32.bf16 %r198, %rs66;
cvt.f32.bf16 %r199, %rs67;
cvt.f32.bf16 %r200, %rs68;
cvt.f32.bf16 %r201, %rs69;
cvt.f32.bf16 %r202, %rs70;
cvt.f32.bf16 %r203, %rs71;
cvt.f32.bf16 %r204, %rs72;
cvt.f32.bf16 %r205, %rs73;
cvt.f32.bf16 %r206, %rs74;
cvt.f32.bf16 %r207, %rs75;
cvt.f32.bf16 %r208, %rs76;
cvt.f32.bf16 %r209, %rs77;
cvt.f32.bf16 %r210, %rs78;
cvt.f32.bf16 %r211, %rs79;
cvt.f32.bf16 %r212, %rs80;
cvt.f32.bf16 %r213, %rs81;
mad.wide.u32 %rd48, %r197, 2, %rd43;
mad.wide.u32 %rd3, %r195, 2, %rd48;
cvt.u64.u32 %rd49, %r191;
xor.b64 %rd50, %rd49, 1;
shl.b64 %rd51, %rd50, 1;
add.s64 %rd52, %rd48, %rd51;
add.s64 %rd4, %rd52, 16;
add.s64 %rd5, %rd52, 32;
add.s64 %rd6, %rd52, 48;
add.s64 %rd7, %rd52, 64;
add.s64 %rd8, %rd52, 80;
add.s64 %rd9, %rd52, 96;
add.s64 %rd10, %rd52, 112;
add.s64 %rd11, %rd52, 128;
add.s64 %rd12, %rd52, 144;
add.s64 %rd13, %rd52, 160;
add.s64 %rd14, %rd52, 176;
add.s64 %rd15, %rd52, 192;
add.s64 %rd16, %rd52, 208;
add.s64 %rd17, %rd52, 224;
add.s64 %rd18, %rd52, 240;
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
and.b32 %r214, %r185, 4016;
bfe.s32 %r215, %r182, 2, 1;
and.b32 %r216, %r215, 4160;
or.b32 %r217, %r216, %r214;
mov.b32 %r218, global_smem;
add.s32 %r219, %r218, %r217;
mov.b32 %r220, {%rs1, %rs3};
mov.b32 %r221, {%rs4, %rs5};
mov.b32 %r222, {%rs6, %rs7};
mov.b32 %r223, {%rs8, %rs9};
st.shared.v4.b32 [%r219], {%r220, %r221, %r222, %r223};
xor.b32 %r224, %r217, 64;
add.s32 %r225, %r218, %r224;
mov.b32 %r226, {%rs10, %rs11};
mov.b32 %r227, {%rs12, %rs13};
mov.b32 %r228, {%rs14, %rs15};
mov.b32 %r229, {%rs16, %rs17};
st.shared.v4.b32 [%r225], {%r226, %r227, %r228, %r229};
bar.sync 0;
shl.b32 %r230, %r182, 6;
and.b32 %r231, %r230, 1600;
shl.b32 %r232, %r182, 3;
and.b32 %r233, %r232, 48;
shl.b32 %r234, %r182, 1;
and.b32 %r235, %r234, 384;
bfe.s32 %r236, %r182, 5, 1;
and.b32 %r237, %r236, 4160;
or.b32 %r238, %r231, %r233;
or.b32 %r239, %r237, %r235;
xor.b32 %r240, %r239, %r238;
add.s32 %r241, %r218, %r240;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r242, %r243, %r244, %r245}, [%r241];
mov.b32 {%rs82, %rs83}, %r242;
mov.b32 {%rs84, %rs85}, %r243;
mov.b32 {%rs86, %rs87}, %r244;
mov.b32 {%rs88, %rs89}, %r245;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r246, %r247, %r248, %r249}, [%r241+2048];
mov.b32 {%rs90, %rs91}, %r246;
mov.b32 {%rs92, %rs93}, %r247;
mov.b32 {%rs94, %rs95}, %r248;
mov.b32 {%rs96, %rs97}, %r249;
cvt.f32.bf16 %r250, %rs82;
cvt.f32.bf16 %r251, %rs84;
cvt.f32.bf16 %r252, %rs86;
cvt.f32.bf16 %r253, %rs88;
cvt.f32.bf16 %r254, %rs90;
cvt.f32.bf16 %r255, %rs92;
cvt.f32.bf16 %r256, %rs94;
cvt.f32.bf16 %r257, %rs96;
cvt.f32.bf16 %r258, %rs83;
cvt.f32.bf16 %r259, %rs85;
cvt.f32.bf16 %r260, %rs87;
cvt.f32.bf16 %r261, %rs89;
cvt.f32.bf16 %r262, %rs91;
cvt.f32.bf16 %r263, %rs93;
cvt.f32.bf16 %r264, %rs95;
cvt.f32.bf16 %r265, %rs97;
mov.u32 %r10, 0x0;
ld.global.b32 { %r10 }, [ %rd19 + 0 ];
mul.lo.s32 %r266, %r187, 2304;
cvt.s64.s32 %rd53, %r266;
mad.wide.s32 %rd54, %r266, 2, %rd41;
shl.b32 %r267, %r179, 7;
mul.wide.s32 %rd55, %r267, 2;
add.s64 %rd56, %rd54, %rd55;
add.s64 %rd20, %rd56, %rd47;
add.s64 %rd21, %rd20, 16;
"#) ++
  r#"mov.u32 %r11, %r5;
mov.u32 %r12, %r5;
mov.u32 %r13, %r5;
"# ++
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
mov.b32 {%rs65, %rs66}, %r10;
mov.b32 {%rs67, %rs68}, %r11;
mov.b32 {%rs69, %rs70}, %r12;
mov.b32 {%rs71, %rs72}, %r13;
"# else "") ++
  r#"mov.u32 %r14, %r5;
"# ++
  (if trimmed then "" else r#"@%p1 ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd20 + 0 ];
mov.b32 {%rs98, %rs99}, %r11;
mov.b32 {%rs100, %rs101}, %r12;
mov.b32 {%rs102, %rs103}, %r13;
mov.b32 {%rs104, %rs105}, %r14;
"#) ++
  r#"mov.u32 %r15, %r5;
mov.u32 %r16, %r5;
mov.u32 %r17, %r5;
"# ++
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
mov.b32 {%rs73, %rs74}, %r14;
mov.b32 {%rs75, %rs76}, %r15;
mov.b32 {%rs77, %rs78}, %r16;
mov.b32 {%rs79, %rs80}, %r17;
cvt.f32.bf16 %r217, %rs66;
cvt.f32.bf16 %r218, %rs65;
cvt.f32.bf16 %r219, %rs68;
cvt.f32.bf16 %r220, %rs67;
cvt.f32.bf16 %r221, %rs70;
cvt.f32.bf16 %r222, %rs69;
cvt.f32.bf16 %r223, %rs72;
cvt.f32.bf16 %r224, %rs71;
cvt.f32.bf16 %r225, %rs74;
cvt.f32.bf16 %r226, %rs73;
cvt.f32.bf16 %r227, %rs76;
cvt.f32.bf16 %r228, %rs75;
cvt.f32.bf16 %r229, %rs78;
cvt.f32.bf16 %r230, %rs77;
cvt.f32.bf16 %r231, %rs80;
cvt.f32.bf16 %r232, %rs79;
mov.u32 %r18, 0x0;
ld.global.b32 { %r18 }, [ %rd5 + 0 ];
mul.lo.s32 %r233, %r194, 2304;
cvt.s64.s32 %rd36, %r233;
mad.wide.s32 %rd37, %r233, 2, %rd27;
shl.b32 %r234, %r187, 7;
mul.wide.s32 %rd38, %r234, 2;
add.s64 %rd39, %rd37, %rd38;
add.s64 %rd6, %rd39, %rd34;
add.s64 %rd7, %rd6, 16;
"# else r#"mov.u32 %r18, %r5;
@%p1 ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd21 + 0 ];
mov.b32 {%rs106, %rs107}, %r15;
mov.b32 {%rs108, %rs109}, %r16;
mov.b32 {%rs110, %rs111}, %r17;
mov.b32 {%rs112, %rs113}, %r18;
cvt.f32.bf16 %r268, %rs98;
cvt.f32.bf16 %r269, %rs99;
cvt.f32.bf16 %r270, %rs100;
cvt.f32.bf16 %r271, %rs101;
cvt.f32.bf16 %r272, %rs102;
cvt.f32.bf16 %r273, %rs103;
cvt.f32.bf16 %r274, %rs104;
cvt.f32.bf16 %r275, %rs105;
cvt.f32.bf16 %r276, %rs106;
cvt.f32.bf16 %r277, %rs107;
cvt.f32.bf16 %r278, %rs108;
cvt.f32.bf16 %r279, %rs109;
cvt.f32.bf16 %r280, %rs110;
cvt.f32.bf16 %r281, %rs111;
cvt.f32.bf16 %r282, %rs112;
cvt.f32.bf16 %r283, %rs113;
mul.lo.s32 %r284, %r187, 768;
mul.wide.u32 %rd57, %r284, 2;
add.s64 %rd58, %rd38, %rd57;
add.s64 %rd59, %rd58, %rd55;
add.s64 %rd22, %rd59, %rd47;
add.s64 %rd23, %rd22, 16;
"#) ++
  r#"mov.u32 %r19, %r5;
mov.u32 %r20, %r5;
mov.u32 %r21, %r5;
mov.u32 %r22, %r5;
"# ++
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r19, %r20, %r21, %r22 }, [ %rd6 + 0 ];
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
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd7 + 0 ];
mov.b32 {%rs89, %rs90}, %r23;
mov.b32 {%rs91, %rs92}, %r24;
mov.b32 {%rs93, %rs94}, %r25;
mov.b32 {%rs95, %rs96}, %r26;
cvt.f32.bf16 %r235, %rs81;
cvt.f32.bf16 %r236, %rs82;
cvt.f32.bf16 %r237, %rs83;
cvt.f32.bf16 %r238, %rs84;
cvt.f32.bf16 %r239, %rs85;
cvt.f32.bf16 %r240, %rs86;
cvt.f32.bf16 %r241, %rs87;
cvt.f32.bf16 %r242, %rs88;
cvt.f32.bf16 %r243, %rs89;
cvt.f32.bf16 %r244, %rs90;
cvt.f32.bf16 %r245, %rs91;
cvt.f32.bf16 %r246, %rs92;
cvt.f32.bf16 %r247, %rs93;
cvt.f32.bf16 %r248, %rs94;
cvt.f32.bf16 %r249, %rs95;
cvt.f32.bf16 %r250, %rs96;
mul.lo.s32 %r251, %r194, 768;
mul.wide.u32 %rd40, %r251, 2;
add.s64 %rd41, %rd24, %rd40;
add.s64 %rd42, %rd41, %rd38;
add.s64 %rd8, %rd42, %rd34;
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
cvt.f32.bf16 %r252, %rs97;
cvt.f32.bf16 %r253, %rs98;
cvt.f32.bf16 %r254, %rs99;
cvt.f32.bf16 %r255, %rs100;
cvt.f32.bf16 %r256, %rs101;
cvt.f32.bf16 %r257, %rs102;
cvt.f32.bf16 %r258, %rs103;
cvt.f32.bf16 %r259, %rs104;
cvt.f32.bf16 %r260, %rs105;
cvt.f32.bf16 %r261, %rs106;
cvt.f32.bf16 %r262, %rs107;
cvt.f32.bf16 %r263, %rs108;
cvt.f32.bf16 %r264, %rs109;
cvt.f32.bf16 %r265, %rs110;
cvt.f32.bf16 %r266, %rs111;
cvt.f32.bf16 %r267, %rs112;
mul.f32 %r268, %r236, %r236;
"# else r#"@%p1 ld.global.v4.b32 { %r23, %r24, %r25, %r26 }, [ %rd23 + 0 ];
mov.b32 {%rs122, %rs123}, %r23;
mov.b32 {%rs124, %rs125}, %r24;
mov.b32 {%rs126, %rs127}, %r25;
mov.b32 {%rs128, %rs129}, %r26;
cvt.f32.bf16 %r285, %rs114;
cvt.f32.bf16 %r286, %rs115;
cvt.f32.bf16 %r287, %rs116;
cvt.f32.bf16 %r288, %rs117;
cvt.f32.bf16 %r289, %rs118;
cvt.f32.bf16 %r290, %rs119;
cvt.f32.bf16 %r291, %rs120;
cvt.f32.bf16 %r292, %rs121;
cvt.f32.bf16 %r293, %rs122;
cvt.f32.bf16 %r294, %rs123;
cvt.f32.bf16 %r295, %rs124;
cvt.f32.bf16 %r296, %rs125;
cvt.f32.bf16 %r297, %rs126;
cvt.f32.bf16 %r298, %rs127;
cvt.f32.bf16 %r299, %rs128;
cvt.f32.bf16 %r300, %rs129;
mul.f32 %r301, %r269, %r269;
"#) ++
  r#"$L__tmp1:
"# ++
  (if trimmed then r#"fma.rn.f32 %r269, %r235, %r235, %r268;
fma.rn.f32 %r270, %r237, %r237, %r269;
fma.rn.f32 %r271, %r238, %r238, %r270;
fma.rn.f32 %r272, %r239, %r239, %r271;
fma.rn.f32 %r273, %r240, %r240, %r272;
fma.rn.f32 %r274, %r241, %r241, %r273;
fma.rn.f32 %r275, %r242, %r242, %r274;
fma.rn.f32 %r276, %r243, %r243, %r275;
fma.rn.f32 %r277, %r244, %r244, %r276;
fma.rn.f32 %r278, %r245, %r245, %r277;
fma.rn.f32 %r279, %r246, %r246, %r278;
fma.rn.f32 %r280, %r247, %r247, %r279;
fma.rn.f32 %r281, %r248, %r248, %r280;
fma.rn.f32 %r282, %r249, %r249, %r281;
fma.rn.f32 %r283, %r250, %r250, %r282;
"# else r#"fma.rn.f32 %r302, %r268, %r268, %r301;
fma.rn.f32 %r303, %r270, %r270, %r302;
fma.rn.f32 %r304, %r271, %r271, %r303;
fma.rn.f32 %r305, %r272, %r272, %r304;
fma.rn.f32 %r306, %r273, %r273, %r305;
fma.rn.f32 %r307, %r274, %r274, %r306;
fma.rn.f32 %r308, %r275, %r275, %r307;
fma.rn.f32 %r309, %r276, %r276, %r308;
fma.rn.f32 %r310, %r277, %r277, %r309;
fma.rn.f32 %r311, %r278, %r278, %r310;
fma.rn.f32 %r312, %r279, %r279, %r311;
fma.rn.f32 %r313, %r280, %r280, %r312;
fma.rn.f32 %r314, %r281, %r281, %r313;
fma.rn.f32 %r315, %r282, %r282, %r314;
fma.rn.f32 %r316, %r283, %r283, %r315;
"#) ++
  r#"$L__tmp2:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r284, %r283, 4, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r317, %r316, 4, 31, -1;
"#) ++
  r#"$L__tmp3:
"# ++
  (if trimmed then r#"add.f32 %r285, %r283, %r284;
"# else r#"add.f32 %r318, %r316, %r317;
"#) ++
  r#"$L__tmp4:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r286, %r285, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r319, %r318, 2, 31, -1;
"#) ++
  r#"$L__tmp5:
"# ++
  (if trimmed then r#"add.f32 %r287, %r285, %r286;
"# else r#"add.f32 %r320, %r318, %r319;
"#) ++
  r#"$L__tmp6:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r288, %r287, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r321, %r320, 1, 31, -1;
"#) ++
  r#"$L__tmp7:
"# ++
  (if trimmed then r#"add.f32 %r289, %r287, %r288;
mov.b32 %r290, 0f43000000;
"# else r#"add.f32 %r322, %r320, %r321;
mov.b32 %r323, 0f43000000;
"#) ++
  r#"$L__tmp8:
"# ++
  (if trimmed then r#"div.full.f32 %r291, %r289, %r290;
add.f32 %r292, %r291, 0f34000000;
rsqrt.approx.ftz.f32 %r293, %r292;
mul.f32 %r294, %r293, %r235;
mul.f32 %r295, %r293, %r236;
mul.f32 %r296, %r293, %r237;
mul.f32 %r297, %r293, %r238;
mul.f32 %r298, %r293, %r239;
mul.f32 %r299, %r293, %r240;
mul.f32 %r300, %r293, %r241;
mul.f32 %r301, %r293, %r242;
mul.f32 %r302, %r293, %r243;
mul.f32 %r303, %r293, %r244;
mul.f32 %r304, %r293, %r245;
mul.f32 %r305, %r293, %r246;
mul.f32 %r306, %r293, %r247;
mul.f32 %r307, %r293, %r248;
mul.f32 %r308, %r293, %r249;
mul.f32 %r309, %r293, %r250;
mul.f32 %r310, %r217, %r253;
mul.f32 %r311, %r218, %r252;
mul.f32 %r312, %r219, %r255;
mul.f32 %r313, %r220, %r254;
mul.f32 %r314, %r221, %r257;
mul.f32 %r315, %r222, %r256;
mul.f32 %r316, %r223, %r259;
mul.f32 %r317, %r224, %r258;
mul.f32 %r318, %r225, %r261;
mul.f32 %r319, %r226, %r260;
mul.f32 %r320, %r227, %r263;
mul.f32 %r321, %r228, %r262;
mul.f32 %r322, %r229, %r265;
mul.f32 %r323, %r230, %r264;
mul.f32 %r324, %r231, %r267;
mul.f32 %r325, %r232, %r266;
fma.rn.f32 %r326, %r201, %r252, %r310;
fma.rn.f32 %r327, %r202, %r253, %r311;
fma.rn.f32 %r328, %r203, %r254, %r312;
fma.rn.f32 %r329, %r204, %r255, %r313;
fma.rn.f32 %r330, %r205, %r256, %r314;
fma.rn.f32 %r331, %r206, %r257, %r315;
fma.rn.f32 %r332, %r207, %r258, %r316;
fma.rn.f32 %r333, %r208, %r259, %r317;
fma.rn.f32 %r334, %r209, %r260, %r318;
fma.rn.f32 %r335, %r210, %r261, %r319;
fma.rn.f32 %r336, %r211, %r262, %r320;
fma.rn.f32 %r337, %r212, %r263, %r321;
fma.rn.f32 %r338, %r213, %r264, %r322;
fma.rn.f32 %r339, %r214, %r265, %r323;
fma.rn.f32 %r340, %r215, %r266, %r324;
fma.rn.f32 %r341, %r216, %r267, %r325;
mul.f32 %r342, %r327, %r295;
"# else r#"div.full.f32 %r324, %r322, %r323;
add.f32 %r325, %r324, 0f34000000;
rsqrt.approx.ftz.f32 %r326, %r325;
mul.f32 %r327, %r326, %r268;
mul.f32 %r328, %r326, %r269;
mul.f32 %r329, %r326, %r270;
mul.f32 %r330, %r326, %r271;
mul.f32 %r331, %r326, %r272;
mul.f32 %r332, %r326, %r273;
mul.f32 %r333, %r326, %r274;
mul.f32 %r334, %r326, %r275;
mul.f32 %r335, %r326, %r276;
mul.f32 %r336, %r326, %r277;
mul.f32 %r337, %r326, %r278;
mul.f32 %r338, %r326, %r279;
mul.f32 %r339, %r326, %r280;
mul.f32 %r340, %r326, %r281;
mul.f32 %r341, %r326, %r282;
mul.f32 %r342, %r326, %r283;
mul.f32 %r343, %r250, %r286;
mul.f32 %r344, %r251, %r285;
mul.f32 %r345, %r252, %r288;
mul.f32 %r346, %r253, %r287;
mul.f32 %r347, %r254, %r290;
mul.f32 %r348, %r255, %r289;
mul.f32 %r349, %r256, %r292;
mul.f32 %r350, %r257, %r291;
mul.f32 %r351, %r258, %r294;
mul.f32 %r352, %r259, %r293;
mul.f32 %r353, %r260, %r296;
mul.f32 %r354, %r261, %r295;
mul.f32 %r355, %r262, %r298;
mul.f32 %r356, %r263, %r297;
mul.f32 %r357, %r264, %r300;
mul.f32 %r358, %r265, %r299;
fma.rn.f32 %r359, %r198, %r285, %r343;
fma.rn.f32 %r360, %r199, %r286, %r344;
fma.rn.f32 %r361, %r200, %r287, %r345;
fma.rn.f32 %r362, %r201, %r288, %r346;
fma.rn.f32 %r363, %r202, %r289, %r347;
fma.rn.f32 %r364, %r203, %r290, %r348;
fma.rn.f32 %r365, %r204, %r291, %r349;
fma.rn.f32 %r366, %r205, %r292, %r350;
fma.rn.f32 %r367, %r206, %r293, %r351;
fma.rn.f32 %r368, %r207, %r294, %r352;
fma.rn.f32 %r369, %r208, %r295, %r353;
fma.rn.f32 %r370, %r209, %r296, %r354;
fma.rn.f32 %r371, %r210, %r297, %r355;
fma.rn.f32 %r372, %r211, %r298, %r356;
fma.rn.f32 %r373, %r212, %r299, %r357;
fma.rn.f32 %r374, %r213, %r300, %r358;
mul.f32 %r375, %r360, %r328;
"#) ++
  r#"$L__tmp9:
"# ++
  (if trimmed then r#"fma.rn.f32 %r343, %r326, %r294, %r342;
fma.rn.f32 %r344, %r328, %r296, %r343;
fma.rn.f32 %r345, %r329, %r297, %r344;
fma.rn.f32 %r346, %r330, %r298, %r345;
fma.rn.f32 %r347, %r331, %r299, %r346;
fma.rn.f32 %r348, %r332, %r300, %r347;
fma.rn.f32 %r349, %r333, %r301, %r348;
fma.rn.f32 %r350, %r334, %r302, %r349;
fma.rn.f32 %r351, %r335, %r303, %r350;
fma.rn.f32 %r352, %r336, %r304, %r351;
fma.rn.f32 %r353, %r337, %r305, %r352;
fma.rn.f32 %r354, %r338, %r306, %r353;
fma.rn.f32 %r355, %r339, %r307, %r354;
fma.rn.f32 %r356, %r340, %r308, %r355;
fma.rn.f32 %r357, %r341, %r309, %r356;
"# else r#"fma.rn.f32 %r376, %r359, %r327, %r375;
fma.rn.f32 %r377, %r361, %r329, %r376;
fma.rn.f32 %r378, %r362, %r330, %r377;
fma.rn.f32 %r379, %r363, %r331, %r378;
fma.rn.f32 %r380, %r364, %r332, %r379;
fma.rn.f32 %r381, %r365, %r333, %r380;
fma.rn.f32 %r382, %r366, %r334, %r381;
fma.rn.f32 %r383, %r367, %r335, %r382;
fma.rn.f32 %r384, %r368, %r336, %r383;
fma.rn.f32 %r385, %r369, %r337, %r384;
fma.rn.f32 %r386, %r370, %r338, %r385;
fma.rn.f32 %r387, %r371, %r339, %r386;
fma.rn.f32 %r388, %r372, %r340, %r387;
fma.rn.f32 %r389, %r373, %r341, %r388;
fma.rn.f32 %r390, %r374, %r342, %r389;
"#) ++
  r#"$L__tmp10:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r358, %r357, 4, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r391, %r390, 4, 31, -1;
"#) ++
  r#"$L__tmp11:
"# ++
  (if trimmed then r#"add.f32 %r359, %r357, %r358;
"# else r#"add.f32 %r392, %r390, %r391;
"#) ++
  r#"$L__tmp12:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r360, %r359, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r393, %r392, 2, 31, -1;
"#) ++
  r#"$L__tmp13:
"# ++
  (if trimmed then r#"add.f32 %r361, %r359, %r360;
"# else r#"add.f32 %r394, %r392, %r393;
"#) ++
  r#"$L__tmp14:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r362, %r361, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r395, %r394, 1, 31, -1;
"#) ++
  r#"$L__tmp15:
"# ++
  (if trimmed then r#"add.f32 %r363, %r361, %r362;
"# else r#"add.f32 %r396, %r394, %r395;
"#) ++
  r#"$L__tmp16:
"# ++
  (if trimmed then r#"div.full.f32 %r364, %r363, %r290;
neg.f32 %r365, %r294;
fma.rn.f32 %r366, %r365, %r364, %r326;
neg.f32 %r367, %r295;
fma.rn.f32 %r368, %r367, %r364, %r327;
neg.f32 %r369, %r296;
fma.rn.f32 %r370, %r369, %r364, %r328;
neg.f32 %r371, %r297;
fma.rn.f32 %r372, %r371, %r364, %r329;
neg.f32 %r373, %r298;
fma.rn.f32 %r374, %r373, %r364, %r330;
neg.f32 %r375, %r299;
fma.rn.f32 %r376, %r375, %r364, %r331;
neg.f32 %r377, %r300;
fma.rn.f32 %r378, %r377, %r364, %r332;
neg.f32 %r379, %r301;
fma.rn.f32 %r380, %r379, %r364, %r333;
neg.f32 %r381, %r302;
fma.rn.f32 %r382, %r381, %r364, %r334;
neg.f32 %r383, %r303;
fma.rn.f32 %r384, %r383, %r364, %r335;
neg.f32 %r385, %r304;
fma.rn.f32 %r386, %r385, %r364, %r336;
neg.f32 %r387, %r305;
fma.rn.f32 %r388, %r387, %r364, %r337;
neg.f32 %r389, %r306;
fma.rn.f32 %r390, %r389, %r364, %r338;
neg.f32 %r391, %r307;
fma.rn.f32 %r392, %r391, %r364, %r339;
neg.f32 %r393, %r308;
fma.rn.f32 %r394, %r393, %r364, %r340;
neg.f32 %r395, %r309;
fma.rn.f32 %r396, %r395, %r364, %r341;
mul.f32 %r397, %r293, %r366;
mul.f32 %r398, %r293, %r368;
mul.f32 %r399, %r293, %r370;
mul.f32 %r400, %r293, %r372;
mul.f32 %r401, %r293, %r374;
mul.f32 %r402, %r293, %r376;
mul.f32 %r403, %r293, %r378;
mul.f32 %r404, %r293, %r380;
mul.f32 %r405, %r293, %r382;
mul.f32 %r406, %r293, %r384;
mul.f32 %r407, %r293, %r386;
mul.f32 %r408, %r293, %r388;
mul.f32 %r409, %r293, %r390;
mul.f32 %r410, %r293, %r392;
mul.f32 %r411, %r293, %r394;
mul.f32 %r412, %r293, %r396;
cvt.rn.bf16.f32 %rs113, %r397;
cvt.rn.bf16.f32 %rs114, %r398;
cvt.rn.bf16.f32 %rs115, %r399;
cvt.rn.bf16.f32 %rs116, %r400;
cvt.rn.bf16.f32 %rs117, %r401;
cvt.rn.bf16.f32 %rs118, %r402;
cvt.rn.bf16.f32 %rs119, %r403;
cvt.rn.bf16.f32 %rs120, %r404;
cvt.rn.bf16.f32 %rs121, %r405;
cvt.rn.bf16.f32 %rs122, %r406;
cvt.rn.bf16.f32 %rs123, %r407;
cvt.rn.bf16.f32 %rs124, %r408;
cvt.rn.bf16.f32 %rs125, %r409;
cvt.rn.bf16.f32 %rs126, %r410;
cvt.rn.bf16.f32 %rs127, %r411;
cvt.rn.bf16.f32 %rs128, %r412;
shl.b32 %r413, %r190, 3;
and.b32 %r414, %r413, 896;
bfe.s32 %r415, %r190, 3, 1;
and.b32 %r416, %r415, 1040;
bfe.s32 %r417, %r190, 7, 1;
and.b32 %r418, %r417, 2080;
or.b32 %r419, %r197, %r414;
or.b32 %r420, %r416, %r418;
xor.b32 %r421, %r420, %r419;
mov.b32 %r422, global_smem;
add.s32 %r423, %r422, %r421;
mov.b32 %r424, {%rs125, %rs127};
mov.b32 %r425, {%rs121, %rs123};
mov.b32 %r426, {%rs117, %rs119};
mov.b32 %r427, {%rs113, %rs115};
st.shared.v4.b32 [%r423], {%r427, %r426, %r425, %r424};
xor.b32 %r428, %r421, 64;
add.s32 %r429, %r422, %r428;
mov.b32 %r430, {%rs126, %rs128};
mov.b32 %r431, {%rs122, %rs124};
mov.b32 %r432, {%rs118, %rs120};
mov.b32 %r433, {%rs114, %rs116};
st.shared.v4.b32 [%r429+4096], {%r433, %r432, %r431, %r430};
"# else r#"div.full.f32 %r397, %r396, %r323;
neg.f32 %r398, %r327;
fma.rn.f32 %r399, %r398, %r397, %r359;
neg.f32 %r400, %r328;
fma.rn.f32 %r401, %r400, %r397, %r360;
neg.f32 %r402, %r329;
fma.rn.f32 %r403, %r402, %r397, %r361;
neg.f32 %r404, %r330;
fma.rn.f32 %r405, %r404, %r397, %r362;
neg.f32 %r406, %r331;
fma.rn.f32 %r407, %r406, %r397, %r363;
neg.f32 %r408, %r332;
fma.rn.f32 %r409, %r408, %r397, %r364;
neg.f32 %r410, %r333;
fma.rn.f32 %r411, %r410, %r397, %r365;
neg.f32 %r412, %r334;
fma.rn.f32 %r413, %r412, %r397, %r366;
neg.f32 %r414, %r335;
fma.rn.f32 %r415, %r414, %r397, %r367;
neg.f32 %r416, %r336;
fma.rn.f32 %r417, %r416, %r397, %r368;
neg.f32 %r418, %r337;
fma.rn.f32 %r419, %r418, %r397, %r369;
neg.f32 %r420, %r338;
fma.rn.f32 %r421, %r420, %r397, %r370;
neg.f32 %r422, %r339;
fma.rn.f32 %r423, %r422, %r397, %r371;
neg.f32 %r424, %r340;
fma.rn.f32 %r425, %r424, %r397, %r372;
neg.f32 %r426, %r341;
fma.rn.f32 %r427, %r426, %r397, %r373;
neg.f32 %r428, %r342;
fma.rn.f32 %r429, %r428, %r397, %r374;
mul.f32 %r430, %r326, %r399;
mul.f32 %r431, %r326, %r401;
mul.f32 %r432, %r326, %r403;
mul.f32 %r433, %r326, %r405;
mul.f32 %r434, %r326, %r407;
mul.f32 %r435, %r326, %r409;
mul.f32 %r436, %r326, %r411;
mul.f32 %r437, %r326, %r413;
mul.f32 %r438, %r326, %r415;
mul.f32 %r439, %r326, %r417;
mul.f32 %r440, %r326, %r419;
mul.f32 %r441, %r326, %r421;
mul.f32 %r442, %r326, %r423;
mul.f32 %r443, %r326, %r425;
mul.f32 %r444, %r326, %r427;
mul.f32 %r445, %r326, %r429;
cvt.rn.bf16.f32 %rs130, %r430;
cvt.rn.bf16.f32 %rs131, %r431;
cvt.rn.bf16.f32 %rs132, %r432;
cvt.rn.bf16.f32 %rs133, %r433;
cvt.rn.bf16.f32 %rs134, %r434;
cvt.rn.bf16.f32 %rs135, %r435;
cvt.rn.bf16.f32 %rs136, %r436;
cvt.rn.bf16.f32 %rs137, %r437;
cvt.rn.bf16.f32 %rs138, %r438;
cvt.rn.bf16.f32 %rs139, %r439;
cvt.rn.bf16.f32 %rs140, %r440;
cvt.rn.bf16.f32 %rs141, %r441;
cvt.rn.bf16.f32 %rs142, %r442;
cvt.rn.bf16.f32 %rs143, %r443;
cvt.rn.bf16.f32 %rs144, %r444;
cvt.rn.bf16.f32 %rs145, %r445;
"#) ++
  r#"bar.sync 0;
"# ++
  (if trimmed then r#"shl.b32 %r434, %r196, 10;
and.b32 %r435, %r192, 496;
and.b32 %r436, %r198, 112;
xor.b32 %r437, %r435, %r436;
add.s32 %r438, %r422, %r434;
add.s32 %r439, %r438, %r437;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r440, %r441, %r442, %r443}, [%r439];
mov.b32 {%rs129, %rs130}, %r440;
mov.b32 {%rs131, %rs132}, %r441;
mov.b32 {%rs133, %rs134}, %r442;
mov.b32 {%rs135, %rs136}, %r443;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r444, %r445, %r446, %r447}, [%r439+512];
mov.b32 {%rs137, %rs138}, %r444;
mov.b32 {%rs139, %rs140}, %r445;
mov.b32 {%rs141, %rs142}, %r446;
mov.b32 {%rs143, %rs144}, %r447;
cvt.f32.bf16 %r448, %rs129;
cvt.f32.bf16 %r449, %rs130;
cvt.f32.bf16 %r450, %rs131;
cvt.f32.bf16 %r451, %rs132;
cvt.f32.bf16 %r452, %rs133;
cvt.f32.bf16 %r453, %rs134;
cvt.f32.bf16 %r454, %rs135;
cvt.f32.bf16 %r455, %rs136;
cvt.f32.bf16 %r456, %rs137;
cvt.f32.bf16 %r457, %rs138;
cvt.f32.bf16 %r458, %rs139;
cvt.f32.bf16 %r459, %rs140;
cvt.f32.bf16 %r460, %rs141;
cvt.f32.bf16 %r461, %rs142;
cvt.f32.bf16 %r462, %rs143;
cvt.f32.bf16 %r463, %rs144;
cvt.f32.bf16 %r464, %rs113;
cvt.f32.bf16 %r465, %rs114;
cvt.f32.bf16 %r466, %rs115;
cvt.f32.bf16 %r467, %rs116;
cvt.f32.bf16 %r468, %rs117;
cvt.f32.bf16 %r469, %rs118;
cvt.f32.bf16 %r470, %rs119;
cvt.f32.bf16 %r471, %rs120;
cvt.f32.bf16 %r472, %rs121;
cvt.f32.bf16 %r473, %rs122;
cvt.f32.bf16 %r474, %rs123;
cvt.f32.bf16 %r475, %rs124;
cvt.f32.bf16 %r476, %rs125;
cvt.f32.bf16 %r477, %rs126;
cvt.f32.bf16 %r478, %rs127;
cvt.f32.bf16 %r479, %rs128;
div.full.f32 %r35, %r448, %r18;
div.full.f32 %r36, %r449, %r18;
div.full.f32 %r37, %r450, %r18;
div.full.f32 %r38, %r451, %r18;
div.full.f32 %r39, %r452, %r18;
div.full.f32 %r40, %r453, %r18;
div.full.f32 %r41, %r454, %r18;
div.full.f32 %r42, %r455, %r18;
div.full.f32 %r43, %r456, %r18;
div.full.f32 %r44, %r457, %r18;
div.full.f32 %r45, %r458, %r18;
div.full.f32 %r46, %r459, %r18;
div.full.f32 %r47, %r460, %r18;
div.full.f32 %r48, %r461, %r18;
div.full.f32 %r49, %r462, %r18;
div.full.f32 %r50, %r463, %r18;
div.full.f32 %r51, %r464, %r18;
div.full.f32 %r52, %r465, %r18;
div.full.f32 %r53, %r466, %r18;
div.full.f32 %r54, %r467, %r18;
div.full.f32 %r55, %r468, %r18;
div.full.f32 %r56, %r469, %r18;
div.full.f32 %r57, %r470, %r18;
div.full.f32 %r58, %r471, %r18;
div.full.f32 %r59, %r472, %r18;
div.full.f32 %r60, %r473, %r18;
div.full.f32 %r61, %r474, %r18;
div.full.f32 %r62, %r475, %r18;
div.full.f32 %r63, %r476, %r18;
div.full.f32 %r64, %r477, %r18;
div.full.f32 %r65, %r478, %r18;
div.full.f32 %r66, %r479, %r18;
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
or.b32 %r480, %r197, %r234;
or.b32 %r481, %r199, %r234;
add.s64 %rd43, %rd30, %rd36;
cvt.s64.s32 %rd44, %r480;
add.s64 %rd10, %rd43, %rd44;
mov.b32 %r67, {%rs9, %rs10};
mov.b32 %r68, {%rs11, %rs12};
mov.b32 %r69, {%rs13, %rs14};
mov.b32 %r70, {%rs15, %rs16};
@%p1 st.global.v4.b32 [ %rd10 + 0 ], { %r67, %r68, %r69, %r70 };
@PITCH482_481@
cvt.s64.s32 %rd45, %r482;
add.s64 %rd46, %rd31, %rd45;
cvt.u64.u32 %rd47, %r195;
add.s64 %rd11, %rd46, %rd47;
mov.b32 %r71, {%rs1, %rs2};
mov.b32 %r72, {%rs3, %rs4};
mov.b32 %r73, {%rs5, %rs6};
mov.b32 %r74, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd11 + 0 ], { %r71, %r72, %r73, %r74 };
add.s32 %r483, %r234, 768;
mad.wide.s32 %rd48, %r483, 2, %rd37;
add.s64 %rd12, %rd48, %rd34;
add.s64 %rd13, %rd12, 16;
"# else r#"and.b32 %r446, %r232, 896;
bfe.s32 %r447, %r182, 3, 1;
and.b32 %r448, %r447, 1040;
bfe.s32 %r449, %r182, 7, 1;
and.b32 %r450, %r449, 2080;
or.b32 %r451, %r193, %r446;
or.b32 %r452, %r448, %r450;
xor.b32 %r453, %r452, %r451;
add.s32 %r454, %r218, %r453;
mov.b32 %r455, {%rs142, %rs144};
mov.b32 %r456, {%rs138, %rs140};
mov.b32 %r457, {%rs134, %rs136};
mov.b32 %r458, {%rs130, %rs132};
st.shared.v4.b32 [%r454], {%r458, %r457, %r456, %r455};
xor.b32 %r459, %r453, 64;
add.s32 %r460, %r218, %r459;
mov.b32 %r461, {%rs143, %rs145};
mov.b32 %r462, {%rs139, %rs141};
mov.b32 %r463, {%rs135, %rs137};
mov.b32 %r464, {%rs131, %rs133};
st.shared.v4.b32 [%r460+4096], {%r464, %r463, %r462, %r461};
bar.sync 0;
shl.b32 %r465, %r192, 10;
shl.b32 %r466, %r184, 4;
shr.u32 %r467, %r190, 1;
xor.b32 %r468, %r466, %r467;
add.s32 %r469, %r218, %r465;
add.s32 %r470, %r469, %r468;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r471, %r472, %r473, %r474}, [%r470];
mov.b32 {%rs146, %rs147}, %r471;
mov.b32 {%rs148, %rs149}, %r472;
mov.b32 {%rs150, %rs151}, %r473;
mov.b32 {%rs152, %rs153}, %r474;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r475, %r476, %r477, %r478}, [%r470+512];
mov.b32 {%rs154, %rs155}, %r475;
mov.b32 {%rs156, %rs157}, %r476;
mov.b32 {%rs158, %rs159}, %r477;
mov.b32 {%rs160, %rs161}, %r478;
cvt.f32.bf16 %r479, %rs146;
cvt.f32.bf16 %r480, %rs147;
cvt.f32.bf16 %r481, %rs148;
cvt.f32.bf16 %r482, %rs149;
cvt.f32.bf16 %r483, %rs150;
cvt.f32.bf16 %r484, %rs151;
cvt.f32.bf16 %r485, %rs152;
cvt.f32.bf16 %r486, %rs153;
cvt.f32.bf16 %r487, %rs154;
cvt.f32.bf16 %r488, %rs155;
cvt.f32.bf16 %r489, %rs156;
cvt.f32.bf16 %r490, %rs157;
cvt.f32.bf16 %r491, %rs158;
cvt.f32.bf16 %r492, %rs159;
cvt.f32.bf16 %r493, %rs160;
cvt.f32.bf16 %r494, %rs161;
cvt.f32.bf16 %r495, %rs130;
cvt.f32.bf16 %r496, %rs131;
cvt.f32.bf16 %r497, %rs132;
cvt.f32.bf16 %r498, %rs133;
cvt.f32.bf16 %r499, %rs134;
cvt.f32.bf16 %r500, %rs135;
cvt.f32.bf16 %r501, %rs136;
cvt.f32.bf16 %r502, %rs137;
cvt.f32.bf16 %r503, %rs138;
cvt.f32.bf16 %r504, %rs139;
cvt.f32.bf16 %r505, %rs140;
cvt.f32.bf16 %r506, %rs141;
cvt.f32.bf16 %r507, %rs142;
cvt.f32.bf16 %r508, %rs143;
cvt.f32.bf16 %r509, %rs144;
cvt.f32.bf16 %r510, %rs145;
div.full.f32 %r27, %r479, %r10;
div.full.f32 %r28, %r480, %r10;
div.full.f32 %r29, %r481, %r10;
div.full.f32 %r30, %r482, %r10;
div.full.f32 %r31, %r483, %r10;
div.full.f32 %r32, %r484, %r10;
div.full.f32 %r33, %r485, %r10;
div.full.f32 %r34, %r486, %r10;
div.full.f32 %r35, %r487, %r10;
div.full.f32 %r36, %r488, %r10;
div.full.f32 %r37, %r489, %r10;
div.full.f32 %r38, %r490, %r10;
div.full.f32 %r39, %r491, %r10;
div.full.f32 %r40, %r492, %r10;
div.full.f32 %r41, %r493, %r10;
div.full.f32 %r42, %r494, %r10;
div.full.f32 %r43, %r495, %r10;
div.full.f32 %r44, %r496, %r10;
div.full.f32 %r45, %r497, %r10;
div.full.f32 %r46, %r498, %r10;
div.full.f32 %r47, %r499, %r10;
div.full.f32 %r48, %r500, %r10;
div.full.f32 %r49, %r501, %r10;
div.full.f32 %r50, %r502, %r10;
div.full.f32 %r51, %r503, %r10;
div.full.f32 %r52, %r504, %r10;
div.full.f32 %r53, %r505, %r10;
div.full.f32 %r54, %r506, %r10;
div.full.f32 %r55, %r507, %r10;
div.full.f32 %r56, %r508, %r10;
div.full.f32 %r57, %r509, %r10;
div.full.f32 %r58, %r510, %r10;
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
or.b32 %r511, %r193, %r267;
or.b32 %r512, %r194, %r267;
add.s64 %rd60, %rd44, %rd53;
cvt.s64.s32 %rd61, %r511;
add.s64 %rd24, %rd60, %rd61;
mov.b32 %r59, {%rs26, %rs27};
mov.b32 %r60, {%rs28, %rs29};
mov.b32 %r61, {%rs30, %rs31};
mov.b32 %r62, {%rs32, %rs33};
@%p1 st.global.v4.b32 [ %rd24 + 0 ], { %r59, %r60, %r61, %r62 };
@PITCH513_512@
cvt.s64.s32 %rd62, %r513;
add.s64 %rd63, %rd45, %rd62;
cvt.u64.u32 %rd64, %r189;
add.s64 %rd25, %rd63, %rd64;
mov.b32 %r63, {%rs18, %rs19};
mov.b32 %r64, {%rs20, %rs21};
mov.b32 %r65, {%rs22, %rs23};
mov.b32 %r66, {%rs24, %rs25};
@%p1 st.global.v4.b32 [ %rd25 + 0 ], { %r63, %r64, %r65, %r66 };
add.s32 %r514, %r267, 768;
mad.wide.s32 %rd65, %r514, 2, %rd54;
add.s64 %rd26, %rd65, %rd47;
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
cvt.f32.bf16 %r515, %rs162;
cvt.f32.bf16 %r516, %rs163;
cvt.f32.bf16 %r517, %rs164;
cvt.f32.bf16 %r518, %rs165;
cvt.f32.bf16 %r519, %rs166;
cvt.f32.bf16 %r520, %rs167;
cvt.f32.bf16 %r521, %rs168;
cvt.f32.bf16 %r522, %rs169;
cvt.f32.bf16 %r523, %rs170;
cvt.f32.bf16 %r524, %rs171;
cvt.f32.bf16 %r525, %rs172;
cvt.f32.bf16 %r526, %rs173;
cvt.f32.bf16 %r527, %rs174;
cvt.f32.bf16 %r528, %rs175;
cvt.f32.bf16 %r529, %rs176;
cvt.f32.bf16 %r530, %rs177;
add.s64 %rd66, %rd39, %rd57;
add.s64 %rd67, %rd66, %rd55;
add.s64 %rd28, %rd67, %rd47;
add.s64 %rd29, %rd28, 16;
"#) ++
  r#"mov.u32 %r75, %r5;
mov.u32 %r76, %r5;
mov.u32 %r77, %r5;
mov.u32 %r78, %r5;
"# ++
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r75, %r76, %r77, %r78 }, [ %rd12 + 0 ];
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
  (if trimmed then r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd13 + 0 ];
mov.b32 {%rs153, %rs154}, %r79;
mov.b32 {%rs155, %rs156}, %r80;
mov.b32 {%rs157, %rs158}, %r81;
mov.b32 {%rs159, %rs160}, %r82;
cvt.f32.bf16 %r484, %rs145;
cvt.f32.bf16 %r485, %rs146;
cvt.f32.bf16 %r486, %rs147;
cvt.f32.bf16 %r487, %rs148;
cvt.f32.bf16 %r488, %rs149;
cvt.f32.bf16 %r489, %rs150;
cvt.f32.bf16 %r490, %rs151;
cvt.f32.bf16 %r491, %rs152;
cvt.f32.bf16 %r492, %rs153;
cvt.f32.bf16 %r493, %rs154;
cvt.f32.bf16 %r494, %rs155;
cvt.f32.bf16 %r495, %rs156;
cvt.f32.bf16 %r496, %rs157;
cvt.f32.bf16 %r497, %rs158;
cvt.f32.bf16 %r498, %rs159;
cvt.f32.bf16 %r499, %rs160;
add.s64 %rd49, %rd25, %rd40;
add.s64 %rd50, %rd49, %rd38;
add.s64 %rd14, %rd50, %rd34;
add.s64 %rd15, %rd14, 16;
setp.lt.u32 %p4, %r196, 4;
setp.eq.b32 %p5, %r194, 0;
or.pred %p6, %p4, %p5;
and.pred %p2, %p1, %p6;
"# else r#"@%p1 ld.global.v4.b32 { %r79, %r80, %r81, %r82 }, [ %rd29 + 0 ];
mul.f32 %r531, %r516, %r516;
$L__tmp17:
fma.rn.f32 %r532, %r515, %r515, %r531;
fma.rn.f32 %r533, %r517, %r517, %r532;
fma.rn.f32 %r534, %r518, %r518, %r533;
fma.rn.f32 %r535, %r519, %r519, %r534;
fma.rn.f32 %r536, %r520, %r520, %r535;
fma.rn.f32 %r537, %r521, %r521, %r536;
fma.rn.f32 %r538, %r522, %r522, %r537;
fma.rn.f32 %r539, %r523, %r523, %r538;
fma.rn.f32 %r540, %r524, %r524, %r539;
fma.rn.f32 %r541, %r525, %r525, %r540;
fma.rn.f32 %r542, %r526, %r526, %r541;
fma.rn.f32 %r543, %r527, %r527, %r542;
fma.rn.f32 %r544, %r528, %r528, %r543;
fma.rn.f32 %r545, %r529, %r529, %r544;
fma.rn.f32 %r546, %r530, %r530, %r545;
$L__tmp18:
shfl.sync.bfly.b32 %r547, %r546, 4, 31, -1;
$L__tmp19:
add.f32 %r548, %r546, %r547;
$L__tmp20:
shfl.sync.bfly.b32 %r549, %r548, 2, 31, -1;
$L__tmp21:
add.f32 %r550, %r548, %r549;
$L__tmp22:
shfl.sync.bfly.b32 %r551, %r550, 1, 31, -1;
$L__tmp23:
add.f32 %r552, %r550, %r551;
$L__tmp24:
div.full.f32 %r553, %r552, %r323;
add.f32 %r554, %r553, 0f34000000;
rsqrt.approx.ftz.f32 %r555, %r554;
mov.b32 {%rs178, %rs179}, %r82;
cvt.f32.bf16 %r556, %rs179;
cvt.f32.bf16 %r557, %rs178;
mov.b32 {%rs180, %rs181}, %r81;
cvt.f32.bf16 %r558, %rs181;
cvt.f32.bf16 %r559, %rs180;
mov.b32 {%rs182, %rs183}, %r80;
cvt.f32.bf16 %r560, %rs183;
cvt.f32.bf16 %r561, %rs182;
mov.b32 {%rs184, %rs185}, %r79;
cvt.f32.bf16 %r562, %rs185;
cvt.f32.bf16 %r563, %rs184;
mov.b32 {%rs186, %rs187}, %r78;
cvt.f32.bf16 %r564, %rs187;
cvt.f32.bf16 %r565, %rs186;
mov.b32 {%rs188, %rs189}, %r77;
cvt.f32.bf16 %r566, %rs189;
cvt.f32.bf16 %r567, %rs188;
mov.b32 {%rs190, %rs191}, %r76;
cvt.f32.bf16 %r568, %rs191;
cvt.f32.bf16 %r569, %rs190;
mov.b32 {%rs192, %rs193}, %r75;
cvt.f32.bf16 %r570, %rs193;
cvt.f32.bf16 %r571, %rs192;
mul.f32 %r572, %r555, %r515;
mul.f32 %r573, %r555, %r516;
mul.f32 %r574, %r555, %r517;
mul.f32 %r575, %r555, %r518;
mul.f32 %r576, %r555, %r519;
mul.f32 %r577, %r555, %r520;
mul.f32 %r578, %r555, %r521;
mul.f32 %r579, %r555, %r522;
mul.f32 %r580, %r555, %r523;
mul.f32 %r581, %r555, %r524;
mul.f32 %r582, %r555, %r525;
mul.f32 %r583, %r555, %r526;
mul.f32 %r584, %r555, %r527;
mul.f32 %r585, %r555, %r528;
mul.f32 %r586, %r555, %r529;
mul.f32 %r587, %r555, %r530;
mul.f32 %r588, %r250, %r570;
mul.f32 %r589, %r251, %r571;
mul.f32 %r590, %r252, %r568;
mul.f32 %r591, %r253, %r569;
mul.f32 %r592, %r254, %r566;
mul.f32 %r593, %r255, %r567;
mul.f32 %r594, %r256, %r564;
mul.f32 %r595, %r257, %r565;
mul.f32 %r596, %r258, %r562;
mul.f32 %r597, %r259, %r563;
mul.f32 %r598, %r260, %r560;
mul.f32 %r599, %r261, %r561;
mul.f32 %r600, %r262, %r558;
mul.f32 %r601, %r263, %r559;
mul.f32 %r602, %r264, %r556;
mul.f32 %r603, %r265, %r557;
fma.rn.f32 %r604, %r198, %r571, %r588;
fma.rn.f32 %r605, %r199, %r570, %r589;
fma.rn.f32 %r606, %r200, %r569, %r590;
fma.rn.f32 %r607, %r201, %r568, %r591;
fma.rn.f32 %r608, %r202, %r567, %r592;
fma.rn.f32 %r609, %r203, %r566, %r593;
fma.rn.f32 %r610, %r204, %r565, %r594;
fma.rn.f32 %r611, %r205, %r564, %r595;
fma.rn.f32 %r612, %r206, %r563, %r596;
fma.rn.f32 %r613, %r207, %r562, %r597;
fma.rn.f32 %r614, %r208, %r561, %r598;
fma.rn.f32 %r615, %r209, %r560, %r599;
fma.rn.f32 %r616, %r210, %r559, %r600;
fma.rn.f32 %r617, %r211, %r558, %r601;
fma.rn.f32 %r618, %r212, %r557, %r602;
fma.rn.f32 %r619, %r213, %r556, %r603;
min.u32 %r620, %r187, @PRELAST@;
setp.lt.u32 %p3, %r187, @LAST@;
and.pred %p2, %p1, %p3;
mul.lo.s32 %r621, %r620, 768;
mad.wide.u32 %rd68, %r621, 2, %rd39;
add.s64 %rd69, %rd68, %rd55;
add.s64 %rd70, %rd69, %rd47;
add.s64 %rd30, %rd70, 1536;
add.s64 %rd31, %rd70, 1552;
"#) ++
  r#"mov.u32 %r83, %r5;
mov.u32 %r84, %r5;
mov.u32 %r85, %r5;
mov.u32 %r86, %r5;
"# ++
  (if trimmed then r#"@%p2 ld.global.v4.b32 { %r83, %r84, %r85, %r86 }, [ %rd14 + 0 ];
"# else r#"@%p2 ld.global.v4.b32 { %r83, %r84, %r85, %r86 }, [ %rd30 + 0 ];
mov.b32 {%rs194, %rs195}, %r83;
mov.b32 {%rs196, %rs197}, %r84;
mov.b32 {%rs198, %rs199}, %r85;
mov.b32 {%rs200, %rs201}, %r86;
"#) ++
  r#"mov.u32 %r87, %r5;
mov.u32 %r88, %r5;
mov.u32 %r89, %r5;
mov.u32 %r90, %r5;
"# ++
  (if trimmed then r#"@%p2 ld.global.v4.b32 { %r87, %r88, %r89, %r90 }, [ %rd15 + 0 ];
mul.f32 %r500, %r485, %r485;
$L__tmp17:
fma.rn.f32 %r501, %r484, %r484, %r500;
fma.rn.f32 %r502, %r486, %r486, %r501;
fma.rn.f32 %r503, %r487, %r487, %r502;
fma.rn.f32 %r504, %r488, %r488, %r503;
fma.rn.f32 %r505, %r489, %r489, %r504;
fma.rn.f32 %r506, %r490, %r490, %r505;
fma.rn.f32 %r507, %r491, %r491, %r506;
fma.rn.f32 %r508, %r492, %r492, %r507;
fma.rn.f32 %r509, %r493, %r493, %r508;
fma.rn.f32 %r510, %r494, %r494, %r509;
fma.rn.f32 %r511, %r495, %r495, %r510;
fma.rn.f32 %r512, %r496, %r496, %r511;
fma.rn.f32 %r513, %r497, %r497, %r512;
fma.rn.f32 %r514, %r498, %r498, %r513;
fma.rn.f32 %r515, %r499, %r499, %r514;
$L__tmp18:
shfl.sync.bfly.b32 %r516, %r515, 4, 31, -1;
$L__tmp19:
add.f32 %r517, %r515, %r516;
$L__tmp20:
shfl.sync.bfly.b32 %r518, %r517, 2, 31, -1;
$L__tmp21:
add.f32 %r519, %r517, %r518;
$L__tmp22:
shfl.sync.bfly.b32 %r520, %r519, 1, 31, -1;
$L__tmp23:
add.f32 %r521, %r519, %r520;
$L__tmp24:
div.full.f32 %r522, %r521, %r290;
add.f32 %r523, %r522, 0f34000000;
rsqrt.approx.ftz.f32 %r524, %r523;
mov.b32 {%rs161, %rs162}, %r90;
cvt.f32.bf16 %r525, %rs162;
cvt.f32.bf16 %r526, %rs161;
mov.b32 {%rs163, %rs164}, %r89;
cvt.f32.bf16 %r527, %rs164;
cvt.f32.bf16 %r528, %rs163;
mov.b32 {%rs165, %rs166}, %r88;
cvt.f32.bf16 %r529, %rs166;
cvt.f32.bf16 %r530, %rs165;
mov.b32 {%rs167, %rs168}, %r87;
cvt.f32.bf16 %r531, %rs168;
cvt.f32.bf16 %r532, %rs167;
mov.b32 {%rs169, %rs170}, %r86;
cvt.f32.bf16 %r533, %rs170;
cvt.f32.bf16 %r534, %rs169;
mov.b32 {%rs171, %rs172}, %r85;
cvt.f32.bf16 %r535, %rs172;
cvt.f32.bf16 %r536, %rs171;
mov.b32 {%rs173, %rs174}, %r84;
cvt.f32.bf16 %r537, %rs174;
cvt.f32.bf16 %r538, %rs173;
mov.b32 {%rs175, %rs176}, %r83;
cvt.f32.bf16 %r539, %rs176;
cvt.f32.bf16 %r540, %rs175;
mul.f32 %r541, %r524, %r484;
mul.f32 %r542, %r524, %r485;
mul.f32 %r543, %r524, %r486;
mul.f32 %r544, %r524, %r487;
mul.f32 %r545, %r524, %r488;
mul.f32 %r546, %r524, %r489;
mul.f32 %r547, %r524, %r490;
mul.f32 %r548, %r524, %r491;
mul.f32 %r549, %r524, %r492;
mul.f32 %r550, %r524, %r493;
mul.f32 %r551, %r524, %r494;
mul.f32 %r552, %r524, %r495;
mul.f32 %r553, %r524, %r496;
mul.f32 %r554, %r524, %r497;
mul.f32 %r555, %r524, %r498;
mul.f32 %r556, %r524, %r499;
mul.f32 %r557, %r217, %r539;
mul.f32 %r558, %r218, %r540;
mul.f32 %r559, %r219, %r537;
mul.f32 %r560, %r220, %r538;
mul.f32 %r561, %r221, %r535;
mul.f32 %r562, %r222, %r536;
mul.f32 %r563, %r223, %r533;
mul.f32 %r564, %r224, %r534;
mul.f32 %r565, %r225, %r531;
mul.f32 %r566, %r226, %r532;
mul.f32 %r567, %r227, %r529;
mul.f32 %r568, %r228, %r530;
mul.f32 %r569, %r229, %r527;
mul.f32 %r570, %r230, %r528;
mul.f32 %r571, %r231, %r525;
mul.f32 %r572, %r232, %r526;
fma.rn.f32 %r573, %r201, %r540, %r557;
fma.rn.f32 %r574, %r202, %r539, %r558;
fma.rn.f32 %r575, %r203, %r538, %r559;
fma.rn.f32 %r576, %r204, %r537, %r560;
fma.rn.f32 %r577, %r205, %r536, %r561;
fma.rn.f32 %r578, %r206, %r535, %r562;
fma.rn.f32 %r579, %r207, %r534, %r563;
fma.rn.f32 %r580, %r208, %r533, %r564;
fma.rn.f32 %r581, %r209, %r532, %r565;
fma.rn.f32 %r582, %r210, %r531, %r566;
fma.rn.f32 %r583, %r211, %r530, %r567;
fma.rn.f32 %r584, %r212, %r529, %r568;
fma.rn.f32 %r585, %r213, %r528, %r569;
fma.rn.f32 %r586, %r214, %r527, %r570;
fma.rn.f32 %r587, %r215, %r526, %r571;
fma.rn.f32 %r588, %r216, %r525, %r572;
min.u32 %r589, %r194, @PRELAST@;
setp.lt.u32 %p7, %r194, @LAST@;
and.pred %p8, %p1, %p7;
setp.gt.u32 %p9, %r196, 3;
and.pred %p3, %p9, %p8;
mul.lo.s32 %r590, %r589, 768;
mad.wide.u32 %rd51, %r590, 2, %rd25;
add.s64 %rd52, %rd51, %rd38;
add.s64 %rd53, %rd52, %rd34;
add.s64 %rd16, %rd53, 1536;
add.s64 %rd17, %rd53, 1552;
mov.u32 %r91, %r5;
mov.u32 %r92, %r5;
mov.u32 %r93, %r5;
mov.u32 %r94, %r5;
@%p3 ld.global.v4.b32 { %r91, %r92, %r93, %r94 }, [ %rd16 + 0 ];
mov.b32 {%rs177, %rs178}, %r91;
mov.b32 {%rs179, %rs180}, %r92;
mov.b32 {%rs181, %rs182}, %r93;
mov.b32 {%rs183, %rs184}, %r94;
mov.u32 %r95, %r5;
mov.u32 %r96, %r5;
mov.u32 %r97, %r5;
mov.u32 %r98, %r5;
@%p3 ld.global.v4.b32 { %r95, %r96, %r97, %r98 }, [ %rd17 + 0 ];
mov.b32 {%rs185, %rs186}, %r95;
mov.b32 {%rs187, %rs188}, %r96;
mov.b32 {%rs189, %rs190}, %r97;
mov.b32 {%rs191, %rs192}, %r98;
cvt.f32.bf16 %r591, %rs177;
cvt.f32.bf16 %r592, %rs178;
cvt.f32.bf16 %r593, %rs179;
cvt.f32.bf16 %r594, %rs180;
cvt.f32.bf16 %r595, %rs181;
cvt.f32.bf16 %r596, %rs182;
cvt.f32.bf16 %r597, %rs183;
cvt.f32.bf16 %r598, %rs184;
cvt.f32.bf16 %r599, %rs185;
cvt.f32.bf16 %r600, %rs186;
cvt.f32.bf16 %r601, %rs187;
cvt.f32.bf16 %r602, %rs188;
cvt.f32.bf16 %r603, %rs189;
cvt.f32.bf16 %r604, %rs190;
cvt.f32.bf16 %r605, %rs191;
cvt.f32.bf16 %r606, %rs192;
add.f32 %r607, %r540, %r591;
add.f32 %r608, %r539, %r592;
add.f32 %r609, %r538, %r593;
add.f32 %r610, %r537, %r594;
add.f32 %r611, %r536, %r595;
add.f32 %r612, %r535, %r596;
add.f32 %r613, %r534, %r597;
add.f32 %r614, %r533, %r598;
add.f32 %r615, %r532, %r599;
add.f32 %r616, %r531, %r600;
add.f32 %r617, %r530, %r601;
add.f32 %r618, %r529, %r602;
add.f32 %r619, %r528, %r603;
add.f32 %r620, %r527, %r604;
add.f32 %r621, %r526, %r605;
add.f32 %r622, %r525, %r606;
selp.f32 %r623, %r607, %r591, %p5;
selp.f32 %r624, %r608, %r592, %p5;
selp.f32 %r625, %r609, %r593, %p5;
selp.f32 %r626, %r610, %r594, %p5;
selp.f32 %r627, %r611, %r595, %p5;
selp.f32 %r628, %r612, %r596, %p5;
selp.f32 %r629, %r613, %r597, %p5;
selp.f32 %r630, %r614, %r598, %p5;
selp.f32 %r631, %r615, %r599, %p5;
selp.f32 %r632, %r616, %r600, %p5;
selp.f32 %r633, %r617, %r601, %p5;
selp.f32 %r634, %r618, %r602, %p5;
selp.f32 %r635, %r619, %r603, %p5;
selp.f32 %r636, %r620, %r604, %p5;
selp.f32 %r637, %r621, %r605, %p5;
selp.f32 %r638, %r622, %r606, %p5;
selp.f32 %r639, %r623, %r573, %p9;
selp.f32 %r640, %r624, %r574, %p9;
selp.f32 %r641, %r625, %r575, %p9;
selp.f32 %r642, %r626, %r576, %p9;
selp.f32 %r643, %r627, %r577, %p9;
selp.f32 %r644, %r628, %r578, %p9;
selp.f32 %r645, %r629, %r579, %p9;
selp.f32 %r646, %r630, %r580, %p9;
selp.f32 %r647, %r631, %r581, %p9;
selp.f32 %r648, %r632, %r582, %p9;
selp.f32 %r649, %r633, %r583, %p9;
selp.f32 %r650, %r634, %r584, %p9;
selp.f32 %r651, %r635, %r585, %p9;
selp.f32 %r652, %r636, %r586, %p9;
selp.f32 %r653, %r637, %r587, %p9;
selp.f32 %r654, %r638, %r588, %p9;
mul.f32 %r655, %r542, %r640;
"# else r#"@%p2 ld.global.v4.b32 { %r87, %r88, %r89, %r90 }, [ %rd31 + 0 ];
mov.b32 {%rs202, %rs203}, %r87;
mov.b32 {%rs204, %rs205}, %r88;
mov.b32 {%rs206, %rs207}, %r89;
mov.b32 {%rs208, %rs209}, %r90;
cvt.f32.bf16 %r622, %rs194;
cvt.f32.bf16 %r623, %rs195;
cvt.f32.bf16 %r624, %rs196;
cvt.f32.bf16 %r625, %rs197;
cvt.f32.bf16 %r626, %rs198;
cvt.f32.bf16 %r627, %rs199;
cvt.f32.bf16 %r628, %rs200;
cvt.f32.bf16 %r629, %rs201;
cvt.f32.bf16 %r630, %rs202;
cvt.f32.bf16 %r631, %rs203;
cvt.f32.bf16 %r632, %rs204;
cvt.f32.bf16 %r633, %rs205;
cvt.f32.bf16 %r634, %rs206;
cvt.f32.bf16 %r635, %rs207;
cvt.f32.bf16 %r636, %rs208;
cvt.f32.bf16 %r637, %rs209;
setp.eq.b32 %p4, %r187, 0;
add.f32 %r638, %r571, %r622;
add.f32 %r639, %r570, %r623;
add.f32 %r640, %r569, %r624;
add.f32 %r641, %r568, %r625;
add.f32 %r642, %r567, %r626;
add.f32 %r643, %r566, %r627;
add.f32 %r644, %r565, %r628;
add.f32 %r645, %r564, %r629;
add.f32 %r646, %r563, %r630;
add.f32 %r647, %r562, %r631;
add.f32 %r648, %r561, %r632;
add.f32 %r649, %r560, %r633;
add.f32 %r650, %r559, %r634;
add.f32 %r651, %r558, %r635;
add.f32 %r652, %r557, %r636;
add.f32 %r653, %r556, %r637;
selp.f32 %r654, %r638, %r622, %p4;
selp.f32 %r655, %r639, %r623, %p4;
selp.f32 %r656, %r640, %r624, %p4;
selp.f32 %r657, %r641, %r625, %p4;
selp.f32 %r658, %r642, %r626, %p4;
selp.f32 %r659, %r643, %r627, %p4;
selp.f32 %r660, %r644, %r628, %p4;
selp.f32 %r661, %r645, %r629, %p4;
selp.f32 %r662, %r646, %r630, %p4;
selp.f32 %r663, %r647, %r631, %p4;
selp.f32 %r664, %r648, %r632, %p4;
selp.f32 %r665, %r649, %r633, %p4;
selp.f32 %r666, %r650, %r634, %p4;
selp.f32 %r667, %r651, %r635, %p4;
selp.f32 %r668, %r652, %r636, %p4;
selp.f32 %r669, %r653, %r637, %p4;
setp.gt.u32 %p5, %r192, 3;
selp.f32 %r670, %r654, %r604, %p5;
selp.f32 %r671, %r655, %r605, %p5;
selp.f32 %r672, %r656, %r606, %p5;
selp.f32 %r673, %r657, %r607, %p5;
selp.f32 %r674, %r658, %r608, %p5;
selp.f32 %r675, %r659, %r609, %p5;
selp.f32 %r676, %r660, %r610, %p5;
selp.f32 %r677, %r661, %r611, %p5;
selp.f32 %r678, %r662, %r612, %p5;
selp.f32 %r679, %r663, %r613, %p5;
selp.f32 %r680, %r664, %r614, %p5;
selp.f32 %r681, %r665, %r615, %p5;
selp.f32 %r682, %r666, %r616, %p5;
selp.f32 %r683, %r667, %r617, %p5;
selp.f32 %r684, %r668, %r618, %p5;
selp.f32 %r685, %r669, %r619, %p5;
mul.f32 %r686, %r573, %r671;
"#) ++
  r#"$L__tmp25:
"# ++
  (if trimmed then r#"fma.rn.f32 %r656, %r541, %r639, %r655;
fma.rn.f32 %r657, %r543, %r641, %r656;
fma.rn.f32 %r658, %r544, %r642, %r657;
fma.rn.f32 %r659, %r545, %r643, %r658;
fma.rn.f32 %r660, %r546, %r644, %r659;
fma.rn.f32 %r661, %r547, %r645, %r660;
fma.rn.f32 %r662, %r548, %r646, %r661;
fma.rn.f32 %r663, %r549, %r647, %r662;
fma.rn.f32 %r664, %r550, %r648, %r663;
fma.rn.f32 %r665, %r551, %r649, %r664;
fma.rn.f32 %r666, %r552, %r650, %r665;
fma.rn.f32 %r667, %r553, %r651, %r666;
fma.rn.f32 %r668, %r554, %r652, %r667;
fma.rn.f32 %r669, %r555, %r653, %r668;
fma.rn.f32 %r670, %r556, %r654, %r669;
"# else r#"fma.rn.f32 %r687, %r572, %r670, %r686;
fma.rn.f32 %r688, %r574, %r672, %r687;
fma.rn.f32 %r689, %r575, %r673, %r688;
fma.rn.f32 %r690, %r576, %r674, %r689;
fma.rn.f32 %r691, %r577, %r675, %r690;
fma.rn.f32 %r692, %r578, %r676, %r691;
fma.rn.f32 %r693, %r579, %r677, %r692;
fma.rn.f32 %r694, %r580, %r678, %r693;
fma.rn.f32 %r695, %r581, %r679, %r694;
fma.rn.f32 %r696, %r582, %r680, %r695;
fma.rn.f32 %r697, %r583, %r681, %r696;
fma.rn.f32 %r698, %r584, %r682, %r697;
fma.rn.f32 %r699, %r585, %r683, %r698;
fma.rn.f32 %r700, %r586, %r684, %r699;
fma.rn.f32 %r701, %r587, %r685, %r700;
"#) ++
  r#"$L__tmp26:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r671, %r670, 4, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r702, %r701, 4, 31, -1;
"#) ++
  r#"$L__tmp27:
"# ++
  (if trimmed then r#"add.f32 %r672, %r670, %r671;
"# else r#"add.f32 %r703, %r701, %r702;
"#) ++
  r#"$L__tmp28:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r673, %r672, 2, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r704, %r703, 2, 31, -1;
"#) ++
  r#"$L__tmp29:
"# ++
  (if trimmed then r#"add.f32 %r674, %r672, %r673;
"# else r#"add.f32 %r705, %r703, %r704;
"#) ++
  r#"$L__tmp30:
"# ++
  (if trimmed then r#"shfl.sync.bfly.b32 %r675, %r674, 1, 31, -1;
"# else r#"shfl.sync.bfly.b32 %r706, %r705, 1, 31, -1;
"#) ++
  r#"$L__tmp31:
"# ++
  (if trimmed then r#"add.f32 %r676, %r674, %r675;
"# else r#"add.f32 %r707, %r705, %r706;
"#) ++
  r#"$L__tmp32:
"# ++
  (if trimmed then r#"div.full.f32 %r677, %r676, %r290;
neg.f32 %r678, %r541;
fma.rn.f32 %r679, %r678, %r677, %r639;
neg.f32 %r680, %r542;
fma.rn.f32 %r681, %r680, %r677, %r640;
neg.f32 %r682, %r543;
fma.rn.f32 %r683, %r682, %r677, %r641;
neg.f32 %r684, %r544;
fma.rn.f32 %r685, %r684, %r677, %r642;
neg.f32 %r686, %r545;
fma.rn.f32 %r687, %r686, %r677, %r643;
neg.f32 %r688, %r546;
fma.rn.f32 %r689, %r688, %r677, %r644;
neg.f32 %r690, %r547;
fma.rn.f32 %r691, %r690, %r677, %r645;
neg.f32 %r692, %r548;
fma.rn.f32 %r693, %r692, %r677, %r646;
neg.f32 %r694, %r549;
fma.rn.f32 %r695, %r694, %r677, %r647;
neg.f32 %r696, %r550;
fma.rn.f32 %r697, %r696, %r677, %r648;
neg.f32 %r698, %r551;
fma.rn.f32 %r699, %r698, %r677, %r649;
neg.f32 %r700, %r552;
fma.rn.f32 %r701, %r700, %r677, %r650;
neg.f32 %r702, %r553;
fma.rn.f32 %r703, %r702, %r677, %r651;
neg.f32 %r704, %r554;
fma.rn.f32 %r705, %r704, %r677, %r652;
neg.f32 %r706, %r555;
fma.rn.f32 %r707, %r706, %r677, %r653;
neg.f32 %r708, %r556;
fma.rn.f32 %r709, %r708, %r677, %r654;
mul.f32 %r710, %r524, %r679;
mul.f32 %r711, %r524, %r681;
mul.f32 %r712, %r524, %r683;
mul.f32 %r713, %r524, %r685;
mul.f32 %r714, %r524, %r687;
mul.f32 %r715, %r524, %r689;
mul.f32 %r716, %r524, %r691;
mul.f32 %r717, %r524, %r693;
mul.f32 %r718, %r524, %r695;
mul.f32 %r719, %r524, %r697;
mul.f32 %r720, %r524, %r699;
mul.f32 %r721, %r524, %r701;
mul.f32 %r722, %r524, %r703;
mul.f32 %r723, %r524, %r705;
mul.f32 %r724, %r524, %r707;
mul.f32 %r725, %r524, %r709;
cvt.rn.bf16.f32 %rs193, %r710;
cvt.rn.bf16.f32 %rs194, %r711;
cvt.rn.bf16.f32 %rs195, %r712;
cvt.rn.bf16.f32 %rs196, %r713;
cvt.rn.bf16.f32 %rs197, %r714;
cvt.rn.bf16.f32 %rs198, %r715;
cvt.rn.bf16.f32 %rs199, %r716;
cvt.rn.bf16.f32 %rs200, %r717;
cvt.rn.bf16.f32 %rs201, %r718;
cvt.rn.bf16.f32 %rs202, %r719;
cvt.rn.bf16.f32 %rs203, %r720;
cvt.rn.bf16.f32 %rs204, %r721;
cvt.rn.bf16.f32 %rs205, %r722;
cvt.rn.bf16.f32 %rs206, %r723;
cvt.rn.bf16.f32 %rs207, %r724;
cvt.rn.bf16.f32 %rs208, %r725;
"# else r#"div.full.f32 %r708, %r707, %r323;
neg.f32 %r709, %r572;
fma.rn.f32 %r710, %r709, %r708, %r670;
neg.f32 %r711, %r573;
fma.rn.f32 %r712, %r711, %r708, %r671;
neg.f32 %r713, %r574;
fma.rn.f32 %r714, %r713, %r708, %r672;
neg.f32 %r715, %r575;
fma.rn.f32 %r716, %r715, %r708, %r673;
neg.f32 %r717, %r576;
fma.rn.f32 %r718, %r717, %r708, %r674;
neg.f32 %r719, %r577;
fma.rn.f32 %r720, %r719, %r708, %r675;
neg.f32 %r721, %r578;
fma.rn.f32 %r722, %r721, %r708, %r676;
neg.f32 %r723, %r579;
fma.rn.f32 %r724, %r723, %r708, %r677;
neg.f32 %r725, %r580;
fma.rn.f32 %r726, %r725, %r708, %r678;
neg.f32 %r727, %r581;
fma.rn.f32 %r728, %r727, %r708, %r679;
neg.f32 %r729, %r582;
fma.rn.f32 %r730, %r729, %r708, %r680;
neg.f32 %r731, %r583;
fma.rn.f32 %r732, %r731, %r708, %r681;
neg.f32 %r733, %r584;
fma.rn.f32 %r734, %r733, %r708, %r682;
neg.f32 %r735, %r585;
fma.rn.f32 %r736, %r735, %r708, %r683;
neg.f32 %r737, %r586;
fma.rn.f32 %r738, %r737, %r708, %r684;
neg.f32 %r739, %r587;
fma.rn.f32 %r740, %r739, %r708, %r685;
mul.f32 %r741, %r555, %r710;
mul.f32 %r742, %r555, %r712;
mul.f32 %r743, %r555, %r714;
mul.f32 %r744, %r555, %r716;
mul.f32 %r745, %r555, %r718;
mul.f32 %r746, %r555, %r720;
mul.f32 %r747, %r555, %r722;
mul.f32 %r748, %r555, %r724;
mul.f32 %r749, %r555, %r726;
mul.f32 %r750, %r555, %r728;
mul.f32 %r751, %r555, %r730;
mul.f32 %r752, %r555, %r732;
mul.f32 %r753, %r555, %r734;
mul.f32 %r754, %r555, %r736;
mul.f32 %r755, %r555, %r738;
mul.f32 %r756, %r555, %r740;
cvt.rn.bf16.f32 %rs210, %r741;
cvt.rn.bf16.f32 %rs211, %r742;
cvt.rn.bf16.f32 %rs212, %r743;
cvt.rn.bf16.f32 %rs213, %r744;
cvt.rn.bf16.f32 %rs214, %r745;
cvt.rn.bf16.f32 %rs215, %r746;
cvt.rn.bf16.f32 %rs216, %r747;
cvt.rn.bf16.f32 %rs217, %r748;
cvt.rn.bf16.f32 %rs218, %r749;
cvt.rn.bf16.f32 %rs219, %r750;
cvt.rn.bf16.f32 %rs220, %r751;
cvt.rn.bf16.f32 %rs221, %r752;
cvt.rn.bf16.f32 %rs222, %r753;
cvt.rn.bf16.f32 %rs223, %r754;
cvt.rn.bf16.f32 %rs224, %r755;
cvt.rn.bf16.f32 %rs225, %r756;
"#) ++
  r#"bar.sync 0;
"# ++
  (if trimmed then r#"mov.b32 %r726, {%rs205, %rs207};
mov.b32 %r727, {%rs201, %rs203};
mov.b32 %r728, {%rs197, %rs199};
mov.b32 %r729, {%rs193, %rs195};
st.shared.v4.b32 [%r423], {%r729, %r728, %r727, %r726};
mov.b32 %r730, {%rs206, %rs208};
mov.b32 %r731, {%rs202, %rs204};
mov.b32 %r732, {%rs198, %rs200};
mov.b32 %r733, {%rs194, %rs196};
st.shared.v4.b32 [%r429+4096], {%r733, %r732, %r731, %r730};
"# else r#"mov.b32 %r757, {%rs222, %rs224};
mov.b32 %r758, {%rs218, %rs220};
mov.b32 %r759, {%rs214, %rs216};
mov.b32 %r760, {%rs210, %rs212};
st.shared.v4.b32 [%r454], {%r760, %r759, %r758, %r757};
mov.b32 %r761, {%rs223, %rs225};
mov.b32 %r762, {%rs219, %rs221};
mov.b32 %r763, {%rs215, %rs217};
mov.b32 %r764, {%rs211, %rs213};
st.shared.v4.b32 [%r460+4096], {%r764, %r763, %r762, %r761};
"#) ++
  r#"bar.sync 0;
"# ++
  (if trimmed then r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r734, %r735, %r736, %r737}, [%r439];
mov.b32 {%rs209, %rs210}, %r734;
mov.b32 {%rs211, %rs212}, %r735;
mov.b32 {%rs213, %rs214}, %r736;
mov.b32 {%rs215, %rs216}, %r737;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r738, %r739, %r740, %r741}, [%r439+512];
mov.b32 {%rs217, %rs218}, %r738;
mov.b32 {%rs219, %rs220}, %r739;
mov.b32 {%rs221, %rs222}, %r740;
mov.b32 {%rs223, %rs224}, %r741;
cvt.f32.bf16 %r742, %rs209;
cvt.f32.bf16 %r743, %rs210;
cvt.f32.bf16 %r744, %rs211;
cvt.f32.bf16 %r745, %rs212;
cvt.f32.bf16 %r746, %rs213;
cvt.f32.bf16 %r747, %rs214;
cvt.f32.bf16 %r748, %rs215;
cvt.f32.bf16 %r749, %rs216;
cvt.f32.bf16 %r750, %rs217;
cvt.f32.bf16 %r751, %rs218;
cvt.f32.bf16 %r752, %rs219;
cvt.f32.bf16 %r753, %rs220;
cvt.f32.bf16 %r754, %rs221;
cvt.f32.bf16 %r755, %rs222;
cvt.f32.bf16 %r756, %rs223;
cvt.f32.bf16 %r757, %rs224;
cvt.f32.bf16 %r758, %rs193;
cvt.f32.bf16 %r759, %rs194;
cvt.f32.bf16 %r760, %rs195;
cvt.f32.bf16 %r761, %rs196;
cvt.f32.bf16 %r762, %rs197;
cvt.f32.bf16 %r763, %rs198;
cvt.f32.bf16 %r764, %rs199;
cvt.f32.bf16 %r765, %rs200;
cvt.f32.bf16 %r766, %rs201;
cvt.f32.bf16 %r767, %rs202;
cvt.f32.bf16 %r768, %rs203;
cvt.f32.bf16 %r769, %rs204;
cvt.f32.bf16 %r770, %rs205;
cvt.f32.bf16 %r771, %rs206;
cvt.f32.bf16 %r772, %rs207;
cvt.f32.bf16 %r773, %rs208;
div.full.f32 %r99, %r742, %r18;
div.full.f32 %r100, %r743, %r18;
div.full.f32 %r101, %r744, %r18;
div.full.f32 %r102, %r745, %r18;
div.full.f32 %r103, %r746, %r18;
div.full.f32 %r104, %r747, %r18;
div.full.f32 %r105, %r748, %r18;
div.full.f32 %r106, %r749, %r18;
div.full.f32 %r107, %r750, %r18;
div.full.f32 %r108, %r751, %r18;
div.full.f32 %r109, %r752, %r18;
div.full.f32 %r110, %r753, %r18;
div.full.f32 %r111, %r754, %r18;
div.full.f32 %r112, %r755, %r18;
div.full.f32 %r113, %r756, %r18;
div.full.f32 %r114, %r757, %r18;
div.full.f32 %r115, %r758, %r18;
div.full.f32 %r116, %r759, %r18;
div.full.f32 %r117, %r760, %r18;
div.full.f32 %r118, %r761, %r18;
div.full.f32 %r119, %r762, %r18;
div.full.f32 %r120, %r763, %r18;
div.full.f32 %r121, %r764, %r18;
div.full.f32 %r122, %r765, %r18;
div.full.f32 %r123, %r766, %r18;
div.full.f32 %r124, %r767, %r18;
div.full.f32 %r125, %r768, %r18;
div.full.f32 %r126, %r769, %r18;
div.full.f32 %r127, %r770, %r18;
div.full.f32 %r128, %r771, %r18;
div.full.f32 %r129, %r772, %r18;
div.full.f32 %r130, %r773, %r18;
cvt.rn.satfinite.e4m3x2.f32 %rs17, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r114, %r113;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r116, %r115;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r118, %r117;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r120, %r119;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r122, %r121;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r124, %r123;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r126, %r125;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r128, %r127;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r130, %r129;
or.b32 %r774, %r483, %r197;
or.b32 %r775, %r483, %r199;
cvt.s64.s32 %rd54, %r774;
add.s64 %rd18, %rd43, %rd54;
mov.b32 %r131, {%rs25, %rs26};
mov.b32 %r132, {%rs27, %rs28};
mov.b32 %r133, {%rs29, %rs30};
mov.b32 %r134, {%rs31, %rs32};
@%p1 st.global.v4.b32 [ %rd18 + 0 ], { %r131, %r132, %r133, %r134 };
@PITCH776_775@
cvt.s64.s32 %rd55, %r776;
add.s64 %rd56, %rd31, %rd55;
add.s64 %rd19, %rd56, %rd47;
mov.b32 %r135, {%rs17, %rs18};
mov.b32 %r136, {%rs19, %rs20};
mov.b32 %r137, {%rs21, %rs22};
mov.b32 %r138, {%rs23, %rs24};
@%p1 st.global.v4.b32 [ %rd19 + 0 ], { %r135, %r136, %r137, %r138 };
add.s64 %rd57, %rd26, %rd40;
add.s64 %rd58, %rd57, %rd38;
add.s64 %rd20, %rd58, %rd34;
add.s64 %rd21, %rd20, 16;
mov.u32 %r139, %r5;
mov.u32 %r140, %r5;
mov.u32 %r141, %r5;
mov.u32 %r142, %r5;
@%p1 ld.global.v4.b32 { %r139, %r140, %r141, %r142 }, [ %rd20 + 0 ];
mov.b32 {%rs225, %rs226}, %r139;
mov.b32 {%rs227, %rs228}, %r140;
mov.b32 {%rs229, %rs230}, %r141;
mov.b32 {%rs231, %rs232}, %r142;
mov.u32 %r143, %r5;
mov.u32 %r144, %r5;
mov.u32 %r145, %r5;
mov.u32 %r146, %r5;
@%p1 ld.global.v4.b32 { %r143, %r144, %r145, %r146 }, [ %rd21 + 0 ];
mov.b32 {%rs233, %rs234}, %r143;
mov.b32 {%rs235, %rs236}, %r144;
mov.b32 {%rs237, %rs238}, %r145;
mov.b32 {%rs239, %rs240}, %r146;
"# else r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r765, %r766, %r767, %r768}, [%r470];
mov.b32 {%rs226, %rs227}, %r765;
mov.b32 {%rs228, %rs229}, %r766;
mov.b32 {%rs230, %rs231}, %r767;
mov.b32 {%rs232, %rs233}, %r768;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r769, %r770, %r771, %r772}, [%r470+512];
mov.b32 {%rs234, %rs235}, %r769;
mov.b32 {%rs236, %rs237}, %r770;
mov.b32 {%rs238, %rs239}, %r771;
mov.b32 {%rs240, %rs241}, %r772;
cvt.f32.bf16 %r773, %rs226;
cvt.f32.bf16 %r774, %rs227;
cvt.f32.bf16 %r775, %rs228;
cvt.f32.bf16 %r776, %rs229;
cvt.f32.bf16 %r777, %rs230;
cvt.f32.bf16 %r778, %rs231;
cvt.f32.bf16 %r779, %rs232;
cvt.f32.bf16 %r780, %rs233;
cvt.f32.bf16 %r781, %rs234;
cvt.f32.bf16 %r782, %rs235;
cvt.f32.bf16 %r783, %rs236;
cvt.f32.bf16 %r784, %rs237;
cvt.f32.bf16 %r785, %rs238;
cvt.f32.bf16 %r786, %rs239;
cvt.f32.bf16 %r787, %rs240;
cvt.f32.bf16 %r788, %rs241;
cvt.f32.bf16 %r789, %rs210;
cvt.f32.bf16 %r790, %rs211;
cvt.f32.bf16 %r791, %rs212;
cvt.f32.bf16 %r792, %rs213;
cvt.f32.bf16 %r793, %rs214;
cvt.f32.bf16 %r794, %rs215;
cvt.f32.bf16 %r795, %rs216;
cvt.f32.bf16 %r796, %rs217;
cvt.f32.bf16 %r797, %rs218;
cvt.f32.bf16 %r798, %rs219;
cvt.f32.bf16 %r799, %rs220;
cvt.f32.bf16 %r800, %rs221;
cvt.f32.bf16 %r801, %rs222;
cvt.f32.bf16 %r802, %rs223;
cvt.f32.bf16 %r803, %rs224;
cvt.f32.bf16 %r804, %rs225;
div.full.f32 %r91, %r773, %r10;
div.full.f32 %r92, %r774, %r10;
div.full.f32 %r93, %r775, %r10;
div.full.f32 %r94, %r776, %r10;
div.full.f32 %r95, %r777, %r10;
div.full.f32 %r96, %r778, %r10;
div.full.f32 %r97, %r779, %r10;
div.full.f32 %r98, %r780, %r10;
div.full.f32 %r99, %r781, %r10;
div.full.f32 %r100, %r782, %r10;
div.full.f32 %r101, %r783, %r10;
div.full.f32 %r102, %r784, %r10;
div.full.f32 %r103, %r785, %r10;
div.full.f32 %r104, %r786, %r10;
div.full.f32 %r105, %r787, %r10;
div.full.f32 %r106, %r788, %r10;
div.full.f32 %r107, %r789, %r10;
div.full.f32 %r108, %r790, %r10;
div.full.f32 %r109, %r791, %r10;
div.full.f32 %r110, %r792, %r10;
div.full.f32 %r111, %r793, %r10;
div.full.f32 %r112, %r794, %r10;
div.full.f32 %r113, %r795, %r10;
div.full.f32 %r114, %r796, %r10;
div.full.f32 %r115, %r797, %r10;
div.full.f32 %r116, %r798, %r10;
div.full.f32 %r117, %r799, %r10;
div.full.f32 %r118, %r800, %r10;
div.full.f32 %r119, %r801, %r10;
div.full.f32 %r120, %r802, %r10;
div.full.f32 %r121, %r803, %r10;
div.full.f32 %r122, %r804, %r10;
cvt.rn.satfinite.e4m3x2.f32 %rs34, %r92, %r91;
cvt.rn.satfinite.e4m3x2.f32 %rs35, %r94, %r93;
cvt.rn.satfinite.e4m3x2.f32 %rs36, %r96, %r95;
cvt.rn.satfinite.e4m3x2.f32 %rs37, %r98, %r97;
cvt.rn.satfinite.e4m3x2.f32 %rs38, %r100, %r99;
cvt.rn.satfinite.e4m3x2.f32 %rs39, %r102, %r101;
cvt.rn.satfinite.e4m3x2.f32 %rs40, %r104, %r103;
cvt.rn.satfinite.e4m3x2.f32 %rs41, %r106, %r105;
cvt.rn.satfinite.e4m3x2.f32 %rs42, %r108, %r107;
cvt.rn.satfinite.e4m3x2.f32 %rs43, %r110, %r109;
cvt.rn.satfinite.e4m3x2.f32 %rs44, %r112, %r111;
cvt.rn.satfinite.e4m3x2.f32 %rs45, %r114, %r113;
cvt.rn.satfinite.e4m3x2.f32 %rs46, %r116, %r115;
cvt.rn.satfinite.e4m3x2.f32 %rs47, %r118, %r117;
cvt.rn.satfinite.e4m3x2.f32 %rs48, %r120, %r119;
cvt.rn.satfinite.e4m3x2.f32 %rs49, %r122, %r121;
or.b32 %r805, %r514, %r193;
or.b32 %r806, %r514, %r194;
cvt.s64.s32 %rd71, %r805;
add.s64 %rd32, %rd60, %rd71;
mov.b32 %r123, {%rs42, %rs43};
mov.b32 %r124, {%rs44, %rs45};
mov.b32 %r125, {%rs46, %rs47};
mov.b32 %r126, {%rs48, %rs49};
@%p1 st.global.v4.b32 [ %rd32 + 0 ], { %r123, %r124, %r125, %r126 };
@PITCH807_806@
cvt.s64.s32 %rd72, %r807;
add.s64 %rd73, %rd45, %rd72;
add.s64 %rd33, %rd73, %rd64;
mov.b32 %r127, {%rs34, %rs35};
mov.b32 %r128, {%rs36, %rs37};
mov.b32 %r129, {%rs38, %rs39};
mov.b32 %r130, {%rs40, %rs41};
@%p1 st.global.v4.b32 [ %rd33 + 0 ], { %r127, %r128, %r129, %r130 };
add.s64 %rd74, %rd40, %rd57;
add.s64 %rd75, %rd74, %rd55;
add.s64 %rd34, %rd75, %rd47;
add.s64 %rd35, %rd34, 16;
mov.u32 %r131, %r5;
mov.u32 %r132, %r5;
mov.u32 %r133, %r5;
mov.u32 %r134, %r5;
@%p1 ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd34 + 0 ];
mov.b32 {%rs242, %rs243}, %r131;
mov.b32 {%rs244, %rs245}, %r132;
mov.b32 {%rs246, %rs247}, %r133;
mov.b32 {%rs248, %rs249}, %r134;
mov.u32 %r135, %r5;
mov.u32 %r136, %r5;
mov.u32 %r137, %r5;
mov.u32 %r138, %r5;
@%p1 ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd35 + 0 ];
mov.b32 {%rs250, %rs251}, %r135;
mov.b32 {%rs252, %rs253}, %r136;
mov.b32 {%rs254, %rs255}, %r137;
mov.b32 {%rs256, %rs257}, %r138;
"#) ++
  r#"bar.sync 0;
"# ++
  (if trimmed then r#"mov.b32 %r777, {%rs225, %rs227};
mov.b32 %r778, {%rs229, %rs231};
mov.b32 %r779, {%rs233, %rs235};
mov.b32 %r780, {%rs237, %rs239};
st.shared.v4.b32 [%r423], {%r777, %r778, %r779, %r780};
mov.b32 {_, %rs241}, %r143;
mov.b32 {_, %rs242}, %r145;
mov.b32 {_, %rs243}, %r139;
mov.b32 {_, %rs244}, %r141;
mov.b32 %r781, {%rs244, %rs232};
mov.b32 %r782, {%rs243, %rs228};
mov.b32 %r783, {%rs242, %rs240};
mov.b32 %r784, {%rs241, %rs236};
st.shared.v4.b32 [%r429+4096], {%r782, %r781, %r784, %r783};
"# else r#"mov.b32 %r808, {%rs242, %rs244};
mov.b32 %r809, {%rs246, %rs248};
mov.b32 %r810, {%rs250, %rs252};
mov.b32 %r811, {%rs254, %rs256};
st.shared.v4.b32 [%r454], {%r808, %r809, %r810, %r811};
mov.b32 {_, %rs258}, %r135;
mov.b32 {_, %rs259}, %r137;
mov.b32 {_, %rs260}, %r131;
mov.b32 {_, %rs261}, %r133;
mov.b32 %r812, {%rs261, %rs249};
mov.b32 %r813, {%rs260, %rs245};
mov.b32 %r814, {%rs259, %rs257};
mov.b32 %r815, {%rs258, %rs253};
st.shared.v4.b32 [%r460+4096], {%r813, %r812, %r815, %r814};
"#) ++
  r#"bar.sync 0;
"# ++
  (if trimmed then r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r785, %r786, %r787, %r788}, [%r439];
mov.b32 {%rs245, %rs246}, %r785;
mov.b32 {%rs247, %rs248}, %r786;
mov.b32 {%rs249, %rs250}, %r787;
mov.b32 {%rs251, %rs252}, %r788;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r789, %r790, %r791, %r792}, [%r439+512];
mov.b32 {%rs253, %rs254}, %r789;
mov.b32 {%rs255, %rs256}, %r790;
mov.b32 {%rs257, %rs258}, %r791;
mov.b32 {%rs259, %rs260}, %r792;
cvt.f32.bf16 %r793, %rs245;
cvt.f32.bf16 %r794, %rs246;
cvt.f32.bf16 %r795, %rs247;
cvt.f32.bf16 %r796, %rs248;
cvt.f32.bf16 %r797, %rs249;
cvt.f32.bf16 %r798, %rs250;
cvt.f32.bf16 %r799, %rs251;
cvt.f32.bf16 %r800, %rs252;
cvt.f32.bf16 %r801, %rs253;
cvt.f32.bf16 %r802, %rs254;
cvt.f32.bf16 %r803, %rs255;
cvt.f32.bf16 %r804, %rs256;
cvt.f32.bf16 %r805, %rs257;
cvt.f32.bf16 %r806, %rs258;
cvt.f32.bf16 %r807, %rs259;
cvt.f32.bf16 %r808, %rs260;
cvt.f32.bf16 %r809, %rs225;
cvt.f32.bf16 %r810, %rs226;
cvt.f32.bf16 %r811, %rs227;
cvt.f32.bf16 %r812, %rs228;
cvt.f32.bf16 %r813, %rs229;
cvt.f32.bf16 %r814, %rs230;
cvt.f32.bf16 %r815, %rs231;
cvt.f32.bf16 %r816, %rs232;
cvt.f32.bf16 %r817, %rs233;
cvt.f32.bf16 %r818, %rs234;
cvt.f32.bf16 %r819, %rs235;
cvt.f32.bf16 %r820, %rs236;
cvt.f32.bf16 %r821, %rs237;
cvt.f32.bf16 %r822, %rs238;
cvt.f32.bf16 %r823, %rs239;
cvt.f32.bf16 %r824, %rs240;
div.full.f32 %r147, %r793, %r18;
div.full.f32 %r148, %r794, %r18;
div.full.f32 %r149, %r795, %r18;
div.full.f32 %r150, %r796, %r18;
div.full.f32 %r151, %r797, %r18;
div.full.f32 %r152, %r798, %r18;
div.full.f32 %r153, %r799, %r18;
div.full.f32 %r154, %r800, %r18;
div.full.f32 %r155, %r801, %r18;
div.full.f32 %r156, %r802, %r18;
div.full.f32 %r157, %r803, %r18;
div.full.f32 %r158, %r804, %r18;
div.full.f32 %r159, %r805, %r18;
div.full.f32 %r160, %r806, %r18;
div.full.f32 %r161, %r807, %r18;
div.full.f32 %r162, %r808, %r18;
div.full.f32 %r163, %r809, %r18;
div.full.f32 %r164, %r810, %r18;
div.full.f32 %r165, %r811, %r18;
div.full.f32 %r166, %r812, %r18;
div.full.f32 %r167, %r813, %r18;
div.full.f32 %r168, %r814, %r18;
div.full.f32 %r169, %r815, %r18;
div.full.f32 %r170, %r816, %r18;
div.full.f32 %r171, %r817, %r18;
div.full.f32 %r172, %r818, %r18;
div.full.f32 %r173, %r819, %r18;
div.full.f32 %r174, %r820, %r18;
div.full.f32 %r175, %r821, %r18;
div.full.f32 %r176, %r822, %r18;
div.full.f32 %r177, %r823, %r18;
div.full.f32 %r178, %r824, %r18;
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
add.s32 %r825, %r234, 1536;
or.b32 %r826, %r825, %r197;
or.b32 %r827, %r825, %r199;
cvt.s64.s32 %rd59, %r826;
add.s64 %rd22, %rd43, %rd59;
mov.b32 %r179, {%rs41, %rs42};
mov.b32 %r180, {%rs43, %rs44};
mov.b32 %r181, {%rs45, %rs46};
mov.b32 %r182, {%rs47, %rs48};
@%p1 st.global.v4.b32 [ %rd22 + 0 ], { %r179, %r180, %r181, %r182 };
@PITCH828_827@
cvt.s64.s32 %rd60, %r828;
add.s64 %rd61, %rd31, %rd60;
add.s64 %rd23, %rd61, %rd47;
mov.b32 %r183, {%rs33, %rs34};
mov.b32 %r184, {%rs35, %rs36};
mov.b32 %r185, {%rs37, %rs38};
mov.b32 %r186, {%rs39, %rs40};
@%p1 st.global.v4.b32 [ %rd23 + 0 ], { %r183, %r184, %r185, %r186 };
"# else r#"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r816, %r817, %r818, %r819}, [%r470];
mov.b32 {%rs262, %rs263}, %r816;
mov.b32 {%rs264, %rs265}, %r817;
mov.b32 {%rs266, %rs267}, %r818;
mov.b32 {%rs268, %rs269}, %r819;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r820, %r821, %r822, %r823}, [%r470+512];
mov.b32 {%rs270, %rs271}, %r820;
mov.b32 {%rs272, %rs273}, %r821;
mov.b32 {%rs274, %rs275}, %r822;
mov.b32 {%rs276, %rs277}, %r823;
cvt.f32.bf16 %r824, %rs262;
cvt.f32.bf16 %r825, %rs263;
cvt.f32.bf16 %r826, %rs264;
cvt.f32.bf16 %r827, %rs265;
cvt.f32.bf16 %r828, %rs266;
cvt.f32.bf16 %r829, %rs267;
cvt.f32.bf16 %r830, %rs268;
cvt.f32.bf16 %r831, %rs269;
cvt.f32.bf16 %r832, %rs270;
cvt.f32.bf16 %r833, %rs271;
cvt.f32.bf16 %r834, %rs272;
cvt.f32.bf16 %r835, %rs273;
cvt.f32.bf16 %r836, %rs274;
cvt.f32.bf16 %r837, %rs275;
cvt.f32.bf16 %r838, %rs276;
cvt.f32.bf16 %r839, %rs277;
cvt.f32.bf16 %r840, %rs242;
cvt.f32.bf16 %r841, %rs243;
cvt.f32.bf16 %r842, %rs244;
cvt.f32.bf16 %r843, %rs245;
cvt.f32.bf16 %r844, %rs246;
cvt.f32.bf16 %r845, %rs247;
cvt.f32.bf16 %r846, %rs248;
cvt.f32.bf16 %r847, %rs249;
cvt.f32.bf16 %r848, %rs250;
cvt.f32.bf16 %r849, %rs251;
cvt.f32.bf16 %r850, %rs252;
cvt.f32.bf16 %r851, %rs253;
cvt.f32.bf16 %r852, %rs254;
cvt.f32.bf16 %r853, %rs255;
cvt.f32.bf16 %r854, %rs256;
cvt.f32.bf16 %r855, %rs257;
div.full.f32 %r139, %r824, %r10;
div.full.f32 %r140, %r825, %r10;
div.full.f32 %r141, %r826, %r10;
div.full.f32 %r142, %r827, %r10;
div.full.f32 %r143, %r828, %r10;
div.full.f32 %r144, %r829, %r10;
div.full.f32 %r145, %r830, %r10;
div.full.f32 %r146, %r831, %r10;
div.full.f32 %r147, %r832, %r10;
div.full.f32 %r148, %r833, %r10;
div.full.f32 %r149, %r834, %r10;
div.full.f32 %r150, %r835, %r10;
div.full.f32 %r151, %r836, %r10;
div.full.f32 %r152, %r837, %r10;
div.full.f32 %r153, %r838, %r10;
div.full.f32 %r154, %r839, %r10;
div.full.f32 %r155, %r840, %r10;
div.full.f32 %r156, %r841, %r10;
div.full.f32 %r157, %r842, %r10;
div.full.f32 %r158, %r843, %r10;
div.full.f32 %r159, %r844, %r10;
div.full.f32 %r160, %r845, %r10;
div.full.f32 %r161, %r846, %r10;
div.full.f32 %r162, %r847, %r10;
div.full.f32 %r163, %r848, %r10;
div.full.f32 %r164, %r849, %r10;
div.full.f32 %r165, %r850, %r10;
div.full.f32 %r166, %r851, %r10;
div.full.f32 %r167, %r852, %r10;
div.full.f32 %r168, %r853, %r10;
div.full.f32 %r169, %r854, %r10;
div.full.f32 %r170, %r855, %r10;
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
add.s32 %r856, %r267, 1536;
or.b32 %r857, %r856, %r193;
or.b32 %r858, %r856, %r194;
cvt.s64.s32 %rd76, %r857;
add.s64 %rd36, %rd60, %rd76;
mov.b32 %r171, {%rs58, %rs59};
mov.b32 %r172, {%rs60, %rs61};
mov.b32 %r173, {%rs62, %rs63};
mov.b32 %r174, {%rs64, %rs65};
@%p1 st.global.v4.b32 [ %rd36 + 0 ], { %r171, %r172, %r173, %r174 };
@PITCH859_858@
cvt.s64.s32 %rd77, %r859;
add.s64 %rd78, %rd45, %rd77;
add.s64 %rd37, %rd78, %rd64;
mov.b32 %r175, {%rs50, %rs51};
mov.b32 %r176, {%rs52, %rs53};
mov.b32 %r177, {%rs54, %rs55};
mov.b32 %r178, {%rs56, %rs57};
@%p1 st.global.v4.b32 [ %rd37 + 0 ], { %r175, %r176, %r177, %r178 };
"#) ++
  r#"ret;
$L__tmp33:
$L__func_end0:
}
"#

structure Artifacts where
  referenceIR : String
  referencePTX : String
  trimmedIR : String
  trimmedPTX : String
  deriving BEq

private def pitch (n : Length) (dst src : Nat) : String :=
  let regs := s!"%r{dst}, %r{src}, "
  match n with
  | .t16k => "shl.b32 " ++ regs ++ "14;"
  | .t32k => "shl.b32 " ++ regs ++ "15;"
  | _ => s!"mul.lo.s32 {regs}{n.tokens};"

private def lines (n : Length) (text : String) : List String :=
  let text := text.replace "@TOKENS@" (toString n.tokens)
  let text := text.replace "@BLOCKS@" (toString (n.tokens / 32))
  let text := (text.replace "@PRELAST@" (toString (n.tokens - 2))).replace "@LAST@" (toString (n.tokens - 1))
  let sites : List (Nat × Nat) := [(482, 481), (513, 512), (776, 775), (807, 806), (828, 827), (859, 858)]
  let text := sites.foldl (fun text (dst, src) =>
    text.replace s!"@PITCH{dst}_{src}@" (pitch n dst src)) text
  (text.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (n : Length) (a : Artifacts) : Bool :=
  n != .validation &&
    QKVTTIR.normalizeTTIR a.referenceIR == some (lines n (irTemplate false)) &&
    QKVTTIR.normalizePTX a.referencePTX == lines n (ptxTemplate false) &&
    QKVTTIR.normalizeTTIR a.trimmedIR == some (lines n (irTemplate true)) &&
    QKVTTIR.normalizePTX a.trimmedPTX == lines n (ptxTemplate true)

structure Certificate (n : Length) (a : Artifacts) : Type where
  matched : matchesArtifacts n a = true
  factorFlip : ∀ lo hi : BitVec 16, (lo ^^^ (lo ^^^ hi), hi ^^^ (lo ^^^ hi)) = (hi, lo)
  tileCoverage : ∀ t : Fin n.tokens, t.val / 32 < n.tokens / 32 ∧ 32 * (t.val / 32) + t.val % 32 = t.val
  range : ∀ p : Point n, rowAddress p < n.tokens * 2304 ∧ transposedAddress p < n.tokens * 2304 ∧
    n.tokens * 2304 < 2^31
  rowUnique : ∀ p q : Point n, rowAddress p = rowAddress q → p = q
  transposeUnique : ∀ p q : Point n, transposedAddress p = transposedAddress q → p = q
  reads : ∀ (t : Fin n.tokens), t.val ≠ 0 → t.val + 1 < n.tokens →
    keyReadBytes false t = 512 ∧ keyReadBytes true t = 256
  stores : ∀ (op : Target) (read : Inputs n) (p : Point n), writes true op read p = writes false op read p

def certify (n : Length) (a : Artifacts) : Option (Certificate n a) :=
  if h : matchesArtifacts n a = true then
    some ⟨h, reversed_pair_bits, token_tile_covers, addresses_bounded, row_address_unique, transpose_address_unique,
      interior_key_reads, fun op read p => both_layouts_equal op read p⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.RopeBackward
