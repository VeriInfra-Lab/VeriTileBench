import models.NanoGPT.Track1.YarnTable
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Bind the compact factor decoder to the emitted TTGIR fragments for
the three layouts. The PTX replay separately checks the complete output
expressions and the physical factor load addresses. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.YarnTable

inductive Phase where | forward | backward
  deriving DecidableEq

private def decoder : Layout → Phase → String
  | .wide, .forward => r#"%offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offsets_59 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%offsets_60 = tt.broadcast %0 : tensor<4x1xi32, #blocked1> -> tensor<4x64xi32, #blocked1>
%offsets_61 = tt.broadcast %offsets_59 : tensor<1x64xi32, #blocked1> -> tensor<4x64xi32, #blocked1>
%offsets_62 = arith.addi %offsets_60, %offsets_61 : tensor<4x64xi32, #blocked1>
%mask_63 = tt.expand_dims %factor_rows {axis = 1 : i32} : tensor<4xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<4x1xi1, #blocked1>
%mask_64 = arith.cmpi slt, %offsets_59, %cst_10 : tensor<1x64xi32, #blocked1>
%mask_65 = tt.broadcast %mask_63 : tensor<4x1xi1, #blocked1> -> tensor<4x64xi1, #blocked1>
%mask_66 = tt.broadcast %mask_64 : tensor<1x64xi1, #blocked1> -> tensor<4x64xi1, #blocked1>
%mask_67 = arith.andi %mask_65, %mask_66 : tensor<4x64xi1, #blocked1>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<4x64x!tt.ptr<bf16>, #blocked1>
%cosine_68 = tt.addptr %cosine, %offsets_62 : tensor<4x64x!tt.ptr<bf16>, #blocked1>, tensor<4x64xi32, #blocked1>
%cosine_69 = tt.load %cosine_68, %mask_67, %cst_18 : tensor<4x64x!tt.ptr<bf16>, #blocked1>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<4x64x!tt.ptr<bf16>, #blocked1>
%sine_70 = tt.addptr %sine, %offsets_62 : tensor<4x64x!tt.ptr<bf16>, #blocked1>, tensor<4x64xi32, #blocked1>
%sine_71 = tt.load %sine_70, %mask_67, %cst_18 : tensor<4x64x!tt.ptr<bf16>, #blocked1>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_71 : tensor<4x64xbf16, #blocked1> -> tensor<4x64xbf16, #blocked1>
%c = tt.join %sine_71, %negative_sine : tensor<4x64xbf16, #blocked1> -> tensor<4x64x2xbf16, #blocked2>
%f2 = tt.reshape %c : tensor<4x64x2xbf16, #blocked2> -> tensor<4x128xbf16, #blocked>
%c_72 = tt.join %cosine_69, %cosine_69 : tensor<4x64xbf16, #blocked1> -> tensor<4x64x2xbf16, #blocked2>
%1 = tt.reshape %c_72 : tensor<4x64x2xbf16, #blocked2> -> tensor<4x128xbf16, #blocked>
%2 = arith.extf %1 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%3 = arith.extf %f2 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%y_73 = tt.expand_dims %rstd_58 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
%y_74 = tt.broadcast %y_73 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
%y_75 = arith.mulf %y_74, %x_53 : tensor<4x128xf32, #blocked>
%y_76 = arith.mulf %y_74, %x_flip_54 : tensor<4x128xf32, #blocked>
%y_77 = arith.mulf %y_76, %3 : tensor<4x128xf32, #blocked>
%y_78 = math.fma %y_75, %2, %y_77 : tensor<4x128xf32, #blocked>"#
  | .wide, .backward => r#"%offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offsets_37 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
%offsets_38 = tt.broadcast %2 : tensor<32x1xi32, #blocked2> -> tensor<32x64xi32, #blocked2>
%offsets_39 = tt.broadcast %offsets_37 : tensor<1x64xi32, #blocked2> -> tensor<32x64xi32, #blocked2>
%offsets_40 = arith.addi %offsets_38, %offsets_39 : tensor<32x64xi32, #blocked2>
%mask = arith.cmpi slt, %offsets_37, %cst_6 : tensor<1x64xi32, #blocked2>
%mask_41 = tt.broadcast %qk_mask_31 : tensor<32x1xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
%mask_42 = tt.broadcast %mask : tensor<1x64xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
%mask_43 = arith.andi %mask_41, %mask_42 : tensor<32x64xi1, #blocked2>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
%cosine_44 = tt.addptr %cosine, %offsets_40 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi32, #blocked2>
%cosine_45 = tt.load %cosine_44, %mask_43, %cst_14 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
%sine_46 = tt.addptr %sine, %offsets_40 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi32, #blocked2>
%sine_47 = tt.load %sine_46, %mask_43, %cst_14 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_47 : tensor<32x64xbf16, #blocked2> -> tensor<32x64xbf16, #blocked2>
%c = tt.join %negative_sine, %sine_47 : tensor<32x64xbf16, #blocked2> -> tensor<32x64x2xbf16, #blocked3>
%f2 = tt.reshape %c : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%c_48 = tt.join %cosine_45, %cosine_45 : tensor<32x64xbf16, #blocked2> -> tensor<32x64x2xbf16, #blocked3>
%3 = tt.reshape %c_48 : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%4 = arith.extf %3 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%5 = arith.extf %f2 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>"#
  | .paired, .forward => r#"%offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offsets_65 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
%offsets_66 = tt.broadcast %2 : tensor<8x1xi32, #blocked1> -> tensor<8x32xi32, #blocked1>
%offsets_67 = tt.broadcast %offsets_65 : tensor<1x32xi32, #blocked1> -> tensor<8x32xi32, #blocked1>
%offsets_68 = arith.addi %offsets_66, %offsets_67 : tensor<8x32xi32, #blocked1>
%mask_69 = tt.expand_dims %factor_rows {axis = 1 : i32} : tensor<8xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi1, #blocked1>
%mask_70 = arith.cmpi slt, %offsets_65, %cst_9 : tensor<1x32xi32, #blocked1>
%mask_71 = tt.broadcast %mask_69 : tensor<8x1xi1, #blocked1> -> tensor<8x32xi1, #blocked1>
%mask_72 = tt.broadcast %mask_70 : tensor<1x32xi1, #blocked1> -> tensor<8x32xi1, #blocked1>
%mask_73 = arith.andi %mask_71, %mask_72 : tensor<8x32xi1, #blocked1>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked1>
%cosine_74 = tt.addptr %cosine, %offsets_68 : tensor<8x32x!tt.ptr<bf16>, #blocked1>, tensor<8x32xi32, #blocked1>
%cosine_75 = tt.load %cosine_74, %mask_73, %cst_18 : tensor<8x32x!tt.ptr<bf16>, #blocked1>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked1>
%sine_76 = tt.addptr %sine, %offsets_68 : tensor<8x32x!tt.ptr<bf16>, #blocked1>, tensor<8x32xi32, #blocked1>
%sine_77 = tt.load %sine_76, %mask_73, %cst_18 : tensor<8x32x!tt.ptr<bf16>, #blocked1>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_77 : tensor<8x32xbf16, #blocked1> -> tensor<8x32xbf16, #blocked1>
%c = tt.join %sine_77, %negative_sine : tensor<8x32xbf16, #blocked1> -> tensor<8x32x2xbf16, #blocked2>
%f2 = tt.reshape %c : tensor<8x32x2xbf16, #blocked2> -> tensor<8x64xbf16, #blocked>
%c_78 = tt.join %cosine_75, %cosine_75 : tensor<8x32xbf16, #blocked1> -> tensor<8x32x2xbf16, #blocked2>
%3 = tt.reshape %c_78 : tensor<8x32x2xbf16, #blocked2> -> tensor<8x64xbf16, #blocked>
%4 = arith.extf %3 : tensor<8x64xbf16, #blocked> to tensor<8x64xf32, #blocked>
%5 = arith.extf %f2 : tensor<8x64xbf16, #blocked> to tensor<8x64xf32, #blocked>
%y_79 = tt.expand_dims %rstd_59 {axis = 1 : i32} : tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xf32, #blocked>
%y_80 = tt.broadcast %y_79 : tensor<8x1xf32, #blocked> -> tensor<8x64xf32, #blocked>
%y_81 = arith.mulf %y_80, %x_54 : tensor<8x64xf32, #blocked>
%y_82 = arith.mulf %y_80, %x_flip_55 : tensor<8x64xf32, #blocked>
%y_83 = arith.mulf %y_82, %5 : tensor<8x64xf32, #blocked>
%y_84 = math.fma %y_81, %4, %y_83 : tensor<8x64xf32, #blocked>"#
  | .paired, .backward => r#"%offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
%offsets_50 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
%offsets_51 = tt.broadcast %6 : tensor<32x1xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
%offsets_52 = tt.broadcast %offsets_50 : tensor<1x32xi32, #blocked3> -> tensor<32x32xi32, #blocked3>
%offsets_53 = arith.addi %offsets_51, %offsets_52 : tensor<32x32xi32, #blocked3>
%mask = arith.cmpi slt, %offsets_50, %cst_6 : tensor<1x32xi32, #blocked3>
%mask_54 = tt.broadcast %qk_mask_39 : tensor<32x1xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
%mask_55 = tt.broadcast %mask : tensor<1x32xi1, #blocked3> -> tensor<32x32xi1, #blocked3>
%mask_56 = arith.andi %mask_54, %mask_55 : tensor<32x32xi1, #blocked3>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked3>
%cosine_57 = tt.addptr %cosine, %offsets_53 : tensor<32x32x!tt.ptr<bf16>, #blocked3>, tensor<32x32xi32, #blocked3>
%cosine_58 = tt.load %cosine_57, %mask_56, %cst_17 : tensor<32x32x!tt.ptr<bf16>, #blocked3>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked3>
%sine_59 = tt.addptr %sine, %offsets_53 : tensor<32x32x!tt.ptr<bf16>, #blocked3>, tensor<32x32xi32, #blocked3>
%sine_60 = tt.load %sine_59, %mask_56, %cst_17 : tensor<32x32x!tt.ptr<bf16>, #blocked3>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_60 : tensor<32x32xbf16, #blocked3> -> tensor<32x32xbf16, #blocked3>
%c = tt.join %negative_sine, %sine_60 : tensor<32x32xbf16, #blocked3> -> tensor<32x32x2xbf16, #blocked4>
%f2 = tt.reshape %c : tensor<32x32x2xbf16, #blocked4> -> tensor<32x64xbf16, #blocked1>
%c_61 = tt.join %cosine_58, %cosine_58 : tensor<32x32xbf16, #blocked3> -> tensor<32x32x2xbf16, #blocked4>
%7 = tt.reshape %c_61 : tensor<32x32x2xbf16, #blocked4> -> tensor<32x64xbf16, #blocked1>
%8 = arith.extf %7 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>
%9 = arith.extf %f2 : tensor<32x64xbf16, #blocked1> to tensor<32x64xf32, #blocked1>"#
  | .narrow, .forward => r#"%offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offsets_53 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
%offsets_54 = tt.broadcast %0 : tensor<8x1xi32, #blocked1> -> tensor<8x32xi32, #blocked1>
%offsets_55 = tt.broadcast %offsets_53 : tensor<1x32xi32, #blocked1> -> tensor<8x32xi32, #blocked1>
%offsets_56 = arith.addi %offsets_54, %offsets_55 : tensor<8x32xi32, #blocked1>
%mask_57 = tt.expand_dims %factor_rows {axis = 1 : i32} : tensor<8xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<8x1xi1, #blocked1>
%mask_58 = arith.cmpi slt, %offsets_53, %cst_4 : tensor<1x32xi32, #blocked1>
%mask_59 = tt.broadcast %mask_57 : tensor<8x1xi1, #blocked1> -> tensor<8x32xi1, #blocked1>
%mask_60 = tt.broadcast %mask_58 : tensor<1x32xi1, #blocked1> -> tensor<8x32xi1, #blocked1>
%mask_61 = arith.andi %mask_59, %mask_60 : tensor<8x32xi1, #blocked1>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked1>
%cosine_62 = tt.addptr %cosine, %offsets_56 : tensor<8x32x!tt.ptr<bf16>, #blocked1>, tensor<8x32xi32, #blocked1>
%cosine_63 = tt.load %cosine_62, %mask_61, %cst_12 : tensor<8x32x!tt.ptr<bf16>, #blocked1>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked1>
%sine_64 = tt.addptr %sine, %offsets_56 : tensor<8x32x!tt.ptr<bf16>, #blocked1>, tensor<8x32xi32, #blocked1>
%sine_65 = tt.load %sine_64, %mask_61, %cst_12 : tensor<8x32x!tt.ptr<bf16>, #blocked1>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_65 : tensor<8x32xbf16, #blocked1> -> tensor<8x32xbf16, #blocked1>
%c = tt.join %sine_65, %negative_sine : tensor<8x32xbf16, #blocked1> -> tensor<8x32x2xbf16, #blocked2>
%f2 = tt.reshape %c : tensor<8x32x2xbf16, #blocked2> -> tensor<8x64xbf16, #blocked>
%c_66 = tt.join %cosine_63, %cosine_63 : tensor<8x32xbf16, #blocked1> -> tensor<8x32x2xbf16, #blocked2>
%1 = tt.reshape %c_66 : tensor<8x32x2xbf16, #blocked2> -> tensor<8x64xbf16, #blocked>
%2 = arith.extf %1 : tensor<8x64xbf16, #blocked> to tensor<8x64xf32, #blocked>
%3 = arith.extf %f2 : tensor<8x64xbf16, #blocked> to tensor<8x64xf32, #blocked>
%y_67 = tt.expand_dims %rstd_52 {axis = 1 : i32} : tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xf32, #blocked>
%y_68 = tt.broadcast %y_67 : tensor<8x1xf32, #blocked> -> tensor<8x64xf32, #blocked>
%y_69 = arith.mulf %y_68, %x_47 : tensor<8x64xf32, #blocked>
%y_70 = arith.mulf %y_68, %x_flip_48 : tensor<8x64xf32, #blocked>
%y_71 = arith.mulf %y_70, %3 : tensor<8x64xf32, #blocked>
%y_72 = math.fma %y_69, %2, %y_71 : tensor<8x64xf32, #blocked>"#
  | .narrow, .backward => r#"%offsets = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offsets_32 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
%offsets_33 = tt.broadcast %2 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
%offsets_34 = tt.broadcast %offsets_32 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
%offsets_35 = arith.addi %offsets_33, %offsets_34 : tensor<32x32xi32, #blocked1>
%mask = arith.cmpi slt, %offsets_32, %cst_1 : tensor<1x32xi32, #blocked1>
%mask_36 = tt.broadcast %qk_mask_26 : tensor<32x1xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
%mask_37 = tt.broadcast %mask : tensor<1x32xi1, #blocked1> -> tensor<32x32xi1, #blocked1>
%mask_38 = arith.andi %mask_36, %mask_37 : tensor<32x32xi1, #blocked1>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked1>
%cosine_39 = tt.addptr %cosine, %offsets_35 : tensor<32x32x!tt.ptr<bf16>, #blocked1>, tensor<32x32xi32, #blocked1>
%cosine_40 = tt.load %cosine_39, %mask_38, %cst_9 : tensor<32x32x!tt.ptr<bf16>, #blocked1>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>, #blocked1>
%sine_41 = tt.addptr %sine, %offsets_35 : tensor<32x32x!tt.ptr<bf16>, #blocked1>, tensor<32x32xi32, #blocked1>
%sine_42 = tt.load %sine_41, %mask_38, %cst_9 : tensor<32x32x!tt.ptr<bf16>, #blocked1>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_42 : tensor<32x32xbf16, #blocked1> -> tensor<32x32xbf16, #blocked1>
%c = tt.join %negative_sine, %sine_42 : tensor<32x32xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%f2 = tt.reshape %c : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked2>
%c_43 = tt.join %cosine_40, %cosine_40 : tensor<32x32xbf16, #blocked1> -> tensor<32x32x2xbf16, #blocked3>
%3 = tt.reshape %c_43 : tensor<32x32x2xbf16, #blocked3> -> tensor<32x64xbf16, #blocked2>
%4 = arith.extf %3 : tensor<32x64xbf16, #blocked2> to tensor<32x64xf32, #blocked2>
%5 = arith.extf %f2 : tensor<32x64xbf16, #blocked2> to tensor<32x64xf32, #blocked2>"#

def matchesDecoder (s : Layout) (phase : Phase) (ir ptx : String) : Bool :=
  match QKVTTIR.normalizeTTIR ir with
  | none => false
  | some lines =>
    let xorInstructions := (QKVTTIR.normalizePTX ptx).filter (·.startsWith "xor.b16 ")
    let count := if phase = .forward then 4 else 8
    ((String.intercalate "\n" lines).splitOn (decoder s phase)).length == 2 &&
      xorInstructions.length == count && xorInstructions.all (·.endsWith "0x8000;")

structure Certificate (s : Layout) (phase : Phase) (ir ptx : String) : Type where
  matched : matchesDecoder s phase ir ptx = true
  sign : ∀ word : BF16, signBit word = word.neg
  reverseSine : ∀ (pair : Factors) (d : Nat),
    expandPair pair (decide ((d ^^^ 1) % 2 = 1)) = expandPair pair (decide (d % 2 = 0))
  table : ∀ a maxRows state steps, observed a s maxRows state steps = reference a s maxRows state steps

def certify (s : Layout) (phase : Phase) (ir ptx : String) : Option (Certificate s phase ir ptx) :=
  if h : matchesDecoder s phase ir ptx = true then
    some ⟨h, sign_bit_is_negation, backward_factor, fun a rows state steps => observed_equal a s rows state steps⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.YarnTable
