import models.NanoGPT.Track1.YarnStationary
import models.NanoGPT.Track1.YarnBinding

/-! Bind the wide factor loads to the half-width pitch, the stationary
predicate, the cosine and sine fills, and the ordered RoPE expressions. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.YarnStationary

private def decoder : YarnTable.Phase → String × List String
  | .forward => (r#"%0 = arith.muli %x_37, %cst_0 : tensor<4x1xi32, #blocked1>
%offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offsets_60 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%offsets_61 = tt.broadcast %0 : tensor<4x1xi32, #blocked1> -> tensor<4x64xi32, #blocked1>
%offsets_62 = tt.broadcast %offsets_60 : tensor<1x64xi32, #blocked1> -> tensor<4x64xi32, #blocked1>
%offsets_63 = arith.addi %offsets_61, %offsets_62 : tensor<4x64xi32, #blocked1>
%mask_64 = arith.cmpi slt, %offsets_60, %cst_10 : tensor<1x64xi32, #blocked1>
%mask_65 = arith.cmpi slt, %offsets_60, %cst_9 : tensor<1x64xi32, #blocked1>
%mask_66 = arith.andi %mask_64, %mask_65 : tensor<1x64xi1, #blocked1>
%mask_67 = tt.broadcast %mask_66 : tensor<1x64xi1, #blocked1> -> tensor<4x64xi1, #blocked1>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<4x64x!tt.ptr<bf16>, #blocked1>
%cosine_68 = tt.addptr %cosine, %offsets_63 : tensor<4x64x!tt.ptr<bf16>, #blocked1>, tensor<4x64xi32, #blocked1>
%cosine_69 = tt.load %cosine_68, %mask_67, %cst_18 : tensor<4x64x!tt.ptr<bf16>, #blocked1>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<4x64x!tt.ptr<bf16>, #blocked1>
%sine_70 = tt.addptr %sine, %offsets_63 : tensor<4x64x!tt.ptr<bf16>, #blocked1>, tensor<4x64xi32, #blocked1>
%sine_71 = tt.load %sine_70, %mask_67, %cst_19 : tensor<4x64x!tt.ptr<bf16>, #blocked1>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_71 : tensor<4x64xbf16, #blocked1> -> tensor<4x64xbf16, #blocked1>
%c = tt.join %sine_71, %negative_sine : tensor<4x64xbf16, #blocked1> -> tensor<4x64x2xbf16, #blocked2>
%f2 = tt.reshape %c : tensor<4x64x2xbf16, #blocked2> -> tensor<4x128xbf16, #blocked>
%c_72 = tt.join %cosine_69, %cosine_69 : tensor<4x64xbf16, #blocked1> -> tensor<4x64x2xbf16, #blocked2>
%1 = tt.reshape %c_72 : tensor<4x64x2xbf16, #blocked2> -> tensor<4x128xbf16, #blocked>
%2 = arith.extf %1 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%3 = arith.extf %f2 : tensor<4x128xbf16, #blocked> to tensor<4x128xf32, #blocked>
%y_73 = tt.expand_dims %rstd_59 {axis = 1 : i32} : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4x1xf32, #blocked>
%y_74 = tt.broadcast %y_73 : tensor<4x1xf32, #blocked> -> tensor<4x128xf32, #blocked>
%y_75 = arith.mulf %y_74, %x_54 : tensor<4x128xf32, #blocked>
%y_76 = arith.mulf %y_74, %x_flip_55 : tensor<4x128xf32, #blocked>
%y_77 = arith.mulf %y_76, %3 : tensor<4x128xf32, #blocked>
%y_78 = math.fma %y_75, %2, %y_77 : tensor<4x128xf32, #blocked>"#, [
      r#"%cst_0 = arith.constant dense<32> : tensor<4x1xi32, #blocked1>"#,
      r#"%cst_9 = arith.constant dense<32> : tensor<1x64xi32, #blocked1>"#,
      r#"%cst_10 = arith.constant dense<64> : tensor<1x64xi32, #blocked1>"#,
      r#"%cst_18 = arith.constant dense<1.000000e+00> : tensor<4x64xbf16, #blocked1>"#,
      r#"%cst_19 = arith.constant dense<0.000000e+00> : tensor<4x64xbf16, #blocked1>"#])
  | .backward => (r#"%2 = arith.muli %0, %cst_10 : tensor<32x1xi32, #blocked2>
%offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
%offsets_36 = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
%offsets_37 = tt.broadcast %2 : tensor<32x1xi32, #blocked2> -> tensor<32x64xi32, #blocked2>
%offsets_38 = tt.broadcast %offsets_36 : tensor<1x64xi32, #blocked2> -> tensor<32x64xi32, #blocked2>
%offsets_39 = arith.addi %offsets_37, %offsets_38 : tensor<32x64xi32, #blocked2>
%mask = arith.cmpi slt, %offsets_36, %cst_7 : tensor<1x64xi32, #blocked2>
%mask_40 = arith.cmpi slt, %offsets_36, %cst_6 : tensor<1x64xi32, #blocked2>
%mask_41 = arith.andi %mask, %mask_40 : tensor<1x64xi1, #blocked2>
%mask_42 = tt.broadcast %mask_41 : tensor<1x64xi1, #blocked2> -> tensor<32x64xi1, #blocked2>
%cosine = tt.splat %factor1 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
%cosine_43 = tt.addptr %cosine, %offsets_39 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi32, #blocked2>
%cosine_44 = tt.load %cosine_43, %mask_42, %cst_14 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
%sine = tt.splat %factor2 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>, #blocked2>
%sine_45 = tt.addptr %sine, %offsets_39 : tensor<32x64x!tt.ptr<bf16>, #blocked2>, tensor<32x64xi32, #blocked2>
%sine_46 = tt.load %sine_45, %mask_42, %cst_15 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
%negative_sine = tt.elementwise_inline_asm "xor.b16 $0, $1, 0x8000;" {constraints = "=h,h", packed_element = 1 : i32, pure = true} %sine_46 : tensor<32x64xbf16, #blocked2> -> tensor<32x64xbf16, #blocked2>
%c = tt.join %negative_sine, %sine_46 : tensor<32x64xbf16, #blocked2> -> tensor<32x64x2xbf16, #blocked3>
%f2 = tt.reshape %c : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%c_47 = tt.join %cosine_44, %cosine_44 : tensor<32x64xbf16, #blocked2> -> tensor<32x64x2xbf16, #blocked3>
%3 = tt.reshape %c_47 : tensor<32x64x2xbf16, #blocked3> -> tensor<32x128xbf16, #blocked1>
%4 = arith.extf %3 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>
%5 = arith.extf %f2 : tensor<32x128xbf16, #blocked1> to tensor<32x128xf32, #blocked1>"#, [
      r#"%cst_6 = arith.constant dense<32> : tensor<1x64xi32, #blocked2>"#,
      r#"%cst_7 = arith.constant dense<64> : tensor<1x64xi32, #blocked2>"#,
      r#"%cst_10 = arith.constant dense<32> : tensor<32x1xi32, #blocked2>"#,
      r#"%cst_14 = arith.constant dense<1.000000e+00> : tensor<32x64xbf16, #blocked2>"#,
      r#"%cst_15 = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked2>"#])

def matchesDecoder (phase : YarnTable.Phase) (ir ptx : String) : Bool :=
  match QKVTTIR.normalizeTTIR ir with
  | none => false
  | some lines =>
    let (body, args) := decoder phase
    let code := QKVTTIR.normalizePTX ptx
    let signOps := code.filter (·.startsWith "xor.b16 ")
    let count := if phase = .forward then 4 else 8
    let one := code.any fun line => line.startsWith "mov.b32 " && line.endsWith "1065369472;"
    ((String.intercalate "\n" lines).splitOn body).length == 2 &&
      args.all (fun arg => lines.contains arg) && one &&
      signOps.length == count && signOps.all (·.endsWith "0x8000;")

structure Certificate (phase : YarnTable.Phase) (ir ptx : String) : Type where
  matched : matchesDecoder phase ir ptx = true
  storeRange : ∀ {tokens} (row : Fin tokens) (pair : Fin 32),
    2 * (row.val * 32 + pair.val) + 1 < 2 * tokens * 32
  liveTile : ∀ {tiles width} (block : Fin tiles) (lane : Fin width),
    block.val * width + lane.val < tiles * width
  tables : ∀ a rows state steps, candidate a rows state steps = reference a rows state steps

def certify (phase : YarnTable.Phase) (ir ptx : String) : Option (Certificate phase ir ptx) :=
  if h : matchesDecoder phase ir ptx = true then
    some ⟨h, factor_address_in_bounds, aligned_tile_live, observed_equal⟩
  else none

#axiomsClean certify
end VeriTile.Bench.Models.NanoGPT.Track1.YarnStationary
