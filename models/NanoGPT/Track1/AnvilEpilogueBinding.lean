import models.NanoGPT.Track1.AnvilRails
import models.NanoGPT.Track1.AnvilGramBinding

/-! Captured pointwise ANVIL epilogue profiles. The original equalizer
publishes a BF16 word, and the optimizer reads that word for the two
FP32 FMAs. The fused kernel rounds the same word in registers. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilEpilogue.Binding
open VeriTile.Triton.FP
open TailNorm (F32 BF16)
open AnvilGram (Code)

def tileCells : Tile → Nat
  | .v1024 => 1024
  | .v512 => 512

theorem all_cells_tiled (bank : Bank) (tile : Tile) :
    AnvilRails.cells bank / (tileCells tile) * (tileCells tile) = AnvilRails.cells bank := by
  cases bank <;> cases tile <;> decide

def linearCell (bank : Bank) (tile : Tile)
    (block : Fin (AnvilRails.cells bank / (tileCells tile))) (lane : Fin (tileCells tile)) :
    Fin (AnvilRails.cells bank) :=
  ⟨block.val * (tileCells tile) + lane.val, by
    have hb := block.isLt
    have hl := lane.isLt
    have exactSize := all_cells_tiled bank tile
    cases tile <;> simp only [tileCells] at * <;> omega⟩

theorem covers_cells (bank : Bank) (tile : Tile) (i : Fin (AnvilRails.cells bank)) :
    ∃ block lane, linearCell bank tile block lane = i := by
  have hi := i.isLt
  have exactSize := all_cells_tiled bank tile
  let block : Fin (AnvilRails.cells bank / (tileCells tile)) := ⟨i.val / (tileCells tile), by
    cases tile <;> simp only [tileCells] at * <;> omega⟩
  let lane : Fin (tileCells tile) := ⟨i.val % (tileCells tile), by
    apply Nat.mod_lt
    cases tile <;> decide⟩
  refine ⟨block, lane, Fin.ext ?_⟩
  change i.val / (tileCells tile) * (tileCells tile) + i.val % (tileCells tile) = i.val
  simpa [Nat.mul_comm] using Nat.div_add_mod i.val (tileCells tile)

theorem cell_unique (bank : Bank) (tile : Tile) (block next : Fin (AnvilRails.cells bank / (tileCells tile)))
    (lane other : Fin (tileCells tile))
    (same : linearCell bank tile block lane = linearCell bank tile next other) :
    block = next ∧ lane = other := by
  have h := congrArg Fin.val same
  have h₁ := lane.isLt
  have h₂ := other.isLt
  cases tile <;> simp only [linearCell, tileCells] at * <;>
    exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

inductive Part where
  | equalize | decay | fused

private def graph : Bank → Tile → Part → String
  | .qk, .v1024, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<2.560000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<1024xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<1024xi32, #blocked>
%x0 = arith.remsi %xindex_6, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = arith.muli %x2, %cst_0 : tensor<1024xi32, #blocked>
%tmp1_10 = arith.addi %x0, %tmp1 : tensor<1024xi32, #blocked>
%tmp1_11 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1_11, %tmp1_10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %tmp1_10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_3 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_20 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_21 = arith.ori %mask, %mask_20 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_21, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_3 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<1024xf32, #blocked>
%mask_22 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_23 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_24 = arith.ori %mask_22, %mask_23 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_24, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<1024xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %1, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .qk, .v1024, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<1024xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<1024xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<1024xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<1024xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<1024xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<1024xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<1024xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<1024xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<1024xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<1024xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<1024xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp26 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp27 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .qk, .v1024, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_5 = arith.constant dense<2.560000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<1024xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<1024xi32, #blocked>
%x0 = arith.remsi %xindex_8, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = arith.muli %x2, %cst_0 : tensor<1024xi32, #blocked>
%tmp1_12 = arith.addi %x0, %tmp1 : tensor<1024xi32, #blocked>
%tmp1_13 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_14 = tt.addptr %tmp1_13, %tmp1_12 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_15 = tt.load %tmp1_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_16 = tt.addptr %tmp4, %tmp1_12 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_17 = tt.load %tmp4_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_18 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_19 = tt.load %tmp14_18 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_20 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_21 = tt.load %tmp17_20 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29_22 = tt.addptr %tmp29, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp29_23 = tt.load %tmp29_22 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp31_24 = tt.addptr %tmp31, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp31_25 = tt.load %tmp31_24 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35_26 = tt.addptr %tmp35, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp35_27 = tt.load %tmp35_26 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp39_28 = tt.addptr %tmp39, %xindex_8 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp39_29 = tt.load %tmp39_28 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<1024xf32, #blocked>
%tmp3 = arith.divf %tmp1_15, %cst_5 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_17 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_17 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_30 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_31 = arith.ori %mask, %mask_30 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_31, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_19, %cst_5 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_21 : tensor<1024xf32, #blocked>
%mask_32 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_33 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_34 = arith.ori %mask_32, %mask_33 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_34, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<1024xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
%tmp24_35 = tt.bitcast %tmp24 : tensor<1024xbf16, #blocked> -> tensor<1024xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_35 : tensor<1024xi16, #blocked> -> tensor<1024xbf16, #blocked>
%tmp27_36 = arith.extf %tmp27 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp30 = arith.mulf %tmp27_36, %tmp29_23 : tensor<1024xf32, #blocked>
%tmp32 = arith.extui %tmp31_25 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<1024xi32, #blocked>
%tmp36 = arith.extui %tmp35_27 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<1024xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp40 = arith.mulf %tmp39_29, %tmp38 : tensor<1024xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<1024xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<1024xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<1024xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_23 : tensor<1024xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<1024xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<1024xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<1024xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp53 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp54 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .qk, .v512, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<2.560000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<512xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<512xi32, #blocked>
%x0 = arith.remsi %xindex_6, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = arith.muli %x2, %cst_0 : tensor<512xi32, #blocked>
%tmp1_10 = arith.addi %x0, %tmp1 : tensor<512xi32, #blocked>
%tmp1_11 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1_11, %tmp1_10 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %tmp1_10 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_3 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_20 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_21 = arith.ori %mask, %mask_20 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_21, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_3 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<512xf32, #blocked>
%mask_22 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_23 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_24 = arith.ori %mask_22, %mask_23 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_24, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<512xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
tt.store %1, %2 : tensor<512x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .qk, .v512, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<512xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<512xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<512xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<512xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<512xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<512xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<512xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<512xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<512xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<512xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<512xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<512xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<512xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<512xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<512xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp26 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp27 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .qk, .v512, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<196608> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<512xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_5 = arith.constant dense<2.560000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<512xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<512xi32, #blocked>
%x0 = arith.remsi %xindex_8, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = arith.muli %x2, %cst_0 : tensor<512xi32, #blocked>
%tmp1_12 = arith.addi %x0, %tmp1 : tensor<512xi32, #blocked>
%tmp1_13 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_14 = tt.addptr %tmp1_13, %tmp1_12 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_15 = tt.load %tmp1_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_16 = tt.addptr %tmp4, %tmp1_12 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_17 = tt.load %tmp4_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_18 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_19 = tt.load %tmp14_18 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_20 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_21 = tt.load %tmp17_20 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp29_22 = tt.addptr %tmp29, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp29_23 = tt.load %tmp29_22 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp31_24 = tt.addptr %tmp31, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp31_25 = tt.load %tmp31_24 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp35_26 = tt.addptr %tmp35, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp35_27 = tt.load %tmp35_26 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp39_28 = tt.addptr %tmp39, %xindex_8 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp39_29 = tt.load %tmp39_28 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<512xf32, #blocked>
%tmp3 = arith.divf %tmp1_15, %cst_5 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_17 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_17 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_30 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_31 = arith.ori %mask, %mask_30 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_31, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_19, %cst_5 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_21 : tensor<512xf32, #blocked>
%mask_32 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_33 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_34 = arith.ori %mask_32, %mask_33 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_34, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<512xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
%tmp24_35 = tt.bitcast %tmp24 : tensor<512xbf16, #blocked> -> tensor<512xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_35 : tensor<512xi16, #blocked> -> tensor<512xbf16, #blocked>
%tmp27_36 = arith.extf %tmp27 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp30 = arith.mulf %tmp27_36, %tmp29_23 : tensor<512xf32, #blocked>
%tmp32 = arith.extui %tmp31_25 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<512xi32, #blocked>
%tmp36 = arith.extui %tmp35_27 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<512xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp40 = arith.mulf %tmp39_29, %tmp38 : tensor<512xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<512xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<512xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<512xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_23 : tensor<512xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<512xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<512xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<512xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp53 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp54 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v1024, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<7.680000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<1024xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<1024xi32, #blocked>
%x4 = arith.divsi %xindex_6, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_10 = tt.addptr %tmp1, %x4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_11 = tt.load %tmp1_10 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_12 = tt.addptr %tmp4, %x4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_13 = tt.load %tmp4_12 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_14 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_15 = tt.load %tmp14_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_16 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_17 = tt.load %tmp17_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_11, %cst_3 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_13 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_13 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_18 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_19 = arith.ori %mask, %mask_18 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_19, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_15, %cst_3 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_17 : tensor<1024xf32, #blocked>
%mask_20 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_21 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_22 = arith.ori %mask_20, %mask_21 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_22, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<1024xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %1, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v1024, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32>, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<1024xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<1024xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<1024xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<1024xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<1024xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<1024xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<1024xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<1024xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<1024xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<1024xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<1024xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp26 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp27 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v1024, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32>, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_5 = arith.constant dense<7.680000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<1024xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<1024xi32, #blocked>
%x5 = arith.divsi %xindex_8, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1, %x5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %x5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29_20 = tt.addptr %tmp29, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp29_21 = tt.load %tmp29_20 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp31_22 = tt.addptr %tmp31, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp31_23 = tt.load %tmp31_22 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35_24 = tt.addptr %tmp35, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp35_25 = tt.load %tmp35_24 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp39_26 = tt.addptr %tmp39, %xindex_8 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp39_27 = tt.load %tmp39_26 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<1024xf32, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_5 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_28 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_29 = arith.ori %mask, %mask_28 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_29, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_5 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<1024xf32, #blocked>
%mask_30 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_31 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_32 = arith.ori %mask_30, %mask_31 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_32, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<1024xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
%tmp24_33 = tt.bitcast %tmp24 : tensor<1024xbf16, #blocked> -> tensor<1024xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_33 : tensor<1024xi16, #blocked> -> tensor<1024xbf16, #blocked>
%tmp27_34 = arith.extf %tmp27 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp30 = arith.mulf %tmp27_34, %tmp29_21 : tensor<1024xf32, #blocked>
%tmp32 = arith.extui %tmp31_23 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<1024xi32, #blocked>
%tmp36 = arith.extui %tmp35_25 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<1024xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp40 = arith.mulf %tmp39_27, %tmp38 : tensor<1024xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<1024xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<1024xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<1024xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_21 : tensor<1024xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<1024xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<1024xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<1024xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp53 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp54 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v512, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<7.680000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<512xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<512xi32, #blocked>
%x4 = arith.divsi %xindex_6, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_10 = tt.addptr %tmp1, %x4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_11 = tt.load %tmp1_10 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_12 = tt.addptr %tmp4, %x4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_13 = tt.load %tmp4_12 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_14 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_15 = tt.load %tmp14_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_16 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_17 = tt.load %tmp17_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_11, %cst_3 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_13 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_13 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_18 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_19 = arith.ori %mask, %mask_18 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_19, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_15, %cst_3 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_17 : tensor<512xf32, #blocked>
%mask_20 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_21 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_22 = arith.ori %mask_20, %mask_21 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_22, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<512xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
tt.store %1, %2 : tensor<512x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v512, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32>, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<512xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<512xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<512xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<512xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<512xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<512xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<512xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<512xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<512xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<512xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<512xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<512xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<512xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<512xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<512xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp26 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp27 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .vo, .v512, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32>, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<589824> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<512xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_5 = arith.constant dense<7.680000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<512xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<512xi32, #blocked>
%x5 = arith.divsi %xindex_8, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1, %x5 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %x5 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp29_20 = tt.addptr %tmp29, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp29_21 = tt.load %tmp29_20 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp31_22 = tt.addptr %tmp31, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp31_23 = tt.load %tmp31_22 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp35_24 = tt.addptr %tmp35, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp35_25 = tt.load %tmp35_24 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp39_26 = tt.addptr %tmp39, %xindex_8 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp39_27 = tt.load %tmp39_26 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<512xf32, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_5 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_28 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_29 = arith.ori %mask, %mask_28 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_29, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_5 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<512xf32, #blocked>
%mask_30 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_31 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_32 = arith.ori %mask_30, %mask_31 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_32, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<512xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
%tmp24_33 = tt.bitcast %tmp24 : tensor<512xbf16, #blocked> -> tensor<512xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_33 : tensor<512xi16, #blocked> -> tensor<512xbf16, #blocked>
%tmp27_34 = arith.extf %tmp27 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp30 = arith.mulf %tmp27_34, %tmp29_21 : tensor<512xf32, #blocked>
%tmp32 = arith.extui %tmp31_23 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<512xi32, #blocked>
%tmp36 = arith.extui %tmp35_25 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<512xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp40 = arith.mulf %tmp39_27, %tmp38 : tensor<512xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<512xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<512xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<512xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_21 : tensor<512xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<512xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<512xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<512xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp53 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp54 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v1024, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<7.680000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<1024xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<1024xi32, #blocked>
%x4 = arith.divsi %xindex_6, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_10 = tt.addptr %tmp1, %x4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_11 = tt.load %tmp1_10 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_12 = tt.addptr %tmp4, %x4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_13 = tt.load %tmp4_12 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_14 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_15 = tt.load %tmp14_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_16 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_17 = tt.load %tmp17_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_11, %cst_3 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_13 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_13 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_18 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_19 = arith.ori %mask, %mask_18 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_19, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_15, %cst_3 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_17 : tensor<1024xf32, #blocked>
%mask_20 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_21 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_22 = arith.ori %mask_20, %mask_21 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_22, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<1024xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
tt.store %1, %2 : tensor<1024x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v1024, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<1024xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<1024xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<1024xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<1024xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<1024xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<1024xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<1024xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<1024xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<1024xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<1024xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<1024xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<1024xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp26 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp27 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v1024, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<1024xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1024xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<1024xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<1024xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<1024xf32, #blocked>
%cst_5 = arith.constant dense<7.680000e+02> : tensor<1024xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 1024 : i32
%xindex = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<1024xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<1024xi32, #blocked>
%x5 = arith.divsi %xindex_8, %cst_0 : tensor<1024xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<1024xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<1024x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1, %x5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %x5 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp29_20 = tt.addptr %tmp29, %x2 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp29_21 = tt.load %tmp29_20 evictionPolicy = evict_last : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp31_22 = tt.addptr %tmp31, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp31_23 = tt.load %tmp31_22 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%tmp35_24 = tt.addptr %tmp35, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
%tmp35_25 = tt.load %tmp35_24 : tensor<1024x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%tmp39_26 = tt.addptr %tmp39, %xindex_8 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%tmp39_27 = tt.load %tmp39_26 : tensor<1024x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<1024xf32, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_5 : tensor<1024xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<1024xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<1024xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<1024xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<1024xf32, #blocked>
%mask_28 = arith.cmpf une, %tmp10, %tmp10 : tensor<1024xf32, #blocked>
%mask_29 = arith.ori %mask, %mask_28 : tensor<1024xi1, #blocked>
%tmp12 = arith.select %mask_29, %tmp10, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_5 : tensor<1024xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<1024xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<1024xf32, #blocked>
%mask_30 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<1024xf32, #blocked>
%mask_31 = arith.cmpf une, %tmp18, %tmp18 : tensor<1024xf32, #blocked>
%mask_32 = arith.ori %mask_30, %mask_31 : tensor<1024xi1, #blocked>
%tmp19 = arith.select %mask_32, %tmp18, %cst_1 : tensor<1024xi1, #blocked>, tensor<1024xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<1024xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<1024xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<1024xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<1024xf32, #blocked> to tensor<1024xbf16, #blocked>
%tmp24_33 = tt.bitcast %tmp24 : tensor<1024xbf16, #blocked> -> tensor<1024xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_33 : tensor<1024xi16, #blocked> -> tensor<1024xbf16, #blocked>
%tmp27_34 = arith.extf %tmp27 : tensor<1024xbf16, #blocked> to tensor<1024xf32, #blocked>
%tmp30 = arith.mulf %tmp27_34, %tmp29_21 : tensor<1024xf32, #blocked>
%tmp32 = arith.extui %tmp31_23 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<1024xi32, #blocked>
%tmp36 = arith.extui %tmp35_25 : tensor<1024xi16, #blocked> to tensor<1024xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<1024xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<1024xi32, #blocked> -> tensor<1024xf32, #blocked>
%tmp40 = arith.mulf %tmp39_27, %tmp38 : tensor<1024xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<1024xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<1024xi1, #blocked> to tensor<1024xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<1024xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<1024xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_21 : tensor<1024xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<1024xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<1024xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<1024xf32, #blocked> -> tensor<1024xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<1024xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<1024xi32, #blocked> to tensor<1024xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %1, %tmp53 : tensor<1024x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<1024x!tt.ptr<i16>, #blocked>, tensor<1024xi32, #blocked>
tt.store %3, %tmp54 : tensor<1024x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v512, .equalize => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<7.680000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_4 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_5 = tt.splat %xoffset_4 : i32 -> tensor<512xi32, #blocked>
%xindex_6 = arith.addi %xindex_5, %xindex : tensor<512xi32, #blocked>
%x4 = arith.divsi %xindex_6, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_6, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_7 = tt.addptr %tmp0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_8 = tt.load %tmp0_7 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = arith.extf %tmp0_8 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_10 = tt.addptr %tmp1, %x4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_11 = tt.load %tmp1_10 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_12 = tt.addptr %tmp4, %x4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_13 = tt.load %tmp4_12 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_14 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_15 = tt.load %tmp14_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_16 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_17 = tt.load %tmp17_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp3 = arith.divf %tmp1_11, %cst_3 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_13 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_2 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_13 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_18 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_19 = arith.ori %mask, %mask_18 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_19, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_15, %cst_3 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_17 : tensor<512xf32, #blocked>
%mask_20 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_21 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_22 = arith.ori %mask_20, %mask_21 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_22, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_9, %tmp21 : tensor<512xf32, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%1 = tt.addptr %0, %xindex_6 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%2 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
tt.store %1, %2 : tensor<512x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v512, .decay => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(%in_ptr0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32>, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_1 = arith.constant dense<16> : tensor<512xi32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_2 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_3 = tt.splat %xoffset_2 : i32 -> tensor<512xi32, #blocked>
%xindex_4 = arith.addi %xindex_3, %xindex : tensor<512xi32, #blocked>
%x1 = arith.divsi %xindex_4, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp0_5 = tt.addptr %tmp0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_6 = tt.load %tmp0_5 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp4 = tt.splat %in_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp4_7 = tt.addptr %tmp4, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp4_8 = tt.load %tmp4_7 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp8 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp8_9 = tt.addptr %tmp8, %xindex_4 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp8_10 = tt.load %tmp8_9 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.load %in_ptr3 : !tt.ptr<f32>
%tmp15 = tt.splat %tmp14 : f32 -> tensor<512xf32, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_11 = tt.addptr %tmp17, %x1 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_12 = tt.load %tmp17_11 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp20 = tt.splat %in_ptr5 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_13 = tt.addptr %tmp20, %xindex_4 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp20_14 = tt.load %tmp20_13 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp20_15 = arith.extf %tmp20_14 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = arith.extui %tmp0_6 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp3 = arith.shli %tmp1, %cst_1 : tensor<512xi32, #blocked>
%tmp5 = arith.extui %tmp4_8 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp6 = arith.ori %tmp3, %tmp5 : tensor<512xi32, #blocked>
%tmp7 = tt.bitcast %tmp6 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp9 = arith.mulf %tmp8_10, %tmp7 : tensor<512xf32, #blocked>
%tmp11 = arith.cmpf oge, %tmp9, %cst_0 : tensor<512xf32, #blocked>
%tmp12 = arith.uitofp %tmp11 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp13 = arith.mulf %tmp7, %tmp12 : tensor<512xf32, #blocked>
%tmp16 = arith.mulf %tmp13, %tmp15 : tensor<512xf32, #blocked>
%tmp18 = arith.mulf %tmp16, %tmp17_12 : tensor<512xf32, #blocked>
%tmp19 = arith.subf %tmp7, %tmp18 : tensor<512xf32, #blocked>
%tmp22 = arith.mulf %tmp20_15, %tmp17_12 : tensor<512xf32, #blocked>
%tmp23 = arith.subf %tmp19, %tmp22 : tensor<512xf32, #blocked>
%tmp24 = tt.bitcast %tmp23 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp25 = arith.shrui %tmp24, %cst_1 : tensor<512xi32, #blocked>
%tmp26 = arith.trunci %tmp25 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp27 = arith.trunci %tmp24 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp26 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_4 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp27 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#
  | .mlp, .v512, .fused => r#"
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr6: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr7: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %in_ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr9: !tt.ptr<f32>, %out_ptr2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %out_ptr3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %xnumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<2162688> : tensor<512xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<512xi32, #blocked>
%cst_1 = arith.constant dense<1.000000e-10> : tensor<512xf32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked>
%cst_3 = arith.constant dense<16> : tensor<512xi32, #blocked>
%cst_4 = arith.constant dense<1.000000e-01> : tensor<512xf32, #blocked>
%cst_5 = arith.constant dense<7.680000e+02> : tensor<512xf32, #blocked>
%xoffset = tt.get_program_id x : i32
%xoffset_6 = arith.muli %xoffset, 512 : i32
%xindex = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
%xindex_7 = tt.splat %xoffset_6 : i32 -> tensor<512xi32, #blocked>
%xindex_8 = arith.addi %xindex_7, %xindex : tensor<512xi32, #blocked>
%x5 = arith.divsi %xindex_8, %cst_0 : tensor<512xi32, #blocked>
%x2 = arith.divsi %xindex_8, %cst : tensor<512xi32, #blocked>
%tmp0 = tt.splat %in_ptr0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_9 = tt.addptr %tmp0, %xindex_8 : tensor<512x!tt.ptr<bf16>, #blocked>, tensor<512xi32, #blocked>
%tmp0_10 = tt.load %tmp0_9 : tensor<512x!tt.ptr<bf16>, #blocked>
%tmp0_11 = arith.extf %tmp0_10 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp1 = tt.splat %in_ptr1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp1_12 = tt.addptr %tmp1, %x5 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp1_13 = tt.load %tmp1_12 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp4 = tt.splat %in_ptr2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp4_14 = tt.addptr %tmp4, %x5 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp4_15 = tt.load %tmp4_14 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp14 = tt.splat %in_ptr3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp14_16 = tt.addptr %tmp14, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp14_17 = tt.load %tmp14_16 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp17 = tt.splat %in_ptr4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp17_18 = tt.addptr %tmp17, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp17_19 = tt.load %tmp17_18 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp29 = tt.splat %in_ptr5 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp29_20 = tt.addptr %tmp29, %x2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp29_21 = tt.load %tmp29_20 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>, #blocked>
%tmp31 = tt.splat %in_ptr6 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp31_22 = tt.addptr %tmp31, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp31_23 = tt.load %tmp31_22 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp35 = tt.splat %in_ptr7 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%tmp35_24 = tt.addptr %tmp35, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
%tmp35_25 = tt.load %tmp35_24 : tensor<512x!tt.ptr<i16>, #blocked>
%tmp39 = tt.splat %in_ptr8 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
%tmp39_26 = tt.addptr %tmp39, %xindex_8 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
%tmp39_27 = tt.load %tmp39_26 : tensor<512x!tt.ptr<f32>, #blocked>
%tmp45 = tt.load %in_ptr9 : !tt.ptr<f32>
%tmp46 = tt.splat %tmp45 : f32 -> tensor<512xf32, #blocked>
%tmp3 = arith.divf %tmp1_13, %cst_5 : tensor<512xf32, #blocked>
%tmp5 = arith.subf %tmp3, %tmp4_15 : tensor<512xf32, #blocked>
%tmp7 = arith.mulf %tmp5, %cst_4 : tensor<512xf32, #blocked>
%tmp10 = arith.addf %tmp7, %tmp4_15 : tensor<512xf32, #blocked>
%mask = arith.cmpf ogt, %tmp10, %cst_1 : tensor<512xf32, #blocked>
%mask_28 = arith.cmpf une, %tmp10, %tmp10 : tensor<512xf32, #blocked>
%mask_29 = arith.ori %mask, %mask_28 : tensor<512xi1, #blocked>
%tmp12 = arith.select %mask_29, %tmp10, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp13 = tt.extern_elementwise %tmp12 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<512xf32, #blocked>) -> tensor<512xf32, #blocked>
%tmp15 = arith.mulf %tmp14_17, %cst_5 : tensor<512xf32, #blocked>
%tmp16 = tt.precise_sqrt %tmp15 : tensor<512xf32, #blocked>
%tmp18 = tt.precise_sqrt %tmp17_19 : tensor<512xf32, #blocked>
%mask_30 = arith.cmpf ogt, %tmp18, %cst_1 : tensor<512xf32, #blocked>
%mask_31 = arith.cmpf une, %tmp18, %tmp18 : tensor<512xf32, #blocked>
%mask_32 = arith.ori %mask_30, %mask_31 : tensor<512xi1, #blocked>
%tmp19 = arith.select %mask_32, %tmp18, %cst_1 : tensor<512xi1, #blocked>, tensor<512xf32, #blocked>
%tmp20 = arith.divf %tmp16, %tmp19 : tensor<512xf32, #blocked>
%tmp21 = arith.mulf %tmp13, %tmp20 : tensor<512xf32, #blocked>
%tmp23 = arith.mulf %tmp0_11, %tmp21 : tensor<512xf32, #blocked>
%tmp24 = arith.truncf %tmp23 : tensor<512xf32, #blocked> to tensor<512xbf16, #blocked>
%tmp24_33 = tt.bitcast %tmp24 : tensor<512xbf16, #blocked> -> tensor<512xi16, #blocked>
%tmp27 = tt.bitcast %tmp24_33 : tensor<512xi16, #blocked> -> tensor<512xbf16, #blocked>
%tmp27_34 = arith.extf %tmp27 : tensor<512xbf16, #blocked> to tensor<512xf32, #blocked>
%tmp30 = arith.mulf %tmp27_34, %tmp29_21 : tensor<512xf32, #blocked>
%tmp32 = arith.extui %tmp31_23 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp34 = arith.shli %tmp32, %cst_3 : tensor<512xi32, #blocked>
%tmp36 = arith.extui %tmp35_25 : tensor<512xi16, #blocked> to tensor<512xi32, #blocked>
%tmp37 = arith.ori %tmp34, %tmp36 : tensor<512xi32, #blocked>
%tmp38 = tt.bitcast %tmp37 : tensor<512xi32, #blocked> -> tensor<512xf32, #blocked>
%tmp40 = arith.mulf %tmp39_27, %tmp38 : tensor<512xf32, #blocked>
%tmp42 = arith.cmpf oge, %tmp40, %cst_2 : tensor<512xf32, #blocked>
%tmp43 = arith.uitofp %tmp42 : tensor<512xi1, #blocked> to tensor<512xf32, #blocked>
%tmp44 = arith.mulf %tmp38, %tmp43 : tensor<512xf32, #blocked>
%tmp47 = arith.mulf %tmp44, %tmp46 : tensor<512xf32, #blocked>
%tmp48 = arith.mulf %tmp47, %tmp29_21 : tensor<512xf32, #blocked>
%tmp49 = arith.subf %tmp38, %tmp48 : tensor<512xf32, #blocked>
%tmp50 = arith.subf %tmp49, %tmp30 : tensor<512xf32, #blocked>
%tmp51 = tt.bitcast %tmp50 : tensor<512xf32, #blocked> -> tensor<512xi32, #blocked>
%tmp52 = arith.shrui %tmp51, %cst_3 : tensor<512xi32, #blocked>
%tmp53 = arith.trunci %tmp52 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%tmp54 = arith.trunci %tmp51 : tensor<512xi32, #blocked> to tensor<512xi16, #blocked>
%0 = tt.splat %out_ptr2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%1 = tt.addptr %0, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %1, %tmp53 : tensor<512x!tt.ptr<i16>, #blocked>
%2 = tt.splat %out_ptr3 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>, #blocked>
%3 = tt.addptr %2, %xindex_8 : tensor<512x!tt.ptr<i16>, #blocked>, tensor<512xi32, #blocked>
tt.store %3, %tmp54 : tensor<512x!tt.ptr<i16>, #blocked>
tt.return
}
}
  "#

private def assembly : Bank → Tile → Part → String
  | .qk, .v1024, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_8
)
.reqntid 128
{
.reg .pred %p<33>;
.reg .b16 %rs<9>;
.reg .b32 %r<186>;
.reg .b64 %rd<39>;
$L__func_begin0:
ld.param.b64 %rd29, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_0];
ld.param.b64 %rd30, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_1];
$L__tmp0:
mov.u32 %r41, %ctaid.x;
shl.b32 %r42, %r41, 10;
ld.param.b64 %rd31, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_2];
ld.param.b64 %rd32, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_3];
mov.u32 %r43, %tid.x;
shl.b32 %r44, %r43, 3;
ld.param.b64 %rd33, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_4];
and.b32 %r45, %r44, 1016;
ld.param.b64 %rd34, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_5];
or.b32 %r46, %r45, %r42;
or.b32 %r47, %r46, 4;
mul.hi.s32 %r48, %r46, 715827883;
shr.u32 %r49, %r48, 7;
shr.u32 %r50, %r48, 31;
add.s32 %r51, %r49, %r50;
mul.lo.s32 %r52, %r51, 768;
sub.s32 %r53, %r46, %r52;
mul.hi.s32 %r54, %r47, 715827883;
shr.u32 %r55, %r54, 31;
shr.u32 %r56, %r54, 7;
add.s32 %r57, %r56, %r55;
mul.lo.s32 %r58, %r57, 768;
sub.s32 %r59, %r47, %r58;
shr.s32 %r60, %r48, 15;
add.s32 %r61, %r60, %r50;
mul.wide.s32 %rd35, %r46, 2;
add.s64 %rd1, %rd29, %rd35;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mul.lo.s32 %r62, %r61, 768;
add.s32 %r63, %r62, %r53;
add.s32 %r64, %r59, %r62;
mul.wide.s32 %rd36, %r63, 4;
add.s64 %rd2, %rd30, %rd36;
mul.wide.s32 %rd37, %r64, 4;
add.s64 %rd4, %rd30, %rd37;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd4 + 0 ], %rd5;
add.s64 %rd6, %rd31, %rd36;
add.s64 %rd8, %rd31, %rd37;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd8 + 0 ], %rd9;
mul.wide.s32 %rd38, %r61, 4;
add.s64 %rd10, %rd32, %rd38;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd10 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd10 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd10 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd10 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd10 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd10 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd10 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd10 + 0 ], %rd18;
add.s64 %rd19, %rd33, %rd38;
mov.u64 %rd20, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd20, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd19 + 0 ], %rd20;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd19 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd19 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd19 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd19 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd19 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd19 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd19 + 0 ], %rd27;
mov.b32 %r65, 0f43800000;
div.full.f32 %r66, %r5, %r65;
div.full.f32 %r67, %r6, %r65;
div.full.f32 %r68, %r7, %r65;
div.full.f32 %r69, %r8, %r65;
div.full.f32 %r70, %r9, %r65;
div.full.f32 %r71, %r10, %r65;
div.full.f32 %r72, %r11, %r65;
div.full.f32 %r73, %r12, %r65;
sub.f32 %r74, %r73, %r20;
sub.f32 %r75, %r72, %r19;
sub.f32 %r76, %r71, %r18;
sub.f32 %r77, %r70, %r17;
sub.f32 %r78, %r69, %r16;
sub.f32 %r79, %r68, %r15;
sub.f32 %r80, %r67, %r14;
sub.f32 %r81, %r66, %r13;
fma.rn.f32 %r82, %r81, 0f3DCCCCCD, %r13;
fma.rn.f32 %r83, %r80, 0f3DCCCCCD, %r14;
fma.rn.f32 %r84, %r79, 0f3DCCCCCD, %r15;
fma.rn.f32 %r85, %r78, 0f3DCCCCCD, %r16;
fma.rn.f32 %r86, %r77, 0f3DCCCCCD, %r17;
fma.rn.f32 %r87, %r76, 0f3DCCCCCD, %r18;
fma.rn.f32 %r88, %r75, 0f3DCCCCCD, %r19;
fma.rn.f32 %r89, %r74, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r89, 0f2EDBE6FF;
setp.gt.f32 %p2, %r88, 0f2EDBE6FF;
setp.gt.f32 %p3, %r87, 0f2EDBE6FF;
setp.gt.f32 %p4, %r86, 0f2EDBE6FF;
setp.gt.f32 %p5, %r85, 0f2EDBE6FF;
setp.gt.f32 %p6, %r84, 0f2EDBE6FF;
setp.gt.f32 %p7, %r83, 0f2EDBE6FF;
setp.gt.f32 %p8, %r82, 0f2EDBE6FF;
setp.nan.f32 %p9, %r82, %r82;
setp.nan.f32 %p10, %r83, %r83;
setp.nan.f32 %p11, %r84, %r84;
setp.nan.f32 %p12, %r85, %r85;
setp.nan.f32 %p13, %r86, %r86;
setp.nan.f32 %p14, %r87, %r87;
setp.nan.f32 %p15, %r88, %r88;
setp.nan.f32 %p16, %r89, %r89;
selp.f32 %r90, %r82, 0f2EDBE6FF, %p9;
selp.f32 %r91, %r82, %r90, %p8;
selp.f32 %r92, %r83, 0f2EDBE6FF, %p10;
selp.f32 %r93, %r83, %r92, %p7;
selp.f32 %r94, %r84, 0f2EDBE6FF, %p11;
selp.f32 %r95, %r84, %r94, %p6;
selp.f32 %r96, %r85, 0f2EDBE6FF, %p12;
selp.f32 %r97, %r85, %r96, %p5;
selp.f32 %r98, %r86, 0f2EDBE6FF, %p13;
selp.f32 %r99, %r86, %r98, %p4;
selp.f32 %r100, %r87, 0f2EDBE6FF, %p14;
selp.f32 %r101, %r87, %r100, %p3;
selp.f32 %r102, %r88, 0f2EDBE6FF, %p15;
selp.f32 %r103, %r88, %r102, %p2;
selp.f32 %r104, %r89, 0f2EDBE6FF, %p16;
selp.f32 %r105, %r89, %r104, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r106, %r91;
rsqrt.approx.ftz.f32 %r107, %r93;
rsqrt.approx.ftz.f32 %r108, %r95;
rsqrt.approx.ftz.f32 %r109, %r97;
rsqrt.approx.ftz.f32 %r110, %r99;
rsqrt.approx.ftz.f32 %r111, %r101;
rsqrt.approx.ftz.f32 %r112, %r103;
rsqrt.approx.ftz.f32 %r113, %r105;
mul.f32 %r114, %r21, 0f43800000;
mul.f32 %r115, %r22, 0f43800000;
mul.f32 %r116, %r23, 0f43800000;
mul.f32 %r117, %r24, 0f43800000;
mul.f32 %r118, %r25, 0f43800000;
mul.f32 %r119, %r26, 0f43800000;
mul.f32 %r120, %r27, 0f43800000;
mul.f32 %r121, %r28, 0f43800000;
sqrt.rn.f32 %r122, %r114;
sqrt.rn.f32 %r123, %r115;
sqrt.rn.f32 %r124, %r116;
sqrt.rn.f32 %r125, %r117;
sqrt.rn.f32 %r126, %r118;
sqrt.rn.f32 %r127, %r119;
sqrt.rn.f32 %r128, %r120;
sqrt.rn.f32 %r129, %r121;
sqrt.rn.f32 %r130, %r29;
sqrt.rn.f32 %r131, %r30;
sqrt.rn.f32 %r132, %r31;
sqrt.rn.f32 %r133, %r32;
sqrt.rn.f32 %r134, %r33;
sqrt.rn.f32 %r135, %r34;
sqrt.rn.f32 %r136, %r35;
sqrt.rn.f32 %r137, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r130, 0f2EDBE6FF;
setp.gt.f32 %p18, %r131, 0f2EDBE6FF;
setp.gt.f32 %p19, %r132, 0f2EDBE6FF;
setp.gt.f32 %p20, %r133, 0f2EDBE6FF;
setp.gt.f32 %p21, %r134, 0f2EDBE6FF;
setp.gt.f32 %p22, %r135, 0f2EDBE6FF;
setp.gt.f32 %p23, %r136, 0f2EDBE6FF;
setp.gt.f32 %p24, %r137, 0f2EDBE6FF;
setp.nan.f32 %p25, %r130, %r130;
setp.nan.f32 %p26, %r131, %r131;
setp.nan.f32 %p27, %r132, %r132;
setp.nan.f32 %p28, %r133, %r133;
setp.nan.f32 %p29, %r134, %r134;
setp.nan.f32 %p30, %r135, %r135;
setp.nan.f32 %p31, %r136, %r136;
setp.nan.f32 %p32, %r137, %r137;
selp.f32 %r138, %r130, 0f2EDBE6FF, %p25;
selp.f32 %r139, %r130, %r138, %p17;
selp.f32 %r140, %r131, 0f2EDBE6FF, %p26;
selp.f32 %r141, %r131, %r140, %p18;
selp.f32 %r142, %r132, 0f2EDBE6FF, %p27;
selp.f32 %r143, %r132, %r142, %p19;
selp.f32 %r144, %r133, 0f2EDBE6FF, %p28;
selp.f32 %r145, %r133, %r144, %p20;
selp.f32 %r146, %r134, 0f2EDBE6FF, %p29;
selp.f32 %r147, %r134, %r146, %p21;
selp.f32 %r148, %r135, 0f2EDBE6FF, %p30;
selp.f32 %r149, %r135, %r148, %p22;
selp.f32 %r150, %r136, 0f2EDBE6FF, %p31;
selp.f32 %r151, %r136, %r150, %p23;
selp.f32 %r152, %r137, 0f2EDBE6FF, %p32;
selp.f32 %r153, %r137, %r152, %p24;
$L__tmp4:
div.full.f32 %r154, %r122, %r139;
div.full.f32 %r155, %r123, %r141;
div.full.f32 %r156, %r124, %r143;
div.full.f32 %r157, %r125, %r145;
div.full.f32 %r158, %r126, %r147;
div.full.f32 %r159, %r127, %r149;
div.full.f32 %r160, %r128, %r151;
div.full.f32 %r161, %r129, %r153;
add.s64 %rd28, %rd34, %rd35;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r162, %rs2;
cvt.f32.bf16 %r163, %rs1;
mul.f32 %r164, %r107, %r155;
mul.f32 %r165, %r106, %r154;
mul.f32 %r166, %r165, %r163;
mul.f32 %r167, %r164, %r162;
cvt.rn.bf16x2.f32 %r37, %r167, %r166;
mov.b32 {%rs3, %rs4}, %r2;
cvt.f32.bf16 %r168, %rs4;
cvt.f32.bf16 %r169, %rs3;
mul.f32 %r170, %r109, %r157;
mul.f32 %r171, %r108, %r156;
mul.f32 %r172, %r171, %r169;
mul.f32 %r173, %r170, %r168;
cvt.rn.bf16x2.f32 %r38, %r173, %r172;
mov.b32 {%rs5, %rs6}, %r3;
cvt.f32.bf16 %r174, %rs6;
cvt.f32.bf16 %r175, %rs5;
mul.f32 %r176, %r111, %r159;
mul.f32 %r177, %r110, %r158;
mul.f32 %r178, %r177, %r175;
mul.f32 %r179, %r176, %r174;
cvt.rn.bf16x2.f32 %r39, %r179, %r178;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r180, %rs8;
cvt.f32.bf16 %r181, %rs7;
mul.f32 %r182, %r113, %r161;
mul.f32 %r183, %r112, %r160;
mul.f32 %r184, %r183, %r181;
mul.f32 %r185, %r182, %r180;
cvt.rn.bf16x2.f32 %r40, %r185, %r184;
st.global.v4.b32 [ %rd28 + 0 ], { %r37, %r38, %r39, %r40 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .qk, .v1024, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 128
{
.reg .pred %p<9>;
.reg .b16 %rs<33>;
.reg .b32 %r<152>;
.reg .b64 %rd<26>;
$L__func_begin0:
ld.param.b64 %rd18, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd19, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r38, %ctaid.x;
shl.b32 %r39, %r38, 10;
ld.param.b64 %rd20, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd5, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r40, %tid.x;
shl.b32 %r41, %r40, 3;
ld.param.b64 %rd21, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r42, %r41, 1016;
ld.param.b64 %rd22, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r43, %r42, %r39;
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r44, %r43, 715827883;
shr.u32 %r45, %r44, 31;
shr.s32 %r46, %r44, 15;
add.s32 %r47, %r46, %r45;
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd25, %r43, 2;
add.s64 %rd1, %rd18, %rd25;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd19, %rd25;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r43, 4, %rd20;
add.s64 %rd4, %rd3, 16;
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
mov.u32 %r17, 0x0;
ld.global.b32 { %r17 }, [ %rd5 + 0 ];
mad.wide.s32 %rd6, %r47, 4, %rd21;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd6 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd6 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd6 + 0 ], %rd10;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd6 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd6 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd6 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd6 + 0 ], %rd14;
add.s64 %rd15, %rd22, %rd25;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd15 + 0 ];
mov.b32 {%rs1, %rs2}, %r26;
cvt.f32.bf16 %r48, %rs1;
cvt.f32.bf16 %r49, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r50, %rs3;
cvt.u32.u16 %r51, %rs4;
shl.b32 %r52, %r51, 16;
shl.b32 %r53, %r50, 16;
mov.b32 {%rs5, %rs6}, %r5;
cvt.u32.u16 %r54, %rs6;
cvt.u32.u16 %r55, %rs5;
or.b32 %r56, %r53, %r55;
or.b32 %r57, %r52, %r54;
mul.f32 %r58, %r10, %r57;
mul.f32 %r59, %r9, %r56;
setp.ge.f32 %p1, %r59, 0f00000000;
setp.ge.f32 %p2, %r58, 0f00000000;
selp.f32 %r60, 0f3F800000, 0f00000000, %p2;
selp.f32 %r61, 0f3F800000, 0f00000000, %p1;
mul.f32 %r62, %r56, %r61;
mul.f32 %r63, %r57, %r60;
mul.f32 %r64, %r63, %r17;
mul.f32 %r65, %r62, %r17;
neg.f32 %r66, %r65;
fma.rn.f32 %r67, %r66, %r18, %r56;
neg.f32 %r68, %r64;
fma.rn.f32 %r69, %r68, %r19, %r57;
neg.f32 %r70, %r19;
fma.rn.f32 %r71, %r70, %r49, %r69;
neg.f32 %r72, %r18;
fma.rn.f32 %r73, %r72, %r48, %r67;
prmt.b32 %r30, %r73, %r71, 0x7632U;
mov.b32 {%rs7, %rs8}, %r27;
cvt.f32.bf16 %r74, %rs7;
cvt.f32.bf16 %r75, %rs8;
mov.b32 {%rs9, %rs10}, %r2;
cvt.u32.u16 %r76, %rs9;
cvt.u32.u16 %r77, %rs10;
shl.b32 %r78, %r77, 16;
shl.b32 %r79, %r76, 16;
mov.b32 {%rs11, %rs12}, %r6;
cvt.u32.u16 %r80, %rs12;
cvt.u32.u16 %r81, %rs11;
or.b32 %r82, %r79, %r81;
or.b32 %r83, %r78, %r80;
mul.f32 %r84, %r12, %r83;
mul.f32 %r85, %r11, %r82;
setp.ge.f32 %p3, %r85, 0f00000000;
setp.ge.f32 %p4, %r84, 0f00000000;
selp.f32 %r86, 0f3F800000, 0f00000000, %p4;
selp.f32 %r87, 0f3F800000, 0f00000000, %p3;
mul.f32 %r88, %r82, %r87;
mul.f32 %r89, %r83, %r86;
mul.f32 %r90, %r89, %r17;
mul.f32 %r91, %r88, %r17;
neg.f32 %r92, %r91;
fma.rn.f32 %r93, %r92, %r20, %r82;
neg.f32 %r94, %r90;
fma.rn.f32 %r95, %r94, %r21, %r83;
neg.f32 %r96, %r21;
fma.rn.f32 %r97, %r96, %r75, %r95;
neg.f32 %r98, %r20;
fma.rn.f32 %r99, %r98, %r74, %r93;
prmt.b32 %r31, %r99, %r97, 0x7632U;
mov.b32 {%rs13, %rs14}, %r28;
cvt.f32.bf16 %r100, %rs13;
cvt.f32.bf16 %r101, %rs14;
mov.b32 {%rs15, %rs16}, %r3;
cvt.u32.u16 %r102, %rs15;
cvt.u32.u16 %r103, %rs16;
shl.b32 %r104, %r103, 16;
shl.b32 %r105, %r102, 16;
mov.b32 {%rs17, %rs18}, %r7;
cvt.u32.u16 %r106, %rs18;
cvt.u32.u16 %r107, %rs17;
or.b32 %r108, %r105, %r107;
or.b32 %r109, %r104, %r106;
mul.f32 %r110, %r14, %r109;
mul.f32 %r111, %r13, %r108;
setp.ge.f32 %p5, %r111, 0f00000000;
setp.ge.f32 %p6, %r110, 0f00000000;
selp.f32 %r112, 0f3F800000, 0f00000000, %p6;
selp.f32 %r113, 0f3F800000, 0f00000000, %p5;
mul.f32 %r114, %r108, %r113;
mul.f32 %r115, %r109, %r112;
mul.f32 %r116, %r115, %r17;
mul.f32 %r117, %r114, %r17;
neg.f32 %r118, %r117;
fma.rn.f32 %r119, %r118, %r22, %r108;
neg.f32 %r120, %r116;
fma.rn.f32 %r121, %r120, %r23, %r109;
neg.f32 %r122, %r23;
fma.rn.f32 %r123, %r122, %r101, %r121;
neg.f32 %r124, %r22;
fma.rn.f32 %r125, %r124, %r100, %r119;
prmt.b32 %r32, %r125, %r123, 0x7632U;
mov.b32 {%rs19, %rs20}, %r29;
cvt.f32.bf16 %r126, %rs19;
cvt.f32.bf16 %r127, %rs20;
mov.b32 {%rs21, %rs22}, %r4;
cvt.u32.u16 %r128, %rs21;
cvt.u32.u16 %r129, %rs22;
shl.b32 %r130, %r129, 16;
shl.b32 %r131, %r128, 16;
mov.b32 {%rs23, %rs24}, %r8;
cvt.u32.u16 %r132, %rs24;
cvt.u32.u16 %r133, %rs23;
or.b32 %r134, %r131, %r133;
or.b32 %r135, %r130, %r132;
mul.f32 %r136, %r16, %r135;
mul.f32 %r137, %r15, %r134;
setp.ge.f32 %p7, %r137, 0f00000000;
setp.ge.f32 %p8, %r136, 0f00000000;
selp.f32 %r138, 0f3F800000, 0f00000000, %p8;
selp.f32 %r139, 0f3F800000, 0f00000000, %p7;
mul.f32 %r140, %r134, %r139;
mul.f32 %r141, %r135, %r138;
mul.f32 %r142, %r141, %r17;
mul.f32 %r143, %r140, %r17;
neg.f32 %r144, %r143;
fma.rn.f32 %r145, %r144, %r24, %r134;
neg.f32 %r146, %r142;
fma.rn.f32 %r147, %r146, %r25, %r135;
neg.f32 %r148, %r25;
fma.rn.f32 %r149, %r148, %r127, %r147;
neg.f32 %r150, %r24;
fma.rn.f32 %r151, %r150, %r126, %r145;
prmt.b32 %r33, %r151, %r149, 0x7632U;
add.s64 %rd16, %rd23, %rd25;
st.global.v4.b32 [ %rd16 + 0 ], { %r30, %r31, %r32, %r33 };
add.s64 %rd17, %rd24, %rd25;
mov.b32 {%rs25, _}, %r71;
mov.b32 {%rs26, _}, %r73;
mov.b32 %r34, {%rs26, %rs25};
mov.b32 {%rs27, _}, %r97;
mov.b32 {%rs28, _}, %r99;
mov.b32 %r35, {%rs28, %rs27};
mov.b32 {%rs29, _}, %r123;
mov.b32 {%rs30, _}, %r125;
mov.b32 %r36, {%rs30, %rs29};
mov.b32 {%rs31, _}, %r149;
mov.b32 {%rs32, _}, %r151;
mov.b32 %r37, {%rs32, %rs31};
st.global.v4.b32 [ %rd17 + 0 ], { %r34, %r35, %r36, %r37 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .qk, .v1024, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_14
)
.reqntid 128
{
.reg .pred %p<41>;
.reg .b16 %rs<41>;
.reg .b32 %r<319>;
.reg .b64 %rd<59>;
$L__func_begin0:
ld.param.b64 %rd44, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_0];
ld.param.b64 %rd45, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_1];
$L__tmp0:
mov.u32 %r70, %ctaid.x;
shl.b32 %r71, %r70, 10;
ld.param.b64 %rd46, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_2];
ld.param.b64 %rd47, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_3];
mov.u32 %r72, %tid.x;
shl.b32 %r73, %r72, 3;
ld.param.b64 %rd48, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_4];
and.b32 %r74, %r73, 1016;
ld.param.b64 %rd49, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_5];
or.b32 %r75, %r74, %r71;
ld.param.b64 %rd50, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_6];
or.b32 %r76, %r75, 4;
ld.param.b64 %rd51, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_7];
mul.hi.s32 %r77, %r75, 715827883;
shr.u32 %r78, %r77, 7;
shr.u32 %r79, %r77, 31;
add.s32 %r80, %r78, %r79;
mul.lo.s32 %r81, %r80, 768;
sub.s32 %r82, %r75, %r81;
ld.param.b64 %rd52, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_8];
mul.hi.s32 %r83, %r76, 715827883;
shr.u32 %r84, %r83, 31;
shr.u32 %r85, %r83, 7;
add.s32 %r86, %r85, %r84;
mul.lo.s32 %r87, %r86, 768;
sub.s32 %r88, %r76, %r87;
ld.param.b64 %rd41, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_9];
shr.s32 %r89, %r77, 15;
add.s32 %r90, %r89, %r79;
ld.param.b64 %rd53, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_10];
ld.param.b64 %rd54, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_11];
mul.wide.s32 %rd55, %r75, 2;
add.s64 %rd1, %rd44, %rd55;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mul.lo.s32 %r91, %r90, 768;
add.s32 %r92, %r91, %r82;
add.s32 %r93, %r88, %r91;
mul.wide.s32 %rd56, %r92, 4;
add.s64 %rd2, %rd45, %rd56;
mul.wide.s32 %rd57, %r93, 4;
add.s64 %rd4, %rd45, %rd57;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r9, 0x0;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd4 + 0 ], %rd5;
add.s64 %rd6, %rd46, %rd56;
add.s64 %rd8, %rd46, %rd57;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r13, 0x0;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r17, 0x0;
mov.u32 %r18, 0x0;
mov.u32 %r19, 0x0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd8 + 0 ], %rd9;
mul.wide.s32 %rd58, %r90, 4;
add.s64 %rd10, %rd47, %rd58;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd10 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd10 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd10 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd10 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd10 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd10 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd10 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd10 + 0 ], %rd18;
add.s64 %rd19, %rd48, %rd58;
mov.u64 %rd20, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd20, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd19 + 0 ], %rd20;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd19 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd19 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd19 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd19 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd19 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd19 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd19 + 0 ], %rd27;
add.s64 %rd28, %rd49, %rd58;
mov.u64 %rd29, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd29, 1.0;
mov.u32 %r37, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r37 }, [ %rd28 + 0 ], %rd29;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r38, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r38 }, [ %rd28 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r39, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r39 }, [ %rd28 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r40, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r40 }, [ %rd28 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r41, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r41 }, [ %rd28 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r42, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r42 }, [ %rd28 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r43, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r43 }, [ %rd28 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r44, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r44 }, [ %rd28 + 0 ], %rd36;
add.s64 %rd37, %rd50, %rd55;
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
ld.global.v4.b32 { %r45, %r46, %r47, %r48 }, [ %rd37 + 0 ];
add.s64 %rd38, %rd51, %rd55;
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
ld.global.v4.b32 { %r49, %r50, %r51, %r52 }, [ %rd38 + 0 ];
mad.wide.s32 %rd39, %r75, 4, %rd52;
add.s64 %rd40, %rd39, 16;
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
ld.global.v4.b32 { %r53, %r54, %r55, %r56 }, [ %rd39 + 0 ];
mov.u32 %r57, 0x0;
mov.u32 %r58, 0x0;
mov.u32 %r59, 0x0;
mov.u32 %r60, 0x0;
ld.global.v4.b32 { %r57, %r58, %r59, %r60 }, [ %rd40 + 0 ];
mov.u32 %r61, 0x0;
ld.global.b32 { %r61 }, [ %rd41 + 0 ];
mov.b32 %r94, 0f43800000;
div.full.f32 %r95, %r5, %r94;
div.full.f32 %r96, %r6, %r94;
div.full.f32 %r97, %r7, %r94;
div.full.f32 %r98, %r8, %r94;
div.full.f32 %r99, %r9, %r94;
div.full.f32 %r100, %r10, %r94;
div.full.f32 %r101, %r11, %r94;
div.full.f32 %r102, %r12, %r94;
sub.f32 %r103, %r102, %r20;
sub.f32 %r104, %r101, %r19;
sub.f32 %r105, %r100, %r18;
sub.f32 %r106, %r99, %r17;
sub.f32 %r107, %r98, %r16;
sub.f32 %r108, %r97, %r15;
sub.f32 %r109, %r96, %r14;
sub.f32 %r110, %r95, %r13;
fma.rn.f32 %r111, %r110, 0f3DCCCCCD, %r13;
fma.rn.f32 %r112, %r109, 0f3DCCCCCD, %r14;
fma.rn.f32 %r113, %r108, 0f3DCCCCCD, %r15;
fma.rn.f32 %r114, %r107, 0f3DCCCCCD, %r16;
fma.rn.f32 %r115, %r106, 0f3DCCCCCD, %r17;
fma.rn.f32 %r116, %r105, 0f3DCCCCCD, %r18;
fma.rn.f32 %r117, %r104, 0f3DCCCCCD, %r19;
fma.rn.f32 %r118, %r103, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r118, 0f2EDBE6FF;
setp.gt.f32 %p2, %r117, 0f2EDBE6FF;
setp.gt.f32 %p3, %r116, 0f2EDBE6FF;
setp.gt.f32 %p4, %r115, 0f2EDBE6FF;
setp.gt.f32 %p5, %r114, 0f2EDBE6FF;
setp.gt.f32 %p6, %r113, 0f2EDBE6FF;
setp.gt.f32 %p7, %r112, 0f2EDBE6FF;
setp.gt.f32 %p8, %r111, 0f2EDBE6FF;
setp.nan.f32 %p9, %r111, %r111;
setp.nan.f32 %p10, %r112, %r112;
setp.nan.f32 %p11, %r113, %r113;
setp.nan.f32 %p12, %r114, %r114;
setp.nan.f32 %p13, %r115, %r115;
setp.nan.f32 %p14, %r116, %r116;
setp.nan.f32 %p15, %r117, %r117;
setp.nan.f32 %p16, %r118, %r118;
selp.f32 %r119, %r111, 0f2EDBE6FF, %p9;
selp.f32 %r120, %r111, %r119, %p8;
selp.f32 %r121, %r112, 0f2EDBE6FF, %p10;
selp.f32 %r122, %r112, %r121, %p7;
selp.f32 %r123, %r113, 0f2EDBE6FF, %p11;
selp.f32 %r124, %r113, %r123, %p6;
selp.f32 %r125, %r114, 0f2EDBE6FF, %p12;
selp.f32 %r126, %r114, %r125, %p5;
selp.f32 %r127, %r115, 0f2EDBE6FF, %p13;
selp.f32 %r128, %r115, %r127, %p4;
selp.f32 %r129, %r116, 0f2EDBE6FF, %p14;
selp.f32 %r130, %r116, %r129, %p3;
selp.f32 %r131, %r117, 0f2EDBE6FF, %p15;
selp.f32 %r132, %r117, %r131, %p2;
selp.f32 %r133, %r118, 0f2EDBE6FF, %p16;
selp.f32 %r134, %r118, %r133, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r135, %r120;
rsqrt.approx.ftz.f32 %r136, %r122;
rsqrt.approx.ftz.f32 %r137, %r124;
rsqrt.approx.ftz.f32 %r138, %r126;
rsqrt.approx.ftz.f32 %r139, %r128;
rsqrt.approx.ftz.f32 %r140, %r130;
rsqrt.approx.ftz.f32 %r141, %r132;
rsqrt.approx.ftz.f32 %r142, %r134;
mul.f32 %r143, %r21, 0f43800000;
mul.f32 %r144, %r22, 0f43800000;
mul.f32 %r145, %r23, 0f43800000;
mul.f32 %r146, %r24, 0f43800000;
mul.f32 %r147, %r25, 0f43800000;
mul.f32 %r148, %r26, 0f43800000;
mul.f32 %r149, %r27, 0f43800000;
mul.f32 %r150, %r28, 0f43800000;
sqrt.rn.f32 %r151, %r143;
sqrt.rn.f32 %r152, %r144;
sqrt.rn.f32 %r153, %r145;
sqrt.rn.f32 %r154, %r146;
sqrt.rn.f32 %r155, %r147;
sqrt.rn.f32 %r156, %r148;
sqrt.rn.f32 %r157, %r149;
sqrt.rn.f32 %r158, %r150;
sqrt.rn.f32 %r159, %r29;
sqrt.rn.f32 %r160, %r30;
sqrt.rn.f32 %r161, %r31;
sqrt.rn.f32 %r162, %r32;
sqrt.rn.f32 %r163, %r33;
sqrt.rn.f32 %r164, %r34;
sqrt.rn.f32 %r165, %r35;
sqrt.rn.f32 %r166, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r159, 0f2EDBE6FF;
setp.gt.f32 %p18, %r160, 0f2EDBE6FF;
setp.gt.f32 %p19, %r161, 0f2EDBE6FF;
setp.gt.f32 %p20, %r162, 0f2EDBE6FF;
setp.gt.f32 %p21, %r163, 0f2EDBE6FF;
setp.gt.f32 %p22, %r164, 0f2EDBE6FF;
setp.gt.f32 %p23, %r165, 0f2EDBE6FF;
setp.gt.f32 %p24, %r166, 0f2EDBE6FF;
setp.nan.f32 %p25, %r159, %r159;
setp.nan.f32 %p26, %r160, %r160;
setp.nan.f32 %p27, %r161, %r161;
setp.nan.f32 %p28, %r162, %r162;
setp.nan.f32 %p29, %r163, %r163;
setp.nan.f32 %p30, %r164, %r164;
setp.nan.f32 %p31, %r165, %r165;
setp.nan.f32 %p32, %r166, %r166;
selp.f32 %r167, %r159, 0f2EDBE6FF, %p25;
selp.f32 %r168, %r159, %r167, %p17;
selp.f32 %r169, %r160, 0f2EDBE6FF, %p26;
selp.f32 %r170, %r160, %r169, %p18;
selp.f32 %r171, %r161, 0f2EDBE6FF, %p27;
selp.f32 %r172, %r161, %r171, %p19;
selp.f32 %r173, %r162, 0f2EDBE6FF, %p28;
selp.f32 %r174, %r162, %r173, %p20;
selp.f32 %r175, %r163, 0f2EDBE6FF, %p29;
selp.f32 %r176, %r163, %r175, %p21;
selp.f32 %r177, %r164, 0f2EDBE6FF, %p30;
selp.f32 %r178, %r164, %r177, %p22;
selp.f32 %r179, %r165, 0f2EDBE6FF, %p31;
selp.f32 %r180, %r165, %r179, %p23;
selp.f32 %r181, %r166, 0f2EDBE6FF, %p32;
selp.f32 %r182, %r166, %r181, %p24;
$L__tmp4:
div.full.f32 %r183, %r151, %r168;
div.full.f32 %r184, %r152, %r170;
div.full.f32 %r185, %r153, %r172;
div.full.f32 %r186, %r154, %r174;
div.full.f32 %r187, %r155, %r176;
div.full.f32 %r188, %r156, %r178;
div.full.f32 %r189, %r157, %r180;
div.full.f32 %r190, %r158, %r182;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r191, %rs2;
cvt.f32.bf16 %r192, %rs1;
mul.f32 %r193, %r136, %r184;
mul.f32 %r194, %r135, %r183;
mul.f32 %r195, %r194, %r192;
mul.f32 %r196, %r193, %r191;
cvt.rn.bf16.f32 %rs3, %r196;
cvt.rn.bf16.f32 %rs4, %r195;
cvt.f32.bf16 %r197, %rs4;
cvt.f32.bf16 %r198, %rs3;
mov.b32 {%rs5, %rs6}, %r45;
cvt.u32.u16 %r199, %rs5;
cvt.u32.u16 %r200, %rs6;
shl.b32 %r201, %r200, 16;
shl.b32 %r202, %r199, 16;
mov.b32 {%rs7, %rs8}, %r49;
cvt.u32.u16 %r203, %rs8;
cvt.u32.u16 %r204, %rs7;
or.b32 %r205, %r202, %r204;
or.b32 %r206, %r201, %r203;
mul.f32 %r207, %r54, %r206;
mul.f32 %r208, %r53, %r205;
setp.ge.f32 %p33, %r208, 0f00000000;
setp.ge.f32 %p34, %r207, 0f00000000;
selp.f32 %r209, 0f3F800000, 0f00000000, %p34;
selp.f32 %r210, 0f3F800000, 0f00000000, %p33;
mul.f32 %r211, %r205, %r210;
mul.f32 %r212, %r206, %r209;
mul.f32 %r213, %r212, %r61;
mul.f32 %r214, %r211, %r61;
neg.f32 %r215, %r214;
fma.rn.f32 %r216, %r215, %r37, %r205;
neg.f32 %r217, %r213;
fma.rn.f32 %r218, %r217, %r38, %r206;
neg.f32 %r219, %r38;
fma.rn.f32 %r220, %r219, %r198, %r218;
neg.f32 %r221, %r37;
fma.rn.f32 %r222, %r221, %r197, %r216;
prmt.b32 %r62, %r222, %r220, 0x7632U;
mov.b32 {%rs9, %rs10}, %r2;
cvt.f32.bf16 %r223, %rs10;
cvt.f32.bf16 %r224, %rs9;
mul.f32 %r225, %r138, %r186;
mul.f32 %r226, %r137, %r185;
mul.f32 %r227, %r226, %r224;
mul.f32 %r228, %r225, %r223;
cvt.rn.bf16.f32 %rs11, %r228;
cvt.rn.bf16.f32 %rs12, %r227;
cvt.f32.bf16 %r229, %rs12;
cvt.f32.bf16 %r230, %rs11;
mov.b32 {%rs13, %rs14}, %r46;
cvt.u32.u16 %r231, %rs13;
cvt.u32.u16 %r232, %rs14;
shl.b32 %r233, %r232, 16;
shl.b32 %r234, %r231, 16;
mov.b32 {%rs15, %rs16}, %r50;
cvt.u32.u16 %r235, %rs16;
cvt.u32.u16 %r236, %rs15;
or.b32 %r237, %r234, %r236;
or.b32 %r238, %r233, %r235;
mul.f32 %r239, %r56, %r238;
mul.f32 %r240, %r55, %r237;
setp.ge.f32 %p35, %r240, 0f00000000;
setp.ge.f32 %p36, %r239, 0f00000000;
selp.f32 %r241, 0f3F800000, 0f00000000, %p36;
selp.f32 %r242, 0f3F800000, 0f00000000, %p35;
mul.f32 %r243, %r237, %r242;
mul.f32 %r244, %r238, %r241;
mul.f32 %r245, %r244, %r61;
mul.f32 %r246, %r243, %r61;
neg.f32 %r247, %r246;
fma.rn.f32 %r248, %r247, %r39, %r237;
neg.f32 %r249, %r245;
fma.rn.f32 %r250, %r249, %r40, %r238;
neg.f32 %r251, %r40;
fma.rn.f32 %r252, %r251, %r230, %r250;
neg.f32 %r253, %r39;
fma.rn.f32 %r254, %r253, %r229, %r248;
prmt.b32 %r63, %r254, %r252, 0x7632U;
mov.b32 {%rs17, %rs18}, %r3;
cvt.f32.bf16 %r255, %rs18;
cvt.f32.bf16 %r256, %rs17;
mul.f32 %r257, %r140, %r188;
mul.f32 %r258, %r139, %r187;
mul.f32 %r259, %r258, %r256;
mul.f32 %r260, %r257, %r255;
cvt.rn.bf16.f32 %rs19, %r260;
cvt.rn.bf16.f32 %rs20, %r259;
cvt.f32.bf16 %r261, %rs20;
cvt.f32.bf16 %r262, %rs19;
mov.b32 {%rs21, %rs22}, %r47;
cvt.u32.u16 %r263, %rs21;
cvt.u32.u16 %r264, %rs22;
shl.b32 %r265, %r264, 16;
shl.b32 %r266, %r263, 16;
mov.b32 {%rs23, %rs24}, %r51;
cvt.u32.u16 %r267, %rs24;
cvt.u32.u16 %r268, %rs23;
or.b32 %r269, %r266, %r268;
or.b32 %r270, %r265, %r267;
mul.f32 %r271, %r58, %r270;
mul.f32 %r272, %r57, %r269;
setp.ge.f32 %p37, %r272, 0f00000000;
setp.ge.f32 %p38, %r271, 0f00000000;
selp.f32 %r273, 0f3F800000, 0f00000000, %p38;
selp.f32 %r274, 0f3F800000, 0f00000000, %p37;
mul.f32 %r275, %r269, %r274;
mul.f32 %r276, %r270, %r273;
mul.f32 %r277, %r276, %r61;
mul.f32 %r278, %r275, %r61;
neg.f32 %r279, %r278;
fma.rn.f32 %r280, %r279, %r41, %r269;
neg.f32 %r281, %r277;
fma.rn.f32 %r282, %r281, %r42, %r270;
neg.f32 %r283, %r42;
fma.rn.f32 %r284, %r283, %r262, %r282;
neg.f32 %r285, %r41;
fma.rn.f32 %r286, %r285, %r261, %r280;
prmt.b32 %r64, %r286, %r284, 0x7632U;
mov.b32 {%rs25, %rs26}, %r4;
cvt.f32.bf16 %r287, %rs26;
cvt.f32.bf16 %r288, %rs25;
mul.f32 %r289, %r142, %r190;
mul.f32 %r290, %r141, %r189;
mul.f32 %r291, %r290, %r288;
mul.f32 %r292, %r289, %r287;
cvt.rn.bf16.f32 %rs27, %r292;
cvt.rn.bf16.f32 %rs28, %r291;
cvt.f32.bf16 %r293, %rs28;
cvt.f32.bf16 %r294, %rs27;
mov.b32 {%rs29, %rs30}, %r48;
cvt.u32.u16 %r295, %rs29;
cvt.u32.u16 %r296, %rs30;
shl.b32 %r297, %r296, 16;
shl.b32 %r298, %r295, 16;
mov.b32 {%rs31, %rs32}, %r52;
cvt.u32.u16 %r299, %rs32;
cvt.u32.u16 %r300, %rs31;
or.b32 %r301, %r298, %r300;
or.b32 %r302, %r297, %r299;
mul.f32 %r303, %r60, %r302;
mul.f32 %r304, %r59, %r301;
setp.ge.f32 %p39, %r304, 0f00000000;
setp.ge.f32 %p40, %r303, 0f00000000;
selp.f32 %r305, 0f3F800000, 0f00000000, %p40;
selp.f32 %r306, 0f3F800000, 0f00000000, %p39;
mul.f32 %r307, %r301, %r306;
mul.f32 %r308, %r302, %r305;
mul.f32 %r309, %r308, %r61;
mul.f32 %r310, %r307, %r61;
neg.f32 %r311, %r310;
fma.rn.f32 %r312, %r311, %r43, %r301;
neg.f32 %r313, %r309;
fma.rn.f32 %r314, %r313, %r44, %r302;
neg.f32 %r315, %r44;
fma.rn.f32 %r316, %r315, %r294, %r314;
neg.f32 %r317, %r43;
fma.rn.f32 %r318, %r317, %r293, %r312;
prmt.b32 %r65, %r318, %r316, 0x7632U;
add.s64 %rd42, %rd53, %rd55;
st.global.v4.b32 [ %rd42 + 0 ], { %r62, %r63, %r64, %r65 };
add.s64 %rd43, %rd54, %rd55;
mov.b32 {%rs33, _}, %r220;
mov.b32 {%rs34, _}, %r222;
mov.b32 %r66, {%rs34, %rs33};
mov.b32 {%rs35, _}, %r252;
mov.b32 {%rs36, _}, %r254;
mov.b32 %r67, {%rs36, %rs35};
mov.b32 {%rs37, _}, %r284;
mov.b32 {%rs38, _}, %r286;
mov.b32 %r68, {%rs38, %rs37};
mov.b32 {%rs39, _}, %r316;
mov.b32 {%rs40, _}, %r318;
mov.b32 %r69, {%rs40, %rs39};
st.global.v4.b32 [ %rd43 + 0 ], { %r66, %r67, %r68, %r69 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .qk, .v512, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_8
)
.reqntid 256
{
.reg .pred %p<9>;
.reg .b16 %rs<3>;
.reg .b32 %r<57>;
.reg .b64 %rd<22>;
$L__func_begin0:
ld.param.b64 %rd13, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_0];
ld.param.b64 %rd14, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd15, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_2];
ld.param.b64 %rd16, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd17, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd18, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_3_param_5];
or.b32 %r16, %r15, %r12;
mul.hi.s32 %r17, %r16, 715827883;
shr.u32 %r18, %r17, 7;
shr.u32 %r19, %r17, 31;
add.s32 %r20, %r18, %r19;
mul.lo.s32 %r21, %r20, 768;
sub.s32 %r22, %r16, %r21;
shr.s32 %r23, %r17, 15;
add.s32 %r24, %r23, %r19;
mul.wide.s32 %rd19, %r16, 2;
add.s64 %rd1, %rd13, %rd19;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mad.lo.s32 %r25, %r24, 768, %r22;
mul.wide.s32 %rd20, %r25, 4;
add.s64 %rd2, %rd14, %rd20;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v2.b32 { %r2, %r3 }, [ %rd2 + 0 ], %rd3;
add.s64 %rd4, %rd15, %rd20;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v2.b32 { %r4, %r5 }, [ %rd4 + 0 ], %rd5;
mul.wide.s32 %rd21, %r24, 4;
add.s64 %rd6, %rd16, %rd21;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd6 + 0 ], %rd8;
add.s64 %rd9, %rd17, %rd21;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd9 + 0 ], %rd10;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd9 + 0 ], %rd11;
mov.b32 %r26, 0f43800000;
div.full.f32 %r27, %r2, %r26;
div.full.f32 %r28, %r3, %r26;
sub.f32 %r29, %r27, %r4;
sub.f32 %r30, %r28, %r5;
fma.rn.f32 %r31, %r29, 0f3DCCCCCD, %r4;
fma.rn.f32 %r32, %r30, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r31, 0f2EDBE6FF;
setp.gt.f32 %p2, %r32, 0f2EDBE6FF;
setp.nan.f32 %p3, %r31, %r31;
setp.nan.f32 %p4, %r32, %r32;
selp.f32 %r33, %r31, 0f2EDBE6FF, %p3;
selp.f32 %r34, %r31, %r33, %p1;
selp.f32 %r35, %r32, 0f2EDBE6FF, %p4;
selp.f32 %r36, %r32, %r35, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r37, %r34;
rsqrt.approx.ftz.f32 %r38, %r36;
mul.f32 %r39, %r6, 0f43800000;
mul.f32 %r40, %r7, 0f43800000;
sqrt.rn.f32 %r41, %r39;
sqrt.rn.f32 %r42, %r40;
sqrt.rn.f32 %r43, %r8;
sqrt.rn.f32 %r44, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r43, 0f2EDBE6FF;
setp.gt.f32 %p6, %r44, 0f2EDBE6FF;
setp.nan.f32 %p7, %r43, %r43;
setp.nan.f32 %p8, %r44, %r44;
selp.f32 %r45, %r43, 0f2EDBE6FF, %p7;
selp.f32 %r46, %r43, %r45, %p5;
selp.f32 %r47, %r44, 0f2EDBE6FF, %p8;
selp.f32 %r48, %r44, %r47, %p6;
$L__tmp4:
div.full.f32 %r49, %r41, %r46;
div.full.f32 %r50, %r42, %r48;
add.s64 %rd12, %rd18, %rd19;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r51, %rs2;
cvt.f32.bf16 %r52, %rs1;
mul.f32 %r53, %r38, %r50;
mul.f32 %r54, %r37, %r49;
mul.f32 %r55, %r54, %r52;
mul.f32 %r56, %r53, %r51;
cvt.rn.bf16x2.f32 %r10, %r56, %r55;
st.global.b32 [ %rd12 + 0 ], { %r10 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .qk, .v512, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 256
{
.reg .pred %p<3>;
.reg .b16 %rs<9>;
.reg .b32 %r<47>;
.reg .b64 %rd<19>;
$L__func_begin0:
ld.param.b64 %rd11, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd12, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd13, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd4, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd14, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd15, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r16, %r15, %r12;
ld.param.b64 %rd16, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r17, %r16, 715827883;
shr.u32 %r18, %r17, 31;
shr.s32 %r19, %r17, 15;
add.s32 %r20, %r19, %r18;
ld.param.b64 %rd17, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd18, %r16, 2;
add.s64 %rd1, %rd11, %rd18;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd12, %rd18;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r16, 4, %rd13;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v2.b32 { %r3, %r4 }, [ %rd3 + 0 ];
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd4 + 0 ];
mad.wide.s32 %rd5, %r20, 4, %rd14;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd5 + 0 ], %rd7;
add.s64 %rd8, %rd15, %rd18;
mov.u32 %r8, 0x0;
ld.global.b32 { %r8 }, [ %rd8 + 0 ];
mov.b32 {%rs1, %rs2}, %r8;
cvt.f32.bf16 %r21, %rs1;
cvt.f32.bf16 %r22, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r23, %rs3;
cvt.u32.u16 %r24, %rs4;
shl.b32 %r25, %r24, 16;
shl.b32 %r26, %r23, 16;
mov.b32 {%rs5, %rs6}, %r2;
cvt.u32.u16 %r27, %rs6;
cvt.u32.u16 %r28, %rs5;
or.b32 %r29, %r26, %r28;
or.b32 %r30, %r25, %r27;
mul.f32 %r31, %r4, %r30;
mul.f32 %r32, %r3, %r29;
setp.ge.f32 %p1, %r32, 0f00000000;
setp.ge.f32 %p2, %r31, 0f00000000;
selp.f32 %r33, 0f3F800000, 0f00000000, %p2;
selp.f32 %r34, 0f3F800000, 0f00000000, %p1;
mul.f32 %r35, %r29, %r34;
mul.f32 %r36, %r30, %r33;
mul.f32 %r37, %r36, %r5;
mul.f32 %r38, %r35, %r5;
neg.f32 %r39, %r38;
fma.rn.f32 %r40, %r39, %r6, %r29;
neg.f32 %r41, %r37;
fma.rn.f32 %r42, %r41, %r7, %r30;
neg.f32 %r43, %r7;
fma.rn.f32 %r44, %r43, %r22, %r42;
neg.f32 %r45, %r6;
fma.rn.f32 %r46, %r45, %r21, %r40;
prmt.b32 %r9, %r46, %r44, 0x7632U;
add.s64 %rd9, %rd16, %rd18;
st.global.b32 [ %rd9 + 0 ], { %r9 };
add.s64 %rd10, %rd17, %rd18;
mov.b32 {%rs7, _}, %r44;
mov.b32 {%rs8, _}, %r46;
mov.b32 %r10, {%rs8, %rs7};
st.global.b32 [ %rd10 + 0 ], { %r10 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .qk, .v512, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_14
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<11>;
.reg .b32 %r<91>;
.reg .b64 %rd<35>;
$L__func_begin0:
ld.param.b64 %rd21, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_0];
ld.param.b64 %rd22, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_1];
$L__tmp0:
mov.u32 %r19, %ctaid.x;
shl.b32 %r20, %r19, 9;
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_2];
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_3];
mov.u32 %r21, %tid.x;
shl.b32 %r22, %r21, 1;
ld.param.b64 %rd25, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_4];
and.b32 %r23, %r22, 510;
ld.param.b64 %rd26, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_5];
or.b32 %r24, %r23, %r20;
ld.param.b64 %rd27, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_6];
mul.hi.s32 %r25, %r24, 715827883;
shr.u32 %r26, %r25, 7;
shr.u32 %r27, %r25, 31;
add.s32 %r28, %r26, %r27;
mul.lo.s32 %r29, %r28, 768;
sub.s32 %r30, %r24, %r29;
ld.param.b64 %rd28, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_7];
shr.s32 %r31, %r25, 15;
add.s32 %r32, %r31, %r27;
ld.param.b64 %rd29, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_8];
ld.param.b64 %rd18, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_9];
mul.wide.s32 %rd30, %r24, 2;
add.s64 %rd1, %rd21, %rd30;
ld.param.b64 %rd31, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_10];
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
ld.param.b64 %rd32, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_3_param_11];
mad.lo.s32 %r33, %r32, 768, %r30;
mul.wide.s32 %rd33, %r33, 4;
add.s64 %rd2, %rd22, %rd33;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v2.b32 { %r2, %r3 }, [ %rd2 + 0 ], %rd3;
add.s64 %rd4, %rd23, %rd33;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.v2.b32 { %r4, %r5 }, [ %rd4 + 0 ], %rd5;
mul.wide.s32 %rd34, %r32, 4;
add.s64 %rd6, %rd24, %rd34;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd6 + 0 ], %rd8;
add.s64 %rd9, %rd25, %rd34;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd9 + 0 ], %rd10;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd9 + 0 ], %rd11;
add.s64 %rd12, %rd26, %rd34;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd12 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd12 + 0 ], %rd14;
add.s64 %rd15, %rd27, %rd30;
mov.u32 %r12, 0x0;
ld.global.b32 { %r12 }, [ %rd15 + 0 ];
add.s64 %rd16, %rd28, %rd30;
mov.u32 %r13, 0x0;
ld.global.b32 { %r13 }, [ %rd16 + 0 ];
mad.wide.s32 %rd17, %r24, 4, %rd29;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
ld.global.v2.b32 { %r14, %r15 }, [ %rd17 + 0 ];
mov.u32 %r16, 0x0;
ld.global.b32 { %r16 }, [ %rd18 + 0 ];
mov.b32 %r34, 0f43800000;
div.full.f32 %r35, %r2, %r34;
div.full.f32 %r36, %r3, %r34;
sub.f32 %r37, %r35, %r4;
sub.f32 %r38, %r36, %r5;
fma.rn.f32 %r39, %r37, 0f3DCCCCCD, %r4;
fma.rn.f32 %r40, %r38, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r39, 0f2EDBE6FF;
setp.gt.f32 %p2, %r40, 0f2EDBE6FF;
setp.nan.f32 %p3, %r39, %r39;
setp.nan.f32 %p4, %r40, %r40;
selp.f32 %r41, %r39, 0f2EDBE6FF, %p3;
selp.f32 %r42, %r39, %r41, %p1;
selp.f32 %r43, %r40, 0f2EDBE6FF, %p4;
selp.f32 %r44, %r40, %r43, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r45, %r42;
rsqrt.approx.ftz.f32 %r46, %r44;
mul.f32 %r47, %r6, 0f43800000;
mul.f32 %r48, %r7, 0f43800000;
sqrt.rn.f32 %r49, %r47;
sqrt.rn.f32 %r50, %r48;
sqrt.rn.f32 %r51, %r8;
sqrt.rn.f32 %r52, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r51, 0f2EDBE6FF;
setp.gt.f32 %p6, %r52, 0f2EDBE6FF;
setp.nan.f32 %p7, %r51, %r51;
setp.nan.f32 %p8, %r52, %r52;
selp.f32 %r53, %r51, 0f2EDBE6FF, %p7;
selp.f32 %r54, %r51, %r53, %p5;
selp.f32 %r55, %r52, 0f2EDBE6FF, %p8;
selp.f32 %r56, %r52, %r55, %p6;
$L__tmp4:
div.full.f32 %r57, %r49, %r54;
div.full.f32 %r58, %r50, %r56;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r59, %rs2;
cvt.f32.bf16 %r60, %rs1;
mul.f32 %r61, %r46, %r58;
mul.f32 %r62, %r45, %r57;
mul.f32 %r63, %r62, %r60;
mul.f32 %r64, %r61, %r59;
cvt.rn.bf16.f32 %rs3, %r64;
cvt.rn.bf16.f32 %rs4, %r63;
cvt.f32.bf16 %r65, %rs4;
cvt.f32.bf16 %r66, %rs3;
mov.b32 {%rs5, %rs6}, %r12;
cvt.u32.u16 %r67, %rs5;
cvt.u32.u16 %r68, %rs6;
shl.b32 %r69, %r68, 16;
shl.b32 %r70, %r67, 16;
mov.b32 {%rs7, %rs8}, %r13;
cvt.u32.u16 %r71, %rs8;
cvt.u32.u16 %r72, %rs7;
or.b32 %r73, %r70, %r72;
or.b32 %r74, %r69, %r71;
mul.f32 %r75, %r15, %r74;
mul.f32 %r76, %r14, %r73;
setp.ge.f32 %p9, %r76, 0f00000000;
setp.ge.f32 %p10, %r75, 0f00000000;
selp.f32 %r77, 0f3F800000, 0f00000000, %p10;
selp.f32 %r78, 0f3F800000, 0f00000000, %p9;
mul.f32 %r79, %r73, %r78;
mul.f32 %r80, %r74, %r77;
mul.f32 %r81, %r80, %r16;
mul.f32 %r82, %r79, %r16;
neg.f32 %r83, %r82;
fma.rn.f32 %r84, %r83, %r10, %r73;
neg.f32 %r85, %r81;
fma.rn.f32 %r86, %r85, %r11, %r74;
neg.f32 %r87, %r11;
fma.rn.f32 %r88, %r87, %r66, %r86;
neg.f32 %r89, %r10;
fma.rn.f32 %r90, %r89, %r65, %r84;
prmt.b32 %r17, %r90, %r88, 0x7632U;
add.s64 %rd19, %rd31, %rd30;
st.global.b32 [ %rd19 + 0 ], { %r17 };
add.s64 %rd20, %rd32, %rd30;
mov.b32 {%rs9, _}, %r88;
mov.b32 {%rs10, _}, %r90;
mov.b32 %r18, {%rs10, %rs9};
st.global.b32 [ %rd20 + 0 ], { %r18 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .vo, .v1024, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_8
)
.reqntid 128
{
.reg .pred %p<33>;
.reg .b16 %rs<9>;
.reg .b32 %r<176>;
.reg .b64 %rd<48>;
$L__func_begin0:
ld.param.b64 %rd39, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0];
ld.param.b64 %rd40, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1];
$L__tmp0:
mov.u32 %r41, %ctaid.x;
shl.b32 %r42, %r41, 10;
ld.param.b64 %rd41, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2];
ld.param.b64 %rd42, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3];
mov.u32 %r43, %tid.x;
shl.b32 %r44, %r43, 3;
ld.param.b64 %rd43, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4];
and.b32 %r45, %r44, 1016;
ld.param.b64 %rd44, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5];
or.b32 %r46, %r45, %r42;
mul.hi.s32 %r47, %r46, 715827883;
shr.u32 %r48, %r47, 31;
shr.s32 %r49, %r47, 7;
add.s32 %r50, %r49, %r48;
mul.hi.s32 %r51, %r46, 954437177;
shr.u32 %r52, %r51, 31;
shr.s32 %r53, %r51, 17;
add.s32 %r54, %r53, %r52;
mul.wide.s32 %rd45, %r46, 2;
add.s64 %rd1, %rd39, %rd45;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mul.wide.s32 %rd46, %r50, 4;
add.s64 %rd2, %rd40, %rd46;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd2 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd2 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd2 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd2 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd2 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd2 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd2 + 0 ], %rd10;
add.s64 %rd11, %rd41, %rd46;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r13, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r14, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd11 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r15, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd11 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd11 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r17, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd11 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd11 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd11 + 0 ], %rd18;
mov.u64 %rd19, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd19, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd11 + 0 ], %rd19;
mul.wide.s32 %rd47, %r54, 4;
add.s64 %rd20, %rd42, %rd47;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd20 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd20 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd20 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd20 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd20 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd20 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd20 + 0 ], %rd27;
mov.u64 %rd28, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd28, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd20 + 0 ], %rd28;
add.s64 %rd29, %rd43, %rd47;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd29 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd29 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd29 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd29 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd29 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd29 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd29 + 0 ], %rd36;
mov.u64 %rd37, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd37, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd29 + 0 ], %rd37;
mov.b32 %r55, 0f44400000;
div.full.f32 %r56, %r5, %r55;
div.full.f32 %r57, %r6, %r55;
div.full.f32 %r58, %r7, %r55;
div.full.f32 %r59, %r8, %r55;
div.full.f32 %r60, %r9, %r55;
div.full.f32 %r61, %r10, %r55;
div.full.f32 %r62, %r11, %r55;
div.full.f32 %r63, %r12, %r55;
sub.f32 %r64, %r63, %r20;
sub.f32 %r65, %r62, %r19;
sub.f32 %r66, %r61, %r18;
sub.f32 %r67, %r60, %r17;
sub.f32 %r68, %r59, %r16;
sub.f32 %r69, %r58, %r15;
sub.f32 %r70, %r57, %r14;
sub.f32 %r71, %r56, %r13;
fma.rn.f32 %r72, %r71, 0f3DCCCCCD, %r13;
fma.rn.f32 %r73, %r70, 0f3DCCCCCD, %r14;
fma.rn.f32 %r74, %r69, 0f3DCCCCCD, %r15;
fma.rn.f32 %r75, %r68, 0f3DCCCCCD, %r16;
fma.rn.f32 %r76, %r67, 0f3DCCCCCD, %r17;
fma.rn.f32 %r77, %r66, 0f3DCCCCCD, %r18;
fma.rn.f32 %r78, %r65, 0f3DCCCCCD, %r19;
fma.rn.f32 %r79, %r64, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r79, 0f2EDBE6FF;
setp.gt.f32 %p2, %r78, 0f2EDBE6FF;
setp.gt.f32 %p3, %r77, 0f2EDBE6FF;
setp.gt.f32 %p4, %r76, 0f2EDBE6FF;
setp.gt.f32 %p5, %r75, 0f2EDBE6FF;
setp.gt.f32 %p6, %r74, 0f2EDBE6FF;
setp.gt.f32 %p7, %r73, 0f2EDBE6FF;
setp.gt.f32 %p8, %r72, 0f2EDBE6FF;
setp.nan.f32 %p9, %r72, %r72;
setp.nan.f32 %p10, %r73, %r73;
setp.nan.f32 %p11, %r74, %r74;
setp.nan.f32 %p12, %r75, %r75;
setp.nan.f32 %p13, %r76, %r76;
setp.nan.f32 %p14, %r77, %r77;
setp.nan.f32 %p15, %r78, %r78;
setp.nan.f32 %p16, %r79, %r79;
selp.f32 %r80, %r72, 0f2EDBE6FF, %p9;
selp.f32 %r81, %r72, %r80, %p8;
selp.f32 %r82, %r73, 0f2EDBE6FF, %p10;
selp.f32 %r83, %r73, %r82, %p7;
selp.f32 %r84, %r74, 0f2EDBE6FF, %p11;
selp.f32 %r85, %r74, %r84, %p6;
selp.f32 %r86, %r75, 0f2EDBE6FF, %p12;
selp.f32 %r87, %r75, %r86, %p5;
selp.f32 %r88, %r76, 0f2EDBE6FF, %p13;
selp.f32 %r89, %r76, %r88, %p4;
selp.f32 %r90, %r77, 0f2EDBE6FF, %p14;
selp.f32 %r91, %r77, %r90, %p3;
selp.f32 %r92, %r78, 0f2EDBE6FF, %p15;
selp.f32 %r93, %r78, %r92, %p2;
selp.f32 %r94, %r79, 0f2EDBE6FF, %p16;
selp.f32 %r95, %r79, %r94, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r96, %r81;
rsqrt.approx.ftz.f32 %r97, %r83;
rsqrt.approx.ftz.f32 %r98, %r85;
rsqrt.approx.ftz.f32 %r99, %r87;
rsqrt.approx.ftz.f32 %r100, %r89;
rsqrt.approx.ftz.f32 %r101, %r91;
rsqrt.approx.ftz.f32 %r102, %r93;
rsqrt.approx.ftz.f32 %r103, %r95;
mul.f32 %r104, %r21, 0f44400000;
mul.f32 %r105, %r22, 0f44400000;
mul.f32 %r106, %r23, 0f44400000;
mul.f32 %r107, %r24, 0f44400000;
mul.f32 %r108, %r25, 0f44400000;
mul.f32 %r109, %r26, 0f44400000;
mul.f32 %r110, %r27, 0f44400000;
mul.f32 %r111, %r28, 0f44400000;
sqrt.rn.f32 %r112, %r104;
sqrt.rn.f32 %r113, %r105;
sqrt.rn.f32 %r114, %r106;
sqrt.rn.f32 %r115, %r107;
sqrt.rn.f32 %r116, %r108;
sqrt.rn.f32 %r117, %r109;
sqrt.rn.f32 %r118, %r110;
sqrt.rn.f32 %r119, %r111;
sqrt.rn.f32 %r120, %r29;
sqrt.rn.f32 %r121, %r30;
sqrt.rn.f32 %r122, %r31;
sqrt.rn.f32 %r123, %r32;
sqrt.rn.f32 %r124, %r33;
sqrt.rn.f32 %r125, %r34;
sqrt.rn.f32 %r126, %r35;
sqrt.rn.f32 %r127, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r120, 0f2EDBE6FF;
setp.gt.f32 %p18, %r121, 0f2EDBE6FF;
setp.gt.f32 %p19, %r122, 0f2EDBE6FF;
setp.gt.f32 %p20, %r123, 0f2EDBE6FF;
setp.gt.f32 %p21, %r124, 0f2EDBE6FF;
setp.gt.f32 %p22, %r125, 0f2EDBE6FF;
setp.gt.f32 %p23, %r126, 0f2EDBE6FF;
setp.gt.f32 %p24, %r127, 0f2EDBE6FF;
setp.nan.f32 %p25, %r120, %r120;
setp.nan.f32 %p26, %r121, %r121;
setp.nan.f32 %p27, %r122, %r122;
setp.nan.f32 %p28, %r123, %r123;
setp.nan.f32 %p29, %r124, %r124;
setp.nan.f32 %p30, %r125, %r125;
setp.nan.f32 %p31, %r126, %r126;
setp.nan.f32 %p32, %r127, %r127;
selp.f32 %r128, %r120, 0f2EDBE6FF, %p25;
selp.f32 %r129, %r120, %r128, %p17;
selp.f32 %r130, %r121, 0f2EDBE6FF, %p26;
selp.f32 %r131, %r121, %r130, %p18;
selp.f32 %r132, %r122, 0f2EDBE6FF, %p27;
selp.f32 %r133, %r122, %r132, %p19;
selp.f32 %r134, %r123, 0f2EDBE6FF, %p28;
selp.f32 %r135, %r123, %r134, %p20;
selp.f32 %r136, %r124, 0f2EDBE6FF, %p29;
selp.f32 %r137, %r124, %r136, %p21;
selp.f32 %r138, %r125, 0f2EDBE6FF, %p30;
selp.f32 %r139, %r125, %r138, %p22;
selp.f32 %r140, %r126, 0f2EDBE6FF, %p31;
selp.f32 %r141, %r126, %r140, %p23;
selp.f32 %r142, %r127, 0f2EDBE6FF, %p32;
selp.f32 %r143, %r127, %r142, %p24;
$L__tmp4:
div.full.f32 %r144, %r112, %r129;
div.full.f32 %r145, %r113, %r131;
div.full.f32 %r146, %r114, %r133;
div.full.f32 %r147, %r115, %r135;
div.full.f32 %r148, %r116, %r137;
div.full.f32 %r149, %r117, %r139;
div.full.f32 %r150, %r118, %r141;
div.full.f32 %r151, %r119, %r143;
add.s64 %rd38, %rd44, %rd45;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r152, %rs2;
cvt.f32.bf16 %r153, %rs1;
mul.f32 %r154, %r97, %r145;
mul.f32 %r155, %r96, %r144;
mul.f32 %r156, %r155, %r153;
mul.f32 %r157, %r154, %r152;
cvt.rn.bf16x2.f32 %r37, %r157, %r156;
mov.b32 {%rs3, %rs4}, %r2;
cvt.f32.bf16 %r158, %rs4;
cvt.f32.bf16 %r159, %rs3;
mul.f32 %r160, %r99, %r147;
mul.f32 %r161, %r98, %r146;
mul.f32 %r162, %r161, %r159;
mul.f32 %r163, %r160, %r158;
cvt.rn.bf16x2.f32 %r38, %r163, %r162;
mov.b32 {%rs5, %rs6}, %r3;
cvt.f32.bf16 %r164, %rs6;
cvt.f32.bf16 %r165, %rs5;
mul.f32 %r166, %r101, %r149;
mul.f32 %r167, %r100, %r148;
mul.f32 %r168, %r167, %r165;
mul.f32 %r169, %r166, %r164;
cvt.rn.bf16x2.f32 %r39, %r169, %r168;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r170, %rs8;
cvt.f32.bf16 %r171, %rs7;
mul.f32 %r172, %r103, %r151;
mul.f32 %r173, %r102, %r150;
mul.f32 %r174, %r173, %r171;
mul.f32 %r175, %r172, %r170;
cvt.rn.bf16x2.f32 %r40, %r175, %r174;
st.global.v4.b32 [ %rd38 + 0 ], { %r37, %r38, %r39, %r40 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .vo, .v1024, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 128
{
.reg .pred %p<9>;
.reg .b16 %rs<33>;
.reg .b32 %r<152>;
.reg .b64 %rd<26>;
$L__func_begin0:
ld.param.b64 %rd18, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd19, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r38, %ctaid.x;
shl.b32 %r39, %r38, 10;
ld.param.b64 %rd20, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd5, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r40, %tid.x;
shl.b32 %r41, %r40, 3;
ld.param.b64 %rd21, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r42, %r41, 1016;
ld.param.b64 %rd22, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r43, %r42, %r39;
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r44, %r43, 954437177;
shr.u32 %r45, %r44, 31;
shr.s32 %r46, %r44, 17;
add.s32 %r47, %r46, %r45;
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd25, %r43, 2;
add.s64 %rd1, %rd18, %rd25;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd19, %rd25;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r43, 4, %rd20;
add.s64 %rd4, %rd3, 16;
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
mov.u32 %r17, 0x0;
ld.global.b32 { %r17 }, [ %rd5 + 0 ];
mad.wide.s32 %rd6, %r47, 4, %rd21;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd6 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd6 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd6 + 0 ], %rd10;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd6 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd6 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd6 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd6 + 0 ], %rd14;
add.s64 %rd15, %rd22, %rd25;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd15 + 0 ];
mov.b32 {%rs1, %rs2}, %r26;
cvt.f32.bf16 %r48, %rs1;
cvt.f32.bf16 %r49, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r50, %rs3;
cvt.u32.u16 %r51, %rs4;
shl.b32 %r52, %r51, 16;
shl.b32 %r53, %r50, 16;
mov.b32 {%rs5, %rs6}, %r5;
cvt.u32.u16 %r54, %rs6;
cvt.u32.u16 %r55, %rs5;
or.b32 %r56, %r53, %r55;
or.b32 %r57, %r52, %r54;
mul.f32 %r58, %r10, %r57;
mul.f32 %r59, %r9, %r56;
setp.ge.f32 %p1, %r59, 0f00000000;
setp.ge.f32 %p2, %r58, 0f00000000;
selp.f32 %r60, 0f3F800000, 0f00000000, %p2;
selp.f32 %r61, 0f3F800000, 0f00000000, %p1;
mul.f32 %r62, %r56, %r61;
mul.f32 %r63, %r57, %r60;
mul.f32 %r64, %r63, %r17;
mul.f32 %r65, %r62, %r17;
neg.f32 %r66, %r65;
fma.rn.f32 %r67, %r66, %r18, %r56;
neg.f32 %r68, %r64;
fma.rn.f32 %r69, %r68, %r19, %r57;
neg.f32 %r70, %r19;
fma.rn.f32 %r71, %r70, %r49, %r69;
neg.f32 %r72, %r18;
fma.rn.f32 %r73, %r72, %r48, %r67;
prmt.b32 %r30, %r73, %r71, 0x7632U;
mov.b32 {%rs7, %rs8}, %r27;
cvt.f32.bf16 %r74, %rs7;
cvt.f32.bf16 %r75, %rs8;
mov.b32 {%rs9, %rs10}, %r2;
cvt.u32.u16 %r76, %rs9;
cvt.u32.u16 %r77, %rs10;
shl.b32 %r78, %r77, 16;
shl.b32 %r79, %r76, 16;
mov.b32 {%rs11, %rs12}, %r6;
cvt.u32.u16 %r80, %rs12;
cvt.u32.u16 %r81, %rs11;
or.b32 %r82, %r79, %r81;
or.b32 %r83, %r78, %r80;
mul.f32 %r84, %r12, %r83;
mul.f32 %r85, %r11, %r82;
setp.ge.f32 %p3, %r85, 0f00000000;
setp.ge.f32 %p4, %r84, 0f00000000;
selp.f32 %r86, 0f3F800000, 0f00000000, %p4;
selp.f32 %r87, 0f3F800000, 0f00000000, %p3;
mul.f32 %r88, %r82, %r87;
mul.f32 %r89, %r83, %r86;
mul.f32 %r90, %r89, %r17;
mul.f32 %r91, %r88, %r17;
neg.f32 %r92, %r91;
fma.rn.f32 %r93, %r92, %r20, %r82;
neg.f32 %r94, %r90;
fma.rn.f32 %r95, %r94, %r21, %r83;
neg.f32 %r96, %r21;
fma.rn.f32 %r97, %r96, %r75, %r95;
neg.f32 %r98, %r20;
fma.rn.f32 %r99, %r98, %r74, %r93;
prmt.b32 %r31, %r99, %r97, 0x7632U;
mov.b32 {%rs13, %rs14}, %r28;
cvt.f32.bf16 %r100, %rs13;
cvt.f32.bf16 %r101, %rs14;
mov.b32 {%rs15, %rs16}, %r3;
cvt.u32.u16 %r102, %rs15;
cvt.u32.u16 %r103, %rs16;
shl.b32 %r104, %r103, 16;
shl.b32 %r105, %r102, 16;
mov.b32 {%rs17, %rs18}, %r7;
cvt.u32.u16 %r106, %rs18;
cvt.u32.u16 %r107, %rs17;
or.b32 %r108, %r105, %r107;
or.b32 %r109, %r104, %r106;
mul.f32 %r110, %r14, %r109;
mul.f32 %r111, %r13, %r108;
setp.ge.f32 %p5, %r111, 0f00000000;
setp.ge.f32 %p6, %r110, 0f00000000;
selp.f32 %r112, 0f3F800000, 0f00000000, %p6;
selp.f32 %r113, 0f3F800000, 0f00000000, %p5;
mul.f32 %r114, %r108, %r113;
mul.f32 %r115, %r109, %r112;
mul.f32 %r116, %r115, %r17;
mul.f32 %r117, %r114, %r17;
neg.f32 %r118, %r117;
fma.rn.f32 %r119, %r118, %r22, %r108;
neg.f32 %r120, %r116;
fma.rn.f32 %r121, %r120, %r23, %r109;
neg.f32 %r122, %r23;
fma.rn.f32 %r123, %r122, %r101, %r121;
neg.f32 %r124, %r22;
fma.rn.f32 %r125, %r124, %r100, %r119;
prmt.b32 %r32, %r125, %r123, 0x7632U;
mov.b32 {%rs19, %rs20}, %r29;
cvt.f32.bf16 %r126, %rs19;
cvt.f32.bf16 %r127, %rs20;
mov.b32 {%rs21, %rs22}, %r4;
cvt.u32.u16 %r128, %rs21;
cvt.u32.u16 %r129, %rs22;
shl.b32 %r130, %r129, 16;
shl.b32 %r131, %r128, 16;
mov.b32 {%rs23, %rs24}, %r8;
cvt.u32.u16 %r132, %rs24;
cvt.u32.u16 %r133, %rs23;
or.b32 %r134, %r131, %r133;
or.b32 %r135, %r130, %r132;
mul.f32 %r136, %r16, %r135;
mul.f32 %r137, %r15, %r134;
setp.ge.f32 %p7, %r137, 0f00000000;
setp.ge.f32 %p8, %r136, 0f00000000;
selp.f32 %r138, 0f3F800000, 0f00000000, %p8;
selp.f32 %r139, 0f3F800000, 0f00000000, %p7;
mul.f32 %r140, %r134, %r139;
mul.f32 %r141, %r135, %r138;
mul.f32 %r142, %r141, %r17;
mul.f32 %r143, %r140, %r17;
neg.f32 %r144, %r143;
fma.rn.f32 %r145, %r144, %r24, %r134;
neg.f32 %r146, %r142;
fma.rn.f32 %r147, %r146, %r25, %r135;
neg.f32 %r148, %r25;
fma.rn.f32 %r149, %r148, %r127, %r147;
neg.f32 %r150, %r24;
fma.rn.f32 %r151, %r150, %r126, %r145;
prmt.b32 %r33, %r151, %r149, 0x7632U;
add.s64 %rd16, %rd23, %rd25;
st.global.v4.b32 [ %rd16 + 0 ], { %r30, %r31, %r32, %r33 };
add.s64 %rd17, %rd24, %rd25;
mov.b32 {%rs25, _}, %r71;
mov.b32 {%rs26, _}, %r73;
mov.b32 %r34, {%rs26, %rs25};
mov.b32 {%rs27, _}, %r97;
mov.b32 {%rs28, _}, %r99;
mov.b32 %r35, {%rs28, %rs27};
mov.b32 {%rs29, _}, %r123;
mov.b32 {%rs30, _}, %r125;
mov.b32 %r36, {%rs30, %rs29};
mov.b32 {%rs31, _}, %r149;
mov.b32 {%rs32, _}, %r151;
mov.b32 %r37, {%rs32, %rs31};
st.global.v4.b32 [ %rd17 + 0 ], { %r34, %r35, %r36, %r37 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .vo, .v1024, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_14
)
.reqntid 128
{
.reg .pred %p<41>;
.reg .b16 %rs<41>;
.reg .b32 %r<309>;
.reg .b64 %rd<68>;
$L__func_begin0:
ld.param.b64 %rd54, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0];
ld.param.b64 %rd55, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1];
$L__tmp0:
mov.u32 %r70, %ctaid.x;
shl.b32 %r71, %r70, 10;
ld.param.b64 %rd56, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2];
ld.param.b64 %rd57, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3];
mov.u32 %r72, %tid.x;
shl.b32 %r73, %r72, 3;
ld.param.b64 %rd58, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4];
and.b32 %r74, %r73, 1016;
ld.param.b64 %rd59, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5];
or.b32 %r75, %r74, %r71;
ld.param.b64 %rd60, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6];
mul.hi.s32 %r76, %r75, 715827883;
shr.u32 %r77, %r76, 31;
shr.s32 %r78, %r76, 7;
add.s32 %r79, %r78, %r77;
ld.param.b64 %rd61, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7];
mul.hi.s32 %r80, %r75, 954437177;
shr.u32 %r81, %r80, 31;
shr.s32 %r82, %r80, 17;
add.s32 %r83, %r82, %r81;
ld.param.b64 %rd62, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8];
ld.param.b64 %rd51, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9];
mul.wide.s32 %rd63, %r75, 2;
add.s64 %rd1, %rd54, %rd63;
ld.param.b64 %rd64, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10];
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
ld.param.b64 %rd65, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11];
mul.wide.s32 %rd66, %r79, 4;
add.s64 %rd2, %rd55, %rd66;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd2 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd2 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd2 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd2 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd2 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd2 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd2 + 0 ], %rd10;
add.s64 %rd11, %rd56, %rd66;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r13, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r14, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd11 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r15, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd11 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd11 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r17, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd11 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd11 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd11 + 0 ], %rd18;
mov.u64 %rd19, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd19, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd11 + 0 ], %rd19;
mul.wide.s32 %rd67, %r83, 4;
add.s64 %rd20, %rd57, %rd67;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd20 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd20 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd20 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd20 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd20 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd20 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd20 + 0 ], %rd27;
mov.u64 %rd28, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd28, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd20 + 0 ], %rd28;
add.s64 %rd29, %rd58, %rd67;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd29 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd29 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd29 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd29 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd29 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd29 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd29 + 0 ], %rd36;
mov.u64 %rd37, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd37, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd29 + 0 ], %rd37;
add.s64 %rd38, %rd59, %rd67;
mov.u64 %rd39, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd39, 1.0;
mov.u32 %r37, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r37 }, [ %rd38 + 0 ], %rd39;
mov.u64 %rd40, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd40, 1.0;
mov.u32 %r38, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r38 }, [ %rd38 + 0 ], %rd40;
mov.u64 %rd41, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd41, 1.0;
mov.u32 %r39, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r39 }, [ %rd38 + 0 ], %rd41;
mov.u64 %rd42, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd42, 1.0;
mov.u32 %r40, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r40 }, [ %rd38 + 0 ], %rd42;
mov.u64 %rd43, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd43, 1.0;
mov.u32 %r41, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r41 }, [ %rd38 + 0 ], %rd43;
mov.u64 %rd44, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd44, 1.0;
mov.u32 %r42, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r42 }, [ %rd38 + 0 ], %rd44;
mov.u64 %rd45, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd45, 1.0;
mov.u32 %r43, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r43 }, [ %rd38 + 0 ], %rd45;
mov.u64 %rd46, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd46, 1.0;
mov.u32 %r44, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r44 }, [ %rd38 + 0 ], %rd46;
add.s64 %rd47, %rd60, %rd63;
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
ld.global.v4.b32 { %r45, %r46, %r47, %r48 }, [ %rd47 + 0 ];
add.s64 %rd48, %rd61, %rd63;
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
ld.global.v4.b32 { %r49, %r50, %r51, %r52 }, [ %rd48 + 0 ];
mad.wide.s32 %rd49, %r75, 4, %rd62;
add.s64 %rd50, %rd49, 16;
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
ld.global.v4.b32 { %r53, %r54, %r55, %r56 }, [ %rd49 + 0 ];
mov.u32 %r57, 0x0;
mov.u32 %r58, 0x0;
mov.u32 %r59, 0x0;
mov.u32 %r60, 0x0;
ld.global.v4.b32 { %r57, %r58, %r59, %r60 }, [ %rd50 + 0 ];
mov.u32 %r61, 0x0;
ld.global.b32 { %r61 }, [ %rd51 + 0 ];
mov.b32 %r84, 0f44400000;
div.full.f32 %r85, %r5, %r84;
div.full.f32 %r86, %r6, %r84;
div.full.f32 %r87, %r7, %r84;
div.full.f32 %r88, %r8, %r84;
div.full.f32 %r89, %r9, %r84;
div.full.f32 %r90, %r10, %r84;
div.full.f32 %r91, %r11, %r84;
div.full.f32 %r92, %r12, %r84;
sub.f32 %r93, %r92, %r20;
sub.f32 %r94, %r91, %r19;
sub.f32 %r95, %r90, %r18;
sub.f32 %r96, %r89, %r17;
sub.f32 %r97, %r88, %r16;
sub.f32 %r98, %r87, %r15;
sub.f32 %r99, %r86, %r14;
sub.f32 %r100, %r85, %r13;
fma.rn.f32 %r101, %r100, 0f3DCCCCCD, %r13;
fma.rn.f32 %r102, %r99, 0f3DCCCCCD, %r14;
fma.rn.f32 %r103, %r98, 0f3DCCCCCD, %r15;
fma.rn.f32 %r104, %r97, 0f3DCCCCCD, %r16;
fma.rn.f32 %r105, %r96, 0f3DCCCCCD, %r17;
fma.rn.f32 %r106, %r95, 0f3DCCCCCD, %r18;
fma.rn.f32 %r107, %r94, 0f3DCCCCCD, %r19;
fma.rn.f32 %r108, %r93, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r108, 0f2EDBE6FF;
setp.gt.f32 %p2, %r107, 0f2EDBE6FF;
setp.gt.f32 %p3, %r106, 0f2EDBE6FF;
setp.gt.f32 %p4, %r105, 0f2EDBE6FF;
setp.gt.f32 %p5, %r104, 0f2EDBE6FF;
setp.gt.f32 %p6, %r103, 0f2EDBE6FF;
setp.gt.f32 %p7, %r102, 0f2EDBE6FF;
setp.gt.f32 %p8, %r101, 0f2EDBE6FF;
setp.nan.f32 %p9, %r101, %r101;
setp.nan.f32 %p10, %r102, %r102;
setp.nan.f32 %p11, %r103, %r103;
setp.nan.f32 %p12, %r104, %r104;
setp.nan.f32 %p13, %r105, %r105;
setp.nan.f32 %p14, %r106, %r106;
setp.nan.f32 %p15, %r107, %r107;
setp.nan.f32 %p16, %r108, %r108;
selp.f32 %r109, %r101, 0f2EDBE6FF, %p9;
selp.f32 %r110, %r101, %r109, %p8;
selp.f32 %r111, %r102, 0f2EDBE6FF, %p10;
selp.f32 %r112, %r102, %r111, %p7;
selp.f32 %r113, %r103, 0f2EDBE6FF, %p11;
selp.f32 %r114, %r103, %r113, %p6;
selp.f32 %r115, %r104, 0f2EDBE6FF, %p12;
selp.f32 %r116, %r104, %r115, %p5;
selp.f32 %r117, %r105, 0f2EDBE6FF, %p13;
selp.f32 %r118, %r105, %r117, %p4;
selp.f32 %r119, %r106, 0f2EDBE6FF, %p14;
selp.f32 %r120, %r106, %r119, %p3;
selp.f32 %r121, %r107, 0f2EDBE6FF, %p15;
selp.f32 %r122, %r107, %r121, %p2;
selp.f32 %r123, %r108, 0f2EDBE6FF, %p16;
selp.f32 %r124, %r108, %r123, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r125, %r110;
rsqrt.approx.ftz.f32 %r126, %r112;
rsqrt.approx.ftz.f32 %r127, %r114;
rsqrt.approx.ftz.f32 %r128, %r116;
rsqrt.approx.ftz.f32 %r129, %r118;
rsqrt.approx.ftz.f32 %r130, %r120;
rsqrt.approx.ftz.f32 %r131, %r122;
rsqrt.approx.ftz.f32 %r132, %r124;
mul.f32 %r133, %r21, 0f44400000;
mul.f32 %r134, %r22, 0f44400000;
mul.f32 %r135, %r23, 0f44400000;
mul.f32 %r136, %r24, 0f44400000;
mul.f32 %r137, %r25, 0f44400000;
mul.f32 %r138, %r26, 0f44400000;
mul.f32 %r139, %r27, 0f44400000;
mul.f32 %r140, %r28, 0f44400000;
sqrt.rn.f32 %r141, %r133;
sqrt.rn.f32 %r142, %r134;
sqrt.rn.f32 %r143, %r135;
sqrt.rn.f32 %r144, %r136;
sqrt.rn.f32 %r145, %r137;
sqrt.rn.f32 %r146, %r138;
sqrt.rn.f32 %r147, %r139;
sqrt.rn.f32 %r148, %r140;
sqrt.rn.f32 %r149, %r29;
sqrt.rn.f32 %r150, %r30;
sqrt.rn.f32 %r151, %r31;
sqrt.rn.f32 %r152, %r32;
sqrt.rn.f32 %r153, %r33;
sqrt.rn.f32 %r154, %r34;
sqrt.rn.f32 %r155, %r35;
sqrt.rn.f32 %r156, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r149, 0f2EDBE6FF;
setp.gt.f32 %p18, %r150, 0f2EDBE6FF;
setp.gt.f32 %p19, %r151, 0f2EDBE6FF;
setp.gt.f32 %p20, %r152, 0f2EDBE6FF;
setp.gt.f32 %p21, %r153, 0f2EDBE6FF;
setp.gt.f32 %p22, %r154, 0f2EDBE6FF;
setp.gt.f32 %p23, %r155, 0f2EDBE6FF;
setp.gt.f32 %p24, %r156, 0f2EDBE6FF;
setp.nan.f32 %p25, %r149, %r149;
setp.nan.f32 %p26, %r150, %r150;
setp.nan.f32 %p27, %r151, %r151;
setp.nan.f32 %p28, %r152, %r152;
setp.nan.f32 %p29, %r153, %r153;
setp.nan.f32 %p30, %r154, %r154;
setp.nan.f32 %p31, %r155, %r155;
setp.nan.f32 %p32, %r156, %r156;
selp.f32 %r157, %r149, 0f2EDBE6FF, %p25;
selp.f32 %r158, %r149, %r157, %p17;
selp.f32 %r159, %r150, 0f2EDBE6FF, %p26;
selp.f32 %r160, %r150, %r159, %p18;
selp.f32 %r161, %r151, 0f2EDBE6FF, %p27;
selp.f32 %r162, %r151, %r161, %p19;
selp.f32 %r163, %r152, 0f2EDBE6FF, %p28;
selp.f32 %r164, %r152, %r163, %p20;
selp.f32 %r165, %r153, 0f2EDBE6FF, %p29;
selp.f32 %r166, %r153, %r165, %p21;
selp.f32 %r167, %r154, 0f2EDBE6FF, %p30;
selp.f32 %r168, %r154, %r167, %p22;
selp.f32 %r169, %r155, 0f2EDBE6FF, %p31;
selp.f32 %r170, %r155, %r169, %p23;
selp.f32 %r171, %r156, 0f2EDBE6FF, %p32;
selp.f32 %r172, %r156, %r171, %p24;
$L__tmp4:
div.full.f32 %r173, %r141, %r158;
div.full.f32 %r174, %r142, %r160;
div.full.f32 %r175, %r143, %r162;
div.full.f32 %r176, %r144, %r164;
div.full.f32 %r177, %r145, %r166;
div.full.f32 %r178, %r146, %r168;
div.full.f32 %r179, %r147, %r170;
div.full.f32 %r180, %r148, %r172;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r181, %rs2;
cvt.f32.bf16 %r182, %rs1;
mul.f32 %r183, %r126, %r174;
mul.f32 %r184, %r125, %r173;
mul.f32 %r185, %r184, %r182;
mul.f32 %r186, %r183, %r181;
cvt.rn.bf16.f32 %rs3, %r186;
cvt.rn.bf16.f32 %rs4, %r185;
cvt.f32.bf16 %r187, %rs4;
cvt.f32.bf16 %r188, %rs3;
mov.b32 {%rs5, %rs6}, %r45;
cvt.u32.u16 %r189, %rs5;
cvt.u32.u16 %r190, %rs6;
shl.b32 %r191, %r190, 16;
shl.b32 %r192, %r189, 16;
mov.b32 {%rs7, %rs8}, %r49;
cvt.u32.u16 %r193, %rs8;
cvt.u32.u16 %r194, %rs7;
or.b32 %r195, %r192, %r194;
or.b32 %r196, %r191, %r193;
mul.f32 %r197, %r54, %r196;
mul.f32 %r198, %r53, %r195;
setp.ge.f32 %p33, %r198, 0f00000000;
setp.ge.f32 %p34, %r197, 0f00000000;
selp.f32 %r199, 0f3F800000, 0f00000000, %p34;
selp.f32 %r200, 0f3F800000, 0f00000000, %p33;
mul.f32 %r201, %r195, %r200;
mul.f32 %r202, %r196, %r199;
mul.f32 %r203, %r202, %r61;
mul.f32 %r204, %r201, %r61;
neg.f32 %r205, %r204;
fma.rn.f32 %r206, %r205, %r37, %r195;
neg.f32 %r207, %r203;
fma.rn.f32 %r208, %r207, %r38, %r196;
neg.f32 %r209, %r38;
fma.rn.f32 %r210, %r209, %r188, %r208;
neg.f32 %r211, %r37;
fma.rn.f32 %r212, %r211, %r187, %r206;
prmt.b32 %r62, %r212, %r210, 0x7632U;
mov.b32 {%rs9, %rs10}, %r2;
cvt.f32.bf16 %r213, %rs10;
cvt.f32.bf16 %r214, %rs9;
mul.f32 %r215, %r128, %r176;
mul.f32 %r216, %r127, %r175;
mul.f32 %r217, %r216, %r214;
mul.f32 %r218, %r215, %r213;
cvt.rn.bf16.f32 %rs11, %r218;
cvt.rn.bf16.f32 %rs12, %r217;
cvt.f32.bf16 %r219, %rs12;
cvt.f32.bf16 %r220, %rs11;
mov.b32 {%rs13, %rs14}, %r46;
cvt.u32.u16 %r221, %rs13;
cvt.u32.u16 %r222, %rs14;
shl.b32 %r223, %r222, 16;
shl.b32 %r224, %r221, 16;
mov.b32 {%rs15, %rs16}, %r50;
cvt.u32.u16 %r225, %rs16;
cvt.u32.u16 %r226, %rs15;
or.b32 %r227, %r224, %r226;
or.b32 %r228, %r223, %r225;
mul.f32 %r229, %r56, %r228;
mul.f32 %r230, %r55, %r227;
setp.ge.f32 %p35, %r230, 0f00000000;
setp.ge.f32 %p36, %r229, 0f00000000;
selp.f32 %r231, 0f3F800000, 0f00000000, %p36;
selp.f32 %r232, 0f3F800000, 0f00000000, %p35;
mul.f32 %r233, %r227, %r232;
mul.f32 %r234, %r228, %r231;
mul.f32 %r235, %r234, %r61;
mul.f32 %r236, %r233, %r61;
neg.f32 %r237, %r236;
fma.rn.f32 %r238, %r237, %r39, %r227;
neg.f32 %r239, %r235;
fma.rn.f32 %r240, %r239, %r40, %r228;
neg.f32 %r241, %r40;
fma.rn.f32 %r242, %r241, %r220, %r240;
neg.f32 %r243, %r39;
fma.rn.f32 %r244, %r243, %r219, %r238;
prmt.b32 %r63, %r244, %r242, 0x7632U;
mov.b32 {%rs17, %rs18}, %r3;
cvt.f32.bf16 %r245, %rs18;
cvt.f32.bf16 %r246, %rs17;
mul.f32 %r247, %r130, %r178;
mul.f32 %r248, %r129, %r177;
mul.f32 %r249, %r248, %r246;
mul.f32 %r250, %r247, %r245;
cvt.rn.bf16.f32 %rs19, %r250;
cvt.rn.bf16.f32 %rs20, %r249;
cvt.f32.bf16 %r251, %rs20;
cvt.f32.bf16 %r252, %rs19;
mov.b32 {%rs21, %rs22}, %r47;
cvt.u32.u16 %r253, %rs21;
cvt.u32.u16 %r254, %rs22;
shl.b32 %r255, %r254, 16;
shl.b32 %r256, %r253, 16;
mov.b32 {%rs23, %rs24}, %r51;
cvt.u32.u16 %r257, %rs24;
cvt.u32.u16 %r258, %rs23;
or.b32 %r259, %r256, %r258;
or.b32 %r260, %r255, %r257;
mul.f32 %r261, %r58, %r260;
mul.f32 %r262, %r57, %r259;
setp.ge.f32 %p37, %r262, 0f00000000;
setp.ge.f32 %p38, %r261, 0f00000000;
selp.f32 %r263, 0f3F800000, 0f00000000, %p38;
selp.f32 %r264, 0f3F800000, 0f00000000, %p37;
mul.f32 %r265, %r259, %r264;
mul.f32 %r266, %r260, %r263;
mul.f32 %r267, %r266, %r61;
mul.f32 %r268, %r265, %r61;
neg.f32 %r269, %r268;
fma.rn.f32 %r270, %r269, %r41, %r259;
neg.f32 %r271, %r267;
fma.rn.f32 %r272, %r271, %r42, %r260;
neg.f32 %r273, %r42;
fma.rn.f32 %r274, %r273, %r252, %r272;
neg.f32 %r275, %r41;
fma.rn.f32 %r276, %r275, %r251, %r270;
prmt.b32 %r64, %r276, %r274, 0x7632U;
mov.b32 {%rs25, %rs26}, %r4;
cvt.f32.bf16 %r277, %rs26;
cvt.f32.bf16 %r278, %rs25;
mul.f32 %r279, %r132, %r180;
mul.f32 %r280, %r131, %r179;
mul.f32 %r281, %r280, %r278;
mul.f32 %r282, %r279, %r277;
cvt.rn.bf16.f32 %rs27, %r282;
cvt.rn.bf16.f32 %rs28, %r281;
cvt.f32.bf16 %r283, %rs28;
cvt.f32.bf16 %r284, %rs27;
mov.b32 {%rs29, %rs30}, %r48;
cvt.u32.u16 %r285, %rs29;
cvt.u32.u16 %r286, %rs30;
shl.b32 %r287, %r286, 16;
shl.b32 %r288, %r285, 16;
mov.b32 {%rs31, %rs32}, %r52;
cvt.u32.u16 %r289, %rs32;
cvt.u32.u16 %r290, %rs31;
or.b32 %r291, %r288, %r290;
or.b32 %r292, %r287, %r289;
mul.f32 %r293, %r60, %r292;
mul.f32 %r294, %r59, %r291;
setp.ge.f32 %p39, %r294, 0f00000000;
setp.ge.f32 %p40, %r293, 0f00000000;
selp.f32 %r295, 0f3F800000, 0f00000000, %p40;
selp.f32 %r296, 0f3F800000, 0f00000000, %p39;
mul.f32 %r297, %r291, %r296;
mul.f32 %r298, %r292, %r295;
mul.f32 %r299, %r298, %r61;
mul.f32 %r300, %r297, %r61;
neg.f32 %r301, %r300;
fma.rn.f32 %r302, %r301, %r43, %r291;
neg.f32 %r303, %r299;
fma.rn.f32 %r304, %r303, %r44, %r292;
neg.f32 %r305, %r44;
fma.rn.f32 %r306, %r305, %r284, %r304;
neg.f32 %r307, %r43;
fma.rn.f32 %r308, %r307, %r283, %r302;
prmt.b32 %r65, %r308, %r306, 0x7632U;
add.s64 %rd52, %rd64, %rd63;
st.global.v4.b32 [ %rd52 + 0 ], { %r62, %r63, %r64, %r65 };
add.s64 %rd53, %rd65, %rd63;
mov.b32 {%rs33, _}, %r210;
mov.b32 {%rs34, _}, %r212;
mov.b32 %r66, {%rs34, %rs33};
mov.b32 {%rs35, _}, %r242;
mov.b32 {%rs36, _}, %r244;
mov.b32 %r67, {%rs36, %rs35};
mov.b32 {%rs37, _}, %r274;
mov.b32 {%rs38, _}, %r276;
mov.b32 %r68, {%rs38, %rs37};
mov.b32 {%rs39, _}, %r306;
mov.b32 {%rs40, _}, %r308;
mov.b32 %r69, {%rs40, %rs39};
st.global.v4.b32 [ %rd53 + 0 ], { %r66, %r67, %r68, %r69 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .vo, .v512, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_8
)
.reqntid 256
{
.reg .pred %p<9>;
.reg .b16 %rs<3>;
.reg .b32 %r<56>;
.reg .b64 %rd<24>;
$L__func_begin0:
ld.param.b64 %rd15, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0];
ld.param.b64 %rd16, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd17, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2];
ld.param.b64 %rd18, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd19, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd20, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5];
or.b32 %r16, %r15, %r12;
mul.hi.s32 %r17, %r16, 715827883;
shr.u32 %r18, %r17, 31;
shr.s32 %r19, %r17, 7;
add.s32 %r20, %r19, %r18;
mul.hi.s32 %r21, %r16, 954437177;
shr.u32 %r22, %r21, 31;
shr.s32 %r23, %r21, 17;
add.s32 %r24, %r23, %r22;
mul.wide.s32 %rd21, %r16, 2;
add.s64 %rd1, %rd15, %rd21;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mul.wide.s32 %rd22, %r20, 4;
add.s64 %rd2, %rd16, %rd22;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd2 + 0 ], %rd4;
add.s64 %rd5, %rd17, %rd22;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd5 + 0 ], %rd7;
mul.wide.s32 %rd23, %r24, 4;
add.s64 %rd8, %rd18, %rd23;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd8 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd8 + 0 ], %rd10;
add.s64 %rd11, %rd19, %rd23;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd11 + 0 ], %rd13;
mov.b32 %r25, 0f44400000;
div.full.f32 %r26, %r2, %r25;
div.full.f32 %r27, %r3, %r25;
sub.f32 %r28, %r26, %r4;
sub.f32 %r29, %r27, %r5;
fma.rn.f32 %r30, %r28, 0f3DCCCCCD, %r4;
fma.rn.f32 %r31, %r29, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r30, 0f2EDBE6FF;
setp.gt.f32 %p2, %r31, 0f2EDBE6FF;
setp.nan.f32 %p3, %r30, %r30;
setp.nan.f32 %p4, %r31, %r31;
selp.f32 %r32, %r30, 0f2EDBE6FF, %p3;
selp.f32 %r33, %r30, %r32, %p1;
selp.f32 %r34, %r31, 0f2EDBE6FF, %p4;
selp.f32 %r35, %r31, %r34, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r36, %r33;
rsqrt.approx.ftz.f32 %r37, %r35;
mul.f32 %r38, %r6, 0f44400000;
mul.f32 %r39, %r7, 0f44400000;
sqrt.rn.f32 %r40, %r38;
sqrt.rn.f32 %r41, %r39;
sqrt.rn.f32 %r42, %r8;
sqrt.rn.f32 %r43, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r42, 0f2EDBE6FF;
setp.gt.f32 %p6, %r43, 0f2EDBE6FF;
setp.nan.f32 %p7, %r42, %r42;
setp.nan.f32 %p8, %r43, %r43;
selp.f32 %r44, %r42, 0f2EDBE6FF, %p7;
selp.f32 %r45, %r42, %r44, %p5;
selp.f32 %r46, %r43, 0f2EDBE6FF, %p8;
selp.f32 %r47, %r43, %r46, %p6;
$L__tmp4:
div.full.f32 %r48, %r40, %r45;
div.full.f32 %r49, %r41, %r47;
add.s64 %rd14, %rd20, %rd21;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r50, %rs2;
cvt.f32.bf16 %r51, %rs1;
mul.f32 %r52, %r37, %r49;
mul.f32 %r53, %r36, %r48;
mul.f32 %r54, %r53, %r51;
mul.f32 %r55, %r52, %r50;
cvt.rn.bf16x2.f32 %r10, %r55, %r54;
st.global.b32 [ %rd14 + 0 ], { %r10 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .vo, .v512, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 256
{
.reg .pred %p<3>;
.reg .b16 %rs<9>;
.reg .b32 %r<47>;
.reg .b64 %rd<19>;
$L__func_begin0:
ld.param.b64 %rd11, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd12, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd13, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd4, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd14, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd15, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r16, %r15, %r12;
ld.param.b64 %rd16, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r17, %r16, 954437177;
shr.u32 %r18, %r17, 31;
shr.s32 %r19, %r17, 17;
add.s32 %r20, %r19, %r18;
ld.param.b64 %rd17, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd18, %r16, 2;
add.s64 %rd1, %rd11, %rd18;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd12, %rd18;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r16, 4, %rd13;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v2.b32 { %r3, %r4 }, [ %rd3 + 0 ];
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd4 + 0 ];
mad.wide.s32 %rd5, %r20, 4, %rd14;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd5 + 0 ], %rd7;
add.s64 %rd8, %rd15, %rd18;
mov.u32 %r8, 0x0;
ld.global.b32 { %r8 }, [ %rd8 + 0 ];
mov.b32 {%rs1, %rs2}, %r8;
cvt.f32.bf16 %r21, %rs1;
cvt.f32.bf16 %r22, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r23, %rs3;
cvt.u32.u16 %r24, %rs4;
shl.b32 %r25, %r24, 16;
shl.b32 %r26, %r23, 16;
mov.b32 {%rs5, %rs6}, %r2;
cvt.u32.u16 %r27, %rs6;
cvt.u32.u16 %r28, %rs5;
or.b32 %r29, %r26, %r28;
or.b32 %r30, %r25, %r27;
mul.f32 %r31, %r4, %r30;
mul.f32 %r32, %r3, %r29;
setp.ge.f32 %p1, %r32, 0f00000000;
setp.ge.f32 %p2, %r31, 0f00000000;
selp.f32 %r33, 0f3F800000, 0f00000000, %p2;
selp.f32 %r34, 0f3F800000, 0f00000000, %p1;
mul.f32 %r35, %r29, %r34;
mul.f32 %r36, %r30, %r33;
mul.f32 %r37, %r36, %r5;
mul.f32 %r38, %r35, %r5;
neg.f32 %r39, %r38;
fma.rn.f32 %r40, %r39, %r6, %r29;
neg.f32 %r41, %r37;
fma.rn.f32 %r42, %r41, %r7, %r30;
neg.f32 %r43, %r7;
fma.rn.f32 %r44, %r43, %r22, %r42;
neg.f32 %r45, %r6;
fma.rn.f32 %r46, %r45, %r21, %r40;
prmt.b32 %r9, %r46, %r44, 0x7632U;
add.s64 %rd9, %rd16, %rd18;
st.global.b32 [ %rd9 + 0 ], { %r9 };
add.s64 %rd10, %rd17, %rd18;
mov.b32 {%rs7, _}, %r44;
mov.b32 {%rs8, _}, %r46;
mov.b32 %r10, {%rs8, %rs7};
st.global.b32 [ %rd10 + 0 ], { %r10 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .vo, .v512, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_14
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<11>;
.reg .b32 %r<90>;
.reg .b64 %rd<37>;
$L__func_begin0:
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0];
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1];
$L__tmp0:
mov.u32 %r19, %ctaid.x;
shl.b32 %r20, %r19, 9;
ld.param.b64 %rd25, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2];
ld.param.b64 %rd26, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3];
mov.u32 %r21, %tid.x;
shl.b32 %r22, %r21, 1;
ld.param.b64 %rd27, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4];
and.b32 %r23, %r22, 510;
ld.param.b64 %rd28, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5];
or.b32 %r24, %r23, %r20;
ld.param.b64 %rd29, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6];
mul.hi.s32 %r25, %r24, 715827883;
shr.u32 %r26, %r25, 31;
shr.s32 %r27, %r25, 7;
add.s32 %r28, %r27, %r26;
ld.param.b64 %rd30, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7];
mul.hi.s32 %r29, %r24, 954437177;
shr.u32 %r30, %r29, 31;
shr.s32 %r31, %r29, 17;
add.s32 %r32, %r31, %r30;
ld.param.b64 %rd31, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8];
ld.param.b64 %rd20, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9];
mul.wide.s32 %rd32, %r24, 2;
add.s64 %rd1, %rd23, %rd32;
ld.param.b64 %rd33, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10];
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
ld.param.b64 %rd34, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11];
mul.wide.s32 %rd35, %r28, 4;
add.s64 %rd2, %rd24, %rd35;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd2 + 0 ], %rd4;
add.s64 %rd5, %rd25, %rd35;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd5 + 0 ], %rd7;
mul.wide.s32 %rd36, %r32, 4;
add.s64 %rd8, %rd26, %rd36;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd8 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd8 + 0 ], %rd10;
add.s64 %rd11, %rd27, %rd36;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd11 + 0 ], %rd13;
add.s64 %rd14, %rd28, %rd36;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd14 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd14 + 0 ], %rd16;
add.s64 %rd17, %rd29, %rd32;
mov.u32 %r12, 0x0;
ld.global.b32 { %r12 }, [ %rd17 + 0 ];
add.s64 %rd18, %rd30, %rd32;
mov.u32 %r13, 0x0;
ld.global.b32 { %r13 }, [ %rd18 + 0 ];
mad.wide.s32 %rd19, %r24, 4, %rd31;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
ld.global.v2.b32 { %r14, %r15 }, [ %rd19 + 0 ];
mov.u32 %r16, 0x0;
ld.global.b32 { %r16 }, [ %rd20 + 0 ];
mov.b32 %r33, 0f44400000;
div.full.f32 %r34, %r2, %r33;
div.full.f32 %r35, %r3, %r33;
sub.f32 %r36, %r34, %r4;
sub.f32 %r37, %r35, %r5;
fma.rn.f32 %r38, %r36, 0f3DCCCCCD, %r4;
fma.rn.f32 %r39, %r37, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r38, 0f2EDBE6FF;
setp.gt.f32 %p2, %r39, 0f2EDBE6FF;
setp.nan.f32 %p3, %r38, %r38;
setp.nan.f32 %p4, %r39, %r39;
selp.f32 %r40, %r38, 0f2EDBE6FF, %p3;
selp.f32 %r41, %r38, %r40, %p1;
selp.f32 %r42, %r39, 0f2EDBE6FF, %p4;
selp.f32 %r43, %r39, %r42, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r44, %r41;
rsqrt.approx.ftz.f32 %r45, %r43;
mul.f32 %r46, %r6, 0f44400000;
mul.f32 %r47, %r7, 0f44400000;
sqrt.rn.f32 %r48, %r46;
sqrt.rn.f32 %r49, %r47;
sqrt.rn.f32 %r50, %r8;
sqrt.rn.f32 %r51, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r50, 0f2EDBE6FF;
setp.gt.f32 %p6, %r51, 0f2EDBE6FF;
setp.nan.f32 %p7, %r50, %r50;
setp.nan.f32 %p8, %r51, %r51;
selp.f32 %r52, %r50, 0f2EDBE6FF, %p7;
selp.f32 %r53, %r50, %r52, %p5;
selp.f32 %r54, %r51, 0f2EDBE6FF, %p8;
selp.f32 %r55, %r51, %r54, %p6;
$L__tmp4:
div.full.f32 %r56, %r48, %r53;
div.full.f32 %r57, %r49, %r55;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r58, %rs2;
cvt.f32.bf16 %r59, %rs1;
mul.f32 %r60, %r45, %r57;
mul.f32 %r61, %r44, %r56;
mul.f32 %r62, %r61, %r59;
mul.f32 %r63, %r60, %r58;
cvt.rn.bf16.f32 %rs3, %r63;
cvt.rn.bf16.f32 %rs4, %r62;
cvt.f32.bf16 %r64, %rs4;
cvt.f32.bf16 %r65, %rs3;
mov.b32 {%rs5, %rs6}, %r12;
cvt.u32.u16 %r66, %rs5;
cvt.u32.u16 %r67, %rs6;
shl.b32 %r68, %r67, 16;
shl.b32 %r69, %r66, 16;
mov.b32 {%rs7, %rs8}, %r13;
cvt.u32.u16 %r70, %rs8;
cvt.u32.u16 %r71, %rs7;
or.b32 %r72, %r69, %r71;
or.b32 %r73, %r68, %r70;
mul.f32 %r74, %r15, %r73;
mul.f32 %r75, %r14, %r72;
setp.ge.f32 %p9, %r75, 0f00000000;
setp.ge.f32 %p10, %r74, 0f00000000;
selp.f32 %r76, 0f3F800000, 0f00000000, %p10;
selp.f32 %r77, 0f3F800000, 0f00000000, %p9;
mul.f32 %r78, %r72, %r77;
mul.f32 %r79, %r73, %r76;
mul.f32 %r80, %r79, %r16;
mul.f32 %r81, %r78, %r16;
neg.f32 %r82, %r81;
fma.rn.f32 %r83, %r82, %r10, %r72;
neg.f32 %r84, %r80;
fma.rn.f32 %r85, %r84, %r11, %r73;
neg.f32 %r86, %r11;
fma.rn.f32 %r87, %r86, %r65, %r85;
neg.f32 %r88, %r10;
fma.rn.f32 %r89, %r88, %r64, %r83;
prmt.b32 %r17, %r89, %r87, 0x7632U;
add.s64 %rd21, %rd33, %rd32;
st.global.b32 [ %rd21 + 0 ], { %r17 };
add.s64 %rd22, %rd34, %rd32;
mov.b32 {%rs9, _}, %r87;
mov.b32 {%rs10, _}, %r89;
mov.b32 %r18, {%rs10, %rs9};
st.global.b32 [ %rd22 + 0 ], { %r18 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .mlp, .v1024, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_8
)
.reqntid 128
{
.reg .pred %p<33>;
.reg .b16 %rs<9>;
.reg .b32 %r<176>;
.reg .b64 %rd<48>;
$L__func_begin0:
ld.param.b64 %rd39, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0];
ld.param.b64 %rd40, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1];
$L__tmp0:
mov.u32 %r41, %ctaid.x;
shl.b32 %r42, %r41, 10;
ld.param.b64 %rd41, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2];
ld.param.b64 %rd42, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3];
mov.u32 %r43, %tid.x;
shl.b32 %r44, %r43, 3;
ld.param.b64 %rd43, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4];
and.b32 %r45, %r44, 1016;
ld.param.b64 %rd44, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5];
or.b32 %r46, %r45, %r42;
mul.hi.s32 %r47, %r46, 715827883;
shr.u32 %r48, %r47, 31;
shr.s32 %r49, %r47, 7;
add.s32 %r50, %r49, %r48;
mul.hi.s32 %r51, %r46, 1041204193;
shr.u32 %r52, %r51, 31;
shr.s32 %r53, %r51, 19;
add.s32 %r54, %r53, %r52;
mul.wide.s32 %rd45, %r46, 2;
add.s64 %rd1, %rd39, %rd45;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
mul.wide.s32 %rd46, %r50, 4;
add.s64 %rd2, %rd40, %rd46;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd2 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd2 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd2 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd2 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd2 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd2 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd2 + 0 ], %rd10;
add.s64 %rd11, %rd41, %rd46;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r13, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r14, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd11 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r15, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd11 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd11 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r17, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd11 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd11 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd11 + 0 ], %rd18;
mov.u64 %rd19, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd19, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd11 + 0 ], %rd19;
mul.wide.s32 %rd47, %r54, 4;
add.s64 %rd20, %rd42, %rd47;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd20 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd20 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd20 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd20 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd20 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd20 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd20 + 0 ], %rd27;
mov.u64 %rd28, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd28, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd20 + 0 ], %rd28;
add.s64 %rd29, %rd43, %rd47;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd29 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd29 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd29 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd29 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd29 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd29 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd29 + 0 ], %rd36;
mov.u64 %rd37, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd37, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd29 + 0 ], %rd37;
mov.b32 %r55, 0f44400000;
div.full.f32 %r56, %r5, %r55;
div.full.f32 %r57, %r6, %r55;
div.full.f32 %r58, %r7, %r55;
div.full.f32 %r59, %r8, %r55;
div.full.f32 %r60, %r9, %r55;
div.full.f32 %r61, %r10, %r55;
div.full.f32 %r62, %r11, %r55;
div.full.f32 %r63, %r12, %r55;
sub.f32 %r64, %r63, %r20;
sub.f32 %r65, %r62, %r19;
sub.f32 %r66, %r61, %r18;
sub.f32 %r67, %r60, %r17;
sub.f32 %r68, %r59, %r16;
sub.f32 %r69, %r58, %r15;
sub.f32 %r70, %r57, %r14;
sub.f32 %r71, %r56, %r13;
fma.rn.f32 %r72, %r71, 0f3DCCCCCD, %r13;
fma.rn.f32 %r73, %r70, 0f3DCCCCCD, %r14;
fma.rn.f32 %r74, %r69, 0f3DCCCCCD, %r15;
fma.rn.f32 %r75, %r68, 0f3DCCCCCD, %r16;
fma.rn.f32 %r76, %r67, 0f3DCCCCCD, %r17;
fma.rn.f32 %r77, %r66, 0f3DCCCCCD, %r18;
fma.rn.f32 %r78, %r65, 0f3DCCCCCD, %r19;
fma.rn.f32 %r79, %r64, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r79, 0f2EDBE6FF;
setp.gt.f32 %p2, %r78, 0f2EDBE6FF;
setp.gt.f32 %p3, %r77, 0f2EDBE6FF;
setp.gt.f32 %p4, %r76, 0f2EDBE6FF;
setp.gt.f32 %p5, %r75, 0f2EDBE6FF;
setp.gt.f32 %p6, %r74, 0f2EDBE6FF;
setp.gt.f32 %p7, %r73, 0f2EDBE6FF;
setp.gt.f32 %p8, %r72, 0f2EDBE6FF;
setp.nan.f32 %p9, %r72, %r72;
setp.nan.f32 %p10, %r73, %r73;
setp.nan.f32 %p11, %r74, %r74;
setp.nan.f32 %p12, %r75, %r75;
setp.nan.f32 %p13, %r76, %r76;
setp.nan.f32 %p14, %r77, %r77;
setp.nan.f32 %p15, %r78, %r78;
setp.nan.f32 %p16, %r79, %r79;
selp.f32 %r80, %r72, 0f2EDBE6FF, %p9;
selp.f32 %r81, %r72, %r80, %p8;
selp.f32 %r82, %r73, 0f2EDBE6FF, %p10;
selp.f32 %r83, %r73, %r82, %p7;
selp.f32 %r84, %r74, 0f2EDBE6FF, %p11;
selp.f32 %r85, %r74, %r84, %p6;
selp.f32 %r86, %r75, 0f2EDBE6FF, %p12;
selp.f32 %r87, %r75, %r86, %p5;
selp.f32 %r88, %r76, 0f2EDBE6FF, %p13;
selp.f32 %r89, %r76, %r88, %p4;
selp.f32 %r90, %r77, 0f2EDBE6FF, %p14;
selp.f32 %r91, %r77, %r90, %p3;
selp.f32 %r92, %r78, 0f2EDBE6FF, %p15;
selp.f32 %r93, %r78, %r92, %p2;
selp.f32 %r94, %r79, 0f2EDBE6FF, %p16;
selp.f32 %r95, %r79, %r94, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r96, %r81;
rsqrt.approx.ftz.f32 %r97, %r83;
rsqrt.approx.ftz.f32 %r98, %r85;
rsqrt.approx.ftz.f32 %r99, %r87;
rsqrt.approx.ftz.f32 %r100, %r89;
rsqrt.approx.ftz.f32 %r101, %r91;
rsqrt.approx.ftz.f32 %r102, %r93;
rsqrt.approx.ftz.f32 %r103, %r95;
mul.f32 %r104, %r21, 0f44400000;
mul.f32 %r105, %r22, 0f44400000;
mul.f32 %r106, %r23, 0f44400000;
mul.f32 %r107, %r24, 0f44400000;
mul.f32 %r108, %r25, 0f44400000;
mul.f32 %r109, %r26, 0f44400000;
mul.f32 %r110, %r27, 0f44400000;
mul.f32 %r111, %r28, 0f44400000;
sqrt.rn.f32 %r112, %r104;
sqrt.rn.f32 %r113, %r105;
sqrt.rn.f32 %r114, %r106;
sqrt.rn.f32 %r115, %r107;
sqrt.rn.f32 %r116, %r108;
sqrt.rn.f32 %r117, %r109;
sqrt.rn.f32 %r118, %r110;
sqrt.rn.f32 %r119, %r111;
sqrt.rn.f32 %r120, %r29;
sqrt.rn.f32 %r121, %r30;
sqrt.rn.f32 %r122, %r31;
sqrt.rn.f32 %r123, %r32;
sqrt.rn.f32 %r124, %r33;
sqrt.rn.f32 %r125, %r34;
sqrt.rn.f32 %r126, %r35;
sqrt.rn.f32 %r127, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r120, 0f2EDBE6FF;
setp.gt.f32 %p18, %r121, 0f2EDBE6FF;
setp.gt.f32 %p19, %r122, 0f2EDBE6FF;
setp.gt.f32 %p20, %r123, 0f2EDBE6FF;
setp.gt.f32 %p21, %r124, 0f2EDBE6FF;
setp.gt.f32 %p22, %r125, 0f2EDBE6FF;
setp.gt.f32 %p23, %r126, 0f2EDBE6FF;
setp.gt.f32 %p24, %r127, 0f2EDBE6FF;
setp.nan.f32 %p25, %r120, %r120;
setp.nan.f32 %p26, %r121, %r121;
setp.nan.f32 %p27, %r122, %r122;
setp.nan.f32 %p28, %r123, %r123;
setp.nan.f32 %p29, %r124, %r124;
setp.nan.f32 %p30, %r125, %r125;
setp.nan.f32 %p31, %r126, %r126;
setp.nan.f32 %p32, %r127, %r127;
selp.f32 %r128, %r120, 0f2EDBE6FF, %p25;
selp.f32 %r129, %r120, %r128, %p17;
selp.f32 %r130, %r121, 0f2EDBE6FF, %p26;
selp.f32 %r131, %r121, %r130, %p18;
selp.f32 %r132, %r122, 0f2EDBE6FF, %p27;
selp.f32 %r133, %r122, %r132, %p19;
selp.f32 %r134, %r123, 0f2EDBE6FF, %p28;
selp.f32 %r135, %r123, %r134, %p20;
selp.f32 %r136, %r124, 0f2EDBE6FF, %p29;
selp.f32 %r137, %r124, %r136, %p21;
selp.f32 %r138, %r125, 0f2EDBE6FF, %p30;
selp.f32 %r139, %r125, %r138, %p22;
selp.f32 %r140, %r126, 0f2EDBE6FF, %p31;
selp.f32 %r141, %r126, %r140, %p23;
selp.f32 %r142, %r127, 0f2EDBE6FF, %p32;
selp.f32 %r143, %r127, %r142, %p24;
$L__tmp4:
div.full.f32 %r144, %r112, %r129;
div.full.f32 %r145, %r113, %r131;
div.full.f32 %r146, %r114, %r133;
div.full.f32 %r147, %r115, %r135;
div.full.f32 %r148, %r116, %r137;
div.full.f32 %r149, %r117, %r139;
div.full.f32 %r150, %r118, %r141;
div.full.f32 %r151, %r119, %r143;
add.s64 %rd38, %rd44, %rd45;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r152, %rs2;
cvt.f32.bf16 %r153, %rs1;
mul.f32 %r154, %r97, %r145;
mul.f32 %r155, %r96, %r144;
mul.f32 %r156, %r155, %r153;
mul.f32 %r157, %r154, %r152;
cvt.rn.bf16x2.f32 %r37, %r157, %r156;
mov.b32 {%rs3, %rs4}, %r2;
cvt.f32.bf16 %r158, %rs4;
cvt.f32.bf16 %r159, %rs3;
mul.f32 %r160, %r99, %r147;
mul.f32 %r161, %r98, %r146;
mul.f32 %r162, %r161, %r159;
mul.f32 %r163, %r160, %r158;
cvt.rn.bf16x2.f32 %r38, %r163, %r162;
mov.b32 {%rs5, %rs6}, %r3;
cvt.f32.bf16 %r164, %rs6;
cvt.f32.bf16 %r165, %rs5;
mul.f32 %r166, %r101, %r149;
mul.f32 %r167, %r100, %r148;
mul.f32 %r168, %r167, %r165;
mul.f32 %r169, %r166, %r164;
cvt.rn.bf16x2.f32 %r39, %r169, %r168;
mov.b32 {%rs7, %rs8}, %r4;
cvt.f32.bf16 %r170, %rs8;
cvt.f32.bf16 %r171, %rs7;
mul.f32 %r172, %r103, %r151;
mul.f32 %r173, %r102, %r150;
mul.f32 %r174, %r173, %r171;
mul.f32 %r175, %r172, %r170;
cvt.rn.bf16x2.f32 %r40, %r175, %r174;
st.global.v4.b32 [ %rd38 + 0 ], { %r37, %r38, %r39, %r40 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .mlp, .v1024, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 128
{
.reg .pred %p<9>;
.reg .b16 %rs<33>;
.reg .b32 %r<152>;
.reg .b64 %rd<26>;
$L__func_begin0:
ld.param.b64 %rd18, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd19, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r38, %ctaid.x;
shl.b32 %r39, %r38, 10;
ld.param.b64 %rd20, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd5, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r40, %tid.x;
shl.b32 %r41, %r40, 3;
ld.param.b64 %rd21, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r42, %r41, 1016;
ld.param.b64 %rd22, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r43, %r42, %r39;
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r44, %r43, 1041204193;
shr.u32 %r45, %r44, 31;
shr.s32 %r46, %r44, 19;
add.s32 %r47, %r46, %r45;
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd25, %r43, 2;
add.s64 %rd1, %rd18, %rd25;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd19, %rd25;
mov.u32 %r5, 0x0;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r43, 4, %rd20;
add.s64 %rd4, %rd3, 16;
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
mov.u32 %r17, 0x0;
ld.global.b32 { %r17 }, [ %rd5 + 0 ];
mad.wide.s32 %rd6, %r47, 4, %rd21;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd6 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd6 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd6 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd6 + 0 ], %rd10;
mov.u64 %rd11, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd11, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd6 + 0 ], %rd11;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd6 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd6 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd6 + 0 ], %rd14;
add.s64 %rd15, %rd22, %rd25;
mov.u32 %r26, 0x0;
mov.u32 %r27, 0x0;
mov.u32 %r28, 0x0;
mov.u32 %r29, 0x0;
ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd15 + 0 ];
mov.b32 {%rs1, %rs2}, %r26;
cvt.f32.bf16 %r48, %rs1;
cvt.f32.bf16 %r49, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r50, %rs3;
cvt.u32.u16 %r51, %rs4;
shl.b32 %r52, %r51, 16;
shl.b32 %r53, %r50, 16;
mov.b32 {%rs5, %rs6}, %r5;
cvt.u32.u16 %r54, %rs6;
cvt.u32.u16 %r55, %rs5;
or.b32 %r56, %r53, %r55;
or.b32 %r57, %r52, %r54;
mul.f32 %r58, %r10, %r57;
mul.f32 %r59, %r9, %r56;
setp.ge.f32 %p1, %r59, 0f00000000;
setp.ge.f32 %p2, %r58, 0f00000000;
selp.f32 %r60, 0f3F800000, 0f00000000, %p2;
selp.f32 %r61, 0f3F800000, 0f00000000, %p1;
mul.f32 %r62, %r56, %r61;
mul.f32 %r63, %r57, %r60;
mul.f32 %r64, %r63, %r17;
mul.f32 %r65, %r62, %r17;
neg.f32 %r66, %r65;
fma.rn.f32 %r67, %r66, %r18, %r56;
neg.f32 %r68, %r64;
fma.rn.f32 %r69, %r68, %r19, %r57;
neg.f32 %r70, %r19;
fma.rn.f32 %r71, %r70, %r49, %r69;
neg.f32 %r72, %r18;
fma.rn.f32 %r73, %r72, %r48, %r67;
prmt.b32 %r30, %r73, %r71, 0x7632U;
mov.b32 {%rs7, %rs8}, %r27;
cvt.f32.bf16 %r74, %rs7;
cvt.f32.bf16 %r75, %rs8;
mov.b32 {%rs9, %rs10}, %r2;
cvt.u32.u16 %r76, %rs9;
cvt.u32.u16 %r77, %rs10;
shl.b32 %r78, %r77, 16;
shl.b32 %r79, %r76, 16;
mov.b32 {%rs11, %rs12}, %r6;
cvt.u32.u16 %r80, %rs12;
cvt.u32.u16 %r81, %rs11;
or.b32 %r82, %r79, %r81;
or.b32 %r83, %r78, %r80;
mul.f32 %r84, %r12, %r83;
mul.f32 %r85, %r11, %r82;
setp.ge.f32 %p3, %r85, 0f00000000;
setp.ge.f32 %p4, %r84, 0f00000000;
selp.f32 %r86, 0f3F800000, 0f00000000, %p4;
selp.f32 %r87, 0f3F800000, 0f00000000, %p3;
mul.f32 %r88, %r82, %r87;
mul.f32 %r89, %r83, %r86;
mul.f32 %r90, %r89, %r17;
mul.f32 %r91, %r88, %r17;
neg.f32 %r92, %r91;
fma.rn.f32 %r93, %r92, %r20, %r82;
neg.f32 %r94, %r90;
fma.rn.f32 %r95, %r94, %r21, %r83;
neg.f32 %r96, %r21;
fma.rn.f32 %r97, %r96, %r75, %r95;
neg.f32 %r98, %r20;
fma.rn.f32 %r99, %r98, %r74, %r93;
prmt.b32 %r31, %r99, %r97, 0x7632U;
mov.b32 {%rs13, %rs14}, %r28;
cvt.f32.bf16 %r100, %rs13;
cvt.f32.bf16 %r101, %rs14;
mov.b32 {%rs15, %rs16}, %r3;
cvt.u32.u16 %r102, %rs15;
cvt.u32.u16 %r103, %rs16;
shl.b32 %r104, %r103, 16;
shl.b32 %r105, %r102, 16;
mov.b32 {%rs17, %rs18}, %r7;
cvt.u32.u16 %r106, %rs18;
cvt.u32.u16 %r107, %rs17;
or.b32 %r108, %r105, %r107;
or.b32 %r109, %r104, %r106;
mul.f32 %r110, %r14, %r109;
mul.f32 %r111, %r13, %r108;
setp.ge.f32 %p5, %r111, 0f00000000;
setp.ge.f32 %p6, %r110, 0f00000000;
selp.f32 %r112, 0f3F800000, 0f00000000, %p6;
selp.f32 %r113, 0f3F800000, 0f00000000, %p5;
mul.f32 %r114, %r108, %r113;
mul.f32 %r115, %r109, %r112;
mul.f32 %r116, %r115, %r17;
mul.f32 %r117, %r114, %r17;
neg.f32 %r118, %r117;
fma.rn.f32 %r119, %r118, %r22, %r108;
neg.f32 %r120, %r116;
fma.rn.f32 %r121, %r120, %r23, %r109;
neg.f32 %r122, %r23;
fma.rn.f32 %r123, %r122, %r101, %r121;
neg.f32 %r124, %r22;
fma.rn.f32 %r125, %r124, %r100, %r119;
prmt.b32 %r32, %r125, %r123, 0x7632U;
mov.b32 {%rs19, %rs20}, %r29;
cvt.f32.bf16 %r126, %rs19;
cvt.f32.bf16 %r127, %rs20;
mov.b32 {%rs21, %rs22}, %r4;
cvt.u32.u16 %r128, %rs21;
cvt.u32.u16 %r129, %rs22;
shl.b32 %r130, %r129, 16;
shl.b32 %r131, %r128, 16;
mov.b32 {%rs23, %rs24}, %r8;
cvt.u32.u16 %r132, %rs24;
cvt.u32.u16 %r133, %rs23;
or.b32 %r134, %r131, %r133;
or.b32 %r135, %r130, %r132;
mul.f32 %r136, %r16, %r135;
mul.f32 %r137, %r15, %r134;
setp.ge.f32 %p7, %r137, 0f00000000;
setp.ge.f32 %p8, %r136, 0f00000000;
selp.f32 %r138, 0f3F800000, 0f00000000, %p8;
selp.f32 %r139, 0f3F800000, 0f00000000, %p7;
mul.f32 %r140, %r134, %r139;
mul.f32 %r141, %r135, %r138;
mul.f32 %r142, %r141, %r17;
mul.f32 %r143, %r140, %r17;
neg.f32 %r144, %r143;
fma.rn.f32 %r145, %r144, %r24, %r134;
neg.f32 %r146, %r142;
fma.rn.f32 %r147, %r146, %r25, %r135;
neg.f32 %r148, %r25;
fma.rn.f32 %r149, %r148, %r127, %r147;
neg.f32 %r150, %r24;
fma.rn.f32 %r151, %r150, %r126, %r145;
prmt.b32 %r33, %r151, %r149, 0x7632U;
add.s64 %rd16, %rd23, %rd25;
st.global.v4.b32 [ %rd16 + 0 ], { %r30, %r31, %r32, %r33 };
add.s64 %rd17, %rd24, %rd25;
mov.b32 {%rs25, _}, %r71;
mov.b32 {%rs26, _}, %r73;
mov.b32 %r34, {%rs26, %rs25};
mov.b32 {%rs27, _}, %r97;
mov.b32 {%rs28, _}, %r99;
mov.b32 %r35, {%rs28, %rs27};
mov.b32 {%rs29, _}, %r123;
mov.b32 {%rs30, _}, %r125;
mov.b32 %r36, {%rs30, %rs29};
mov.b32 {%rs31, _}, %r149;
mov.b32 {%rs32, _}, %r151;
mov.b32 %r37, {%rs32, %rs31};
st.global.v4.b32 [ %rd17 + 0 ], { %r34, %r35, %r36, %r37 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .mlp, .v1024, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_14
)
.reqntid 128
{
.reg .pred %p<41>;
.reg .b16 %rs<41>;
.reg .b32 %r<309>;
.reg .b64 %rd<68>;
$L__func_begin0:
ld.param.b64 %rd54, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0];
ld.param.b64 %rd55, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1];
$L__tmp0:
mov.u32 %r70, %ctaid.x;
shl.b32 %r71, %r70, 10;
ld.param.b64 %rd56, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2];
ld.param.b64 %rd57, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3];
mov.u32 %r72, %tid.x;
shl.b32 %r73, %r72, 3;
ld.param.b64 %rd58, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4];
and.b32 %r74, %r73, 1016;
ld.param.b64 %rd59, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5];
or.b32 %r75, %r74, %r71;
ld.param.b64 %rd60, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6];
mul.hi.s32 %r76, %r75, 715827883;
shr.u32 %r77, %r76, 31;
shr.s32 %r78, %r76, 7;
add.s32 %r79, %r78, %r77;
ld.param.b64 %rd61, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7];
mul.hi.s32 %r80, %r75, 1041204193;
shr.u32 %r81, %r80, 31;
shr.s32 %r82, %r80, 19;
add.s32 %r83, %r82, %r81;
ld.param.b64 %rd62, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8];
ld.param.b64 %rd51, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9];
mul.wide.s32 %rd63, %r75, 2;
add.s64 %rd1, %rd54, %rd63;
ld.param.b64 %rd64, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10];
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
ld.param.b64 %rd65, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11];
mul.wide.s32 %rd66, %r79, 4;
add.s64 %rd2, %rd55, %rd66;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd2 + 0 ], %rd4;
mov.u64 %rd5, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd5, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd2 + 0 ], %rd5;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd2 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd2 + 0 ], %rd7;
mov.u64 %rd8, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd8, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd2 + 0 ], %rd8;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd2 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r12, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r12 }, [ %rd2 + 0 ], %rd10;
add.s64 %rd11, %rd56, %rd66;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r13, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r13 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r14, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r14 }, [ %rd11 + 0 ], %rd13;
mov.u64 %rd14, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd14, 1.0;
mov.u32 %r15, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r15 }, [ %rd11 + 0 ], %rd14;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r16, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r16 }, [ %rd11 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r17, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r17 }, [ %rd11 + 0 ], %rd16;
mov.u64 %rd17, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd17, 1.0;
mov.u32 %r18, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r18 }, [ %rd11 + 0 ], %rd17;
mov.u64 %rd18, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd18, 1.0;
mov.u32 %r19, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r19 }, [ %rd11 + 0 ], %rd18;
mov.u64 %rd19, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd19, 1.0;
mov.u32 %r20, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r20 }, [ %rd11 + 0 ], %rd19;
mul.wide.s32 %rd67, %r83, 4;
add.s64 %rd20, %rd57, %rd67;
mov.u64 %rd21, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd21, 1.0;
mov.u32 %r21, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r21 }, [ %rd20 + 0 ], %rd21;
mov.u64 %rd22, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd22, 1.0;
mov.u32 %r22, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r22 }, [ %rd20 + 0 ], %rd22;
mov.u64 %rd23, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd23, 1.0;
mov.u32 %r23, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r23 }, [ %rd20 + 0 ], %rd23;
mov.u64 %rd24, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd24, 1.0;
mov.u32 %r24, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r24 }, [ %rd20 + 0 ], %rd24;
mov.u64 %rd25, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd25, 1.0;
mov.u32 %r25, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r25 }, [ %rd20 + 0 ], %rd25;
mov.u64 %rd26, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd26, 1.0;
mov.u32 %r26, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r26 }, [ %rd20 + 0 ], %rd26;
mov.u64 %rd27, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd27, 1.0;
mov.u32 %r27, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r27 }, [ %rd20 + 0 ], %rd27;
mov.u64 %rd28, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd28, 1.0;
mov.u32 %r28, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r28 }, [ %rd20 + 0 ], %rd28;
add.s64 %rd29, %rd58, %rd67;
mov.u64 %rd30, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd30, 1.0;
mov.u32 %r29, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r29 }, [ %rd29 + 0 ], %rd30;
mov.u64 %rd31, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd31, 1.0;
mov.u32 %r30, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r30 }, [ %rd29 + 0 ], %rd31;
mov.u64 %rd32, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd32, 1.0;
mov.u32 %r31, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r31 }, [ %rd29 + 0 ], %rd32;
mov.u64 %rd33, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd33, 1.0;
mov.u32 %r32, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r32 }, [ %rd29 + 0 ], %rd33;
mov.u64 %rd34, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd34, 1.0;
mov.u32 %r33, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r33 }, [ %rd29 + 0 ], %rd34;
mov.u64 %rd35, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd35, 1.0;
mov.u32 %r34, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r34 }, [ %rd29 + 0 ], %rd35;
mov.u64 %rd36, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd36, 1.0;
mov.u32 %r35, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r35 }, [ %rd29 + 0 ], %rd36;
mov.u64 %rd37, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd37, 1.0;
mov.u32 %r36, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r36 }, [ %rd29 + 0 ], %rd37;
add.s64 %rd38, %rd59, %rd67;
mov.u64 %rd39, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd39, 1.0;
mov.u32 %r37, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r37 }, [ %rd38 + 0 ], %rd39;
mov.u64 %rd40, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd40, 1.0;
mov.u32 %r38, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r38 }, [ %rd38 + 0 ], %rd40;
mov.u64 %rd41, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd41, 1.0;
mov.u32 %r39, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r39 }, [ %rd38 + 0 ], %rd41;
mov.u64 %rd42, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd42, 1.0;
mov.u32 %r40, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r40 }, [ %rd38 + 0 ], %rd42;
mov.u64 %rd43, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd43, 1.0;
mov.u32 %r41, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r41 }, [ %rd38 + 0 ], %rd43;
mov.u64 %rd44, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd44, 1.0;
mov.u32 %r42, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r42 }, [ %rd38 + 0 ], %rd44;
mov.u64 %rd45, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd45, 1.0;
mov.u32 %r43, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r43 }, [ %rd38 + 0 ], %rd45;
mov.u64 %rd46, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd46, 1.0;
mov.u32 %r44, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r44 }, [ %rd38 + 0 ], %rd46;
add.s64 %rd47, %rd60, %rd63;
mov.u32 %r45, 0x0;
mov.u32 %r46, 0x0;
mov.u32 %r47, 0x0;
mov.u32 %r48, 0x0;
ld.global.v4.b32 { %r45, %r46, %r47, %r48 }, [ %rd47 + 0 ];
add.s64 %rd48, %rd61, %rd63;
mov.u32 %r49, 0x0;
mov.u32 %r50, 0x0;
mov.u32 %r51, 0x0;
mov.u32 %r52, 0x0;
ld.global.v4.b32 { %r49, %r50, %r51, %r52 }, [ %rd48 + 0 ];
mad.wide.s32 %rd49, %r75, 4, %rd62;
add.s64 %rd50, %rd49, 16;
mov.u32 %r53, 0x0;
mov.u32 %r54, 0x0;
mov.u32 %r55, 0x0;
mov.u32 %r56, 0x0;
ld.global.v4.b32 { %r53, %r54, %r55, %r56 }, [ %rd49 + 0 ];
mov.u32 %r57, 0x0;
mov.u32 %r58, 0x0;
mov.u32 %r59, 0x0;
mov.u32 %r60, 0x0;
ld.global.v4.b32 { %r57, %r58, %r59, %r60 }, [ %rd50 + 0 ];
mov.u32 %r61, 0x0;
ld.global.b32 { %r61 }, [ %rd51 + 0 ];
mov.b32 %r84, 0f44400000;
div.full.f32 %r85, %r5, %r84;
div.full.f32 %r86, %r6, %r84;
div.full.f32 %r87, %r7, %r84;
div.full.f32 %r88, %r8, %r84;
div.full.f32 %r89, %r9, %r84;
div.full.f32 %r90, %r10, %r84;
div.full.f32 %r91, %r11, %r84;
div.full.f32 %r92, %r12, %r84;
sub.f32 %r93, %r92, %r20;
sub.f32 %r94, %r91, %r19;
sub.f32 %r95, %r90, %r18;
sub.f32 %r96, %r89, %r17;
sub.f32 %r97, %r88, %r16;
sub.f32 %r98, %r87, %r15;
sub.f32 %r99, %r86, %r14;
sub.f32 %r100, %r85, %r13;
fma.rn.f32 %r101, %r100, 0f3DCCCCCD, %r13;
fma.rn.f32 %r102, %r99, 0f3DCCCCCD, %r14;
fma.rn.f32 %r103, %r98, 0f3DCCCCCD, %r15;
fma.rn.f32 %r104, %r97, 0f3DCCCCCD, %r16;
fma.rn.f32 %r105, %r96, 0f3DCCCCCD, %r17;
fma.rn.f32 %r106, %r95, 0f3DCCCCCD, %r18;
fma.rn.f32 %r107, %r94, 0f3DCCCCCD, %r19;
fma.rn.f32 %r108, %r93, 0f3DCCCCCD, %r20;
$L__tmp1:
setp.gt.f32 %p1, %r108, 0f2EDBE6FF;
setp.gt.f32 %p2, %r107, 0f2EDBE6FF;
setp.gt.f32 %p3, %r106, 0f2EDBE6FF;
setp.gt.f32 %p4, %r105, 0f2EDBE6FF;
setp.gt.f32 %p5, %r104, 0f2EDBE6FF;
setp.gt.f32 %p6, %r103, 0f2EDBE6FF;
setp.gt.f32 %p7, %r102, 0f2EDBE6FF;
setp.gt.f32 %p8, %r101, 0f2EDBE6FF;
setp.nan.f32 %p9, %r101, %r101;
setp.nan.f32 %p10, %r102, %r102;
setp.nan.f32 %p11, %r103, %r103;
setp.nan.f32 %p12, %r104, %r104;
setp.nan.f32 %p13, %r105, %r105;
setp.nan.f32 %p14, %r106, %r106;
setp.nan.f32 %p15, %r107, %r107;
setp.nan.f32 %p16, %r108, %r108;
selp.f32 %r109, %r101, 0f2EDBE6FF, %p9;
selp.f32 %r110, %r101, %r109, %p8;
selp.f32 %r111, %r102, 0f2EDBE6FF, %p10;
selp.f32 %r112, %r102, %r111, %p7;
selp.f32 %r113, %r103, 0f2EDBE6FF, %p11;
selp.f32 %r114, %r103, %r113, %p6;
selp.f32 %r115, %r104, 0f2EDBE6FF, %p12;
selp.f32 %r116, %r104, %r115, %p5;
selp.f32 %r117, %r105, 0f2EDBE6FF, %p13;
selp.f32 %r118, %r105, %r117, %p4;
selp.f32 %r119, %r106, 0f2EDBE6FF, %p14;
selp.f32 %r120, %r106, %r119, %p3;
selp.f32 %r121, %r107, 0f2EDBE6FF, %p15;
selp.f32 %r122, %r107, %r121, %p2;
selp.f32 %r123, %r108, 0f2EDBE6FF, %p16;
selp.f32 %r124, %r108, %r123, %p1;
$L__tmp2:
rsqrt.approx.ftz.f32 %r125, %r110;
rsqrt.approx.ftz.f32 %r126, %r112;
rsqrt.approx.ftz.f32 %r127, %r114;
rsqrt.approx.ftz.f32 %r128, %r116;
rsqrt.approx.ftz.f32 %r129, %r118;
rsqrt.approx.ftz.f32 %r130, %r120;
rsqrt.approx.ftz.f32 %r131, %r122;
rsqrt.approx.ftz.f32 %r132, %r124;
mul.f32 %r133, %r21, 0f44400000;
mul.f32 %r134, %r22, 0f44400000;
mul.f32 %r135, %r23, 0f44400000;
mul.f32 %r136, %r24, 0f44400000;
mul.f32 %r137, %r25, 0f44400000;
mul.f32 %r138, %r26, 0f44400000;
mul.f32 %r139, %r27, 0f44400000;
mul.f32 %r140, %r28, 0f44400000;
sqrt.rn.f32 %r141, %r133;
sqrt.rn.f32 %r142, %r134;
sqrt.rn.f32 %r143, %r135;
sqrt.rn.f32 %r144, %r136;
sqrt.rn.f32 %r145, %r137;
sqrt.rn.f32 %r146, %r138;
sqrt.rn.f32 %r147, %r139;
sqrt.rn.f32 %r148, %r140;
sqrt.rn.f32 %r149, %r29;
sqrt.rn.f32 %r150, %r30;
sqrt.rn.f32 %r151, %r31;
sqrt.rn.f32 %r152, %r32;
sqrt.rn.f32 %r153, %r33;
sqrt.rn.f32 %r154, %r34;
sqrt.rn.f32 %r155, %r35;
sqrt.rn.f32 %r156, %r36;
$L__tmp3:
setp.gt.f32 %p17, %r149, 0f2EDBE6FF;
setp.gt.f32 %p18, %r150, 0f2EDBE6FF;
setp.gt.f32 %p19, %r151, 0f2EDBE6FF;
setp.gt.f32 %p20, %r152, 0f2EDBE6FF;
setp.gt.f32 %p21, %r153, 0f2EDBE6FF;
setp.gt.f32 %p22, %r154, 0f2EDBE6FF;
setp.gt.f32 %p23, %r155, 0f2EDBE6FF;
setp.gt.f32 %p24, %r156, 0f2EDBE6FF;
setp.nan.f32 %p25, %r149, %r149;
setp.nan.f32 %p26, %r150, %r150;
setp.nan.f32 %p27, %r151, %r151;
setp.nan.f32 %p28, %r152, %r152;
setp.nan.f32 %p29, %r153, %r153;
setp.nan.f32 %p30, %r154, %r154;
setp.nan.f32 %p31, %r155, %r155;
setp.nan.f32 %p32, %r156, %r156;
selp.f32 %r157, %r149, 0f2EDBE6FF, %p25;
selp.f32 %r158, %r149, %r157, %p17;
selp.f32 %r159, %r150, 0f2EDBE6FF, %p26;
selp.f32 %r160, %r150, %r159, %p18;
selp.f32 %r161, %r151, 0f2EDBE6FF, %p27;
selp.f32 %r162, %r151, %r161, %p19;
selp.f32 %r163, %r152, 0f2EDBE6FF, %p28;
selp.f32 %r164, %r152, %r163, %p20;
selp.f32 %r165, %r153, 0f2EDBE6FF, %p29;
selp.f32 %r166, %r153, %r165, %p21;
selp.f32 %r167, %r154, 0f2EDBE6FF, %p30;
selp.f32 %r168, %r154, %r167, %p22;
selp.f32 %r169, %r155, 0f2EDBE6FF, %p31;
selp.f32 %r170, %r155, %r169, %p23;
selp.f32 %r171, %r156, 0f2EDBE6FF, %p32;
selp.f32 %r172, %r156, %r171, %p24;
$L__tmp4:
div.full.f32 %r173, %r141, %r158;
div.full.f32 %r174, %r142, %r160;
div.full.f32 %r175, %r143, %r162;
div.full.f32 %r176, %r144, %r164;
div.full.f32 %r177, %r145, %r166;
div.full.f32 %r178, %r146, %r168;
div.full.f32 %r179, %r147, %r170;
div.full.f32 %r180, %r148, %r172;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r181, %rs2;
cvt.f32.bf16 %r182, %rs1;
mul.f32 %r183, %r126, %r174;
mul.f32 %r184, %r125, %r173;
mul.f32 %r185, %r184, %r182;
mul.f32 %r186, %r183, %r181;
cvt.rn.bf16.f32 %rs3, %r186;
cvt.rn.bf16.f32 %rs4, %r185;
cvt.f32.bf16 %r187, %rs4;
cvt.f32.bf16 %r188, %rs3;
mov.b32 {%rs5, %rs6}, %r45;
cvt.u32.u16 %r189, %rs5;
cvt.u32.u16 %r190, %rs6;
shl.b32 %r191, %r190, 16;
shl.b32 %r192, %r189, 16;
mov.b32 {%rs7, %rs8}, %r49;
cvt.u32.u16 %r193, %rs8;
cvt.u32.u16 %r194, %rs7;
or.b32 %r195, %r192, %r194;
or.b32 %r196, %r191, %r193;
mul.f32 %r197, %r54, %r196;
mul.f32 %r198, %r53, %r195;
setp.ge.f32 %p33, %r198, 0f00000000;
setp.ge.f32 %p34, %r197, 0f00000000;
selp.f32 %r199, 0f3F800000, 0f00000000, %p34;
selp.f32 %r200, 0f3F800000, 0f00000000, %p33;
mul.f32 %r201, %r195, %r200;
mul.f32 %r202, %r196, %r199;
mul.f32 %r203, %r202, %r61;
mul.f32 %r204, %r201, %r61;
neg.f32 %r205, %r204;
fma.rn.f32 %r206, %r205, %r37, %r195;
neg.f32 %r207, %r203;
fma.rn.f32 %r208, %r207, %r38, %r196;
neg.f32 %r209, %r38;
fma.rn.f32 %r210, %r209, %r188, %r208;
neg.f32 %r211, %r37;
fma.rn.f32 %r212, %r211, %r187, %r206;
prmt.b32 %r62, %r212, %r210, 0x7632U;
mov.b32 {%rs9, %rs10}, %r2;
cvt.f32.bf16 %r213, %rs10;
cvt.f32.bf16 %r214, %rs9;
mul.f32 %r215, %r128, %r176;
mul.f32 %r216, %r127, %r175;
mul.f32 %r217, %r216, %r214;
mul.f32 %r218, %r215, %r213;
cvt.rn.bf16.f32 %rs11, %r218;
cvt.rn.bf16.f32 %rs12, %r217;
cvt.f32.bf16 %r219, %rs12;
cvt.f32.bf16 %r220, %rs11;
mov.b32 {%rs13, %rs14}, %r46;
cvt.u32.u16 %r221, %rs13;
cvt.u32.u16 %r222, %rs14;
shl.b32 %r223, %r222, 16;
shl.b32 %r224, %r221, 16;
mov.b32 {%rs15, %rs16}, %r50;
cvt.u32.u16 %r225, %rs16;
cvt.u32.u16 %r226, %rs15;
or.b32 %r227, %r224, %r226;
or.b32 %r228, %r223, %r225;
mul.f32 %r229, %r56, %r228;
mul.f32 %r230, %r55, %r227;
setp.ge.f32 %p35, %r230, 0f00000000;
setp.ge.f32 %p36, %r229, 0f00000000;
selp.f32 %r231, 0f3F800000, 0f00000000, %p36;
selp.f32 %r232, 0f3F800000, 0f00000000, %p35;
mul.f32 %r233, %r227, %r232;
mul.f32 %r234, %r228, %r231;
mul.f32 %r235, %r234, %r61;
mul.f32 %r236, %r233, %r61;
neg.f32 %r237, %r236;
fma.rn.f32 %r238, %r237, %r39, %r227;
neg.f32 %r239, %r235;
fma.rn.f32 %r240, %r239, %r40, %r228;
neg.f32 %r241, %r40;
fma.rn.f32 %r242, %r241, %r220, %r240;
neg.f32 %r243, %r39;
fma.rn.f32 %r244, %r243, %r219, %r238;
prmt.b32 %r63, %r244, %r242, 0x7632U;
mov.b32 {%rs17, %rs18}, %r3;
cvt.f32.bf16 %r245, %rs18;
cvt.f32.bf16 %r246, %rs17;
mul.f32 %r247, %r130, %r178;
mul.f32 %r248, %r129, %r177;
mul.f32 %r249, %r248, %r246;
mul.f32 %r250, %r247, %r245;
cvt.rn.bf16.f32 %rs19, %r250;
cvt.rn.bf16.f32 %rs20, %r249;
cvt.f32.bf16 %r251, %rs20;
cvt.f32.bf16 %r252, %rs19;
mov.b32 {%rs21, %rs22}, %r47;
cvt.u32.u16 %r253, %rs21;
cvt.u32.u16 %r254, %rs22;
shl.b32 %r255, %r254, 16;
shl.b32 %r256, %r253, 16;
mov.b32 {%rs23, %rs24}, %r51;
cvt.u32.u16 %r257, %rs24;
cvt.u32.u16 %r258, %rs23;
or.b32 %r259, %r256, %r258;
or.b32 %r260, %r255, %r257;
mul.f32 %r261, %r58, %r260;
mul.f32 %r262, %r57, %r259;
setp.ge.f32 %p37, %r262, 0f00000000;
setp.ge.f32 %p38, %r261, 0f00000000;
selp.f32 %r263, 0f3F800000, 0f00000000, %p38;
selp.f32 %r264, 0f3F800000, 0f00000000, %p37;
mul.f32 %r265, %r259, %r264;
mul.f32 %r266, %r260, %r263;
mul.f32 %r267, %r266, %r61;
mul.f32 %r268, %r265, %r61;
neg.f32 %r269, %r268;
fma.rn.f32 %r270, %r269, %r41, %r259;
neg.f32 %r271, %r267;
fma.rn.f32 %r272, %r271, %r42, %r260;
neg.f32 %r273, %r42;
fma.rn.f32 %r274, %r273, %r252, %r272;
neg.f32 %r275, %r41;
fma.rn.f32 %r276, %r275, %r251, %r270;
prmt.b32 %r64, %r276, %r274, 0x7632U;
mov.b32 {%rs25, %rs26}, %r4;
cvt.f32.bf16 %r277, %rs26;
cvt.f32.bf16 %r278, %rs25;
mul.f32 %r279, %r132, %r180;
mul.f32 %r280, %r131, %r179;
mul.f32 %r281, %r280, %r278;
mul.f32 %r282, %r279, %r277;
cvt.rn.bf16.f32 %rs27, %r282;
cvt.rn.bf16.f32 %rs28, %r281;
cvt.f32.bf16 %r283, %rs28;
cvt.f32.bf16 %r284, %rs27;
mov.b32 {%rs29, %rs30}, %r48;
cvt.u32.u16 %r285, %rs29;
cvt.u32.u16 %r286, %rs30;
shl.b32 %r287, %r286, 16;
shl.b32 %r288, %r285, 16;
mov.b32 {%rs31, %rs32}, %r52;
cvt.u32.u16 %r289, %rs32;
cvt.u32.u16 %r290, %rs31;
or.b32 %r291, %r288, %r290;
or.b32 %r292, %r287, %r289;
mul.f32 %r293, %r60, %r292;
mul.f32 %r294, %r59, %r291;
setp.ge.f32 %p39, %r294, 0f00000000;
setp.ge.f32 %p40, %r293, 0f00000000;
selp.f32 %r295, 0f3F800000, 0f00000000, %p40;
selp.f32 %r296, 0f3F800000, 0f00000000, %p39;
mul.f32 %r297, %r291, %r296;
mul.f32 %r298, %r292, %r295;
mul.f32 %r299, %r298, %r61;
mul.f32 %r300, %r297, %r61;
neg.f32 %r301, %r300;
fma.rn.f32 %r302, %r301, %r43, %r291;
neg.f32 %r303, %r299;
fma.rn.f32 %r304, %r303, %r44, %r292;
neg.f32 %r305, %r44;
fma.rn.f32 %r306, %r305, %r284, %r304;
neg.f32 %r307, %r43;
fma.rn.f32 %r308, %r307, %r283, %r302;
prmt.b32 %r65, %r308, %r306, 0x7632U;
add.s64 %rd52, %rd64, %rd63;
st.global.v4.b32 [ %rd52 + 0 ], { %r62, %r63, %r64, %r65 };
add.s64 %rd53, %rd65, %rd63;
mov.b32 {%rs33, _}, %r210;
mov.b32 {%rs34, _}, %r212;
mov.b32 %r66, {%rs34, %rs33};
mov.b32 {%rs35, _}, %r242;
mov.b32 {%rs36, _}, %r244;
mov.b32 %r67, {%rs36, %rs35};
mov.b32 {%rs37, _}, %r274;
mov.b32 {%rs38, _}, %r276;
mov.b32 %r68, {%rs38, %rs37};
mov.b32 {%rs39, _}, %r306;
mov.b32 {%rs40, _}, %r308;
mov.b32 %r69, {%rs40, %rs39};
st.global.v4.b32 [ %rd53 + 0 ], { %r66, %r67, %r68, %r69 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .mlp, .v512, .equalize => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2(
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5,
.param .u32 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_8
)
.reqntid 256
{
.reg .pred %p<9>;
.reg .b16 %rs<3>;
.reg .b32 %r<56>;
.reg .b64 %rd<24>;
$L__func_begin0:
ld.param.b64 %rd15, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_0];
ld.param.b64 %rd16, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd17, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_2];
ld.param.b64 %rd18, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd19, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd20, [triton_poi_fused__to_copy_clamp_min_copy__div_lerp_mean_mul_pow_rsqrt_sqrt_2_param_5];
or.b32 %r16, %r15, %r12;
mul.hi.s32 %r17, %r16, 715827883;
shr.u32 %r18, %r17, 31;
shr.s32 %r19, %r17, 7;
add.s32 %r20, %r19, %r18;
mul.hi.s32 %r21, %r16, 1041204193;
shr.u32 %r22, %r21, 31;
shr.s32 %r23, %r21, 19;
add.s32 %r24, %r23, %r22;
mul.wide.s32 %rd21, %r16, 2;
add.s64 %rd1, %rd15, %rd21;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mul.wide.s32 %rd22, %r20, 4;
add.s64 %rd2, %rd16, %rd22;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd2 + 0 ], %rd4;
add.s64 %rd5, %rd17, %rd22;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd5 + 0 ], %rd7;
mul.wide.s32 %rd23, %r24, 4;
add.s64 %rd8, %rd18, %rd23;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd8 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd8 + 0 ], %rd10;
add.s64 %rd11, %rd19, %rd23;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd11 + 0 ], %rd13;
mov.b32 %r25, 0f44400000;
div.full.f32 %r26, %r2, %r25;
div.full.f32 %r27, %r3, %r25;
sub.f32 %r28, %r26, %r4;
sub.f32 %r29, %r27, %r5;
fma.rn.f32 %r30, %r28, 0f3DCCCCCD, %r4;
fma.rn.f32 %r31, %r29, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r30, 0f2EDBE6FF;
setp.gt.f32 %p2, %r31, 0f2EDBE6FF;
setp.nan.f32 %p3, %r30, %r30;
setp.nan.f32 %p4, %r31, %r31;
selp.f32 %r32, %r30, 0f2EDBE6FF, %p3;
selp.f32 %r33, %r30, %r32, %p1;
selp.f32 %r34, %r31, 0f2EDBE6FF, %p4;
selp.f32 %r35, %r31, %r34, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r36, %r33;
rsqrt.approx.ftz.f32 %r37, %r35;
mul.f32 %r38, %r6, 0f44400000;
mul.f32 %r39, %r7, 0f44400000;
sqrt.rn.f32 %r40, %r38;
sqrt.rn.f32 %r41, %r39;
sqrt.rn.f32 %r42, %r8;
sqrt.rn.f32 %r43, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r42, 0f2EDBE6FF;
setp.gt.f32 %p6, %r43, 0f2EDBE6FF;
setp.nan.f32 %p7, %r42, %r42;
setp.nan.f32 %p8, %r43, %r43;
selp.f32 %r44, %r42, 0f2EDBE6FF, %p7;
selp.f32 %r45, %r42, %r44, %p5;
selp.f32 %r46, %r43, 0f2EDBE6FF, %p8;
selp.f32 %r47, %r43, %r46, %p6;
$L__tmp4:
div.full.f32 %r48, %r40, %r45;
div.full.f32 %r49, %r41, %r47;
add.s64 %rd14, %rd20, %rd21;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r50, %rs2;
cvt.f32.bf16 %r51, %rs1;
mul.f32 %r52, %r37, %r49;
mul.f32 %r53, %r36, %r48;
mul.f32 %r54, %r53, %r51;
mul.f32 %r55, %r52, %r50;
cvt.rn.bf16x2.f32 %r10, %r55, %r54;
st.global.b32 [ %rd14 + 0 ], { %r10 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .mlp, .v512, .decay => r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_10
)
.reqntid 256
{
.reg .pred %p<3>;
.reg .b16 %rs<9>;
.reg .b32 %r<47>;
.reg .b64 %rd<19>;
$L__func_begin0:
ld.param.b64 %rd11, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_0];
ld.param.b64 %rd12, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_1];
$L__tmp0:
mov.u32 %r11, %ctaid.x;
shl.b32 %r12, %r11, 9;
ld.param.b64 %rd13, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_2];
ld.param.b64 %rd4, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_3];
mov.u32 %r13, %tid.x;
shl.b32 %r14, %r13, 1;
ld.param.b64 %rd14, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_4];
and.b32 %r15, %r14, 510;
ld.param.b64 %rd15, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_5];
or.b32 %r16, %r15, %r12;
ld.param.b64 %rd16, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_6];
mul.hi.s32 %r17, %r16, 1041204193;
shr.u32 %r18, %r17, 31;
shr.s32 %r19, %r17, 19;
add.s32 %r20, %r19, %r18;
ld.param.b64 %rd17, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_copy__ge_mul_sub_view_0_param_7];
mul.wide.s32 %rd18, %r16, 2;
add.s64 %rd1, %rd11, %rd18;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
add.s64 %rd2, %rd12, %rd18;
mov.u32 %r2, 0x0;
ld.global.b32 { %r2 }, [ %rd2 + 0 ];
mad.wide.s32 %rd3, %r16, 4, %rd13;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
ld.global.v2.b32 { %r3, %r4 }, [ %rd3 + 0 ];
mov.u32 %r5, 0x0;
ld.global.b32 { %r5 }, [ %rd4 + 0 ];
mad.wide.s32 %rd5, %r20, 4, %rd14;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd5 + 0 ], %rd7;
add.s64 %rd8, %rd15, %rd18;
mov.u32 %r8, 0x0;
ld.global.b32 { %r8 }, [ %rd8 + 0 ];
mov.b32 {%rs1, %rs2}, %r8;
cvt.f32.bf16 %r21, %rs1;
cvt.f32.bf16 %r22, %rs2;
mov.b32 {%rs3, %rs4}, %r1;
cvt.u32.u16 %r23, %rs3;
cvt.u32.u16 %r24, %rs4;
shl.b32 %r25, %r24, 16;
shl.b32 %r26, %r23, 16;
mov.b32 {%rs5, %rs6}, %r2;
cvt.u32.u16 %r27, %rs6;
cvt.u32.u16 %r28, %rs5;
or.b32 %r29, %r26, %r28;
or.b32 %r30, %r25, %r27;
mul.f32 %r31, %r4, %r30;
mul.f32 %r32, %r3, %r29;
setp.ge.f32 %p1, %r32, 0f00000000;
setp.ge.f32 %p2, %r31, 0f00000000;
selp.f32 %r33, 0f3F800000, 0f00000000, %p2;
selp.f32 %r34, 0f3F800000, 0f00000000, %p1;
mul.f32 %r35, %r29, %r34;
mul.f32 %r36, %r30, %r33;
mul.f32 %r37, %r36, %r5;
mul.f32 %r38, %r35, %r5;
neg.f32 %r39, %r38;
fma.rn.f32 %r40, %r39, %r6, %r29;
neg.f32 %r41, %r37;
fma.rn.f32 %r42, %r41, %r7, %r30;
neg.f32 %r43, %r7;
fma.rn.f32 %r44, %r43, %r22, %r42;
neg.f32 %r45, %r6;
fma.rn.f32 %r46, %r45, %r21, %r40;
prmt.b32 %r9, %r46, %r44, 0x7632U;
add.s64 %rd9, %rd16, %rd18;
st.global.b32 [ %rd9 + 0 ], { %r9 };
add.s64 %rd10, %rd17, %rd18;
mov.b32 {%rs7, _}, %r44;
mov.b32 {%rs8, _}, %r46;
mov.b32 %r10, {%rs8, %rs7};
st.global.b32 [ %rd10 + 0 ], { %r10 };
ret;
$L__tmp1:
$L__func_end0:
}
  "#
  | .mlp, .v512, .fused => r#"
.version 8.7
.target sm_90a
.address_size 64
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
.visible .entry triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2(
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11,
.param .u32 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_12,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_13,
.param .u64 .ptr .global .align 1 triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_14
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<11>;
.reg .b32 %r<90>;
.reg .b64 %rd<37>;
$L__func_begin0:
ld.param.b64 %rd23, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_0];
ld.param.b64 %rd24, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_1];
$L__tmp0:
mov.u32 %r19, %ctaid.x;
shl.b32 %r20, %r19, 9;
ld.param.b64 %rd25, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_2];
ld.param.b64 %rd26, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_3];
mov.u32 %r21, %tid.x;
shl.b32 %r22, %r21, 1;
ld.param.b64 %rd27, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_4];
and.b32 %r23, %r22, 510;
ld.param.b64 %rd28, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_5];
or.b32 %r24, %r23, %r20;
ld.param.b64 %rd29, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_6];
mul.hi.s32 %r25, %r24, 715827883;
shr.u32 %r26, %r25, 31;
shr.s32 %r27, %r25, 7;
add.s32 %r28, %r27, %r26;
ld.param.b64 %rd30, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_7];
mul.hi.s32 %r29, %r24, 1041204193;
shr.u32 %r30, %r29, 31;
shr.s32 %r31, %r29, 19;
add.s32 %r32, %r31, %r30;
ld.param.b64 %rd31, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_8];
ld.param.b64 %rd20, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_9];
mul.wide.s32 %rd32, %r24, 2;
add.s64 %rd1, %rd23, %rd32;
ld.param.b64 %rd33, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_10];
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
ld.param.b64 %rd34, [triton_poi_fused___lshift_____rshift____to_copy_bitwise_or_clamp_min_copy__div_ge_lerp_mean_mul_pow_rsqrt_sqrt_sub_view_2_param_11];
mul.wide.s32 %rd35, %r28, 4;
add.s64 %rd2, %rd24, %rd35;
mov.u64 %rd3, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
mov.u32 %r2, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r2 }, [ %rd2 + 0 ], %rd3;
mov.u64 %rd4, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd4, 1.0;
mov.u32 %r3, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r3 }, [ %rd2 + 0 ], %rd4;
add.s64 %rd5, %rd25, %rd35;
mov.u64 %rd6, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd6, 1.0;
mov.u32 %r4, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r4 }, [ %rd5 + 0 ], %rd6;
mov.u64 %rd7, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd7, 1.0;
mov.u32 %r5, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r5 }, [ %rd5 + 0 ], %rd7;
mul.wide.s32 %rd36, %r32, 4;
add.s64 %rd8, %rd26, %rd36;
mov.u64 %rd9, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
mov.u32 %r6, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r6 }, [ %rd8 + 0 ], %rd9;
mov.u64 %rd10, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd10, 1.0;
mov.u32 %r7, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r7 }, [ %rd8 + 0 ], %rd10;
add.s64 %rd11, %rd27, %rd36;
mov.u64 %rd12, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd12, 1.0;
mov.u32 %r8, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r8 }, [ %rd11 + 0 ], %rd12;
mov.u64 %rd13, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd13, 1.0;
mov.u32 %r9, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r9 }, [ %rd11 + 0 ], %rd13;
add.s64 %rd14, %rd28, %rd36;
mov.u64 %rd15, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd15, 1.0;
mov.u32 %r10, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r10 }, [ %rd14 + 0 ], %rd15;
mov.u64 %rd16, 0x0;
createpolicy.fractional.L2::evict_last.b64 %rd16, 1.0;
mov.u32 %r11, 0x0;
ld.global.L1::evict_last.L2::cache_hint.b32 { %r11 }, [ %rd14 + 0 ], %rd16;
add.s64 %rd17, %rd29, %rd32;
mov.u32 %r12, 0x0;
ld.global.b32 { %r12 }, [ %rd17 + 0 ];
add.s64 %rd18, %rd30, %rd32;
mov.u32 %r13, 0x0;
ld.global.b32 { %r13 }, [ %rd18 + 0 ];
mad.wide.s32 %rd19, %r24, 4, %rd31;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
ld.global.v2.b32 { %r14, %r15 }, [ %rd19 + 0 ];
mov.u32 %r16, 0x0;
ld.global.b32 { %r16 }, [ %rd20 + 0 ];
mov.b32 %r33, 0f44400000;
div.full.f32 %r34, %r2, %r33;
div.full.f32 %r35, %r3, %r33;
sub.f32 %r36, %r34, %r4;
sub.f32 %r37, %r35, %r5;
fma.rn.f32 %r38, %r36, 0f3DCCCCCD, %r4;
fma.rn.f32 %r39, %r37, 0f3DCCCCCD, %r5;
$L__tmp1:
setp.gt.f32 %p1, %r38, 0f2EDBE6FF;
setp.gt.f32 %p2, %r39, 0f2EDBE6FF;
setp.nan.f32 %p3, %r38, %r38;
setp.nan.f32 %p4, %r39, %r39;
selp.f32 %r40, %r38, 0f2EDBE6FF, %p3;
selp.f32 %r41, %r38, %r40, %p1;
selp.f32 %r42, %r39, 0f2EDBE6FF, %p4;
selp.f32 %r43, %r39, %r42, %p2;
$L__tmp2:
rsqrt.approx.ftz.f32 %r44, %r41;
rsqrt.approx.ftz.f32 %r45, %r43;
mul.f32 %r46, %r6, 0f44400000;
mul.f32 %r47, %r7, 0f44400000;
sqrt.rn.f32 %r48, %r46;
sqrt.rn.f32 %r49, %r47;
sqrt.rn.f32 %r50, %r8;
sqrt.rn.f32 %r51, %r9;
$L__tmp3:
setp.gt.f32 %p5, %r50, 0f2EDBE6FF;
setp.gt.f32 %p6, %r51, 0f2EDBE6FF;
setp.nan.f32 %p7, %r50, %r50;
setp.nan.f32 %p8, %r51, %r51;
selp.f32 %r52, %r50, 0f2EDBE6FF, %p7;
selp.f32 %r53, %r50, %r52, %p5;
selp.f32 %r54, %r51, 0f2EDBE6FF, %p8;
selp.f32 %r55, %r51, %r54, %p6;
$L__tmp4:
div.full.f32 %r56, %r48, %r53;
div.full.f32 %r57, %r49, %r55;
mov.b32 {%rs1, %rs2}, %r1;
cvt.f32.bf16 %r58, %rs2;
cvt.f32.bf16 %r59, %rs1;
mul.f32 %r60, %r45, %r57;
mul.f32 %r61, %r44, %r56;
mul.f32 %r62, %r61, %r59;
mul.f32 %r63, %r60, %r58;
cvt.rn.bf16.f32 %rs3, %r63;
cvt.rn.bf16.f32 %rs4, %r62;
cvt.f32.bf16 %r64, %rs4;
cvt.f32.bf16 %r65, %rs3;
mov.b32 {%rs5, %rs6}, %r12;
cvt.u32.u16 %r66, %rs5;
cvt.u32.u16 %r67, %rs6;
shl.b32 %r68, %r67, 16;
shl.b32 %r69, %r66, 16;
mov.b32 {%rs7, %rs8}, %r13;
cvt.u32.u16 %r70, %rs8;
cvt.u32.u16 %r71, %rs7;
or.b32 %r72, %r69, %r71;
or.b32 %r73, %r68, %r70;
mul.f32 %r74, %r15, %r73;
mul.f32 %r75, %r14, %r72;
setp.ge.f32 %p9, %r75, 0f00000000;
setp.ge.f32 %p10, %r74, 0f00000000;
selp.f32 %r76, 0f3F800000, 0f00000000, %p10;
selp.f32 %r77, 0f3F800000, 0f00000000, %p9;
mul.f32 %r78, %r72, %r77;
mul.f32 %r79, %r73, %r76;
mul.f32 %r80, %r79, %r16;
mul.f32 %r81, %r78, %r16;
neg.f32 %r82, %r81;
fma.rn.f32 %r83, %r82, %r10, %r72;
neg.f32 %r84, %r80;
fma.rn.f32 %r85, %r84, %r11, %r73;
neg.f32 %r86, %r11;
fma.rn.f32 %r87, %r86, %r65, %r85;
neg.f32 %r88, %r10;
fma.rn.f32 %r89, %r88, %r64, %r83;
prmt.b32 %r17, %r89, %r87, 0x7632U;
add.s64 %rd21, %rd33, %rd32;
st.global.b32 [ %rd21 + 0 ], { %r17 };
add.s64 %rd22, %rd34, %rd32;
mov.b32 {%rs9, _}, %r87;
mov.b32 {%rs10, _}, %r89;
mov.b32 %r18, {%rs10, %rs9};
st.global.b32 [ %rd22 + 0 ], { %r18 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#


private def lines (s : String) : List String :=
  (s.splitOn "\n").map String.trimAscii |>.filterMap fun line =>
    if line.isEmpty then none else some line.toString

def fitsCode (bank : Bank) (tile : Tile) (part : Part) (code : Code) : Bool :=
  QKVTTIR.normalizeTTIR code.ttgir == some (lines (graph bank tile part)) &&
    QKVTTIR.normalizePTX code.ptx == lines (assembly bank tile part)

structure Artifacts where
  equalize : Code
  decay : Code
  fused : Code
  deriving BEq

def fitsArtifacts (bank : Bank) (tile : Tile) (code : Artifacts) : Bool :=
  fitsCode bank tile .equalize code.equalize && fitsCode bank tile .decay code.decay &&
    fitsCode bank tile .fused code.fused

structure Program (bank : Bank) (tile : Tile) where
  artifacts : Artifacts
  checked : fitsArtifacts bank tile artifacts = true

def Program.check (bank : Bank) (tile : Tile) (code : Artifacts) : Option (Program bank tile) :=
  if h : fitsArtifacts bank tile code = true then some ⟨code, h⟩ else none

def Program.run {bank : Bank} {tile : Tile} (program : Program bank tile) : Snapshot → Outputs := fun s =>
  if fitsArtifacts bank tile program.artifacts then fused bank s else reference bank s

theorem Program.run_eq {bank : Bank} {tile : Tile} (program : Program bank tile) :
    program.run = reference bank := by
  funext s
  simp only [Program.run, program.checked, if_true, epilogue_equal]

abbrev Programs := (bank : Bank) → (tile : Tile) → Program bank tile

def Programs.run (programs : Programs) : Epilogue := fun bank input => (programs bank input.tile).run input

theorem Programs.run_eq (programs : Programs) : programs.run = reference := by
  funext bank input
  exact congrFun (programs bank input.tile).run_eq input

#axiomsClean covers_cells
#axiomsClean cell_unique
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.AnvilEpilogue.Binding
