import models.NanoGPT.Track1.AnvilTrace
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Hopper lowering of ANVIL's first Gram trace. The two kernel shapes
differ in the number of live warp rows and the inter-warp addition tree. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilTrace
open VeriTile.Triton.FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 zero widen)

def launchWarps : Bank → Nat
  | .qk => 2
  | .vo | .mlp => 8

def warpMask : Bank → UInt32
  | .qk => 32
  | .vo | .mlp => 224

/-- The warp starts a 128-entry diagonal span. A lane in that warp
contributes the four positions separated by 32 entries. -/
def diagonalBase (bank : Bank) (thread : UInt32) : UInt32 :=
  (((thread &&& warpMask bank) <<< (2 : UInt32)) ||| (thread &&& (31 : UInt32)))

theorem diagonal_base_exact (bank : Bank) (thread : Fin (32 * launchWarps bank)) :
    (diagonalBase bank (UInt32.ofNat thread.val)).toNat = 128 * (thread.val / 32) + thread.val % 32 := by
  cases bank <;> decide +kernel +revert

def activeWarp (bank : Bank) (thread : UInt32) : Bool :=
  match bank with
  | .qk => true
  | .vo | .mlp => decide ((thread &&& warpMask bank) < (192 : UInt32))

theorem active_warp_exact (bank : Bank) (thread : Fin (32 * launchWarps bank)) :
    activeWarp bank (UInt32.ofNat thread.val) = decide (thread.val / 32 < chunks bank) := by
  cases bank <;> decide +kernel +revert

theorem active_diagonal_address (bank : Bank) (batch : Fin bank.dimensions.1)
    (thread : Fin (32 * launchWarps bank)) (part : Fin 4)
    (active : activeWarp bank (UInt32.ofNat thread.val) = true) :
    let index := (diagonalBase bank (UInt32.ofNat thread.val)).toNat + 32 * part.val
    index < width bank ∧ 2 * (batch.val * width bank * width bank + index * (width bank + 1)) < 2^31 := by
  simp only [diagonal_base_exact, active_warp_exact, decide_eq_true_eq] at *
  have hp := part.isLt
  have hb := batch.isLt
  cases bank <;> simp only [width, chunks, Bank.dimensions] at * <;> omega

private theorem zero_partial (tree : Tree) : tree.eval (fun _ => zero) = zero := by
  have h : add Config.ieeeValue zero zero = zero := by decide +kernel
  induction tree <;> simp_all only [Tree.eval]

private def shuffleTree (bank : Bank) : Tree :=
  let lanes := fun lane => if lane < chunks bank then Tree.diagonal lane else .zero
  let shuffles := if width bank = 256 then [1] else [4, 2, 1]
  butterfly shuffles lanes 0

theorem shuffle_tree_exact (bank : Bank) : shuffleTree bank = lastTree bank := by
  cases bank <;> decide

/-- Masked diagonal loads return BF16 positive zero, so all that warp's
butterfly terms evaluate to FP32 positive zero. -/
theorem inactive_warp_value : warpPartial.eval (fun _ => widen (Value.zero .bf16)) = zero := by
  have h : widen (Value.zero .bf16) = zero := by decide +kernel
  simp only [h, zero_partial]

private def compiledDivisors : DivisorKernel := fun bank gram b =>
  let capacity := fun i => (shuffleTree bank).eval (fun chunk => partialSum warpPartial bank gram i.val chunk)
  readDivisors (packed fun i => divisorWords (capacity i)) b

private theorem compiled_exact : compiledDivisors = publish := by
  funext bank gram b
  simp only [compiledDivisors, shuffle_tree_exact, publish, fused]

private def graph (bank : Bank) : String :=
  match bank with
  | .qk => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#linear = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0]], warp = [[1]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_gram_divisors(%Input: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<128> : tensor<2x1xi32, #blocked>
%cst_0 = arith.constant dense<257> : tensor<2x32xi32, #blocked>
%cst_1 = arith.constant dense<2> : tensor<2x1xi32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<2x32xbf16, #blocked>
%cst_3 = arith.constant dense<8224> : tensor<2x32xi32, #blocked>
%cst_4 = arith.constant dense<16448> : tensor<2x32xi32, #blocked>
%cst_5 = arith.constant dense<24672> : tensor<2x32xi32, #blocked>
%cst_6 = arith.constant dense<8> : tensor<2x32xi32, #blocked>
%cst_7 = arith.constant dense<4> : tensor<2x32xi32, #blocked>
%cst_8 = arith.constant dense<2> : tensor<2x32xi32, #blocked>
%cst_9 = arith.constant dense<1> : tensor<2x32xi32, #blocked>
%cst_10 = arith.constant dense<16> : tensor<2x32xi32, #blocked>
%cst_11 = arith.constant dense<0> : tensor<2x1xi32, #blocked>
%cst_12 = arith.constant 1.050000e+00 : f32
%cst_13 = arith.constant 9.99999997E-7 : f32
%0 = tt.get_program_id x : i32
%chunk = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%diagonal = tt.expand_dims %chunk {axis = 1 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<2x1xi32, #blocked>
%diagonal_14 = arith.muli %diagonal, %cst : tensor<2x1xi32, #blocked>
%diagonal_15 = tt.expand_dims %lane {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
%diagonal_16 = tt.broadcast %diagonal_14 : tensor<2x1xi32, #blocked> -> tensor<2x32xi32, #blocked>
%diagonal_17 = tt.broadcast %diagonal_15 : tensor<1x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
%diagonal_18 = arith.addi %diagonal_16, %diagonal_17 : tensor<2x32xi32, #blocked>
%ptr = arith.muli %0, 65536 : i32
%ptr_19 = tt.addptr %Input, %ptr : !tt.ptr<bf16>, i32
%ptr_20 = arith.muli %diagonal_18, %cst_0 : tensor<2x32xi32, #blocked>
%ptr_21 = tt.splat %ptr_19 : !tt.ptr<bf16> -> tensor<2x32x!tt.ptr<bf16>, #blocked>
%ptr_22 = tt.addptr %ptr_21, %ptr_20 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%active = arith.cmpi slt, %diagonal, %cst_1 : tensor<2x1xi32, #blocked>
%x0 = tt.broadcast %active : tensor<2x1xi1, #blocked> -> tensor<2x32xi1, #blocked>
%x0_23 = tt.load %ptr_22, %x0, %cst_2 : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x0_24 = arith.extf %x0_23 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x32 = tt.addptr %ptr_22, %cst_3 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x32_25 = tt.load %x32, %x0, %cst_2 : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x32_26 = arith.extf %x32_25 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x64 = tt.addptr %ptr_22, %cst_4 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x64_27 = tt.load %x64, %x0, %cst_2 : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x64_28 = arith.extf %x64_27 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%x96 = tt.addptr %ptr_22, %cst_5 : tensor<2x32x!tt.ptr<bf16>, #blocked>, tensor<2x32xi32, #blocked>
%x96_29 = tt.load %x96, %x0, %cst_2 : tensor<2x32x!tt.ptr<bf16>, #blocked>
%x96_30 = arith.extf %x96_29 : tensor<2x32xbf16, #blocked> to tensor<2x32xf32, #blocked>
%partial = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x0_24 : tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_31 = arith.addf %partial, %x64_28 : tensor<2x32xf32, #blocked>
%partial_32 = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x32_26 : tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_33 = arith.addf %partial_32, %x96_30 : tensor<2x32xf32, #blocked>
%partial_34 = arith.addf %partial_31, %partial_33 : tensor<2x32xf32, #blocked>
%peer = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_34, %cst_6 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_35 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_34, %peer : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_36 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_35, %cst_7 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_37 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_35, %peer_36 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_38 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_37, %cst_8 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_39 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_37, %peer_38 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_40 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_39, %cst_9 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_41 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_39, %peer_40 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%peer_42 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_41, %cst_10 : tensor<2x32xf32, #blocked>, tensor<2x32xi32, #blocked> -> tensor<2x32xf32, #blocked>
%partial_43 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_41, %peer_42 : tensor<2x32xf32, #blocked>, tensor<2x32xf32, #blocked> -> tensor<2x32xf32, #blocked>
%sums = tt.gather %partial_43[%cst_11] {axis = 1 : i32} : (tensor<2x32xf32, #blocked>, tensor<2x1xi32, #blocked>) -> tensor<2x1xf32, #blocked>
%sums_44 = tt.reshape %sums : tensor<2x1xf32, #blocked> -> tensor<2xf32, #linear>
%sums_45 = ttg.convert_layout %sums_44 : tensor<2xf32, #linear> -> tensor<2xf32, #blocked1>
%trace = "tt.reduce"(%sums_45) <{axis = 0 : i32}> ({
^bb0(%trace_47: f32, %trace_48: f32):
%trace_49 = arith.addf %trace_47, %trace_48 : f32
tt.reduce.return %trace_49 : f32
}) : (tensor<2xf32, #blocked1>) -> f32
%denominator = tt.precise_sqrt %trace : f32
%denominator_46 = math.fma %denominator, %cst_12, %cst_13 : f32
%1 = arith.muli %0, 2 : i32
%2 = tt.addptr %Divisors, %1 : !tt.ptr<f32>, i32
tt.store %2, %denominator_46 : !tt.ptr<f32>
%3 = tt.addptr %2, 1 : !tt.ptr<f32>, i32
%4 = arith.mulf %denominator_46, %denominator_46 : f32
tt.store %3, %4 : !tt.ptr<f32>
tt.return
}
}
"#
  | .vo | .mlp => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#linear = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0]], warp = [[1], [2], [4]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_gram_divisors(%Input: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %Divisors: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<128> : tensor<8x1xi32, #blocked>
%cst_0 = arith.constant dense<769> : tensor<8x32xi32, #blocked>
%cst_1 = arith.constant dense<6> : tensor<8x1xi32, #blocked>
%cst_2 = arith.constant dense<0.000000e+00> : tensor<8x32xbf16, #blocked>
%cst_3 = arith.constant dense<24608> : tensor<8x32xi32, #blocked>
%cst_4 = arith.constant dense<49216> : tensor<8x32xi32, #blocked>
%cst_5 = arith.constant dense<73824> : tensor<8x32xi32, #blocked>
%cst_6 = arith.constant dense<8> : tensor<8x32xi32, #blocked>
%cst_7 = arith.constant dense<4> : tensor<8x32xi32, #blocked>
%cst_8 = arith.constant dense<2> : tensor<8x32xi32, #blocked>
%cst_9 = arith.constant dense<1> : tensor<8x32xi32, #blocked>
%cst_10 = arith.constant dense<16> : tensor<8x32xi32, #blocked>
%cst_11 = arith.constant dense<0> : tensor<8x1xi32, #blocked>
%cst_12 = arith.constant 1.050000e+00 : f32
%cst_13 = arith.constant 9.99999997E-7 : f32
%0 = tt.get_program_id x : i32
%chunk = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%lane = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%diagonal = tt.expand_dims %chunk {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
%diagonal_14 = arith.muli %diagonal, %cst : tensor<8x1xi32, #blocked>
%diagonal_15 = tt.expand_dims %lane {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
%diagonal_16 = tt.broadcast %diagonal_14 : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_17 = tt.broadcast %diagonal_15 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
%diagonal_18 = arith.addi %diagonal_16, %diagonal_17 : tensor<8x32xi32, #blocked>
%ptr = arith.muli %0, 589824 : i32
%ptr_19 = tt.addptr %Input, %ptr : !tt.ptr<bf16>, i32
%ptr_20 = arith.muli %diagonal_18, %cst_0 : tensor<8x32xi32, #blocked>
%ptr_21 = tt.splat %ptr_19 : !tt.ptr<bf16> -> tensor<8x32x!tt.ptr<bf16>, #blocked>
%ptr_22 = tt.addptr %ptr_21, %ptr_20 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%active = arith.cmpi slt, %diagonal, %cst_1 : tensor<8x1xi32, #blocked>
%x0 = tt.broadcast %active : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
%x0_23 = tt.load %ptr_22, %x0, %cst_2 : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x0_24 = arith.extf %x0_23 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x32 = tt.addptr %ptr_22, %cst_3 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x32_25 = tt.load %x32, %x0, %cst_2 : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x32_26 = arith.extf %x32_25 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x64 = tt.addptr %ptr_22, %cst_4 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x64_27 = tt.load %x64, %x0, %cst_2 : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x64_28 = arith.extf %x64_27 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%x96 = tt.addptr %ptr_22, %cst_5 : tensor<8x32x!tt.ptr<bf16>, #blocked>, tensor<8x32xi32, #blocked>
%x96_29 = tt.load %x96, %x0, %cst_2 : tensor<8x32x!tt.ptr<bf16>, #blocked>
%x96_30 = arith.extf %x96_29 : tensor<8x32xbf16, #blocked> to tensor<8x32xf32, #blocked>
%partial = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x0_24 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_31 = arith.addf %partial, %x64_28 : tensor<8x32xf32, #blocked>
%partial_32 = tt.elementwise_inline_asm "add.f32 $0, $1, 0f00000000;" {constraints = "=f,f", packed_element = 1 : i32, pure = true} %x32_26 : tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_33 = arith.addf %partial_32, %x96_30 : tensor<8x32xf32, #blocked>
%partial_34 = arith.addf %partial_31, %partial_33 : tensor<8x32xf32, #blocked>
%peer = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_34, %cst_6 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_35 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_34, %peer : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_36 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_35, %cst_7 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_37 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_35, %peer_36 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_38 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_37, %cst_8 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_39 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_37, %peer_38 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_40 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_39, %cst_9 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_41 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_39, %peer_40 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%peer_42 = tt.elementwise_inline_asm "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;" {constraints = "=f,f,n", packed_element = 1 : i32, pure = true} %partial_41, %cst_10 : tensor<8x32xf32, #blocked>, tensor<8x32xi32, #blocked> -> tensor<8x32xf32, #blocked>
%partial_43 = tt.elementwise_inline_asm "add.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %partial_41, %peer_42 : tensor<8x32xf32, #blocked>, tensor<8x32xf32, #blocked> -> tensor<8x32xf32, #blocked>
%sums = tt.gather %partial_43[%cst_11] {axis = 1 : i32} : (tensor<8x32xf32, #blocked>, tensor<8x1xi32, #blocked>) -> tensor<8x1xf32, #blocked>
%sums_44 = tt.reshape %sums : tensor<8x1xf32, #blocked> -> tensor<8xf32, #linear>
%sums_45 = ttg.convert_layout %sums_44 : tensor<8xf32, #linear> -> tensor<8xf32, #blocked1>
%trace = "tt.reduce"(%sums_45) <{axis = 0 : i32}> ({
^bb0(%trace_47: f32, %trace_48: f32):
%trace_49 = arith.addf %trace_47, %trace_48 : f32
tt.reduce.return %trace_49 : f32
}) : (tensor<8xf32, #blocked1>) -> f32
%denominator = tt.precise_sqrt %trace : f32
%denominator_46 = math.fma %denominator, %cst_12, %cst_13 : f32
%1 = arith.muli %0, 2 : i32
%2 = tt.addptr %Divisors, %1 : !tt.ptr<f32>, i32
tt.store %2, %denominator_46 : !tt.ptr<f32>
%3 = tt.addptr %2, 1 : !tt.ptr<f32>, i32
%4 = arith.mulf %denominator_46, %denominator_46 : f32
tt.store %3, %4 : !tt.ptr<f32>
tt.return
}
}
"#

private def assembly (bank : Bank) : String :=
  match bank with
  | .qk => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _gram_divisors(
.param .u64 .ptr .global .align 1 _gram_divisors_param_0,
.param .u64 .ptr .global .align 1 _gram_divisors_param_1,
.param .u64 .ptr .global .align 1 _gram_divisors_param_2,
.param .u64 .ptr .global .align 1 _gram_divisors_param_3
)
.reqntid 64
{
.reg .pred %p<3>;
.reg .b16 %rs<8>;
.reg .b32 %r<44>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd7, [_gram_divisors_param_0];
ld.param.b64 %rd8, [_gram_divisors_param_1];
$L__tmp0:
mov.u32 %r18, %ctaid.x;
$L__tmp1:
mov.u32 %r19, %tid.x;
and.b32 %r20, %r19, 32;
and.b32 %r21, %r19, 31;
shl.b32 %r22, %r20, 2;
or.b32 %r23, %r22, %r21;
shl.b32 %r24, %r18, 16;
mad.wide.s32 %rd9, %r24, 2, %rd7;
mul.lo.s32 %r25, %r23, 257;
mad.wide.u32 %rd1, %r25, 2, %rd9;
mov.b16 %rs2, 0;
mov.pred %p1, -1;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd1 + 0 ];
cvt.f32.bf16 %r2, %rs1;
add.s64 %rd2, %rd1, 16448;
mov.u16 %rs3, %rs2;
@%p1 ld.global.b16 { %rs3 }, [ %rd2 + 0 ];
cvt.f32.bf16 %r4, %rs3;
add.s64 %rd3, %rd1, 32896;
mov.u16 %rs4, %rs2;
@%p1 ld.global.b16 { %rs4 }, [ %rd3 + 0 ];
add.s64 %rd4, %rd1, 49344;
mov.u16 %rs5, %rs2;
@%p1 ld.global.b16 { %rs5 }, [ %rd4 + 0 ];
$L__tmp2:
add.f32 %r1, %r2, 0f00000000;
$L__tmp3:
add.f32 %r3, %r4, 0f00000000;
$L__tmp4:
mov.b32 %r26, {%rs4, %rs5};
mov.b32 {%rs6, %rs7}, %r26;
cvt.f32.bf16 %r27, %rs6;
cvt.f32.bf16 %r28, %rs7;
add.f32 %r29, %r3, %r28;
add.f32 %r30, %r1, %r27;
add.f32 %r5, %r30, %r29;
$L__tmp5:
shfl.sync.bfly.b32 %r6, %r5, 8, 31, -1;
add.f32 %r7, %r5, %r6;
shfl.sync.bfly.b32 %r8, %r7, 4, 31, -1;
add.f32 %r9, %r7, %r8;
shfl.sync.bfly.b32 %r10, %r9, 2, 31, -1;
add.f32 %r11, %r9, %r10;
shfl.sync.bfly.b32 %r12, %r11, 1, 31, -1;
add.f32 %r13, %r11, %r12;
shfl.sync.bfly.b32 %r15, %r13, 16, 31, -1;
add.f32 %r14, %r13, %r15;
$L__tmp6:
shfl.sync.idx.b32 %r31, %r14, 0, 31, -1;
shr.u32 %r32, %r20, 3;
mov.b32 %r33, global_smem;
add.s32 %r34, %r33, %r32;
st.shared.b32 [%r34], %r31;
bar.sync 0;
shl.b32 %r35, %r19, 2;
and.b32 %r36, %r35, 4;
add.s32 %r37, %r33, %r36;
ld.shared.b32 %r38, [%r37];
$L__tmp7:
shfl.sync.bfly.b32 %r39, %r38, 1, 31, -1;
$L__tmp8:
add.f32 %r40, %r38, %r39;
$L__tmp9:
sqrt.rn.f32 %r41, %r40;
fma.rn.f32 %r16, %r41, 0f3F866666, 0f358637BD;
shl.b32 %r42, %r18, 1;
mad.wide.s32 %rd5, %r42, 4, %rd8;
and.b32 %r43, %r19, 63;
setp.eq.b32 %p2, %r43, 0;
@%p2 st.global.b32 [ %rd5 + 0 ], { %r16 };
add.s64 %rd6, %rd5, 4;
mul.f32 %r17, %r16, %r16;
@%p2 st.global.b32 [ %rd6 + 0 ], { %r17 };
$L__tmp10:
ret;
$L__tmp11:
$L__func_end0:
}
"#
  | .vo | .mlp => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _gram_divisors(
.param .u64 .ptr .global .align 1 _gram_divisors_param_0,
.param .u64 .ptr .global .align 1 _gram_divisors_param_1,
.param .u64 .ptr .global .align 1 _gram_divisors_param_2,
.param .u64 .ptr .global .align 1 _gram_divisors_param_3
)
.reqntid 256
{
.reg .pred %p<3>;
.reg .b16 %rs<8>;
.reg .b32 %r<48>;
.reg .b64 %rd<10>;
$L__func_begin0:
ld.param.b64 %rd7, [_gram_divisors_param_0];
ld.param.b64 %rd8, [_gram_divisors_param_1];
$L__tmp0:
mov.u32 %r18, %ctaid.x;
$L__tmp1:
mov.u32 %r19, %tid.x;
and.b32 %r20, %r19, 224;
and.b32 %r21, %r19, 31;
shl.b32 %r22, %r20, 2;
or.b32 %r23, %r22, %r21;
mul.lo.s32 %r24, %r18, 589824;
mad.wide.s32 %rd9, %r24, 2, %rd7;
mul.lo.s32 %r25, %r23, 769;
mad.wide.u32 %rd1, %r25, 2, %rd9;
setp.lt.u32 %p1, %r20, 192;
mov.b16 %rs2, 0;
mov.u16 %rs1, %rs2;
@%p1 ld.global.b16 { %rs1 }, [ %rd1 + 0 ];
cvt.f32.bf16 %r2, %rs1;
add.s64 %rd2, %rd1, 49216;
mov.u16 %rs3, %rs2;
@%p1 ld.global.b16 { %rs3 }, [ %rd2 + 0 ];
cvt.f32.bf16 %r4, %rs3;
add.s64 %rd3, %rd1, 98432;
mov.u16 %rs4, %rs2;
@%p1 ld.global.b16 { %rs4 }, [ %rd3 + 0 ];
add.s64 %rd4, %rd1, 147648;
mov.u16 %rs5, %rs2;
@%p1 ld.global.b16 { %rs5 }, [ %rd4 + 0 ];
$L__tmp2:
add.f32 %r1, %r2, 0f00000000;
$L__tmp3:
add.f32 %r3, %r4, 0f00000000;
$L__tmp4:
mov.b32 %r26, {%rs4, %rs5};
mov.b32 {%rs6, %rs7}, %r26;
cvt.f32.bf16 %r27, %rs6;
cvt.f32.bf16 %r28, %rs7;
add.f32 %r29, %r3, %r28;
add.f32 %r30, %r1, %r27;
add.f32 %r5, %r30, %r29;
$L__tmp5:
shfl.sync.bfly.b32 %r6, %r5, 8, 31, -1;
add.f32 %r7, %r5, %r6;
shfl.sync.bfly.b32 %r8, %r7, 4, 31, -1;
add.f32 %r9, %r7, %r8;
shfl.sync.bfly.b32 %r10, %r9, 2, 31, -1;
add.f32 %r11, %r9, %r10;
shfl.sync.bfly.b32 %r12, %r11, 1, 31, -1;
add.f32 %r13, %r11, %r12;
shfl.sync.bfly.b32 %r15, %r13, 16, 31, -1;
add.f32 %r14, %r13, %r15;
$L__tmp6:
shfl.sync.idx.b32 %r31, %r14, 0, 31, -1;
shr.u32 %r32, %r20, 3;
mov.b32 %r33, global_smem;
add.s32 %r34, %r33, %r32;
st.shared.b32 [%r34], %r31;
bar.sync 0;
shl.b32 %r35, %r19, 2;
and.b32 %r36, %r35, 28;
add.s32 %r37, %r33, %r36;
ld.shared.b32 %r38, [%r37];
$L__tmp7:
shfl.sync.bfly.b32 %r39, %r38, 4, 31, -1;
$L__tmp8:
add.f32 %r40, %r38, %r39;
$L__tmp9:
shfl.sync.bfly.b32 %r41, %r40, 2, 31, -1;
$L__tmp10:
add.f32 %r42, %r40, %r41;
$L__tmp11:
shfl.sync.bfly.b32 %r43, %r42, 1, 31, -1;
$L__tmp12:
add.f32 %r44, %r42, %r43;
$L__tmp13:
sqrt.rn.f32 %r45, %r44;
fma.rn.f32 %r16, %r45, 0f3F866666, 0f358637BD;
shl.b32 %r46, %r18, 1;
mad.wide.s32 %rd5, %r46, 4, %rd8;
and.b32 %r47, %r19, 255;
setp.eq.b32 %p2, %r47, 0;
@%p2 st.global.b32 [ %rd5 + 0 ], { %r16 };
add.s64 %rd6, %rd5, 4;
mul.f32 %r17, %r16, %r16;
@%p2 st.global.b32 [ %rd6 + 0 ], { %r17 };
$L__tmp14:
ret;
$L__tmp15:
$L__func_end0:
}
"#

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def fitsArtifacts (bank : Bank) (ttgir ptx : String) : Bool :=
  QKVTTIR.normalizeTTIR ttgir == some (lines (graph bank)) &&
    QKVTTIR.normalizePTX ptx == lines (assembly bank)

structure Program (bank : Bank) where
  ttgir : String
  ptx : String
  checked : fitsArtifacts bank ttgir ptx = true

def Program.check (bank : Bank) (ttgir ptx : String) : Option (Program bank) :=
  if h : fitsArtifacts bank ttgir ptx = true then some ⟨ttgir, ptx, h⟩ else none

def Program.run {bank : Bank} (program : Program bank) (eligible : Bool)
    (gram : Nat → BF16) (batch : Fin bank.dimensions.1) : Divisors :=
  if eligible && fitsArtifacts bank program.ttgir program.ptx then compiledDivisors bank gram batch
  else original false bank gram batch

theorem Program.run_eq {bank : Bank} (program : Program bank) (eligible : Bool)
    (gram : Nat → BF16) (batch : Fin bank.dimensions.1) :
    program.run eligible gram batch = original eligible bank gram batch := by
  cases eligible
  · rfl
  · simp [Program.run, program.checked, compiled_exact, publish, original, fused_equal, packed_roundtrip]

abbrev Programs := (bank : Bank) → Program bank

def Programs.run (programs : Programs) : Dispatch := fun eligible bank => (programs bank).run eligible

theorem Programs.run_eq (programs : Programs) : programs.run = original := by
  funext eligible bank gram batch
  exact (programs bank).run_eq eligible gram batch

#axiomsClean diagonal_base_exact
#axiomsClean active_warp_exact
#axiomsClean active_diagonal_address
#axiomsClean inactive_warp_value
#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.AnvilTrace
