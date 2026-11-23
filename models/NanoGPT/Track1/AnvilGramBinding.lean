import models.NanoGPT.Track1.AnvilGramTMA
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Pinned Hopper Gram and polynomial tile artifacts. The checked variant
keeps compact triangular ownership and replays the two-stage TMA pipeline. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram
open AnvilEpilogue (Bank)

inductive Shape where
  | qkGram | voGram | mlpGram | qkPolynomial | widePolynomial
  deriving DecidableEq

def Shape.of (bank : Bank) (operation : Operation) : Shape :=
  match bank, operation with
  | .qk, .gram => .qkGram
  | .vo, .gram => .voGram
  | .mlp, .gram => .mlpGram
  | .qk, .polynomial .. => .qkPolynomial
  | _, .polynomial .. => .widePolynomial

inductive Kind where
  | baseline | compact | full

structure Code where
  ttgir : String
  ptx : String
  deriving BEq

private def graph : Shape → Kind → String
  | .qkGram, .baseline => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XXT_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
%cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%num_pid_m = arith.addi %M, 127 : i32
%num_pid_m_2 = arith.divsi %num_pid_m, 128 : i32
%batch_idx = arith.muli %num_pid_m_2, %num_pid_m_2 : i32
%batch_idx_3 = arith.divsi %pid, %batch_idx : i32
%pid_4 = arith.remsi %pid, %batch_idx : i32
%pid_m = arith.divsi %pid_4, %num_pid_m_2 : i32
%pid_n = arith.remsi %pid_4, %num_pid_m_2 : i32
%ij = arith.muli %pid_m, %num_pid_m_2 : i32
%ij_5 = arith.addi %ij, %pid_n : i32
%size_gj = arith.muli %num_pid_m_2, 8 : i32
%group_id = arith.divsi %ij_5, %size_gj : i32
%off_i = arith.muli %group_id, 8 : i32
%size_g = arith.subi %num_pid_m_2, %off_i : i32
%size_g_6 = arith.minsi %size_g, 8 : i32
%ij_7 = arith.remsi %ij_5, %size_gj : i32
%new_i = arith.remsi %ij_7, %size_g_6 : i32
%new_i_8 = arith.addi %off_i, %new_i : i32
%new_j = arith.divsi %ij_7, %size_g_6 : i32
%m_idx = arith.muli %new_i_8, 128 : i32
%n_idx = arith.muli %new_j, 128 : i32
%skip_block_above_diag = arith.addi %m_idx, 128 : i32
%skip_block_above_diag_9 = arith.cmpi sle, %skip_block_above_diag, %n_idx : i32
cf.cond_br %skip_block_above_diag_9, ^bb1, ^bb2
^bb1: // pred: ^bb0
tt.return
^bb2: // pred: ^bb0
%A_ptr_10 = arith.muli %batch_idx_3, %a_stride_b : i32
%A_ptr_11 = tt.addptr %A_ptr, %A_ptr_10 : !tt.ptr<bf16>, i32
%C_ptr_12 = arith.muli %batch_idx_3, %c_stride_b : i32
%C_ptr_13 = tt.addptr %C_ptr, %C_ptr_12 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_16 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_17 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_18 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_21 = arith.addi %offs_m_18, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_22 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_23 = arith.remsi %offs_m_19, %offs_m_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_24 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_25 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n_24, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_28 = arith.addi %offs_n_25, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_29 = arith.remsi %offs_n_26, %offs_m_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%a_ptrs = tt.expand_dims %offs_m_23 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_ptrs_30 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_31 = arith.muli %a_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%a_ptrs_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%a_ptrs_33 = tt.expand_dims %a_ptrs_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
%a_ptrs_34 = tt.broadcast %a_ptrs_31 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_35 = tt.broadcast %a_ptrs_33 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_36 = arith.addi %a_ptrs_34, %a_ptrs_35 : tensor<128x64xi32, #blocked>
%a_ptrs_37 = tt.splat %A_ptr_11 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
%a_ptrs_38 = tt.addptr %a_ptrs_37, %a_ptrs_36 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs = tt.expand_dims %offs_n_29 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%at_ptrs_39 = arith.muli %at_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%at_ptrs_40 = tt.broadcast %at_ptrs_39 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%at_ptrs_41 = arith.addi %at_ptrs_40, %a_ptrs_35 : tensor<128x64xi32, #blocked>
%at_ptrs_42 = tt.addptr %a_ptrs_37, %at_ptrs_41 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%0 = arith.addi %K, 63 : i32
%1 = arith.divsi %0, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %1, 0 : i32
%a_43 = tt.splat %K : i32 -> tensor<1x64xi32, #blocked>
%a_44 = arith.cmpi slt, %a_ptrs_33, %a_43 : tensor<1x64xi32, #blocked>
%a_45 = tt.broadcast %a_44 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_46 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_47 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked>
%accumulator_48 = arith.andi %accumulator_47, %a_45 : tensor<128x64xi1, #blocked>
%a_49 = ttg.async_copy_global_to_local %a_ptrs_38, %a_46 mask %accumulator_48 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_50 = ttg.async_commit_group tokens %a_49
%at_temp_51 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_52 = ttg.async_copy_global_to_local %at_ptrs_42, %at_temp_51 mask %accumulator_48 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_53 = ttg.async_commit_group tokens %at_temp_52
%accumulator_54 = arith.cmpi sgt, %1, 1 : i32
%a_ptrs_55 = tt.addptr %a_ptrs_38, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_56 = tt.addptr %at_ptrs_42, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining = arith.subi %K, 64 : i32
%a_57 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked>
%a_58 = arith.cmpi slt, %a_ptrs_33, %a_57 : tensor<1x64xi32, #blocked>
%a_59 = tt.broadcast %a_58 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_60 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_61 = tt.splat %accumulator_54 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_62 = arith.andi %accumulator_61, %a_59 : tensor<128x64xi1, #blocked>
%a_63 = ttg.async_copy_global_to_local %a_ptrs_55, %a_60 mask %accumulator_62 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_64 = ttg.async_commit_group tokens %a_63
%at_temp_65 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_66 = ttg.async_copy_global_to_local %at_ptrs_56, %at_temp_65 mask %accumulator_62 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_67 = ttg.async_commit_group tokens %at_temp_66
%accumulator_68 = arith.cmpi sgt, %1, 2 : i32
%a_ptrs_69 = tt.addptr %a_ptrs_55, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_70 = tt.addptr %at_ptrs_56, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining_71 = arith.subi %K, 128 : i32
%a_72 = tt.splat %k_remaining_71 : i32 -> tensor<1x64xi32, #blocked>
%a_73 = arith.cmpi slt, %a_ptrs_33, %a_72 : tensor<1x64xi32, #blocked>
%a_74 = tt.broadcast %a_73 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_75 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_76 = tt.splat %accumulator_68 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_77 = arith.andi %accumulator_76, %a_74 : tensor<128x64xi1, #blocked>
%a_78 = ttg.async_copy_global_to_local %a_ptrs_69, %a_75 mask %accumulator_77 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_79 = ttg.async_commit_group tokens %a_78
%at_temp_80 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_81 = ttg.async_copy_global_to_local %at_ptrs_70, %at_temp_80 mask %accumulator_77 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_82 = ttg.async_commit_group tokens %at_temp_81
%accumulator_83:11 = scf.for %accumulator_110 = 0 to %1 step 1 iter_args(%arg9 = %cst, %a_ptrs_111 = %a_ptrs_69, %at_ptrs_112 = %at_ptrs_70, %arg12 = 2, %arg13 = %c-1_i32, %a_113 = %a_50, %a_114 = %a_64, %a_115 = %a_79, %at_temp_116 = %at_temp_53, %at_temp_117 = %at_temp_67, %at_temp_118 = %at_temp_82) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_119 = arith.subi %1, 3 : i32
%accumulator_120 = arith.cmpi slt, %accumulator_110, %accumulator_119 : i32
%accumulator_121 = arith.addi %arg13, 1 : i32
%accumulator_122 = arith.cmpi sge, %accumulator_121, 4 : i32
%accumulator_123 = arith.select %accumulator_122, 0, %accumulator_121 : i32
%a_124 = ttg.async_wait %a_113, %at_temp_116 {num = 4 : i32}
%a_125 = ttg.memdesc_index %a[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_126 = ttg.memdesc_index %at_temp[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_126 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_127 = ttng.warp_group_dot %a_125, %at, %arg9 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_128:3 = ttng.warp_group_dot_wait %accumulator_127, %a_125, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_129 = tt.addptr %a_ptrs_111, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_130 = tt.addptr %at_ptrs_112, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%accumulator_131 = arith.addi %arg12, 1 : i32
%accumulator_132 = arith.cmpi sge, %accumulator_131, 4 : i32
%accumulator_133 = arith.select %accumulator_132, 0, %accumulator_131 : i32
%accumulator_134 = arith.addi %accumulator_110, 3 : i32
%k_remaining_135 = arith.muli %accumulator_134, 64 : i32
%k_remaining_136 = arith.subi %K, %k_remaining_135 : i32
%a_137 = tt.splat %k_remaining_136 : i32 -> tensor<1x64xi32, #blocked>
%a_138 = arith.cmpi slt, %a_ptrs_33, %a_137 : tensor<1x64xi32, #blocked>
%a_139 = tt.broadcast %a_138 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_140 = ttg.memdesc_index %a[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_141 = tt.splat %accumulator_120 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_142 = arith.andi %accumulator_141, %a_139 : tensor<128x64xi1, #blocked>
%a_143 = ttg.async_copy_global_to_local %a_ptrs_129, %a_140 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_144 = ttg.async_commit_group tokens %a_143
%at_temp_145 = ttg.memdesc_index %at_temp[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_146 = ttg.async_copy_global_to_local %at_ptrs_130, %at_temp_145 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_147 = ttg.async_commit_group tokens %at_temp_146
scf.yield %accumulator_128#0, %a_ptrs_129, %at_ptrs_130, %accumulator_133, %accumulator_123, %a_114, %a_115, %a_144, %at_temp_117, %at_temp_118, %at_temp_147 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_84 = ttng.warp_group_dot_wait %accumulator_83#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_85 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_84 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%2 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_m_20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_86 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%c_ptrs_87 = arith.muli %c_ptrs, %c_ptrs_86 : tensor<128x1xi32, #blocked1>
%c_ptrs_88 = tt.expand_dims %offs_n_28 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_89 = tt.broadcast %c_ptrs_87 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_90 = tt.broadcast %c_ptrs_88 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_91 = arith.addi %c_ptrs_89, %c_ptrs_90 : tensor<128x128xi32, #blocked1>
%c_ptrs_92 = tt.splat %C_ptr_13 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_93 = tt.addptr %c_ptrs_92, %c_ptrs_91 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked1>
%c_mask_94 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_95 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked1>
%c_mask_96 = arith.cmpi slt, %c_ptrs_88, %c_mask_95 : tensor<1x128xi32, #blocked1>
%c_mask_97 = tt.broadcast %c_mask_94 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_98 = tt.broadcast %c_mask_96 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_99 = arith.andi %c_mask_97, %c_mask_98 : tensor<128x128xi1, #blocked1>
%3 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_93, %3, %c_mask_99 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_t = tt.expand_dims %offs_n_27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_t_100 = arith.muli %c_ptrs_t, %c_ptrs_86 : tensor<128x1xi32, #blocked1>
%c_ptrs_t_101 = tt.expand_dims %offs_m_21 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_t_102 = tt.broadcast %c_ptrs_t_100 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_103 = tt.broadcast %c_ptrs_t_101 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_104 = arith.addi %c_ptrs_t_102, %c_ptrs_t_103 : tensor<128x128xi32, #blocked1>
%c_ptrs_t_105 = tt.addptr %c_ptrs_92, %c_ptrs_t_104 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_t_106 = arith.cmpi slt, %c_ptrs_t_101, %c_mask_95 : tensor<1x128xi32, #blocked1>
%c_mask_t_107 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_108 = tt.broadcast %c_mask_t_106 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_109 = arith.andi %c_mask_t_107, %c_mask_t_108 : tensor<128x128xi1, #blocked1>
%4 = ttg.convert_layout %2 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_t_105, %4, %c_mask_t_109 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
  "#
  | .qkGram, .compact => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XXT_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
%cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_2 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_2 : i32 to i64
%row_3 = arith.shrui %c72_i64, %row : i64
%row_4 = arith.andi %row_3, %c7_i64 : i64
%row_5 = arith.trunci %row_4 : i64 to i32
%col = arith.shrui %c64_i64, %row : i64
%col_6 = arith.andi %col, %c7_i64 : i64
%col_7 = arith.trunci %col_6 : i64 to i32
%0 = arith.divsi %pid, 3 : i32
%1 = arith.muli %row_5, 128 : i32
%2 = arith.muli %col_7, 128 : i32
%A_ptr_8 = arith.muli %0, %a_stride_b : i32
%A_ptr_9 = tt.addptr %A_ptr, %A_ptr_8 : !tt.ptr<bf16>, i32
%C_ptr_10 = arith.muli %0, %c_stride_b : i32
%C_ptr_11 = tt.addptr %C_ptr, %C_ptr_10 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_14 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_15 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_16 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_17 = arith.addi %offs_m_14, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_18 = arith.addi %offs_m_15, %offs_m_12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_20 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_21 = arith.remsi %offs_m_17, %offs_m_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_22 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_23 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_24 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_25 = arith.addi %offs_n_22, %offs_m_12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n_23, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_27 = arith.remsi %offs_n_24, %offs_m_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%a_ptrs = tt.expand_dims %offs_m_21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_ptrs_28 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_29 = arith.muli %a_ptrs, %a_ptrs_28 : tensor<128x1xi32, #blocked>
%a_ptrs_30 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%a_ptrs_31 = tt.expand_dims %a_ptrs_30 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
%a_ptrs_32 = tt.broadcast %a_ptrs_29 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_33 = tt.broadcast %a_ptrs_31 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_34 = arith.addi %a_ptrs_32, %a_ptrs_33 : tensor<128x64xi32, #blocked>
%a_ptrs_35 = tt.splat %A_ptr_9 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
%a_ptrs_36 = tt.addptr %a_ptrs_35, %a_ptrs_34 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs = tt.expand_dims %offs_n_27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%at_ptrs_37 = arith.muli %at_ptrs, %a_ptrs_28 : tensor<128x1xi32, #blocked>
%at_ptrs_38 = tt.broadcast %at_ptrs_37 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%at_ptrs_39 = arith.addi %at_ptrs_38, %a_ptrs_33 : tensor<128x64xi32, #blocked>
%at_ptrs_40 = tt.addptr %a_ptrs_35, %at_ptrs_39 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%3 = arith.addi %K, 63 : i32
%4 = arith.divsi %3, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %4, 0 : i32
%a_41 = tt.splat %K : i32 -> tensor<1x64xi32, #blocked>
%a_42 = arith.cmpi slt, %a_ptrs_31, %a_41 : tensor<1x64xi32, #blocked>
%a_43 = tt.broadcast %a_42 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_44 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_45 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked>
%accumulator_46 = arith.andi %accumulator_45, %a_43 : tensor<128x64xi1, #blocked>
%a_47 = ttg.async_copy_global_to_local %a_ptrs_36, %a_44 mask %accumulator_46 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_48 = ttg.async_commit_group tokens %a_47
%at_temp_49 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_50 = ttg.async_copy_global_to_local %at_ptrs_40, %at_temp_49 mask %accumulator_46 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_51 = ttg.async_commit_group tokens %at_temp_50
%accumulator_52 = arith.cmpi sgt, %4, 1 : i32
%a_ptrs_53 = tt.addptr %a_ptrs_36, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_54 = tt.addptr %at_ptrs_40, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining = arith.subi %K, 64 : i32
%a_55 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked>
%a_56 = arith.cmpi slt, %a_ptrs_31, %a_55 : tensor<1x64xi32, #blocked>
%a_57 = tt.broadcast %a_56 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_58 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_59 = tt.splat %accumulator_52 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_60 = arith.andi %accumulator_59, %a_57 : tensor<128x64xi1, #blocked>
%a_61 = ttg.async_copy_global_to_local %a_ptrs_53, %a_58 mask %accumulator_60 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_62 = ttg.async_commit_group tokens %a_61
%at_temp_63 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_64 = ttg.async_copy_global_to_local %at_ptrs_54, %at_temp_63 mask %accumulator_60 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_65 = ttg.async_commit_group tokens %at_temp_64
%accumulator_66 = arith.cmpi sgt, %4, 2 : i32
%a_ptrs_67 = tt.addptr %a_ptrs_53, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_68 = tt.addptr %at_ptrs_54, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining_69 = arith.subi %K, 128 : i32
%a_70 = tt.splat %k_remaining_69 : i32 -> tensor<1x64xi32, #blocked>
%a_71 = arith.cmpi slt, %a_ptrs_31, %a_70 : tensor<1x64xi32, #blocked>
%a_72 = tt.broadcast %a_71 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_73 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_74 = tt.splat %accumulator_66 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_75 = arith.andi %accumulator_74, %a_72 : tensor<128x64xi1, #blocked>
%a_76 = ttg.async_copy_global_to_local %a_ptrs_67, %a_73 mask %accumulator_75 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_77 = ttg.async_commit_group tokens %a_76
%at_temp_78 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_79 = ttg.async_copy_global_to_local %at_ptrs_68, %at_temp_78 mask %accumulator_75 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_80 = ttg.async_commit_group tokens %at_temp_79
%accumulator_81:11 = scf.for %accumulator_108 = 0 to %4 step 1 iter_args(%accumulator_109 = %cst, %a_ptrs_110 = %a_ptrs_67, %at_ptrs_111 = %at_ptrs_68, %arg12 = 2, %accumulator_112 = %c-1_i32, %a_113 = %a_48, %a_114 = %a_62, %a_115 = %a_77, %at_temp_116 = %at_temp_51, %at_temp_117 = %at_temp_65, %at_temp_118 = %at_temp_80) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_119 = arith.subi %4, 3 : i32
%accumulator_120 = arith.cmpi slt, %accumulator_108, %accumulator_119 : i32
%accumulator_121 = arith.addi %accumulator_112, 1 : i32
%accumulator_122 = arith.cmpi sge, %accumulator_121, 4 : i32
%accumulator_123 = arith.select %accumulator_122, 0, %accumulator_121 : i32
%a_124 = ttg.async_wait %a_113, %at_temp_116 {num = 4 : i32}
%a_125 = ttg.memdesc_index %a[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_126 = ttg.memdesc_index %at_temp[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_126 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_127 = ttng.warp_group_dot %a_125, %at, %accumulator_109 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_128:3 = ttng.warp_group_dot_wait %accumulator_127, %a_125, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_129 = tt.addptr %a_ptrs_110, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_130 = tt.addptr %at_ptrs_111, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%accumulator_131 = arith.addi %arg12, 1 : i32
%accumulator_132 = arith.cmpi sge, %accumulator_131, 4 : i32
%accumulator_133 = arith.select %accumulator_132, 0, %accumulator_131 : i32
%accumulator_134 = arith.addi %accumulator_108, 3 : i32
%k_remaining_135 = arith.muli %accumulator_134, 64 : i32
%k_remaining_136 = arith.subi %K, %k_remaining_135 : i32
%a_137 = tt.splat %k_remaining_136 : i32 -> tensor<1x64xi32, #blocked>
%a_138 = arith.cmpi slt, %a_ptrs_31, %a_137 : tensor<1x64xi32, #blocked>
%a_139 = tt.broadcast %a_138 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_140 = ttg.memdesc_index %a[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_141 = tt.splat %accumulator_120 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_142 = arith.andi %accumulator_141, %a_139 : tensor<128x64xi1, #blocked>
%a_143 = ttg.async_copy_global_to_local %a_ptrs_129, %a_140 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_144 = ttg.async_commit_group tokens %a_143
%at_temp_145 = ttg.memdesc_index %at_temp[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_146 = ttg.async_copy_global_to_local %at_ptrs_130, %at_temp_145 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_147 = ttg.async_commit_group tokens %at_temp_146
scf.yield %accumulator_128#0, %a_ptrs_129, %at_ptrs_130, %accumulator_133, %accumulator_123, %a_114, %a_115, %a_144, %at_temp_117, %at_temp_118, %at_temp_147 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_82 = ttng.warp_group_dot_wait %accumulator_81#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_83 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_82 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%5 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_m_18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_84 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%c_ptrs_85 = arith.muli %c_ptrs, %c_ptrs_84 : tensor<128x1xi32, #blocked1>
%c_ptrs_86 = tt.expand_dims %offs_n_26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_87 = tt.broadcast %c_ptrs_85 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_88 = tt.broadcast %c_ptrs_86 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_89 = arith.addi %c_ptrs_87, %c_ptrs_88 : tensor<128x128xi32, #blocked1>
%c_ptrs_90 = tt.splat %C_ptr_11 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_91 = tt.addptr %c_ptrs_90, %c_ptrs_89 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked1>
%c_mask_92 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_93 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked1>
%c_mask_94 = arith.cmpi slt, %c_ptrs_86, %c_mask_93 : tensor<1x128xi32, #blocked1>
%c_mask_95 = tt.broadcast %c_mask_92 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_96 = tt.broadcast %c_mask_94 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_97 = arith.andi %c_mask_95, %c_mask_96 : tensor<128x128xi1, #blocked1>
%6 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_91, %6, %c_mask_97 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_t = tt.expand_dims %offs_n_25 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_t_98 = arith.muli %c_ptrs_t, %c_ptrs_84 : tensor<128x1xi32, #blocked1>
%c_ptrs_t_99 = tt.expand_dims %offs_m_19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_t_100 = tt.broadcast %c_ptrs_t_98 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_101 = tt.broadcast %c_ptrs_t_99 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_102 = arith.addi %c_ptrs_t_100, %c_ptrs_t_101 : tensor<128x128xi32, #blocked1>
%c_ptrs_t_103 = tt.addptr %c_ptrs_90, %c_ptrs_t_102 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_t_104 = arith.cmpi slt, %c_ptrs_t_99, %c_mask_93 : tensor<1x128xi32, #blocked1>
%c_mask_t_105 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_106 = tt.broadcast %c_mask_t_104 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_107 = arith.andi %c_mask_t_105, %c_mask_t_106 : tensor<128x128xi1, #blocked1>
%7 = ttg.convert_layout %5 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_t_103, %7, %c_mask_t_107 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
  "#
  | .qkGram, .full => r#"
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @polar(%A: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %A_0: i32, %A_1: i32, %A_2: i64, %A_3: i64, %C: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %C_4: i32, %C_5: i32, %C_6: i64, %C_7: i64) attributes {noinline = false} {
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_8 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_8 : i32 to i64
%row_9 = arith.shrui %c72_i64, %row : i64
%row_10 = arith.andi %row_9, %c7_i64 : i64
%row_11 = arith.trunci %row_10 : i64 to i32
%row_12 = arith.muli %row_11, 128 : i32
%col = arith.shrui %c64_i64, %row : i64
%col_13 = arith.andi %col, %c7_i64 : i64
%col_14 = arith.trunci %col_13 : i64 to i32
%col_15 = arith.muli %col_14, 128 : i32
%batch = arith.divsi %pid, 3 : i32
%offdiag = arith.cmpi ne, %row_12, %col_15 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 256 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_12 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_19 = arith.addi %acc, %col_15 : i32
ttng.async_tma_copy_global_to_local %A[%coords_19, %c0_i32] %acc_18, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_20 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_21 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_22 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_23 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_23, %phase_22, %true deps %base, %acc_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_18 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_24 = ttng.warp_group_dot %base, %right, %acc_21, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_25 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_26 = ttg.memdesc_index %barriers[%nxt_25] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_27 = ttg.memdesc_index %tiles[%nxt_25] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_26, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_27, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_28 = ttng.warp_group_dot_wait %acc_24 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %A[%coords_19, %k] %acc_18, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_28 : tensor<128x128xf32, #mma>
}
scf.yield %acc_20 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 256 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_12 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_20 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_21 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_22 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_22, %phase_21, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_23 = ttng.warp_group_dot %base, %right, %acc_20, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_24 = ttng.warp_group_dot_wait %acc_23 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_25 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_26 = ttg.memdesc_index %barriers[%nxt_25] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_27 = ttg.memdesc_index %tiles[%nxt_25] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_26, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_27, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_24 : tensor<128x128xf32, #mma>
}
%acc_19 = ttng.warp_group_dot_wait %acc_18 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_19 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_16 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%6 = arith.muli %batch, 256 : i32
%7 = arith.addi %6, %row_12 : i32
ttng.async_tma_copy_local_to_global %C[%7, %col_15] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 256 : i32
%4 = arith.addi %3, %col_15 : i32
ttng.async_tma_copy_local_to_global %C[%4, %row_12] %mirror_16 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .voGram, .baseline => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XXT_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
%cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%num_pid_m = arith.addi %M, 127 : i32
%num_pid_m_2 = arith.divsi %num_pid_m, 128 : i32
%batch_idx = arith.muli %num_pid_m_2, %num_pid_m_2 : i32
%batch_idx_3 = arith.divsi %pid, %batch_idx : i32
%pid_4 = arith.remsi %pid, %batch_idx : i32
%pid_m = arith.divsi %pid_4, %num_pid_m_2 : i32
%pid_n = arith.remsi %pid_4, %num_pid_m_2 : i32
%ij = arith.muli %pid_m, %num_pid_m_2 : i32
%ij_5 = arith.addi %ij, %pid_n : i32
%size_gj = arith.muli %num_pid_m_2, 8 : i32
%group_id = arith.divsi %ij_5, %size_gj : i32
%off_i = arith.muli %group_id, 8 : i32
%size_g = arith.subi %num_pid_m_2, %off_i : i32
%size_g_6 = arith.minsi %size_g, 8 : i32
%ij_7 = arith.remsi %ij_5, %size_gj : i32
%new_i = arith.remsi %ij_7, %size_g_6 : i32
%new_i_8 = arith.addi %off_i, %new_i : i32
%new_j = arith.divsi %ij_7, %size_g_6 : i32
%m_idx = arith.muli %new_i_8, 128 : i32
%n_idx = arith.muli %new_j, 128 : i32
%skip_block_above_diag = arith.addi %m_idx, 128 : i32
%skip_block_above_diag_9 = arith.cmpi sle, %skip_block_above_diag, %n_idx : i32
cf.cond_br %skip_block_above_diag_9, ^bb1, ^bb2
^bb1: // pred: ^bb0
tt.return
^bb2: // pred: ^bb0
%A_ptr_10 = arith.muli %batch_idx_3, %a_stride_b : i32
%A_ptr_11 = tt.addptr %A_ptr, %A_ptr_10 : !tt.ptr<bf16>, i32
%C_ptr_12 = arith.muli %batch_idx_3, %c_stride_b : i32
%C_ptr_13 = tt.addptr %C_ptr, %C_ptr_12 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_16 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_17 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_18 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_21 = arith.addi %offs_m_18, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_22 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_23 = arith.remsi %offs_m_19, %offs_m_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_24 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_25 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n_24, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_28 = arith.addi %offs_n_25, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_29 = arith.remsi %offs_n_26, %offs_m_22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%a_ptrs = tt.expand_dims %offs_m_23 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_ptrs_30 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_31 = arith.muli %a_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%a_ptrs_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%a_ptrs_33 = tt.expand_dims %a_ptrs_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
%a_ptrs_34 = tt.broadcast %a_ptrs_31 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_35 = tt.broadcast %a_ptrs_33 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_36 = arith.addi %a_ptrs_34, %a_ptrs_35 : tensor<128x64xi32, #blocked>
%a_ptrs_37 = tt.splat %A_ptr_11 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
%a_ptrs_38 = tt.addptr %a_ptrs_37, %a_ptrs_36 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs = tt.expand_dims %offs_n_29 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%at_ptrs_39 = arith.muli %at_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%at_ptrs_40 = tt.broadcast %at_ptrs_39 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%at_ptrs_41 = arith.addi %at_ptrs_40, %a_ptrs_35 : tensor<128x64xi32, #blocked>
%at_ptrs_42 = tt.addptr %a_ptrs_37, %at_ptrs_41 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%0 = arith.addi %K, 63 : i32
%1 = arith.divsi %0, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %1, 0 : i32
%a_43 = tt.splat %K : i32 -> tensor<1x64xi32, #blocked>
%a_44 = arith.cmpi slt, %a_ptrs_33, %a_43 : tensor<1x64xi32, #blocked>
%a_45 = tt.broadcast %a_44 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_46 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_47 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked>
%accumulator_48 = arith.andi %accumulator_47, %a_45 : tensor<128x64xi1, #blocked>
%a_49 = ttg.async_copy_global_to_local %a_ptrs_38, %a_46 mask %accumulator_48 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_50 = ttg.async_commit_group tokens %a_49
%at_temp_51 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_52 = ttg.async_copy_global_to_local %at_ptrs_42, %at_temp_51 mask %accumulator_48 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_53 = ttg.async_commit_group tokens %at_temp_52
%accumulator_54 = arith.cmpi sgt, %1, 1 : i32
%a_ptrs_55 = tt.addptr %a_ptrs_38, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_56 = tt.addptr %at_ptrs_42, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining = arith.subi %K, 64 : i32
%a_57 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked>
%a_58 = arith.cmpi slt, %a_ptrs_33, %a_57 : tensor<1x64xi32, #blocked>
%a_59 = tt.broadcast %a_58 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_60 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_61 = tt.splat %accumulator_54 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_62 = arith.andi %accumulator_61, %a_59 : tensor<128x64xi1, #blocked>
%a_63 = ttg.async_copy_global_to_local %a_ptrs_55, %a_60 mask %accumulator_62 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_64 = ttg.async_commit_group tokens %a_63
%at_temp_65 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_66 = ttg.async_copy_global_to_local %at_ptrs_56, %at_temp_65 mask %accumulator_62 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_67 = ttg.async_commit_group tokens %at_temp_66
%accumulator_68 = arith.cmpi sgt, %1, 2 : i32
%a_ptrs_69 = tt.addptr %a_ptrs_55, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_70 = tt.addptr %at_ptrs_56, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining_71 = arith.subi %K, 128 : i32
%a_72 = tt.splat %k_remaining_71 : i32 -> tensor<1x64xi32, #blocked>
%a_73 = arith.cmpi slt, %a_ptrs_33, %a_72 : tensor<1x64xi32, #blocked>
%a_74 = tt.broadcast %a_73 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_75 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_76 = tt.splat %accumulator_68 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_77 = arith.andi %accumulator_76, %a_74 : tensor<128x64xi1, #blocked>
%a_78 = ttg.async_copy_global_to_local %a_ptrs_69, %a_75 mask %accumulator_77 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_79 = ttg.async_commit_group tokens %a_78
%at_temp_80 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_81 = ttg.async_copy_global_to_local %at_ptrs_70, %at_temp_80 mask %accumulator_77 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_82 = ttg.async_commit_group tokens %at_temp_81
%accumulator_83:11 = scf.for %accumulator_110 = 0 to %1 step 1 iter_args(%arg9 = %cst, %a_ptrs_111 = %a_ptrs_69, %at_ptrs_112 = %at_ptrs_70, %arg12 = 2, %arg13 = %c-1_i32, %a_113 = %a_50, %a_114 = %a_64, %a_115 = %a_79, %at_temp_116 = %at_temp_53, %at_temp_117 = %at_temp_67, %at_temp_118 = %at_temp_82) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_119 = arith.subi %1, 3 : i32
%accumulator_120 = arith.cmpi slt, %accumulator_110, %accumulator_119 : i32
%accumulator_121 = arith.addi %arg13, 1 : i32
%accumulator_122 = arith.cmpi sge, %accumulator_121, 4 : i32
%accumulator_123 = arith.select %accumulator_122, 0, %accumulator_121 : i32
%a_124 = ttg.async_wait %a_113, %at_temp_116 {num = 4 : i32}
%a_125 = ttg.memdesc_index %a[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_126 = ttg.memdesc_index %at_temp[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_126 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_127 = ttng.warp_group_dot %a_125, %at, %arg9 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_128:3 = ttng.warp_group_dot_wait %accumulator_127, %a_125, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_129 = tt.addptr %a_ptrs_111, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_130 = tt.addptr %at_ptrs_112, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%accumulator_131 = arith.addi %arg12, 1 : i32
%accumulator_132 = arith.cmpi sge, %accumulator_131, 4 : i32
%accumulator_133 = arith.select %accumulator_132, 0, %accumulator_131 : i32
%accumulator_134 = arith.addi %accumulator_110, 3 : i32
%k_remaining_135 = arith.muli %accumulator_134, 64 : i32
%k_remaining_136 = arith.subi %K, %k_remaining_135 : i32
%a_137 = tt.splat %k_remaining_136 : i32 -> tensor<1x64xi32, #blocked>
%a_138 = arith.cmpi slt, %a_ptrs_33, %a_137 : tensor<1x64xi32, #blocked>
%a_139 = tt.broadcast %a_138 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_140 = ttg.memdesc_index %a[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_141 = tt.splat %accumulator_120 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_142 = arith.andi %accumulator_141, %a_139 : tensor<128x64xi1, #blocked>
%a_143 = ttg.async_copy_global_to_local %a_ptrs_129, %a_140 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_144 = ttg.async_commit_group tokens %a_143
%at_temp_145 = ttg.memdesc_index %at_temp[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_146 = ttg.async_copy_global_to_local %at_ptrs_130, %at_temp_145 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_147 = ttg.async_commit_group tokens %at_temp_146
scf.yield %accumulator_128#0, %a_ptrs_129, %at_ptrs_130, %accumulator_133, %accumulator_123, %a_114, %a_115, %a_144, %at_temp_117, %at_temp_118, %at_temp_147 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_84 = ttng.warp_group_dot_wait %accumulator_83#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_85 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_84 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%2 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_m_20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_86 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%c_ptrs_87 = arith.muli %c_ptrs, %c_ptrs_86 : tensor<128x1xi32, #blocked1>
%c_ptrs_88 = tt.expand_dims %offs_n_28 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_89 = tt.broadcast %c_ptrs_87 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_90 = tt.broadcast %c_ptrs_88 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_91 = arith.addi %c_ptrs_89, %c_ptrs_90 : tensor<128x128xi32, #blocked1>
%c_ptrs_92 = tt.splat %C_ptr_13 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_93 = tt.addptr %c_ptrs_92, %c_ptrs_91 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked1>
%c_mask_94 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_95 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked1>
%c_mask_96 = arith.cmpi slt, %c_ptrs_88, %c_mask_95 : tensor<1x128xi32, #blocked1>
%c_mask_97 = tt.broadcast %c_mask_94 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_98 = tt.broadcast %c_mask_96 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_99 = arith.andi %c_mask_97, %c_mask_98 : tensor<128x128xi1, #blocked1>
%3 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_93, %3, %c_mask_99 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_t = tt.expand_dims %offs_n_27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_t_100 = arith.muli %c_ptrs_t, %c_ptrs_86 : tensor<128x1xi32, #blocked1>
%c_ptrs_t_101 = tt.expand_dims %offs_m_21 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_t_102 = tt.broadcast %c_ptrs_t_100 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_103 = tt.broadcast %c_ptrs_t_101 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_104 = arith.addi %c_ptrs_t_102, %c_ptrs_t_103 : tensor<128x128xi32, #blocked1>
%c_ptrs_t_105 = tt.addptr %c_ptrs_92, %c_ptrs_t_104 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_t_106 = arith.cmpi slt, %c_ptrs_t_101, %c_mask_95 : tensor<1x128xi32, #blocked1>
%c_mask_t_107 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_108 = tt.broadcast %c_mask_t_106 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_109 = arith.andi %c_mask_t_107, %c_mask_t_108 : tensor<128x128xi1, #blocked1>
%4 = ttg.convert_layout %2 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_t_105, %4, %c_mask_t_109 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
  "#
  | .voGram, .compact => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XXT_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked>
%cst_1 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_2 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_2 : i32 to i64
%row_3 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_4 = arith.andi %row_3, %c7_i64 : i64
%row_5 = arith.trunci %row_4 : i64 to i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_6 = arith.andi %col, %c7_i64 : i64
%col_7 = arith.trunci %col_6 : i64 to i32
%0 = arith.divsi %pid, 21 : i32
%1 = arith.muli %row_5, 128 : i32
%2 = arith.muli %col_7, 128 : i32
%A_ptr_8 = arith.muli %0, %a_stride_b : i32
%A_ptr_9 = tt.addptr %A_ptr, %A_ptr_8 : !tt.ptr<bf16>, i32
%C_ptr_10 = arith.muli %0, %c_stride_b : i32
%C_ptr_11 = tt.addptr %C_ptr, %C_ptr_10 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_14 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_15 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_16 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_17 = arith.addi %offs_m_14, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_18 = arith.addi %offs_m_15, %offs_m_12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_m_20 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_21 = arith.remsi %offs_m_17, %offs_m_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_22 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_23 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_24 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_25 = arith.addi %offs_n_22, %offs_m_12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n_23, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%offs_n_27 = arith.remsi %offs_n_24, %offs_m_20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%a_ptrs = tt.expand_dims %offs_m_21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_ptrs_28 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_29 = arith.muli %a_ptrs, %a_ptrs_28 : tensor<128x1xi32, #blocked>
%a_ptrs_30 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%a_ptrs_31 = tt.expand_dims %a_ptrs_30 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
%a_ptrs_32 = tt.broadcast %a_ptrs_29 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_33 = tt.broadcast %a_ptrs_31 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
%a_ptrs_34 = arith.addi %a_ptrs_32, %a_ptrs_33 : tensor<128x64xi32, #blocked>
%a_ptrs_35 = tt.splat %A_ptr_9 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked>
%a_ptrs_36 = tt.addptr %a_ptrs_35, %a_ptrs_34 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs = tt.expand_dims %offs_n_27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%at_ptrs_37 = arith.muli %at_ptrs, %a_ptrs_28 : tensor<128x1xi32, #blocked>
%at_ptrs_38 = tt.broadcast %at_ptrs_37 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
%at_ptrs_39 = arith.addi %at_ptrs_38, %a_ptrs_33 : tensor<128x64xi32, #blocked>
%at_ptrs_40 = tt.addptr %a_ptrs_35, %at_ptrs_39 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%3 = arith.addi %K, 63 : i32
%4 = arith.divsi %3, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %4, 0 : i32
%a_41 = tt.splat %K : i32 -> tensor<1x64xi32, #blocked>
%a_42 = arith.cmpi slt, %a_ptrs_31, %a_41 : tensor<1x64xi32, #blocked>
%a_43 = tt.broadcast %a_42 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_44 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_45 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked>
%accumulator_46 = arith.andi %accumulator_45, %a_43 : tensor<128x64xi1, #blocked>
%a_47 = ttg.async_copy_global_to_local %a_ptrs_36, %a_44 mask %accumulator_46 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_48 = ttg.async_commit_group tokens %a_47
%at_temp_49 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_50 = ttg.async_copy_global_to_local %at_ptrs_40, %at_temp_49 mask %accumulator_46 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_51 = ttg.async_commit_group tokens %at_temp_50
%accumulator_52 = arith.cmpi sgt, %4, 1 : i32
%a_ptrs_53 = tt.addptr %a_ptrs_36, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_54 = tt.addptr %at_ptrs_40, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining = arith.subi %K, 64 : i32
%a_55 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked>
%a_56 = arith.cmpi slt, %a_ptrs_31, %a_55 : tensor<1x64xi32, #blocked>
%a_57 = tt.broadcast %a_56 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_58 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_59 = tt.splat %accumulator_52 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_60 = arith.andi %accumulator_59, %a_57 : tensor<128x64xi1, #blocked>
%a_61 = ttg.async_copy_global_to_local %a_ptrs_53, %a_58 mask %accumulator_60 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_62 = ttg.async_commit_group tokens %a_61
%at_temp_63 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_64 = ttg.async_copy_global_to_local %at_ptrs_54, %at_temp_63 mask %accumulator_60 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_65 = ttg.async_commit_group tokens %at_temp_64
%accumulator_66 = arith.cmpi sgt, %4, 2 : i32
%a_ptrs_67 = tt.addptr %a_ptrs_53, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_68 = tt.addptr %at_ptrs_54, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%k_remaining_69 = arith.subi %K, 128 : i32
%a_70 = tt.splat %k_remaining_69 : i32 -> tensor<1x64xi32, #blocked>
%a_71 = arith.cmpi slt, %a_ptrs_31, %a_70 : tensor<1x64xi32, #blocked>
%a_72 = tt.broadcast %a_71 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_73 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_74 = tt.splat %accumulator_66 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_75 = arith.andi %accumulator_74, %a_72 : tensor<128x64xi1, #blocked>
%a_76 = ttg.async_copy_global_to_local %a_ptrs_67, %a_73 mask %accumulator_75 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_77 = ttg.async_commit_group tokens %a_76
%at_temp_78 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_79 = ttg.async_copy_global_to_local %at_ptrs_68, %at_temp_78 mask %accumulator_75 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_80 = ttg.async_commit_group tokens %at_temp_79
%accumulator_81:11 = scf.for %accumulator_108 = 0 to %4 step 1 iter_args(%accumulator_109 = %cst, %a_ptrs_110 = %a_ptrs_67, %at_ptrs_111 = %at_ptrs_68, %arg12 = 2, %accumulator_112 = %c-1_i32, %a_113 = %a_48, %a_114 = %a_62, %a_115 = %a_77, %at_temp_116 = %at_temp_51, %at_temp_117 = %at_temp_65, %at_temp_118 = %at_temp_80) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_119 = arith.subi %4, 3 : i32
%accumulator_120 = arith.cmpi slt, %accumulator_108, %accumulator_119 : i32
%accumulator_121 = arith.addi %accumulator_112, 1 : i32
%accumulator_122 = arith.cmpi sge, %accumulator_121, 4 : i32
%accumulator_123 = arith.select %accumulator_122, 0, %accumulator_121 : i32
%a_124 = ttg.async_wait %a_113, %at_temp_116 {num = 4 : i32}
%a_125 = ttg.memdesc_index %a[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_126 = ttg.memdesc_index %at_temp[%accumulator_123] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_126 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_127 = ttng.warp_group_dot %a_125, %at, %accumulator_109 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_128:3 = ttng.warp_group_dot_wait %accumulator_127, %a_125, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_129 = tt.addptr %a_ptrs_110, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%at_ptrs_130 = tt.addptr %at_ptrs_111, %cst_1 : tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64xi32, #blocked>
%accumulator_131 = arith.addi %arg12, 1 : i32
%accumulator_132 = arith.cmpi sge, %accumulator_131, 4 : i32
%accumulator_133 = arith.select %accumulator_132, 0, %accumulator_131 : i32
%accumulator_134 = arith.addi %accumulator_108, 3 : i32
%k_remaining_135 = arith.muli %accumulator_134, 64 : i32
%k_remaining_136 = arith.subi %K, %k_remaining_135 : i32
%a_137 = tt.splat %k_remaining_136 : i32 -> tensor<1x64xi32, #blocked>
%a_138 = arith.cmpi slt, %a_ptrs_31, %a_137 : tensor<1x64xi32, #blocked>
%a_139 = tt.broadcast %a_138 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
%a_140 = ttg.memdesc_index %a[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_141 = tt.splat %accumulator_120 : i1 -> tensor<128x64xi1, #blocked>
%accumulator_142 = arith.andi %accumulator_141, %a_139 : tensor<128x64xi1, #blocked>
%a_143 = ttg.async_copy_global_to_local %a_ptrs_129, %a_140 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%a_144 = ttg.async_commit_group tokens %a_143
%at_temp_145 = ttg.memdesc_index %at_temp[%accumulator_133] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_146 = ttg.async_copy_global_to_local %at_ptrs_130, %at_temp_145 mask %accumulator_142 other %cst_0 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_147 = ttg.async_commit_group tokens %at_temp_146
scf.yield %accumulator_128#0, %a_ptrs_129, %at_ptrs_130, %accumulator_133, %accumulator_123, %a_114, %a_115, %a_144, %at_temp_117, %at_temp_118, %at_temp_147 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked>, tensor<128x64x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_82 = ttng.warp_group_dot_wait %accumulator_81#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_83 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_82 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%5 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_m_18 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_84 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%c_ptrs_85 = arith.muli %c_ptrs, %c_ptrs_84 : tensor<128x1xi32, #blocked1>
%c_ptrs_86 = tt.expand_dims %offs_n_26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_87 = tt.broadcast %c_ptrs_85 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_88 = tt.broadcast %c_ptrs_86 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_89 = arith.addi %c_ptrs_87, %c_ptrs_88 : tensor<128x128xi32, #blocked1>
%c_ptrs_90 = tt.splat %C_ptr_11 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_91 = tt.addptr %c_ptrs_90, %c_ptrs_89 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked1>
%c_mask_92 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_93 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked1>
%c_mask_94 = arith.cmpi slt, %c_ptrs_86, %c_mask_93 : tensor<1x128xi32, #blocked1>
%c_mask_95 = tt.broadcast %c_mask_92 : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_96 = tt.broadcast %c_mask_94 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_97 = arith.andi %c_mask_95, %c_mask_96 : tensor<128x128xi1, #blocked1>
%6 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_91, %6, %c_mask_97 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
%c_ptrs_t = tt.expand_dims %offs_n_25 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%c_ptrs_t_98 = arith.muli %c_ptrs_t, %c_ptrs_84 : tensor<128x1xi32, #blocked1>
%c_ptrs_t_99 = tt.expand_dims %offs_m_19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%c_ptrs_t_100 = tt.broadcast %c_ptrs_t_98 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_101 = tt.broadcast %c_ptrs_t_99 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
%c_ptrs_t_102 = arith.addi %c_ptrs_t_100, %c_ptrs_t_101 : tensor<128x128xi32, #blocked1>
%c_ptrs_t_103 = tt.addptr %c_ptrs_90, %c_ptrs_t_102 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi32, #blocked1>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked1>
%c_mask_t_104 = arith.cmpi slt, %c_ptrs_t_99, %c_mask_93 : tensor<1x128xi32, #blocked1>
%c_mask_t_105 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_106 = tt.broadcast %c_mask_t_104 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
%c_mask_t_107 = arith.andi %c_mask_t_105, %c_mask_t_106 : tensor<128x128xi1, #blocked1>
%7 = ttg.convert_layout %5 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked1>
tt.store %c_ptrs_t_103, %7, %c_mask_t_107 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
tt.return
}
}
  "#
  | .voGram, .full => r#"
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @polar(%A: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %A_0: i32, %A_1: i32, %A_2: i64, %A_3: i64, %C: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %C_4: i32, %C_5: i32, %C_6: i64, %C_7: i64) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_8 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_8 : i32 to i64
%row_9 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_10 = arith.andi %row_9, %c7_i64 : i64
%row_11 = arith.trunci %row_10 : i64 to i32
%row_12 = arith.muli %row_11, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_13 = arith.andi %col, %c7_i64 : i64
%col_14 = arith.trunci %col_13 : i64 to i32
%col_15 = arith.muli %col_14, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_12, %col_15 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 768 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_12 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_19 = arith.addi %acc, %col_15 : i32
ttng.async_tma_copy_global_to_local %A[%coords_19, %c0_i32] %acc_18, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_20 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_21 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_22 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_23 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_23, %phase_22, %true deps %base, %acc_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_18 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_24 = ttng.warp_group_dot %base, %right, %acc_21, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_25 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_26 = ttg.memdesc_index %barriers[%nxt_25] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_27 = ttg.memdesc_index %tiles[%nxt_25] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_26, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_27, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_28 = ttng.warp_group_dot_wait %acc_24 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %A[%coords_19, %k] %acc_18, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_28 : tensor<128x128xf32, #mma>
}
scf.yield %acc_20 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 768 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_12 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_20 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_21 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_22 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_22, %phase_21, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_23 = ttng.warp_group_dot %base, %right, %acc_20, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_24 = ttng.warp_group_dot_wait %acc_23 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_25 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_26 = ttg.memdesc_index %barriers[%nxt_25] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_27 = ttg.memdesc_index %tiles[%nxt_25] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_26, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_27, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_24 : tensor<128x128xf32, #mma>
}
%acc_19 = ttng.warp_group_dot_wait %acc_18 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_19 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_16 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%6 = arith.muli %batch, 768 : i32
%7 = arith.addi %6, %row_12 : i32
ttng.async_tma_copy_local_to_global %C[%7, %col_15] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 768 : i32
%4 = arith.addi %3, %col_15 : i32
ttng.async_tma_copy_local_to_global %C[%4, %row_12] %mirror_16 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .mlpGram, .baseline => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XTX_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%num_pid_m = arith.addi %K, 127 : i32
%num_pid_m_1 = arith.divsi %num_pid_m, 128 : i32
%batch_idx = arith.muli %num_pid_m_1, %num_pid_m_1 : i32
%batch_idx_2 = arith.divsi %pid, %batch_idx : i32
%pid_3 = arith.remsi %pid, %batch_idx : i32
%pid_m = arith.divsi %pid_3, %num_pid_m_1 : i32
%pid_n = arith.remsi %pid_3, %num_pid_m_1 : i32
%ij = arith.muli %pid_m, %num_pid_m_1 : i32
%ij_4 = arith.addi %ij, %pid_n : i32
%size_gj = arith.muli %num_pid_m_1, 8 : i32
%group_id = arith.divsi %ij_4, %size_gj : i32
%off_i = arith.muli %group_id, 8 : i32
%size_g = arith.subi %num_pid_m_1, %off_i : i32
%size_g_5 = arith.minsi %size_g, 8 : i32
%ij_6 = arith.remsi %ij_4, %size_gj : i32
%new_i = arith.remsi %ij_6, %size_g_5 : i32
%new_i_7 = arith.addi %off_i, %new_i : i32
%new_j = arith.divsi %ij_6, %size_g_5 : i32
%m_idx = arith.muli %new_i_7, 128 : i32
%n_idx = arith.muli %new_j, 128 : i32
%skip_block_above_diag = arith.addi %m_idx, 128 : i32
%skip_block_above_diag_8 = arith.cmpi sle, %skip_block_above_diag, %n_idx : i32
cf.cond_br %skip_block_above_diag_8, ^bb1, ^bb2
^bb1: // pred: ^bb0
tt.return
^bb2: // pred: ^bb0
%A_ptr_9 = arith.muli %batch_idx_2, %a_stride_b : i32
%A_ptr_10 = tt.addptr %A_ptr, %A_ptr_9 : !tt.ptr<bf16>, i32
%C_ptr_11 = arith.muli %batch_idx_2, %c_stride_b : i32
%C_ptr_12 = tt.addptr %C_ptr, %C_ptr_11 : !tt.ptr<bf16>, i32
%offs_k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_14 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_15 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_16 = arith.addi %offs_k_14, %offs_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_17 = arith.addi %offs_k_15, %offs_k_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_18 = tt.splat %K : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_19 = arith.remsi %offs_k_16, %offs_k_18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_20 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_21 = arith.addi %offs_n, %offs_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_22 = arith.addi %offs_n_20, %offs_k_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_23 = arith.remsi %offs_n_21, %offs_k_18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%at_ptrs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%at_ptrs_24 = tt.expand_dims %at_ptrs {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%at_ptrs_25 = tt.splat %a_stride_r : i32 -> tensor<64x1xi32, #blocked>
%at_ptrs_26 = arith.muli %at_ptrs_24, %at_ptrs_25 : tensor<64x1xi32, #blocked>
%at_ptrs_27 = tt.expand_dims %offs_k_19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%at_ptrs_28 = tt.broadcast %at_ptrs_26 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%at_ptrs_29 = tt.broadcast %at_ptrs_27 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%at_ptrs_30 = arith.addi %at_ptrs_28, %at_ptrs_29 : tensor<64x128xi32, #blocked>
%at_ptrs_31 = tt.splat %A_ptr_10 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%at_ptrs_32 = tt.addptr %at_ptrs_31, %at_ptrs_30 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs = tt.expand_dims %offs_n_23 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_ptrs_33 = tt.broadcast %a_ptrs : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%a_ptrs_34 = arith.addi %at_ptrs_28, %a_ptrs_33 : tensor<64x128xi32, #blocked>
%a_ptrs_35 = tt.addptr %at_ptrs_31, %a_ptrs_34 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%0 = arith.addi %M, 63 : i32
%1 = arith.divsi %0, 64 : i32
%at_ptrs_36 = arith.muli %a_stride_r, 64 : i32
%at_ptrs_37 = tt.splat %at_ptrs_36 : i32 -> tensor<64x128xi32, #blocked>
%at = ttg.local_alloc : () -> !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%a = ttg.local_alloc : () -> !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %1, 0 : i32
%at_38 = tt.splat %M : i32 -> tensor<64x1xi32, #blocked>
%at_39 = arith.cmpi slt, %at_ptrs_24, %at_38 : tensor<64x1xi32, #blocked>
%at_40 = tt.broadcast %at_39 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_41 = ttg.memdesc_index %at[%c0_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_42 = tt.splat %accumulator : i1 -> tensor<64x128xi1, #blocked>
%accumulator_43 = arith.andi %accumulator_42, %at_40 : tensor<64x128xi1, #blocked>
%at_44 = ttg.async_copy_global_to_local %at_ptrs_32, %at_41 mask %accumulator_43 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_45 = ttg.async_commit_group tokens %at_44
%a_46 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_47 = ttg.async_copy_global_to_local %a_ptrs_35, %a_46 mask %accumulator_43 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_48 = ttg.async_commit_group tokens %a_47
%accumulator_49 = arith.cmpi sgt, %1, 1 : i32
%at_ptrs_50 = tt.addptr %at_ptrs_32, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_51 = tt.addptr %a_ptrs_35, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%m_remaining = arith.subi %M, 64 : i32
%at_52 = tt.splat %m_remaining : i32 -> tensor<64x1xi32, #blocked>
%at_53 = arith.cmpi slt, %at_ptrs_24, %at_52 : tensor<64x1xi32, #blocked>
%at_54 = tt.broadcast %at_53 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_55 = ttg.memdesc_index %at[%c1_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_56 = tt.splat %accumulator_49 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_57 = arith.andi %accumulator_56, %at_54 : tensor<64x128xi1, #blocked>
%at_58 = ttg.async_copy_global_to_local %at_ptrs_50, %at_55 mask %accumulator_57 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_59 = ttg.async_commit_group tokens %at_58
%a_60 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_61 = ttg.async_copy_global_to_local %a_ptrs_51, %a_60 mask %accumulator_57 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_62 = ttg.async_commit_group tokens %a_61
%accumulator_63 = arith.cmpi sgt, %1, 2 : i32
%at_ptrs_64 = tt.addptr %at_ptrs_50, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_65 = tt.addptr %a_ptrs_51, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%m_remaining_66 = arith.subi %M, 128 : i32
%at_67 = tt.splat %m_remaining_66 : i32 -> tensor<64x1xi32, #blocked>
%at_68 = arith.cmpi slt, %at_ptrs_24, %at_67 : tensor<64x1xi32, #blocked>
%at_69 = tt.broadcast %at_68 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_70 = ttg.memdesc_index %at[%c2_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_71 = tt.splat %accumulator_63 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_72 = arith.andi %accumulator_71, %at_69 : tensor<64x128xi1, #blocked>
%at_73 = ttg.async_copy_global_to_local %at_ptrs_64, %at_70 mask %accumulator_72 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_74 = ttg.async_commit_group tokens %at_73
%a_75 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_76 = ttg.async_copy_global_to_local %a_ptrs_65, %a_75 mask %accumulator_72 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_77 = ttg.async_commit_group tokens %a_76
%accumulator_78:11 = scf.for %accumulator_105 = 0 to %1 step 1 iter_args(%arg9 = %cst, %at_ptrs_106 = %at_ptrs_64, %a_ptrs_107 = %a_ptrs_65, %arg12 = 2, %arg13 = %c-1_i32, %at_108 = %at_45, %at_109 = %at_59, %at_110 = %at_74, %a_111 = %a_48, %a_112 = %a_62, %a_113 = %a_77) -> (tensor<128x128xf32, #mma>, tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_114 = arith.subi %1, 3 : i32
%accumulator_115 = arith.cmpi slt, %accumulator_105, %accumulator_114 : i32
%accumulator_116 = arith.addi %arg13, 1 : i32
%accumulator_117 = arith.cmpi sge, %accumulator_116, 4 : i32
%accumulator_118 = arith.select %accumulator_117, 0, %accumulator_116 : i32
%at_119 = ttg.async_wait %at_108, %a_111 {num = 4 : i32}
%at_120 = ttg.memdesc_index %at[%accumulator_118] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_121 = ttg.memdesc_trans %at_120 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
%a_122 = ttg.memdesc_index %a[%accumulator_118] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_123 = ttng.warp_group_dot %accumulator_121, %a_122, %arg9 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_124:3 = ttng.warp_group_dot_wait %accumulator_123, %accumulator_121, %a_122 {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%at_ptrs_125 = tt.addptr %at_ptrs_106, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_126 = tt.addptr %a_ptrs_107, %at_ptrs_37 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%accumulator_127 = arith.addi %arg12, 1 : i32
%accumulator_128 = arith.cmpi sge, %accumulator_127, 4 : i32
%accumulator_129 = arith.select %accumulator_128, 0, %accumulator_127 : i32
%accumulator_130 = arith.addi %accumulator_105, 3 : i32
%m_remaining_131 = arith.muli %accumulator_130, 64 : i32
%m_remaining_132 = arith.subi %M, %m_remaining_131 : i32
%at_133 = tt.splat %m_remaining_132 : i32 -> tensor<64x1xi32, #blocked>
%at_134 = arith.cmpi slt, %at_ptrs_24, %at_133 : tensor<64x1xi32, #blocked>
%at_135 = tt.broadcast %at_134 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_136 = ttg.memdesc_index %at[%accumulator_129] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_137 = tt.splat %accumulator_115 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_138 = arith.andi %accumulator_137, %at_135 : tensor<64x128xi1, #blocked>
%at_139 = ttg.async_copy_global_to_local %at_ptrs_125, %at_136 mask %accumulator_138 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_140 = ttg.async_commit_group tokens %at_139
%a_141 = ttg.memdesc_index %a[%accumulator_129] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_142 = ttg.async_copy_global_to_local %a_ptrs_126, %a_141 mask %accumulator_138 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_143 = ttg.async_commit_group tokens %a_142
scf.yield %accumulator_124#0, %at_ptrs_125, %a_ptrs_126, %accumulator_129, %accumulator_118, %at_109, %at_110, %at_140, %a_112, %a_113, %a_143 : tensor<128x128xf32, #mma>, tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_79 = ttng.warp_group_dot_wait %accumulator_78#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_80 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %a : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
ttg.local_dealloc %at : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_79 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%2 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_k_17 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_81 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_82 = arith.muli %c_ptrs, %c_ptrs_81 : tensor<128x1xi32, #blocked>
%c_ptrs_83 = tt.expand_dims %offs_n_21 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_84 = tt.broadcast %c_ptrs_82 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_85 = tt.broadcast %c_ptrs_83 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_86 = arith.addi %c_ptrs_84, %c_ptrs_85 : tensor<128x128xi32, #blocked>
%c_ptrs_87 = tt.splat %C_ptr_12 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_88 = tt.addptr %c_ptrs_87, %c_ptrs_86 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask = tt.splat %K : i32 -> tensor<128x1xi32, #blocked>
%c_mask_89 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked>
%c_mask_90 = tt.splat %K : i32 -> tensor<1x128xi32, #blocked>
%c_mask_91 = arith.cmpi slt, %c_ptrs_83, %c_mask_90 : tensor<1x128xi32, #blocked>
%c_mask_92 = tt.broadcast %c_mask_89 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_93 = tt.broadcast %c_mask_91 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_94 = arith.andi %c_mask_92, %c_mask_93 : tensor<128x128xi1, #blocked>
%3 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
tt.store %c_ptrs_88, %3, %c_mask_94 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_95 = arith.muli %c_ptrs_t, %c_ptrs_81 : tensor<128x1xi32, #blocked>
%c_ptrs_t_96 = tt.expand_dims %offs_k_16 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_97 = tt.broadcast %c_ptrs_t_95 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_98 = tt.broadcast %c_ptrs_t_96 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_99 = arith.addi %c_ptrs_t_97, %c_ptrs_t_98 : tensor<128x128xi32, #blocked>
%c_ptrs_t_100 = tt.addptr %c_ptrs_87, %c_ptrs_t_99 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked>
%c_mask_t_101 = arith.cmpi slt, %c_ptrs_t_96, %c_mask_90 : tensor<1x128xi32, #blocked>
%c_mask_t_102 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_103 = tt.broadcast %c_mask_t_101 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_104 = arith.andi %c_mask_t_102, %c_mask_t_103 : tensor<128x128xi1, #blocked>
%4 = ttg.convert_layout %2 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
tt.store %c_ptrs_t_100, %4, %c_mask_t_104 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .mlpGram, .compact => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @XTX_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_1 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_1 : i32 to i64
%row_2 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_3 = arith.andi %row_2, %c7_i64 : i64
%row_4 = arith.trunci %row_3 : i64 to i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_5 = arith.andi %col, %c7_i64 : i64
%col_6 = arith.trunci %col_5 : i64 to i32
%0 = arith.divsi %pid, 21 : i32
%1 = arith.muli %row_4, 128 : i32
%2 = arith.muli %col_6, 128 : i32
%A_ptr_7 = arith.muli %0, %a_stride_b : i32
%A_ptr_8 = tt.addptr %A_ptr, %A_ptr_7 : !tt.ptr<bf16>, i32
%C_ptr_9 = arith.muli %0, %c_stride_b : i32
%C_ptr_10 = tt.addptr %C_ptr, %C_ptr_9 : !tt.ptr<bf16>, i32
%offs_k = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_12 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_13 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_14 = arith.addi %offs_k_12, %offs_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_15 = arith.addi %offs_k_13, %offs_k_11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_k_16 = tt.splat %K : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_k_17 = arith.remsi %offs_k_14, %offs_k_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_18 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_19 = arith.addi %offs_n, %offs_k : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_20 = arith.addi %offs_n_18, %offs_k_11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_21 = arith.remsi %offs_n_19, %offs_k_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%at_ptrs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%at_ptrs_22 = tt.expand_dims %at_ptrs {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%at_ptrs_23 = tt.splat %a_stride_r : i32 -> tensor<64x1xi32, #blocked>
%at_ptrs_24 = arith.muli %at_ptrs_22, %at_ptrs_23 : tensor<64x1xi32, #blocked>
%at_ptrs_25 = tt.expand_dims %offs_k_17 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%at_ptrs_26 = tt.broadcast %at_ptrs_24 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%at_ptrs_27 = tt.broadcast %at_ptrs_25 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%at_ptrs_28 = arith.addi %at_ptrs_26, %at_ptrs_27 : tensor<64x128xi32, #blocked>
%at_ptrs_29 = tt.splat %A_ptr_8 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%at_ptrs_30 = tt.addptr %at_ptrs_29, %at_ptrs_28 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs = tt.expand_dims %offs_n_21 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_ptrs_31 = tt.broadcast %a_ptrs : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%a_ptrs_32 = arith.addi %at_ptrs_26, %a_ptrs_31 : tensor<64x128xi32, #blocked>
%a_ptrs_33 = tt.addptr %at_ptrs_29, %a_ptrs_32 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%3 = arith.addi %M, 63 : i32
%4 = arith.divsi %3, 64 : i32
%at_ptrs_34 = arith.muli %a_stride_r, 64 : i32
%at_ptrs_35 = tt.splat %at_ptrs_34 : i32 -> tensor<64x128xi32, #blocked>
%at = ttg.local_alloc : () -> !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%a = ttg.local_alloc : () -> !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %4, 0 : i32
%at_36 = tt.splat %M : i32 -> tensor<64x1xi32, #blocked>
%at_37 = arith.cmpi slt, %at_ptrs_22, %at_36 : tensor<64x1xi32, #blocked>
%at_38 = tt.broadcast %at_37 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_39 = ttg.memdesc_index %at[%c0_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_40 = tt.splat %accumulator : i1 -> tensor<64x128xi1, #blocked>
%accumulator_41 = arith.andi %accumulator_40, %at_38 : tensor<64x128xi1, #blocked>
%at_42 = ttg.async_copy_global_to_local %at_ptrs_30, %at_39 mask %accumulator_41 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_43 = ttg.async_commit_group tokens %at_42
%a_44 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_45 = ttg.async_copy_global_to_local %a_ptrs_33, %a_44 mask %accumulator_41 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_46 = ttg.async_commit_group tokens %a_45
%accumulator_47 = arith.cmpi sgt, %4, 1 : i32
%at_ptrs_48 = tt.addptr %at_ptrs_30, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_49 = tt.addptr %a_ptrs_33, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%m_remaining = arith.subi %M, 64 : i32
%at_50 = tt.splat %m_remaining : i32 -> tensor<64x1xi32, #blocked>
%at_51 = arith.cmpi slt, %at_ptrs_22, %at_50 : tensor<64x1xi32, #blocked>
%at_52 = tt.broadcast %at_51 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_53 = ttg.memdesc_index %at[%c1_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_54 = tt.splat %accumulator_47 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_55 = arith.andi %accumulator_54, %at_52 : tensor<64x128xi1, #blocked>
%at_56 = ttg.async_copy_global_to_local %at_ptrs_48, %at_53 mask %accumulator_55 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_57 = ttg.async_commit_group tokens %at_56
%a_58 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_59 = ttg.async_copy_global_to_local %a_ptrs_49, %a_58 mask %accumulator_55 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_60 = ttg.async_commit_group tokens %a_59
%accumulator_61 = arith.cmpi sgt, %4, 2 : i32
%at_ptrs_62 = tt.addptr %at_ptrs_48, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_63 = tt.addptr %a_ptrs_49, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%m_remaining_64 = arith.subi %M, 128 : i32
%at_65 = tt.splat %m_remaining_64 : i32 -> tensor<64x1xi32, #blocked>
%at_66 = arith.cmpi slt, %at_ptrs_22, %at_65 : tensor<64x1xi32, #blocked>
%at_67 = tt.broadcast %at_66 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_68 = ttg.memdesc_index %at[%c2_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_69 = tt.splat %accumulator_61 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_70 = arith.andi %accumulator_69, %at_67 : tensor<64x128xi1, #blocked>
%at_71 = ttg.async_copy_global_to_local %at_ptrs_62, %at_68 mask %accumulator_70 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_72 = ttg.async_commit_group tokens %at_71
%a_73 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_74 = ttg.async_copy_global_to_local %a_ptrs_63, %a_73 mask %accumulator_70 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_75 = ttg.async_commit_group tokens %a_74
%accumulator_76:11 = scf.for %accumulator_103 = 0 to %4 step 1 iter_args(%accumulator_104 = %cst, %at_ptrs_105 = %at_ptrs_62, %a_ptrs_106 = %a_ptrs_63, %arg12 = 2, %accumulator_107 = %c-1_i32, %at_108 = %at_43, %at_109 = %at_57, %at_110 = %at_72, %a_111 = %a_46, %a_112 = %a_60, %a_113 = %a_75) -> (tensor<128x128xf32, #mma>, tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_114 = arith.subi %4, 3 : i32
%accumulator_115 = arith.cmpi slt, %accumulator_103, %accumulator_114 : i32
%accumulator_116 = arith.addi %accumulator_107, 1 : i32
%accumulator_117 = arith.cmpi sge, %accumulator_116, 4 : i32
%accumulator_118 = arith.select %accumulator_117, 0, %accumulator_116 : i32
%at_119 = ttg.async_wait %at_108, %a_111 {num = 4 : i32}
%at_120 = ttg.memdesc_index %at[%accumulator_118] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_121 = ttg.memdesc_trans %at_120 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>
%a_122 = ttg.memdesc_index %a[%accumulator_118] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_123 = ttng.warp_group_dot %accumulator_121, %a_122, %accumulator_104 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_124:3 = ttng.warp_group_dot_wait %accumulator_123, %accumulator_121, %a_122 {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%at_ptrs_125 = tt.addptr %at_ptrs_105, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%a_ptrs_126 = tt.addptr %a_ptrs_106, %at_ptrs_35 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%accumulator_127 = arith.addi %arg12, 1 : i32
%accumulator_128 = arith.cmpi sge, %accumulator_127, 4 : i32
%accumulator_129 = arith.select %accumulator_128, 0, %accumulator_127 : i32
%accumulator_130 = arith.addi %accumulator_103, 3 : i32
%m_remaining_131 = arith.muli %accumulator_130, 64 : i32
%m_remaining_132 = arith.subi %M, %m_remaining_131 : i32
%at_133 = tt.splat %m_remaining_132 : i32 -> tensor<64x1xi32, #blocked>
%at_134 = arith.cmpi slt, %at_ptrs_22, %at_133 : tensor<64x1xi32, #blocked>
%at_135 = tt.broadcast %at_134 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
%at_136 = ttg.memdesc_index %at[%accumulator_129] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%accumulator_137 = tt.splat %accumulator_115 : i1 -> tensor<64x128xi1, #blocked>
%accumulator_138 = arith.andi %accumulator_137, %at_135 : tensor<64x128xi1, #blocked>
%at_139 = ttg.async_copy_global_to_local %at_ptrs_125, %at_136 mask %accumulator_138 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%at_140 = ttg.async_commit_group tokens %at_139
%a_141 = ttg.memdesc_index %a[%accumulator_129] : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared, #smem, mutable>
%a_142 = ttg.async_copy_global_to_local %a_ptrs_126, %a_141 mask %accumulator_138 other %cst_0 {contiguity = 8 : i32} : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #smem, mutable>
%a_143 = ttg.async_commit_group tokens %a_142
scf.yield %accumulator_124#0, %at_ptrs_125, %a_ptrs_126, %accumulator_129, %accumulator_118, %at_109, %at_110, %at_140, %a_112, %a_113, %a_143 : tensor<128x128xf32, #mma>, tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128x!tt.ptr<bf16>, #blocked>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_77 = ttng.warp_group_dot_wait %accumulator_76#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_78 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %a : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
ttg.local_dealloc %at : !ttg.memdesc<4x64x128xbf16, #shared, #smem, mutable>
%output = arith.truncf %accumulator_77 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%5 = tt.trans %output {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%c_ptrs = tt.expand_dims %offs_k_15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_79 = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_80 = arith.muli %c_ptrs, %c_ptrs_79 : tensor<128x1xi32, #blocked>
%c_ptrs_81 = tt.expand_dims %offs_n_19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_82 = tt.broadcast %c_ptrs_80 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_83 = tt.broadcast %c_ptrs_81 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_84 = arith.addi %c_ptrs_82, %c_ptrs_83 : tensor<128x128xi32, #blocked>
%c_ptrs_85 = tt.splat %C_ptr_10 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_86 = tt.addptr %c_ptrs_85, %c_ptrs_84 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask = tt.splat %K : i32 -> tensor<128x1xi32, #blocked>
%c_mask_87 = arith.cmpi slt, %c_ptrs, %c_mask : tensor<128x1xi32, #blocked>
%c_mask_88 = tt.splat %K : i32 -> tensor<1x128xi32, #blocked>
%c_mask_89 = arith.cmpi slt, %c_ptrs_81, %c_mask_88 : tensor<1x128xi32, #blocked>
%c_mask_90 = tt.broadcast %c_mask_87 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_91 = tt.broadcast %c_mask_89 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_92 = arith.andi %c_mask_90, %c_mask_91 : tensor<128x128xi1, #blocked>
%6 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
tt.store %c_ptrs_86, %6, %c_mask_92 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_93 = arith.muli %c_ptrs_t, %c_ptrs_79 : tensor<128x1xi32, #blocked>
%c_ptrs_t_94 = tt.expand_dims %offs_k_14 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_95 = tt.broadcast %c_ptrs_t_93 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_96 = tt.broadcast %c_ptrs_t_94 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_97 = arith.addi %c_ptrs_t_95, %c_ptrs_t_96 : tensor<128x128xi32, #blocked>
%c_ptrs_t_98 = tt.addptr %c_ptrs_85, %c_ptrs_t_97 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %c_mask : tensor<128x1xi32, #blocked>
%c_mask_t_99 = arith.cmpi slt, %c_ptrs_t_94, %c_mask_88 : tensor<1x128xi32, #blocked>
%c_mask_t_100 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_101 = tt.broadcast %c_mask_t_99 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_102 = arith.andi %c_mask_t_100, %c_mask_t_101 : tensor<128x128xi1, #blocked>
%7 = ttg.convert_layout %5 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #blocked>
tt.store %c_ptrs_t_98, %7, %c_mask_t_102 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .mlpGram, .full => r#"
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @polar(%A: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %A_0: i32, %A_1: i32, %A_2: i64, %A_3: i64, %C: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %C_4: i32, %C_5: i32, %C_6: i64, %C_7: i64) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_8 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_8 : i32 to i64
%row_9 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_10 = arith.andi %row_9, %c7_i64 : i64
%row_11 = arith.trunci %row_10 : i64 to i32
%row_12 = arith.muli %row_11, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_13 = arith.andi %col, %c7_i64 : i64
%col_14 = arith.trunci %col_13 : i64 to i32
%col_15 = arith.muli %col_14, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_12, %col_15 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 2816 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%acc, %row_12] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%acc, %col_15] %acc_18, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_19 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_20 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_21 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_22 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_22, %phase_21, %true deps %base, %acc_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_23 = ttng.warp_group_dot %left, %acc_18, %acc_20, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_24 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_25 = ttg.memdesc_index %barriers[%nxt_24] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_26 = ttg.memdesc_index %tiles[%nxt_24] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_25, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %A[%coords, %row_12] %acc_26, %acc_25, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_27 = ttng.warp_group_dot_wait %acc_23 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %A[%coords, %col_15] %acc_18, %acc_25, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_27 : tensor<128x128xf32, #mma>
}
scf.yield %acc_19 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 2816 : i32
%acc_17 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%acc, %row_12] %acc_17, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_18 = scf.for %chunk = 0 to 22 step 1 iter_args(%acc_20 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_21 = arith.remsi %phase, 2 : i32
%base = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_22 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_22, %phase_21, %true deps %base, %base : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%left = ttg.memdesc_trans %base {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_23 = ttng.warp_group_dot %left, %base, %acc_20, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_24 = ttng.warp_group_dot_wait %acc_23 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_25 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 22 : i32
%k = arith.muli %nxt, 128 : i32
%acc_26 = ttg.memdesc_index %barriers[%nxt_25] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_27 = ttg.memdesc_index %tiles[%nxt_25] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_26, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %k : i32
ttng.async_tma_copy_global_to_local %A[%coords, %row_12] %acc_27, %acc_26, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_24 : tensor<128x128xf32, #mma>
}
%acc_19 = ttng.warp_group_dot_wait %acc_18 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_19 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%words = arith.truncf %2 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_16 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%6 = arith.muli %batch, 768 : i32
%7 = arith.addi %6, %row_12 : i32
ttng.async_tma_copy_local_to_global %C[%7, %col_15] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 768 : i32
%4 = arith.addi %3, %col_15 : i32
ttng.async_tma_copy_local_to_global %C[%4, %row_12] %mirror_16 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .qkPolynomial, .baseline => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @ba_plus_cAA_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
%cst_2 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%num_pid_m = arith.addi %M, 127 : i32
%num_pid_m_3 = arith.divsi %num_pid_m, 128 : i32
%batch_idx = arith.muli %num_pid_m_3, %num_pid_m_3 : i32
%batch_idx_4 = arith.divsi %pid, %batch_idx : i32
%pid_5 = arith.remsi %pid, %batch_idx : i32
%pid_m = arith.divsi %pid_5, %num_pid_m_3 : i32
%pid_n = arith.remsi %pid_5, %num_pid_m_3 : i32
%ij = arith.muli %pid_m, %num_pid_m_3 : i32
%ij_6 = arith.addi %ij, %pid_n : i32
%size_gj = arith.muli %num_pid_m_3, 8 : i32
%group_id = arith.divsi %ij_6, %size_gj : i32
%off_i = arith.muli %group_id, 8 : i32
%size_g = arith.subi %num_pid_m_3, %off_i : i32
%size_g_7 = arith.minsi %size_g, 8 : i32
%ij_8 = arith.remsi %ij_6, %size_gj : i32
%new_i = arith.remsi %ij_8, %size_g_7 : i32
%new_i_9 = arith.addi %off_i, %new_i : i32
%new_j = arith.divsi %ij_8, %size_g_7 : i32
%m_idx = arith.muli %new_i_9, 128 : i32
%n_idx = arith.muli %new_j, 128 : i32
%skip_block_above_diag = arith.addi %m_idx, 128 : i32
%skip_block_above_diag_10 = arith.cmpi sle, %skip_block_above_diag, %n_idx : i32
cf.cond_br %skip_block_above_diag_10, ^bb1, ^bb2
^bb1: // pred: ^bb0
tt.return
^bb2: // pred: ^bb0
%A_ptr_11 = arith.muli %batch_idx_4, %a_stride_b : i32
%A_ptr_12 = tt.addptr %A_ptr, %A_ptr_11 : !tt.ptr<bf16>, i32
%C_ptr_13 = arith.muli %batch_idx_4, %c_stride_b : i32
%C_ptr_14 = tt.addptr %C_ptr, %C_ptr_13 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_17 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_18 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_19 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_21 = arith.addi %offs_m_18, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_22 = arith.addi %offs_m_19, %offs_m_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_23 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_24 = arith.remsi %offs_m_20, %offs_m_23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_25 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_26 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_28 = arith.addi %offs_n_25, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_29 = arith.addi %offs_n_26, %offs_m_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_30 = arith.remsi %offs_n_27, %offs_m_23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%a_ptrs = tt.expand_dims %offs_m_24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%a_ptrs_31 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%a_ptrs_32 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_33 = arith.muli %a_ptrs, %a_ptrs_31 : tensor<128x1xi32, #blocked1>
%a_ptrs_34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%a_ptrs_35 = tt.expand_dims %a_ptrs_34 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%a_ptrs_36 = tt.broadcast %a_ptrs_33 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_37 = tt.broadcast %a_ptrs_35 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_38 = arith.addi %a_ptrs_36, %a_ptrs_37 : tensor<128x64xi32, #blocked1>
%a_ptrs_39 = tt.splat %A_ptr_12 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
%a_ptrs_40 = tt.addptr %a_ptrs_39, %a_ptrs_38 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs = tt.expand_dims %offs_n_30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%at_ptrs_41 = arith.muli %at_ptrs, %a_ptrs_31 : tensor<128x1xi32, #blocked1>
%at_ptrs_42 = tt.broadcast %at_ptrs_41 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%at_ptrs_43 = arith.addi %at_ptrs_42, %a_ptrs_37 : tensor<128x64xi32, #blocked1>
%at_ptrs_44 = tt.addptr %a_ptrs_39, %at_ptrs_43 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%0 = arith.addi %M, 63 : i32
%1 = arith.divsi %0, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %1, 0 : i32
%a_45 = tt.splat %M : i32 -> tensor<1x64xi32, #blocked1>
%a_46 = arith.cmpi slt, %a_ptrs_35, %a_45 : tensor<1x64xi32, #blocked1>
%a_47 = tt.broadcast %a_46 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_48 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_49 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_50 = arith.andi %accumulator_49, %a_47 : tensor<128x64xi1, #blocked1>
%a_51 = ttg.async_copy_global_to_local %a_ptrs_40, %a_48 mask %accumulator_50 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_52 = ttg.async_commit_group tokens %a_51
%at_temp_53 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_54 = ttg.async_copy_global_to_local %at_ptrs_44, %at_temp_53 mask %accumulator_50 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_55 = ttg.async_commit_group tokens %at_temp_54
%accumulator_56 = arith.cmpi sgt, %1, 1 : i32
%a_ptrs_57 = tt.addptr %a_ptrs_40, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_58 = tt.addptr %at_ptrs_44, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining = arith.subi %M, 64 : i32
%a_59 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked1>
%a_60 = arith.cmpi slt, %a_ptrs_35, %a_59 : tensor<1x64xi32, #blocked1>
%a_61 = tt.broadcast %a_60 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_62 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_63 = tt.splat %accumulator_56 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_64 = arith.andi %accumulator_63, %a_61 : tensor<128x64xi1, #blocked1>
%a_65 = ttg.async_copy_global_to_local %a_ptrs_57, %a_62 mask %accumulator_64 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_66 = ttg.async_commit_group tokens %a_65
%at_temp_67 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_68 = ttg.async_copy_global_to_local %at_ptrs_58, %at_temp_67 mask %accumulator_64 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_69 = ttg.async_commit_group tokens %at_temp_68
%accumulator_70 = arith.cmpi sgt, %1, 2 : i32
%a_ptrs_71 = tt.addptr %a_ptrs_57, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_72 = tt.addptr %at_ptrs_58, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining_73 = arith.subi %M, 128 : i32
%a_74 = tt.splat %k_remaining_73 : i32 -> tensor<1x64xi32, #blocked1>
%a_75 = arith.cmpi slt, %a_ptrs_35, %a_74 : tensor<1x64xi32, #blocked1>
%a_76 = tt.broadcast %a_75 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_77 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_78 = tt.splat %accumulator_70 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_79 = arith.andi %accumulator_78, %a_76 : tensor<128x64xi1, #blocked1>
%a_80 = ttg.async_copy_global_to_local %a_ptrs_71, %a_77 mask %accumulator_79 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_81 = ttg.async_commit_group tokens %a_80
%at_temp_82 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_83 = ttg.async_copy_global_to_local %at_ptrs_72, %at_temp_82 mask %accumulator_79 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_84 = ttg.async_commit_group tokens %at_temp_83
%accumulator_85:11 = scf.for %accumulator_125 = 0 to %1 step 1 iter_args(%arg10 = %cst, %a_ptrs_126 = %a_ptrs_71, %at_ptrs_127 = %at_ptrs_72, %arg13 = 2, %arg14 = %c-1_i32, %a_128 = %a_52, %a_129 = %a_66, %a_130 = %a_81, %at_temp_131 = %at_temp_55, %at_temp_132 = %at_temp_69, %at_temp_133 = %at_temp_84) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_134 = arith.subi %1, 3 : i32
%accumulator_135 = arith.cmpi slt, %accumulator_125, %accumulator_134 : i32
%accumulator_136 = arith.addi %arg14, 1 : i32
%accumulator_137 = arith.cmpi sge, %accumulator_136, 4 : i32
%accumulator_138 = arith.select %accumulator_137, 0, %accumulator_136 : i32
%a_139 = ttg.async_wait %a_128, %at_temp_131 {num = 4 : i32}
%a_140 = ttg.memdesc_index %a[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_141 = ttg.memdesc_index %at_temp[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_141 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_142 = ttng.warp_group_dot %a_140, %at, %arg10 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_143:3 = ttng.warp_group_dot_wait %accumulator_142, %a_140, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_144 = tt.addptr %a_ptrs_126, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_145 = tt.addptr %at_ptrs_127, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%accumulator_146 = arith.addi %arg13, 1 : i32
%accumulator_147 = arith.cmpi sge, %accumulator_146, 4 : i32
%accumulator_148 = arith.select %accumulator_147, 0, %accumulator_146 : i32
%accumulator_149 = arith.addi %accumulator_125, 3 : i32
%k_remaining_150 = arith.muli %accumulator_149, 64 : i32
%k_remaining_151 = arith.subi %M, %k_remaining_150 : i32
%a_152 = tt.splat %k_remaining_151 : i32 -> tensor<1x64xi32, #blocked1>
%a_153 = arith.cmpi slt, %a_ptrs_35, %a_152 : tensor<1x64xi32, #blocked1>
%a_154 = tt.broadcast %a_153 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_155 = ttg.memdesc_index %a[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_156 = tt.splat %accumulator_135 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_157 = arith.andi %accumulator_156, %a_154 : tensor<128x64xi1, #blocked1>
%a_158 = ttg.async_copy_global_to_local %a_ptrs_144, %a_155 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_159 = ttg.async_commit_group tokens %a_158
%at_temp_160 = ttg.memdesc_index %at_temp[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_161 = ttg.async_copy_global_to_local %at_ptrs_145, %at_temp_160 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_162 = ttg.async_commit_group tokens %at_temp_161
scf.yield %accumulator_143#0, %a_ptrs_144, %at_ptrs_145, %accumulator_148, %accumulator_138, %a_129, %a_130, %a_159, %at_temp_132, %at_temp_133, %at_temp_162 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_86 = ttng.warp_group_dot_wait %accumulator_85#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_87 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%a_add_ptrs = tt.expand_dims %offs_m_21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_add_ptrs_88 = arith.muli %a_add_ptrs, %a_ptrs_32 : tensor<128x1xi32, #blocked>
%a_add_ptrs_89 = tt.expand_dims %offs_n_29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_add_ptrs_90 = tt.broadcast %a_add_ptrs_88 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_91 = tt.broadcast %a_add_ptrs_89 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_92 = arith.addi %a_add_ptrs_90, %a_add_ptrs_91 : tensor<128x128xi32, #blocked>
%a_add_ptrs_93 = tt.splat %A_ptr_12 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%a_add_ptrs_94 = tt.addptr %a_add_ptrs_93, %a_add_ptrs_92 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%a_add_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked>
%a_add_mask_95 = arith.cmpi slt, %a_add_ptrs, %a_add_mask : tensor<128x1xi32, #blocked>
%a_add_mask_96 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked>
%a_add_mask_97 = arith.cmpi slt, %a_add_ptrs_89, %a_add_mask_96 : tensor<1x128xi32, #blocked>
%a_add_mask_98 = tt.broadcast %a_add_mask_95 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_99 = tt.broadcast %a_add_mask_97 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_100 = arith.andi %a_add_mask_98, %a_add_mask_99 : tensor<128x128xi1, #blocked>
%a_add = tt.load %a_add_ptrs_94, %a_add_mask_100, %cst_0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%accumulator_101 = ttg.convert_layout %a_add : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%a_add_102 = arith.extf %accumulator_101 : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%accumulator_103 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%accumulator_104 = arith.mulf %accumulator_86, %accumulator_103 : tensor<128x128xf32, #mma>
%accumulator_105 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%accumulator_106 = arith.mulf %a_add_102, %accumulator_105 : tensor<128x128xf32, #mma>
%accumulator_107 = arith.addf %accumulator_104, %accumulator_106 : tensor<128x128xf32, #mma>
%output = arith.truncf %accumulator_107 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%output_108 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked2>
%2 = tt.trans %output_108 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked2> -> tensor<128x128xbf16, #blocked>
%output_109 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
%c_ptrs = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_110 = arith.muli %a_add_ptrs, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_111 = tt.broadcast %c_ptrs_110 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_112 = arith.addi %c_ptrs_111, %a_add_ptrs_91 : tensor<128x128xi32, #blocked>
%c_ptrs_113 = tt.splat %C_ptr_14 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_114 = tt.addptr %c_ptrs_113, %c_ptrs_112 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
tt.store %c_ptrs_114, %output_109, %a_add_mask_100 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_115 = arith.muli %c_ptrs_t, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_t_116 = tt.expand_dims %offs_m_22 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_117 = tt.broadcast %c_ptrs_t_115 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_118 = tt.broadcast %c_ptrs_t_116 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_119 = arith.addi %c_ptrs_t_117, %c_ptrs_t_118 : tensor<128x128xi32, #blocked>
%c_ptrs_t_120 = tt.addptr %c_ptrs_113, %c_ptrs_t_119 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %a_add_mask : tensor<128x1xi32, #blocked>
%c_mask_t_121 = arith.cmpi slt, %c_ptrs_t_116, %a_add_mask_96 : tensor<1x128xi32, #blocked>
%c_mask_t_122 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_123 = tt.broadcast %c_mask_t_121 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_124 = arith.andi %c_mask_t_122, %c_mask_t_123 : tensor<128x128xi1, #blocked>
tt.store %c_ptrs_t_120, %2, %c_mask_t_124 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .qkPolynomial, .compact => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @ba_plus_cAA_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
%cst_2 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_3 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_3 : i32 to i64
%row_4 = arith.shrui %c72_i64, %row : i64
%row_5 = arith.andi %row_4, %c7_i64 : i64
%row_6 = arith.trunci %row_5 : i64 to i32
%col = arith.shrui %c64_i64, %row : i64
%col_7 = arith.andi %col, %c7_i64 : i64
%col_8 = arith.trunci %col_7 : i64 to i32
%0 = arith.divsi %pid, 3 : i32
%1 = arith.muli %row_6, 128 : i32
%2 = arith.muli %col_8, 128 : i32
%A_ptr_9 = arith.muli %0, %a_stride_b : i32
%A_ptr_10 = tt.addptr %A_ptr, %A_ptr_9 : !tt.ptr<bf16>, i32
%C_ptr_11 = arith.muli %0, %c_stride_b : i32
%C_ptr_12 = tt.addptr %C_ptr, %C_ptr_11 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_15 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_16 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_17 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_18 = arith.addi %offs_m_15, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_21 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_22 = arith.remsi %offs_m_18, %offs_m_21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_23 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_24 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_25 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n_23, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n_24, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_28 = arith.remsi %offs_n_25, %offs_m_21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%a_ptrs = tt.expand_dims %offs_m_22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%a_ptrs_29 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%a_ptrs_30 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_31 = arith.muli %a_ptrs, %a_ptrs_29 : tensor<128x1xi32, #blocked1>
%a_ptrs_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%a_ptrs_33 = tt.expand_dims %a_ptrs_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%a_ptrs_34 = tt.broadcast %a_ptrs_31 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_35 = tt.broadcast %a_ptrs_33 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_36 = arith.addi %a_ptrs_34, %a_ptrs_35 : tensor<128x64xi32, #blocked1>
%a_ptrs_37 = tt.splat %A_ptr_10 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
%a_ptrs_38 = tt.addptr %a_ptrs_37, %a_ptrs_36 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs = tt.expand_dims %offs_n_28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%at_ptrs_39 = arith.muli %at_ptrs, %a_ptrs_29 : tensor<128x1xi32, #blocked1>
%at_ptrs_40 = tt.broadcast %at_ptrs_39 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%at_ptrs_41 = arith.addi %at_ptrs_40, %a_ptrs_35 : tensor<128x64xi32, #blocked1>
%at_ptrs_42 = tt.addptr %a_ptrs_37, %at_ptrs_41 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%3 = arith.addi %M, 63 : i32
%4 = arith.divsi %3, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %4, 0 : i32
%a_43 = tt.splat %M : i32 -> tensor<1x64xi32, #blocked1>
%a_44 = arith.cmpi slt, %a_ptrs_33, %a_43 : tensor<1x64xi32, #blocked1>
%a_45 = tt.broadcast %a_44 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_46 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_47 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_48 = arith.andi %accumulator_47, %a_45 : tensor<128x64xi1, #blocked1>
%a_49 = ttg.async_copy_global_to_local %a_ptrs_38, %a_46 mask %accumulator_48 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_50 = ttg.async_commit_group tokens %a_49
%at_temp_51 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_52 = ttg.async_copy_global_to_local %at_ptrs_42, %at_temp_51 mask %accumulator_48 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_53 = ttg.async_commit_group tokens %at_temp_52
%accumulator_54 = arith.cmpi sgt, %4, 1 : i32
%a_ptrs_55 = tt.addptr %a_ptrs_38, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_56 = tt.addptr %at_ptrs_42, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining = arith.subi %M, 64 : i32
%a_57 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked1>
%a_58 = arith.cmpi slt, %a_ptrs_33, %a_57 : tensor<1x64xi32, #blocked1>
%a_59 = tt.broadcast %a_58 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_60 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_61 = tt.splat %accumulator_54 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_62 = arith.andi %accumulator_61, %a_59 : tensor<128x64xi1, #blocked1>
%a_63 = ttg.async_copy_global_to_local %a_ptrs_55, %a_60 mask %accumulator_62 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_64 = ttg.async_commit_group tokens %a_63
%at_temp_65 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_66 = ttg.async_copy_global_to_local %at_ptrs_56, %at_temp_65 mask %accumulator_62 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_67 = ttg.async_commit_group tokens %at_temp_66
%accumulator_68 = arith.cmpi sgt, %4, 2 : i32
%a_ptrs_69 = tt.addptr %a_ptrs_55, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_70 = tt.addptr %at_ptrs_56, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining_71 = arith.subi %M, 128 : i32
%a_72 = tt.splat %k_remaining_71 : i32 -> tensor<1x64xi32, #blocked1>
%a_73 = arith.cmpi slt, %a_ptrs_33, %a_72 : tensor<1x64xi32, #blocked1>
%a_74 = tt.broadcast %a_73 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_75 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_76 = tt.splat %accumulator_68 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_77 = arith.andi %accumulator_76, %a_74 : tensor<128x64xi1, #blocked1>
%a_78 = ttg.async_copy_global_to_local %a_ptrs_69, %a_75 mask %accumulator_77 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_79 = ttg.async_commit_group tokens %a_78
%at_temp_80 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_81 = ttg.async_copy_global_to_local %at_ptrs_70, %at_temp_80 mask %accumulator_77 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_82 = ttg.async_commit_group tokens %at_temp_81
%accumulator_83:11 = scf.for %accumulator_123 = 0 to %4 step 1 iter_args(%accumulator_124 = %cst, %a_ptrs_125 = %a_ptrs_69, %at_ptrs_126 = %at_ptrs_70, %arg13 = 2, %accumulator_127 = %c-1_i32, %a_128 = %a_50, %a_129 = %a_64, %a_130 = %a_79, %at_temp_131 = %at_temp_53, %at_temp_132 = %at_temp_67, %at_temp_133 = %at_temp_82) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_134 = arith.subi %4, 3 : i32
%accumulator_135 = arith.cmpi slt, %accumulator_123, %accumulator_134 : i32
%accumulator_136 = arith.addi %accumulator_127, 1 : i32
%accumulator_137 = arith.cmpi sge, %accumulator_136, 4 : i32
%accumulator_138 = arith.select %accumulator_137, 0, %accumulator_136 : i32
%a_139 = ttg.async_wait %a_128, %at_temp_131 {num = 4 : i32}
%a_140 = ttg.memdesc_index %a[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_141 = ttg.memdesc_index %at_temp[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_141 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_142 = ttng.warp_group_dot %a_140, %at, %accumulator_124 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_143:3 = ttng.warp_group_dot_wait %accumulator_142, %a_140, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_144 = tt.addptr %a_ptrs_125, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_145 = tt.addptr %at_ptrs_126, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%accumulator_146 = arith.addi %arg13, 1 : i32
%accumulator_147 = arith.cmpi sge, %accumulator_146, 4 : i32
%accumulator_148 = arith.select %accumulator_147, 0, %accumulator_146 : i32
%accumulator_149 = arith.addi %accumulator_123, 3 : i32
%k_remaining_150 = arith.muli %accumulator_149, 64 : i32
%k_remaining_151 = arith.subi %M, %k_remaining_150 : i32
%a_152 = tt.splat %k_remaining_151 : i32 -> tensor<1x64xi32, #blocked1>
%a_153 = arith.cmpi slt, %a_ptrs_33, %a_152 : tensor<1x64xi32, #blocked1>
%a_154 = tt.broadcast %a_153 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_155 = ttg.memdesc_index %a[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_156 = tt.splat %accumulator_135 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_157 = arith.andi %accumulator_156, %a_154 : tensor<128x64xi1, #blocked1>
%a_158 = ttg.async_copy_global_to_local %a_ptrs_144, %a_155 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_159 = ttg.async_commit_group tokens %a_158
%at_temp_160 = ttg.memdesc_index %at_temp[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_161 = ttg.async_copy_global_to_local %at_ptrs_145, %at_temp_160 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_162 = ttg.async_commit_group tokens %at_temp_161
scf.yield %accumulator_143#0, %a_ptrs_144, %at_ptrs_145, %accumulator_148, %accumulator_138, %a_129, %a_130, %a_159, %at_temp_132, %at_temp_133, %at_temp_162 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_84 = ttng.warp_group_dot_wait %accumulator_83#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_85 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%a_add_ptrs = tt.expand_dims %offs_m_19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_add_ptrs_86 = arith.muli %a_add_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%a_add_ptrs_87 = tt.expand_dims %offs_n_27 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_add_ptrs_88 = tt.broadcast %a_add_ptrs_86 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_89 = tt.broadcast %a_add_ptrs_87 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_90 = arith.addi %a_add_ptrs_88, %a_add_ptrs_89 : tensor<128x128xi32, #blocked>
%a_add_ptrs_91 = tt.splat %A_ptr_10 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%a_add_ptrs_92 = tt.addptr %a_add_ptrs_91, %a_add_ptrs_90 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%a_add_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked>
%a_add_mask_93 = arith.cmpi slt, %a_add_ptrs, %a_add_mask : tensor<128x1xi32, #blocked>
%a_add_mask_94 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked>
%a_add_mask_95 = arith.cmpi slt, %a_add_ptrs_87, %a_add_mask_94 : tensor<1x128xi32, #blocked>
%a_add_mask_96 = tt.broadcast %a_add_mask_93 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_97 = tt.broadcast %a_add_mask_95 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_98 = arith.andi %a_add_mask_96, %a_add_mask_97 : tensor<128x128xi1, #blocked>
%a_add = tt.load %a_add_ptrs_92, %a_add_mask_98, %cst_0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%accumulator_99 = ttg.convert_layout %a_add : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%a_add_100 = arith.extf %accumulator_99 : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%accumulator_101 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%accumulator_102 = arith.mulf %accumulator_84, %accumulator_101 : tensor<128x128xf32, #mma>
%accumulator_103 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%accumulator_104 = arith.mulf %a_add_100, %accumulator_103 : tensor<128x128xf32, #mma>
%accumulator_105 = arith.addf %accumulator_102, %accumulator_104 : tensor<128x128xf32, #mma>
%output = arith.truncf %accumulator_105 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%output_106 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked2>
%5 = tt.trans %output_106 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked2> -> tensor<128x128xbf16, #blocked>
%output_107 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
%c_ptrs = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_108 = arith.muli %a_add_ptrs, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_109 = tt.broadcast %c_ptrs_108 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_110 = arith.addi %c_ptrs_109, %a_add_ptrs_89 : tensor<128x128xi32, #blocked>
%c_ptrs_111 = tt.splat %C_ptr_12 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_112 = tt.addptr %c_ptrs_111, %c_ptrs_110 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
tt.store %c_ptrs_112, %output_107, %a_add_mask_98 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_26 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_113 = arith.muli %c_ptrs_t, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_t_114 = tt.expand_dims %offs_m_20 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_115 = tt.broadcast %c_ptrs_t_113 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_116 = tt.broadcast %c_ptrs_t_114 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_117 = arith.addi %c_ptrs_t_115, %c_ptrs_t_116 : tensor<128x128xi32, #blocked>
%c_ptrs_t_118 = tt.addptr %c_ptrs_111, %c_ptrs_t_117 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %a_add_mask : tensor<128x1xi32, #blocked>
%c_mask_t_119 = arith.cmpi slt, %c_ptrs_t_114, %a_add_mask_94 : tensor<1x128xi32, #blocked>
%c_mask_t_120 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_121 = tt.broadcast %c_mask_t_119 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_122 = arith.andi %c_mask_t_120, %c_mask_t_121 : tensor<128x128xi1, #blocked>
tt.store %c_ptrs_t_118, %5, %c_mask_t_122 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .qkPolynomial, .full => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @polar(%A: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %A_0: i32, %A_1: i32, %A_2: i64, %A_3: i64, %C: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %C_4: i32, %C_5: i32, %C_6: i64, %C_7: i64, %A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%c72_i64 = arith.constant 72 : i64
%c7_i64 = arith.constant 7 : i64
%c64_i64 = arith.constant 64 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_8 = arith.constant dense<256> : tensor<128x1xi32, #blocked>
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 3 : i32
%shift_9 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_9 : i32 to i64
%row_10 = arith.shrui %c72_i64, %row : i64
%row_11 = arith.andi %row_10, %c7_i64 : i64
%row_12 = arith.trunci %row_11 : i64 to i32
%row_13 = arith.muli %row_12, 128 : i32
%col = arith.shrui %c64_i64, %row : i64
%col_14 = arith.andi %col, %c7_i64 : i64
%col_15 = arith.trunci %col_14 : i64 to i32
%col_16 = arith.muli %col_15, 128 : i32
%batch = arith.divsi %pid, 3 : i32
%offdiag = arith.cmpi ne, %row_13, %col_16 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 256 : i32
%acc_36 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_13 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_36, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_37 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_38 = arith.addi %acc, %col_16 : i32
ttng.async_tma_copy_global_to_local %A[%coords_38, %c0_i32] %acc_37, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = scf.for %chunk = 0 to 2 step 1 iter_args(%acc_40 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_41 = arith.remsi %phase, 2 : i32
%base_42 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_41, %true deps %base_42, %acc_37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_37 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %base_42, %right, %acc_40, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_45 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 2 : i32
%k = arith.muli %nxt, 128 : i32
%acc_46 = ttg.memdesc_index %barriers[%nxt_45] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_47 = ttg.memdesc_index %tiles[%nxt_45] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_46, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_47, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_48 = ttng.warp_group_dot_wait %acc_44 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %A[%coords_38, %k] %acc_37, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_48 : tensor<128x128xf32, #mma>
}
scf.yield %acc_39 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 256 : i32
%acc_36 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_13 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_36, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_37 = scf.for %chunk = 0 to 2 step 1 iter_args(%acc_39 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_40 = arith.remsi %phase, 2 : i32
%base_41 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_42 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_42, %phase_40, %true deps %base_41, %base_41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_41 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_43 = ttng.warp_group_dot %base_41, %right, %acc_39, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_44 = ttng.warp_group_dot_wait %acc_43 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_45 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 2 : i32
%k = arith.muli %nxt, 128 : i32
%acc_46 = ttg.memdesc_index %barriers[%nxt_45] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_47 = ttg.memdesc_index %tiles[%nxt_45] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_46, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_47, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_44 : tensor<128x128xf32, #mma>
}
%acc_38 = ttng.warp_group_dot_wait %acc_37 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_38 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%i = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%j = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %batch, 65536 : i32
%dst = tt.expand_dims %i {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%dst_17 = tt.splat %row_13 : i32 -> tensor<128x1xi32, #blocked>
%dst_18 = arith.addi %dst_17, %dst : tensor<128x1xi32, #blocked>
%dst_19 = arith.muli %dst_18, %cst_8 : tensor<128x1xi32, #blocked>
%dst_20 = tt.expand_dims %j {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%dst_21 = tt.splat %col_16 : i32 -> tensor<1x128xi32, #blocked>
%dst_22 = arith.addi %dst_21, %dst_20 : tensor<1x128xi32, #blocked>
%dst_23 = tt.broadcast %dst_19 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_24 = tt.broadcast %dst_22 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_25 = arith.addi %dst_23, %dst_24 : tensor<128x128xi32, #blocked>
%bias_words = tt.addptr %A_ptr, %base : !tt.ptr<bf16>, i32
%bias_words_26 = tt.splat %bias_words : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias_words_27 = tt.addptr %bias_words_26, %dst_25 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%bias_words_28 = tt.load %bias_words_27 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias = ttg.convert_layout %bias_words_28 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%bias_29 = arith.extf %bias : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%words = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%words_30 = arith.mulf %2, %words : tensor<128x128xf32, #mma>
%words_31 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%words_32 = arith.mulf %bias_29, %words_31 : tensor<128x128xf32, #mma>
%words_33 = arith.addf %words_30, %words_32 : tensor<128x128xf32, #mma>
%words_34 = arith.truncf %words_33 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words_34, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words_34 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_35 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%6 = arith.muli %batch, 256 : i32
%7 = arith.addi %6, %row_13 : i32
ttng.async_tma_copy_local_to_global %C[%7, %col_16] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 256 : i32
%4 = arith.addi %3, %col_16 : i32
ttng.async_tma_copy_local_to_global %C[%4, %row_13] %mirror_35 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#
  | .widePolynomial, .baseline => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @ba_plus_cAA_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
%cst_2 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%num_pid_m = arith.addi %M, 127 : i32
%num_pid_m_3 = arith.divsi %num_pid_m, 128 : i32
%batch_idx = arith.muli %num_pid_m_3, %num_pid_m_3 : i32
%batch_idx_4 = arith.divsi %pid, %batch_idx : i32
%pid_5 = arith.remsi %pid, %batch_idx : i32
%pid_m = arith.divsi %pid_5, %num_pid_m_3 : i32
%pid_n = arith.remsi %pid_5, %num_pid_m_3 : i32
%ij = arith.muli %pid_m, %num_pid_m_3 : i32
%ij_6 = arith.addi %ij, %pid_n : i32
%size_gj = arith.muli %num_pid_m_3, 8 : i32
%group_id = arith.divsi %ij_6, %size_gj : i32
%off_i = arith.muli %group_id, 8 : i32
%size_g = arith.subi %num_pid_m_3, %off_i : i32
%size_g_7 = arith.minsi %size_g, 8 : i32
%ij_8 = arith.remsi %ij_6, %size_gj : i32
%new_i = arith.remsi %ij_8, %size_g_7 : i32
%new_i_9 = arith.addi %off_i, %new_i : i32
%new_j = arith.divsi %ij_8, %size_g_7 : i32
%m_idx = arith.muli %new_i_9, 128 : i32
%n_idx = arith.muli %new_j, 128 : i32
%skip_block_above_diag = arith.addi %m_idx, 128 : i32
%skip_block_above_diag_10 = arith.cmpi sle, %skip_block_above_diag, %n_idx : i32
cf.cond_br %skip_block_above_diag_10, ^bb1, ^bb2
^bb1: // pred: ^bb0
tt.return
^bb2: // pred: ^bb0
%A_ptr_11 = arith.muli %batch_idx_4, %a_stride_b : i32
%A_ptr_12 = tt.addptr %A_ptr, %A_ptr_11 : !tt.ptr<bf16>, i32
%C_ptr_13 = arith.muli %batch_idx_4, %c_stride_b : i32
%C_ptr_14 = tt.addptr %C_ptr, %C_ptr_13 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_17 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_18 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_19 = tt.splat %m_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_21 = arith.addi %offs_m_18, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_22 = arith.addi %offs_m_19, %offs_m_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_23 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_24 = arith.remsi %offs_m_20, %offs_m_23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_25 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_26 = tt.splat %n_idx : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_28 = arith.addi %offs_n_25, %offs_m_15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_29 = arith.addi %offs_n_26, %offs_m_16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_30 = arith.remsi %offs_n_27, %offs_m_23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%a_ptrs = tt.expand_dims %offs_m_24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%a_ptrs_31 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%a_ptrs_32 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_33 = arith.muli %a_ptrs, %a_ptrs_31 : tensor<128x1xi32, #blocked1>
%a_ptrs_34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%a_ptrs_35 = tt.expand_dims %a_ptrs_34 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%a_ptrs_36 = tt.broadcast %a_ptrs_33 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_37 = tt.broadcast %a_ptrs_35 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_38 = arith.addi %a_ptrs_36, %a_ptrs_37 : tensor<128x64xi32, #blocked1>
%a_ptrs_39 = tt.splat %A_ptr_12 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
%a_ptrs_40 = tt.addptr %a_ptrs_39, %a_ptrs_38 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs = tt.expand_dims %offs_n_30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%at_ptrs_41 = arith.muli %at_ptrs, %a_ptrs_31 : tensor<128x1xi32, #blocked1>
%at_ptrs_42 = tt.broadcast %at_ptrs_41 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%at_ptrs_43 = arith.addi %at_ptrs_42, %a_ptrs_37 : tensor<128x64xi32, #blocked1>
%at_ptrs_44 = tt.addptr %a_ptrs_39, %at_ptrs_43 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%0 = arith.addi %M, 63 : i32
%1 = arith.divsi %0, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %1, 0 : i32
%a_45 = tt.splat %M : i32 -> tensor<1x64xi32, #blocked1>
%a_46 = arith.cmpi slt, %a_ptrs_35, %a_45 : tensor<1x64xi32, #blocked1>
%a_47 = tt.broadcast %a_46 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_48 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_49 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_50 = arith.andi %accumulator_49, %a_47 : tensor<128x64xi1, #blocked1>
%a_51 = ttg.async_copy_global_to_local %a_ptrs_40, %a_48 mask %accumulator_50 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_52 = ttg.async_commit_group tokens %a_51
%at_temp_53 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_54 = ttg.async_copy_global_to_local %at_ptrs_44, %at_temp_53 mask %accumulator_50 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_55 = ttg.async_commit_group tokens %at_temp_54
%accumulator_56 = arith.cmpi sgt, %1, 1 : i32
%a_ptrs_57 = tt.addptr %a_ptrs_40, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_58 = tt.addptr %at_ptrs_44, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining = arith.subi %M, 64 : i32
%a_59 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked1>
%a_60 = arith.cmpi slt, %a_ptrs_35, %a_59 : tensor<1x64xi32, #blocked1>
%a_61 = tt.broadcast %a_60 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_62 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_63 = tt.splat %accumulator_56 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_64 = arith.andi %accumulator_63, %a_61 : tensor<128x64xi1, #blocked1>
%a_65 = ttg.async_copy_global_to_local %a_ptrs_57, %a_62 mask %accumulator_64 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_66 = ttg.async_commit_group tokens %a_65
%at_temp_67 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_68 = ttg.async_copy_global_to_local %at_ptrs_58, %at_temp_67 mask %accumulator_64 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_69 = ttg.async_commit_group tokens %at_temp_68
%accumulator_70 = arith.cmpi sgt, %1, 2 : i32
%a_ptrs_71 = tt.addptr %a_ptrs_57, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_72 = tt.addptr %at_ptrs_58, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining_73 = arith.subi %M, 128 : i32
%a_74 = tt.splat %k_remaining_73 : i32 -> tensor<1x64xi32, #blocked1>
%a_75 = arith.cmpi slt, %a_ptrs_35, %a_74 : tensor<1x64xi32, #blocked1>
%a_76 = tt.broadcast %a_75 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_77 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_78 = tt.splat %accumulator_70 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_79 = arith.andi %accumulator_78, %a_76 : tensor<128x64xi1, #blocked1>
%a_80 = ttg.async_copy_global_to_local %a_ptrs_71, %a_77 mask %accumulator_79 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_81 = ttg.async_commit_group tokens %a_80
%at_temp_82 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_83 = ttg.async_copy_global_to_local %at_ptrs_72, %at_temp_82 mask %accumulator_79 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_84 = ttg.async_commit_group tokens %at_temp_83
%accumulator_85:11 = scf.for %accumulator_125 = 0 to %1 step 1 iter_args(%arg10 = %cst, %a_ptrs_126 = %a_ptrs_71, %at_ptrs_127 = %at_ptrs_72, %arg13 = 2, %arg14 = %c-1_i32, %a_128 = %a_52, %a_129 = %a_66, %a_130 = %a_81, %at_temp_131 = %at_temp_55, %at_temp_132 = %at_temp_69, %at_temp_133 = %at_temp_84) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_134 = arith.subi %1, 3 : i32
%accumulator_135 = arith.cmpi slt, %accumulator_125, %accumulator_134 : i32
%accumulator_136 = arith.addi %arg14, 1 : i32
%accumulator_137 = arith.cmpi sge, %accumulator_136, 4 : i32
%accumulator_138 = arith.select %accumulator_137, 0, %accumulator_136 : i32
%a_139 = ttg.async_wait %a_128, %at_temp_131 {num = 4 : i32}
%a_140 = ttg.memdesc_index %a[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_141 = ttg.memdesc_index %at_temp[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_141 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_142 = ttng.warp_group_dot %a_140, %at, %arg10 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_143:3 = ttng.warp_group_dot_wait %accumulator_142, %a_140, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_144 = tt.addptr %a_ptrs_126, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_145 = tt.addptr %at_ptrs_127, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%accumulator_146 = arith.addi %arg13, 1 : i32
%accumulator_147 = arith.cmpi sge, %accumulator_146, 4 : i32
%accumulator_148 = arith.select %accumulator_147, 0, %accumulator_146 : i32
%accumulator_149 = arith.addi %accumulator_125, 3 : i32
%k_remaining_150 = arith.muli %accumulator_149, 64 : i32
%k_remaining_151 = arith.subi %M, %k_remaining_150 : i32
%a_152 = tt.splat %k_remaining_151 : i32 -> tensor<1x64xi32, #blocked1>
%a_153 = arith.cmpi slt, %a_ptrs_35, %a_152 : tensor<1x64xi32, #blocked1>
%a_154 = tt.broadcast %a_153 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_155 = ttg.memdesc_index %a[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_156 = tt.splat %accumulator_135 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_157 = arith.andi %accumulator_156, %a_154 : tensor<128x64xi1, #blocked1>
%a_158 = ttg.async_copy_global_to_local %a_ptrs_144, %a_155 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_159 = ttg.async_commit_group tokens %a_158
%at_temp_160 = ttg.memdesc_index %at_temp[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_161 = ttg.async_copy_global_to_local %at_ptrs_145, %at_temp_160 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_162 = ttg.async_commit_group tokens %at_temp_161
scf.yield %accumulator_143#0, %a_ptrs_144, %at_ptrs_145, %accumulator_148, %accumulator_138, %a_129, %a_130, %a_159, %at_temp_132, %at_temp_133, %at_temp_162 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_86 = ttng.warp_group_dot_wait %accumulator_85#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_87 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%a_add_ptrs = tt.expand_dims %offs_m_21 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_add_ptrs_88 = arith.muli %a_add_ptrs, %a_ptrs_32 : tensor<128x1xi32, #blocked>
%a_add_ptrs_89 = tt.expand_dims %offs_n_29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_add_ptrs_90 = tt.broadcast %a_add_ptrs_88 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_91 = tt.broadcast %a_add_ptrs_89 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_92 = arith.addi %a_add_ptrs_90, %a_add_ptrs_91 : tensor<128x128xi32, #blocked>
%a_add_ptrs_93 = tt.splat %A_ptr_12 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%a_add_ptrs_94 = tt.addptr %a_add_ptrs_93, %a_add_ptrs_92 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%a_add_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked>
%a_add_mask_95 = arith.cmpi slt, %a_add_ptrs, %a_add_mask : tensor<128x1xi32, #blocked>
%a_add_mask_96 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked>
%a_add_mask_97 = arith.cmpi slt, %a_add_ptrs_89, %a_add_mask_96 : tensor<1x128xi32, #blocked>
%a_add_mask_98 = tt.broadcast %a_add_mask_95 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_99 = tt.broadcast %a_add_mask_97 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_100 = arith.andi %a_add_mask_98, %a_add_mask_99 : tensor<128x128xi1, #blocked>
%a_add = tt.load %a_add_ptrs_94, %a_add_mask_100, %cst_0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%accumulator_101 = ttg.convert_layout %a_add : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%a_add_102 = arith.extf %accumulator_101 : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%accumulator_103 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%accumulator_104 = arith.mulf %accumulator_86, %accumulator_103 : tensor<128x128xf32, #mma>
%accumulator_105 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%accumulator_106 = arith.mulf %a_add_102, %accumulator_105 : tensor<128x128xf32, #mma>
%accumulator_107 = arith.addf %accumulator_104, %accumulator_106 : tensor<128x128xf32, #mma>
%output = arith.truncf %accumulator_107 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%output_108 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked2>
%2 = tt.trans %output_108 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked2> -> tensor<128x128xbf16, #blocked>
%output_109 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
%c_ptrs = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_110 = arith.muli %a_add_ptrs, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_111 = tt.broadcast %c_ptrs_110 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_112 = arith.addi %c_ptrs_111, %a_add_ptrs_91 : tensor<128x128xi32, #blocked>
%c_ptrs_113 = tt.splat %C_ptr_14 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_114 = tt.addptr %c_ptrs_113, %c_ptrs_112 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
tt.store %c_ptrs_114, %output_109, %a_add_mask_100 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_115 = arith.muli %c_ptrs_t, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_t_116 = tt.expand_dims %offs_m_22 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_117 = tt.broadcast %c_ptrs_t_115 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_118 = tt.broadcast %c_ptrs_t_116 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_119 = arith.addi %c_ptrs_t_117, %c_ptrs_t_118 : tensor<128x128xi32, #blocked>
%c_ptrs_t_120 = tt.addptr %c_ptrs_113, %c_ptrs_t_119 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %a_add_mask : tensor<128x1xi32, #blocked>
%c_mask_t_121 = arith.cmpi slt, %c_ptrs_t_116, %a_add_mask_96 : tensor<1x128xi32, #blocked>
%c_mask_t_122 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_123 = tt.broadcast %c_mask_t_121 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_124 = arith.andi %c_mask_t_122, %c_mask_t_123 : tensor<128x128xi1, #blocked>
tt.store %c_ptrs_t_120, %2, %c_mask_t_124 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .widePolynomial, .compact => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @ba_plus_cAA_kernel(%A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %C_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %a_stride_b: i32 {tt.divisibility = 16 : i32}, %a_stride_r: i32 {tt.divisibility = 16 : i32}, %c_stride_b: i32 {tt.divisibility = 16 : i32}, %c_stride_r: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
%cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1>
%cst_2 = arith.constant dense<64> : tensor<128x64xi32, #blocked1>
%c-1_i32 = arith.constant -1 : i32
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_3 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_3 : i32 to i64
%row_4 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_5 = arith.andi %row_4, %c7_i64 : i64
%row_6 = arith.trunci %row_5 : i64 to i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_7 = arith.andi %col, %c7_i64 : i64
%col_8 = arith.trunci %col_7 : i64 to i32
%0 = arith.divsi %pid, 21 : i32
%1 = arith.muli %row_6, 128 : i32
%2 = arith.muli %col_8, 128 : i32
%A_ptr_9 = arith.muli %0, %a_stride_b : i32
%A_ptr_10 = tt.addptr %A_ptr, %A_ptr_9 : !tt.ptr<bf16>, i32
%C_ptr_11 = arith.muli %0, %c_stride_b : i32
%C_ptr_12 = tt.addptr %C_ptr, %C_ptr_11 : !tt.ptr<bf16>, i32
%offs_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_15 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_16 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_17 = tt.splat %1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_18 = arith.addi %offs_m_15, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_19 = arith.addi %offs_m_16, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_m_20 = arith.addi %offs_m_17, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_m_21 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_22 = arith.remsi %offs_m_18, %offs_m_21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_23 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_24 = tt.splat %2 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_25 = arith.addi %offs_n, %offs_m : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_n_26 = arith.addi %offs_n_23, %offs_m_13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n_27 = arith.addi %offs_n_24, %offs_m_14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%offs_n_28 = arith.remsi %offs_n_25, %offs_m_21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%a_ptrs = tt.expand_dims %offs_m_22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%a_ptrs_29 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked1>
%a_ptrs_30 = tt.splat %a_stride_r : i32 -> tensor<128x1xi32, #blocked>
%a_ptrs_31 = arith.muli %a_ptrs, %a_ptrs_29 : tensor<128x1xi32, #blocked1>
%a_ptrs_32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%a_ptrs_33 = tt.expand_dims %a_ptrs_32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
%a_ptrs_34 = tt.broadcast %a_ptrs_31 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_35 = tt.broadcast %a_ptrs_33 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%a_ptrs_36 = arith.addi %a_ptrs_34, %a_ptrs_35 : tensor<128x64xi32, #blocked1>
%a_ptrs_37 = tt.splat %A_ptr_10 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
%a_ptrs_38 = tt.addptr %a_ptrs_37, %a_ptrs_36 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs = tt.expand_dims %offs_n_28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
%at_ptrs_39 = arith.muli %at_ptrs, %a_ptrs_29 : tensor<128x1xi32, #blocked1>
%at_ptrs_40 = tt.broadcast %at_ptrs_39 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
%at_ptrs_41 = arith.addi %at_ptrs_40, %a_ptrs_35 : tensor<128x64xi32, #blocked1>
%at_ptrs_42 = tt.addptr %a_ptrs_37, %at_ptrs_41 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%3 = arith.addi %M, 63 : i32
%4 = arith.divsi %3, 64 : i32
%a = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%at_temp = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%accumulator = arith.cmpi sgt, %4, 0 : i32
%a_43 = tt.splat %M : i32 -> tensor<1x64xi32, #blocked1>
%a_44 = arith.cmpi slt, %a_ptrs_33, %a_43 : tensor<1x64xi32, #blocked1>
%a_45 = tt.broadcast %a_44 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_46 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_47 = tt.splat %accumulator : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_48 = arith.andi %accumulator_47, %a_45 : tensor<128x64xi1, #blocked1>
%a_49 = ttg.async_copy_global_to_local %a_ptrs_38, %a_46 mask %accumulator_48 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_50 = ttg.async_commit_group tokens %a_49
%at_temp_51 = ttg.memdesc_index %at_temp[%c0_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_52 = ttg.async_copy_global_to_local %at_ptrs_42, %at_temp_51 mask %accumulator_48 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_53 = ttg.async_commit_group tokens %at_temp_52
%accumulator_54 = arith.cmpi sgt, %4, 1 : i32
%a_ptrs_55 = tt.addptr %a_ptrs_38, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_56 = tt.addptr %at_ptrs_42, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining = arith.subi %M, 64 : i32
%a_57 = tt.splat %k_remaining : i32 -> tensor<1x64xi32, #blocked1>
%a_58 = arith.cmpi slt, %a_ptrs_33, %a_57 : tensor<1x64xi32, #blocked1>
%a_59 = tt.broadcast %a_58 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_60 = ttg.memdesc_index %a[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_61 = tt.splat %accumulator_54 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_62 = arith.andi %accumulator_61, %a_59 : tensor<128x64xi1, #blocked1>
%a_63 = ttg.async_copy_global_to_local %a_ptrs_55, %a_60 mask %accumulator_62 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_64 = ttg.async_commit_group tokens %a_63
%at_temp_65 = ttg.memdesc_index %at_temp[%c1_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_66 = ttg.async_copy_global_to_local %at_ptrs_56, %at_temp_65 mask %accumulator_62 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_67 = ttg.async_commit_group tokens %at_temp_66
%accumulator_68 = arith.cmpi sgt, %4, 2 : i32
%a_ptrs_69 = tt.addptr %a_ptrs_55, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_70 = tt.addptr %at_ptrs_56, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%k_remaining_71 = arith.subi %M, 128 : i32
%a_72 = tt.splat %k_remaining_71 : i32 -> tensor<1x64xi32, #blocked1>
%a_73 = arith.cmpi slt, %a_ptrs_33, %a_72 : tensor<1x64xi32, #blocked1>
%a_74 = tt.broadcast %a_73 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_75 = ttg.memdesc_index %a[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_76 = tt.splat %accumulator_68 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_77 = arith.andi %accumulator_76, %a_74 : tensor<128x64xi1, #blocked1>
%a_78 = ttg.async_copy_global_to_local %a_ptrs_69, %a_75 mask %accumulator_77 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_79 = ttg.async_commit_group tokens %a_78
%at_temp_80 = ttg.memdesc_index %at_temp[%c2_i32] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_81 = ttg.async_copy_global_to_local %at_ptrs_70, %at_temp_80 mask %accumulator_77 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_82 = ttg.async_commit_group tokens %at_temp_81
%accumulator_83:11 = scf.for %accumulator_123 = 0 to %4 step 1 iter_args(%accumulator_124 = %cst, %a_ptrs_125 = %a_ptrs_69, %at_ptrs_126 = %at_ptrs_70, %arg13 = 2, %accumulator_127 = %c-1_i32, %a_128 = %a_50, %a_129 = %a_64, %a_130 = %a_79, %at_temp_131 = %at_temp_53, %at_temp_132 = %at_temp_67, %at_temp_133 = %at_temp_82) -> (tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) : i32 {
%accumulator_134 = arith.subi %4, 3 : i32
%accumulator_135 = arith.cmpi slt, %accumulator_123, %accumulator_134 : i32
%accumulator_136 = arith.addi %accumulator_127, 1 : i32
%accumulator_137 = arith.cmpi sge, %accumulator_136, 4 : i32
%accumulator_138 = arith.select %accumulator_137, 0, %accumulator_136 : i32
%a_139 = ttg.async_wait %a_128, %at_temp_131 {num = 4 : i32}
%a_140 = ttg.memdesc_index %a[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_141 = ttg.memdesc_index %at_temp[%accumulator_138] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at = ttg.memdesc_trans %at_temp_141 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%accumulator_142 = ttng.warp_group_dot %a_140, %at, %accumulator_124 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xbf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable> -> tensor<128x128xf32, #mma>
%accumulator_143:3 = ttng.warp_group_dot_wait %accumulator_142, %a_140, %at {pendings = 1 : i32} : tensor<128x128xf32, #mma>, !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xbf16, #shared1, #smem, mutable>
%a_ptrs_144 = tt.addptr %a_ptrs_125, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%at_ptrs_145 = tt.addptr %at_ptrs_126, %cst_2 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi32, #blocked1>
%accumulator_146 = arith.addi %arg13, 1 : i32
%accumulator_147 = arith.cmpi sge, %accumulator_146, 4 : i32
%accumulator_148 = arith.select %accumulator_147, 0, %accumulator_146 : i32
%accumulator_149 = arith.addi %accumulator_123, 3 : i32
%k_remaining_150 = arith.muli %accumulator_149, 64 : i32
%k_remaining_151 = arith.subi %M, %k_remaining_150 : i32
%a_152 = tt.splat %k_remaining_151 : i32 -> tensor<1x64xi32, #blocked1>
%a_153 = arith.cmpi slt, %a_ptrs_33, %a_152 : tensor<1x64xi32, #blocked1>
%a_154 = tt.broadcast %a_153 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
%a_155 = ttg.memdesc_index %a[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%accumulator_156 = tt.splat %accumulator_135 : i1 -> tensor<128x64xi1, #blocked1>
%accumulator_157 = arith.andi %accumulator_156, %a_154 : tensor<128x64xi1, #blocked1>
%a_158 = ttg.async_copy_global_to_local %a_ptrs_144, %a_155 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%a_159 = ttg.async_commit_group tokens %a_158
%at_temp_160 = ttg.memdesc_index %at_temp[%accumulator_148] : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xbf16, #shared, #smem, mutable>
%at_temp_161 = ttg.async_copy_global_to_local %at_ptrs_145, %at_temp_160 mask %accumulator_157 other %cst_1 {contiguity = 8 : i32} : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared, #smem, mutable>
%at_temp_162 = ttg.async_commit_group tokens %at_temp_161
scf.yield %accumulator_143#0, %a_ptrs_144, %at_ptrs_145, %accumulator_148, %accumulator_138, %a_129, %a_130, %a_159, %at_temp_132, %at_temp_133, %at_temp_162 : tensor<128x128xf32, #mma>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64x!tt.ptr<bf16>, #blocked1>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
}
%accumulator_84 = ttng.warp_group_dot_wait %accumulator_83#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
%accumulator_85 = ttg.async_wait {num = 0 : i32}
ttg.local_dealloc %at_temp : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
ttg.local_dealloc %a : !ttg.memdesc<4x128x64xbf16, #shared, #smem, mutable>
%a_add_ptrs = tt.expand_dims %offs_m_19 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%a_add_ptrs_86 = arith.muli %a_add_ptrs, %a_ptrs_30 : tensor<128x1xi32, #blocked>
%a_add_ptrs_87 = tt.expand_dims %offs_n_27 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%a_add_ptrs_88 = tt.broadcast %a_add_ptrs_86 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_89 = tt.broadcast %a_add_ptrs_87 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%a_add_ptrs_90 = arith.addi %a_add_ptrs_88, %a_add_ptrs_89 : tensor<128x128xi32, #blocked>
%a_add_ptrs_91 = tt.splat %A_ptr_10 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%a_add_ptrs_92 = tt.addptr %a_add_ptrs_91, %a_add_ptrs_90 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%a_add_mask = tt.splat %M : i32 -> tensor<128x1xi32, #blocked>
%a_add_mask_93 = arith.cmpi slt, %a_add_ptrs, %a_add_mask : tensor<128x1xi32, #blocked>
%a_add_mask_94 = tt.splat %M : i32 -> tensor<1x128xi32, #blocked>
%a_add_mask_95 = arith.cmpi slt, %a_add_ptrs_87, %a_add_mask_94 : tensor<1x128xi32, #blocked>
%a_add_mask_96 = tt.broadcast %a_add_mask_93 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_97 = tt.broadcast %a_add_mask_95 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%a_add_mask_98 = arith.andi %a_add_mask_96, %a_add_mask_97 : tensor<128x128xi1, #blocked>
%a_add = tt.load %a_add_ptrs_92, %a_add_mask_98, %cst_0 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%accumulator_99 = ttg.convert_layout %a_add : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%a_add_100 = arith.extf %accumulator_99 : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%accumulator_101 = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%accumulator_102 = arith.mulf %accumulator_84, %accumulator_101 : tensor<128x128xf32, #mma>
%accumulator_103 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%accumulator_104 = arith.mulf %a_add_100, %accumulator_103 : tensor<128x128xf32, #mma>
%accumulator_105 = arith.addf %accumulator_102, %accumulator_104 : tensor<128x128xf32, #mma>
%output = arith.truncf %accumulator_105 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%output_106 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked2>
%5 = tt.trans %output_106 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #blocked2> -> tensor<128x128xbf16, #blocked>
%output_107 = ttg.convert_layout %output : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #blocked>
%c_ptrs = tt.splat %c_stride_r : i32 -> tensor<128x1xi32, #blocked>
%c_ptrs_108 = arith.muli %a_add_ptrs, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_109 = tt.broadcast %c_ptrs_108 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_110 = arith.addi %c_ptrs_109, %a_add_ptrs_89 : tensor<128x128xi32, #blocked>
%c_ptrs_111 = tt.splat %C_ptr_12 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_112 = tt.addptr %c_ptrs_111, %c_ptrs_110 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
tt.store %c_ptrs_112, %output_107, %a_add_mask_98 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%c_ptrs_t = tt.expand_dims %offs_n_26 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%c_ptrs_t_113 = arith.muli %c_ptrs_t, %c_ptrs : tensor<128x1xi32, #blocked>
%c_ptrs_t_114 = tt.expand_dims %offs_m_20 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%c_ptrs_t_115 = tt.broadcast %c_ptrs_t_113 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_116 = tt.broadcast %c_ptrs_t_114 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%c_ptrs_t_117 = arith.addi %c_ptrs_t_115, %c_ptrs_t_116 : tensor<128x128xi32, #blocked>
%c_ptrs_t_118 = tt.addptr %c_ptrs_111, %c_ptrs_t_117 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%c_mask_t = arith.cmpi slt, %c_ptrs_t, %a_add_mask : tensor<128x1xi32, #blocked>
%c_mask_t_119 = arith.cmpi slt, %c_ptrs_t_114, %a_add_mask_94 : tensor<1x128xi32, #blocked>
%c_mask_t_120 = tt.broadcast %c_mask_t : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_121 = tt.broadcast %c_mask_t_119 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
%c_mask_t_122 = arith.andi %c_mask_t_120, %c_mask_t_121 : tensor<128x128xi1, #blocked>
tt.store %c_ptrs_t_118, %5, %c_mask_t_122 : tensor<128x128x!tt.ptr<bf16>, #blocked>
tt.return
}
}
  "#
  | .widePolynomial, .full => r#"
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [0, 8], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]], warp = [[0, 16], [0, 32], [0, 64]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @polar(%A: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %A_0: i32, %A_1: i32, %A_2: i64, %A_3: i64, %C: !tt.tensordesc<tensor<128x128xbf16, #shared>>, %C_4: i32, %C_5: i32, %C_6: i64, %C_7: i64, %A_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %alpha: f32, %beta: f32) attributes {noinline = false} {
%c6569755928097638024_i64 = arith.constant 6569755928097638024 : i64
%c7_i64 = arith.constant 7 : i64
%c6420841298313609216_i64 = arith.constant 6420841298313609216 : i64
%cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
%true = arith.constant true
%cst_8 = arith.constant dense<768> : tensor<128x1xi32, #blocked>
%pid = tt.get_program_id x : i32
%shift = arith.remsi %pid, 21 : i32
%shift_9 = arith.muli %shift, 3 : i32
%row = arith.extsi %shift_9 : i32 to i64
%row_10 = arith.shrui %c6569755928097638024_i64, %row : i64
%row_11 = arith.andi %row_10, %c7_i64 : i64
%row_12 = arith.trunci %row_11 : i64 to i32
%row_13 = arith.muli %row_12, 128 : i32
%col = arith.shrui %c6420841298313609216_i64, %row : i64
%col_14 = arith.andi %col, %c7_i64 : i64
%col_15 = arith.trunci %col_14 : i64 to i32
%col_16 = arith.muli %col_15, 128 : i32
%batch = arith.divsi %pid, 21 : i32
%offdiag = arith.cmpi ne, %row_13, %col_16 : i32
%tiles = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable>
%barriers = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
%0 = ttg.memdesc_index %barriers[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%1 = ttg.memdesc_index %barriers[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.init_barrier %1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%2 = scf.if %offdiag -> (tensor<128x128xf32, #mma>) {
%acc = arith.muli %batch, 768 : i32
%acc_36 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 65536, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_13 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_36, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_37 = ttg.memdesc_index %tiles[%c2_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%coords_38 = arith.addi %acc, %col_16 : i32
ttng.async_tma_copy_global_to_local %A[%coords_38, %c0_i32] %acc_37, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_39 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_40 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_41 = arith.remsi %phase, 2 : i32
%base_42 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_43 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_43, %phase_41, %true deps %base_42, %acc_37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %acc_37 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_44 = ttng.warp_group_dot %base_42, %right, %acc_40, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_45 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_46 = ttg.memdesc_index %barriers[%nxt_45] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_47 = ttg.memdesc_index %tiles[%nxt_45] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_46, 65536, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_47, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_48 = ttng.warp_group_dot_wait %acc_44 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
ttng.async_tma_copy_global_to_local %A[%coords_38, %k] %acc_37, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_48 : tensor<128x128xf32, #mma>
}
scf.yield %acc_39 : tensor<128x128xf32, #mma>
} else {
%acc = arith.muli %batch, 768 : i32
%acc_36 = ttg.memdesc_index %tiles[%c0_i32] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %0, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%coords = arith.addi %acc, %row_13 : i32
ttng.async_tma_copy_global_to_local %A[%coords, %c0_i32] %acc_36, %0, %true : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_37 = scf.for %chunk = 0 to 6 step 1 iter_args(%acc_39 = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
%buf = arith.remsi %chunk, 2 : i32
%phase = arith.divsi %chunk, 2 : i32
%phase_40 = arith.remsi %phase, 2 : i32
%base_41 = ttg.memdesc_index %tiles[%buf] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%acc_42 = ttg.memdesc_index %barriers[%buf] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.wait_barrier %acc_42, %phase_40, %true deps %base_41, %base_41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%right = ttg.memdesc_trans %base_41 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable>
%acc_43 = ttng.warp_group_dot %base_41, %right, %acc_39, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x128xbf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xbf16, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
%acc_44 = ttng.warp_group_dot_wait %acc_43 {pendings = 1 : i32} : tensor<128x128xf32, #mma>
%nxt = arith.addi %chunk, 1 : i32
%nxt_45 = arith.remsi %nxt, 2 : i32
%more = arith.cmpi slt, %nxt, 6 : i32
%k = arith.muli %nxt, 128 : i32
%acc_46 = ttg.memdesc_index %barriers[%nxt_45] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%acc_47 = ttg.memdesc_index %tiles[%nxt_45] : !ttg.memdesc<3x128x128xbf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.barrier_expect %acc_46, 32768, %more : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.async_tma_copy_global_to_local %A[%coords, %k] %acc_47, %acc_46, %more : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.yield %acc_44 : tensor<128x128xf32, #mma>
}
%acc_38 = ttng.warp_group_dot_wait %acc_37 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
scf.yield %acc_38 : tensor<128x128xf32, #mma>
}
ttng.inval_barrier %0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
ttng.inval_barrier %1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
%i = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%j = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %batch, 589824 : i32
%dst = tt.expand_dims %i {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
%dst_17 = tt.splat %row_13 : i32 -> tensor<128x1xi32, #blocked>
%dst_18 = arith.addi %dst_17, %dst : tensor<128x1xi32, #blocked>
%dst_19 = arith.muli %dst_18, %cst_8 : tensor<128x1xi32, #blocked>
%dst_20 = tt.expand_dims %j {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%dst_21 = tt.splat %col_16 : i32 -> tensor<1x128xi32, #blocked>
%dst_22 = arith.addi %dst_21, %dst_20 : tensor<1x128xi32, #blocked>
%dst_23 = tt.broadcast %dst_19 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_24 = tt.broadcast %dst_22 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
%dst_25 = arith.addi %dst_23, %dst_24 : tensor<128x128xi32, #blocked>
%bias_words = tt.addptr %A_ptr, %base : !tt.ptr<bf16>, i32
%bias_words_26 = tt.splat %bias_words : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias_words_27 = tt.addptr %bias_words_26, %dst_25 : tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xi32, #blocked>
%bias_words_28 = tt.load %bias_words_27 : tensor<128x128x!tt.ptr<bf16>, #blocked>
%bias = ttg.convert_layout %bias_words_28 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #mma>
%bias_29 = arith.extf %bias : tensor<128x128xbf16, #mma> to tensor<128x128xf32, #mma>
%words = tt.splat %alpha : f32 -> tensor<128x128xf32, #mma>
%words_30 = arith.mulf %2, %words : tensor<128x128xf32, #mma>
%words_31 = tt.splat %beta : f32 -> tensor<128x128xf32, #mma>
%words_32 = arith.mulf %bias_29, %words_31 : tensor<128x128xf32, #mma>
%words_33 = arith.addf %words_30, %words_32 : tensor<128x128xf32, #mma>
%words_34 = arith.truncf %words_33 : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
%direct = ttg.local_alloc : () -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
scf.if %offdiag {
ttg.local_store %words_34, %direct : tensor<128x128xbf16, #mma> -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%mirror = tt.trans %words_34 {order = array<i32: 1, 0>} : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #linear>
%mirror_35 = ttg.local_alloc %mirror : (tensor<128x128xbf16, #linear>) -> !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
scf.if %offdiag {
%6 = arith.muli %batch, 768 : i32
%7 = arith.addi %6, %row_13 : i32
ttng.async_tma_copy_local_to_global %C[%7, %col_16] %direct : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
} else {
}
%3 = arith.muli %batch, 768 : i32
%4 = arith.addi %3, %col_16 : i32
ttng.async_tma_copy_local_to_global %C[%4, %row_13] %mirror_35 : !tt.tensordesc<tensor<128x128xbf16, #shared>>, !ttg.memdesc<128x128xbf16, #shared, #smem, mutable>
%5 = tt.elementwise_inline_asm "cp.async.bulk.wait_group 0; mov.u32 $0, 0;" {constraints = "=r,~{memory}", packed_element = 1 : i32, pure = false} -> i32
tt.return
}
}
"#

private def assembly : Shape → Kind → String
  | .qkGram, .baseline => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XXT_kernel(
.param .u64 .ptr .global .align 1 XXT_kernel_param_0,
.param .u64 .ptr .global .align 1 XXT_kernel_param_1,
.param .u32 XXT_kernel_param_2,
.param .u32 XXT_kernel_param_3,
.param .u32 XXT_kernel_param_4,
.param .u32 XXT_kernel_param_5,
.param .u32 XXT_kernel_param_6,
.param .u32 XXT_kernel_param_7,
.param .u64 .ptr .global .align 1 XXT_kernel_param_8,
.param .u64 .ptr .global .align 1 XXT_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<49>;
.reg .b32 %r<386>;
.reg .b64 %rd<89>;
$L__func_begin0:
ld.param.b32 %r37, [XXT_kernel_param_2];
$L__tmp0:
mov.u32 %r38, %ctaid.x;
$L__tmp1:
add.s32 %r39, %r37, 127;
shr.s32 %r40, %r39, 31;
shr.u32 %r41, %r40, 25;
add.s32 %r42, %r39, %r41;
shr.s32 %r43, %r42, 7;
$L__tmp2:
mul.lo.s32 %r44, %r43, %r43;
div.s32 %r1, %r38, %r44;
mul.lo.s32 %r45, %r1, %r44;
sub.s32 %r46, %r38, %r45;
$L__tmp3:
shl.b32 %r47, %r43, 3;
div.s32 %r48, %r46, %r47;
shl.b32 %r49, %r48, 3;
sub.s32 %r50, %r43, %r49;
min.s32 %r51, %r50, 8;
mul.lo.s32 %r52, %r48, %r47;
sub.s32 %r53, %r46, %r52;
div.s32 %r54, %r53, %r51;
mul.lo.s32 %r55, %r54, %r51;
sub.s32 %r56, %r53, %r55;
add.s32 %r57, %r56, %r49;
$L__tmp4:
shl.b32 %r2, %r57, 7;
shl.b32 %r3, %r54, 7;
$L__tmp5:
add.s32 %r58, %r2, 128;
setp.gt.s32 %p1, %r58, %r3;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b32 %r36, [XXT_kernel_param_7];
ld.param.b32 %r35, [XXT_kernel_param_6];
ld.param.b32 %r34, [XXT_kernel_param_5];
ld.param.b32 %r33, [XXT_kernel_param_4];
ld.param.b32 %r32, [XXT_kernel_param_3];
ld.param.b64 %rd19, [XXT_kernel_param_1];
ld.param.b64 %rd18, [XXT_kernel_param_0];
mul.lo.s32 %r86, %r1, %r33;
mad.wide.s32 %rd88, %r86, 2, %rd18;
mul.lo.s32 %r87, %r1, %r35;
mad.wide.s32 %rd1, %r87, 2, %rd19;
mov.u32 %r4, %tid.x;
shr.u32 %r88, %r4, 3;
bfe.u32 %r89, %r4, 3, 5;
or.b32 %r90, %r89, 32;
or.b32 %r91, %r89, 64;
or.b32 %r92, %r88, 96;
and.b32 %r5, %r4, 240;
bfe.u32 %r93, %r4, 4, 4;
or.b32 %r94, %r93, 16;
or.b32 %r95, %r93, 32;
or.b32 %r96, %r93, 48;
or.b32 %r97, %r93, 64;
or.b32 %r98, %r93, 80;
or.b32 %r99, %r93, 96;
or.b32 %r100, %r93, 112;
shl.b32 %r6, %r4, 3;
and.b32 %r101, %r6, 120;
or.b32 %r102, %r2, %r89;
or.b32 %r103, %r2, %r90;
or.b32 %r104, %r2, %r91;
or.b32 %r105, %r2, %r92;
or.b32 %r7, %r2, %r93;
or.b32 %r8, %r2, %r94;
or.b32 %r9, %r2, %r95;
or.b32 %r10, %r2, %r96;
or.b32 %r11, %r2, %r97;
or.b32 %r12, %r2, %r98;
or.b32 %r13, %r2, %r99;
or.b32 %r14, %r2, %r100;
or.b32 %r15, %r2, %r101;
rem.s32 %r106, %r102, %r37;
rem.s32 %r107, %r103, %r37;
rem.s32 %r108, %r104, %r37;
rem.s32 %r109, %r105, %r37;
or.b32 %r110, %r3, %r89;
or.b32 %r111, %r3, %r90;
or.b32 %r112, %r3, %r91;
or.b32 %r113, %r3, %r92;
or.b32 %r16, %r3, %r93;
or.b32 %r17, %r3, %r94;
or.b32 %r18, %r3, %r95;
or.b32 %r19, %r3, %r96;
or.b32 %r20, %r3, %r97;
or.b32 %r21, %r3, %r98;
or.b32 %r22, %r3, %r99;
or.b32 %r23, %r3, %r100;
or.b32 %r24, %r3, %r101;
rem.s32 %r114, %r110, %r37;
rem.s32 %r115, %r111, %r37;
rem.s32 %r116, %r112, %r37;
rem.s32 %r117, %r113, %r37;
and.b32 %r25, %r4, 7;
shl.b32 %r26, %r25, 3;
mad.lo.s32 %r118, %r106, %r34, %r26;
mad.lo.s32 %r119, %r107, %r34, %r26;
mad.lo.s32 %r120, %r108, %r34, %r26;
mad.lo.s32 %r121, %r109, %r34, %r26;
mad.wide.s32 %rd20, %r118, 2, %rd88;
mad.wide.s32 %rd21, %r119, 2, %rd88;
mad.wide.s32 %rd22, %r120, 2, %rd88;
mad.wide.s32 %rd23, %r121, 2, %rd88;
mad.lo.s32 %r122, %r114, %r34, %r26;
mad.lo.s32 %r123, %r115, %r34, %r26;
mad.lo.s32 %r124, %r116, %r34, %r26;
mad.lo.s32 %r125, %r117, %r34, %r26;
mad.wide.s32 %rd24, %r122, 2, %rd88;
mad.wide.s32 %rd25, %r123, 2, %rd88;
mad.wide.s32 %rd26, %r124, 2, %rd88;
mad.wide.s32 %rd27, %r125, 2, %rd88;
$L__tmp6:
add.s32 %r126, %r32, 63;
$L__tmp7:
setp.lt.s32 %p2, %r126, 64;
setp.gt.s32 %p3, %r126, 63;
setp.lt.s32 %p4, %r26, %r32;
shl.b32 %r130, %r4, 4;
and.b32 %r131, %r130, 4080;
shl.b32 %r28, %r4, 1;
and.b32 %r132, %r28, 112;
xor.b32 %r29, %r131, %r132;
mov.b32 %r133, global_smem;
add.s32 %r59, %r133, %r29;
selp.b32 %r134, 16, 0, %p3;
selp.b32 %r61, %r134, 0, %p4;
cp.async.cg.shared.global [ %r59 + 0 ], [ %rd20 + 0 ], 0x10, %r61;
add.s32 %r60, %r59, 4096;
cp.async.cg.shared.global [ %r60 + 0 ], [ %rd21 + 0 ], 0x10, %r61;
add.s32 %r62, %r59, 8192;
cp.async.cg.shared.global [ %r62 + 0 ], [ %rd22 + 0 ], 0x10, %r61;
add.s32 %r63, %r59, 12288;
cp.async.cg.shared.global [ %r63 + 0 ], [ %rd23 + 0 ], 0x10, %r61;
cp.async.commit_group;
add.s32 %r64, %r59, 65536;
cp.async.cg.shared.global [ %r64 + 0 ], [ %rd24 + 0 ], 0x10, %r61;
add.s32 %r65, %r59, 69632;
cp.async.cg.shared.global [ %r65 + 0 ], [ %rd25 + 0 ], 0x10, %r61;
add.s32 %r66, %r59, 73728;
cp.async.cg.shared.global [ %r66 + 0 ], [ %rd26 + 0 ], 0x10, %r61;
add.s32 %r67, %r59, 77824;
cp.async.cg.shared.global [ %r67 + 0 ], [ %rd27 + 0 ], 0x10, %r61;
cp.async.commit_group;
setp.gt.s32 %p5, %r126, 127;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s64 %rd35, %rd27, 128;
add.s32 %r135, %r32, -64;
setp.lt.s32 %p6, %r26, %r135;
bar.sync 0;
add.s32 %r68, %r59, 16384;
selp.b32 %r136, 16, 0, %p6;
selp.b32 %r70, %r136, 0, %p5;
cp.async.cg.shared.global [ %r68 + 0 ], [ %rd28 + 0 ], 0x10, %r70;
add.s32 %r69, %r59, 20480;
cp.async.cg.shared.global [ %r69 + 0 ], [ %rd29 + 0 ], 0x10, %r70;
add.s32 %r71, %r59, 24576;
cp.async.cg.shared.global [ %r71 + 0 ], [ %rd30 + 0 ], 0x10, %r70;
add.s32 %r72, %r59, 28672;
cp.async.cg.shared.global [ %r72 + 0 ], [ %rd31 + 0 ], 0x10, %r70;
cp.async.commit_group;
add.s32 %r73, %r59, 81920;
cp.async.cg.shared.global [ %r73 + 0 ], [ %rd32 + 0 ], 0x10, %r70;
add.s32 %r74, %r59, 86016;
cp.async.cg.shared.global [ %r74 + 0 ], [ %rd33 + 0 ], 0x10, %r70;
add.s32 %r75, %r59, 90112;
cp.async.cg.shared.global [ %r75 + 0 ], [ %rd34 + 0 ], 0x10, %r70;
add.s32 %r76, %r59, 94208;
cp.async.cg.shared.global [ %r76 + 0 ], [ %rd35 + 0 ], 0x10, %r70;
cp.async.commit_group;
setp.gt.s32 %p7, %r126, 191;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s64 %rd43, %rd27, 256;
add.s32 %r137, %r32, -128;
setp.lt.s32 %p8, %r26, %r137;
bar.sync 0;
add.s32 %r77, %r59, 32768;
selp.b32 %r138, 16, 0, %p8;
selp.b32 %r79, %r138, 0, %p7;
cp.async.cg.shared.global [ %r77 + 0 ], [ %rd36 + 0 ], 0x10, %r79;
add.s32 %r78, %r59, 36864;
cp.async.cg.shared.global [ %r78 + 0 ], [ %rd37 + 0 ], 0x10, %r79;
add.s32 %r80, %r59, 40960;
cp.async.cg.shared.global [ %r80 + 0 ], [ %rd38 + 0 ], 0x10, %r79;
add.s32 %r81, %r59, 45056;
cp.async.cg.shared.global [ %r81 + 0 ], [ %rd39 + 0 ], 0x10, %r79;
cp.async.commit_group;
add.s32 %r82, %r59, 98304;
cp.async.cg.shared.global [ %r82 + 0 ], [ %rd40 + 0 ], 0x10, %r79;
add.s32 %r83, %r59, 102400;
cp.async.cg.shared.global [ %r83 + 0 ], [ %rd41 + 0 ], 0x10, %r79;
add.s32 %r84, %r59, 106496;
cp.async.cg.shared.global [ %r84 + 0 ], [ %rd42 + 0 ], 0x10, %r79;
add.s32 %r85, %r59, 110592;
cp.async.cg.shared.global [ %r85 + 0 ], [ %rd43 + 0 ], 0x10, %r79;
cp.async.commit_group;
mov.b32 %r321, 0f00000000;
mov.b32 %r322, %r321;
mov.b32 %r323, %r321;
mov.b32 %r324, %r321;
mov.b32 %r325, %r321;
mov.b32 %r326, %r321;
mov.b32 %r327, %r321;
mov.b32 %r328, %r321;
mov.b32 %r329, %r321;
mov.b32 %r330, %r321;
mov.b32 %r331, %r321;
mov.b32 %r332, %r321;
mov.b32 %r333, %r321;
mov.b32 %r334, %r321;
mov.b32 %r335, %r321;
mov.b32 %r336, %r321;
mov.b32 %r337, %r321;
mov.b32 %r338, %r321;
mov.b32 %r339, %r321;
mov.b32 %r340, %r321;
mov.b32 %r341, %r321;
mov.b32 %r342, %r321;
mov.b32 %r343, %r321;
mov.b32 %r344, %r321;
mov.b32 %r345, %r321;
mov.b32 %r346, %r321;
mov.b32 %r347, %r321;
mov.b32 %r348, %r321;
mov.b32 %r349, %r321;
mov.b32 %r350, %r321;
mov.b32 %r351, %r321;
mov.b32 %r352, %r321;
mov.b32 %r353, %r321;
mov.b32 %r354, %r321;
mov.b32 %r355, %r321;
mov.b32 %r356, %r321;
mov.b32 %r357, %r321;
mov.b32 %r358, %r321;
mov.b32 %r359, %r321;
mov.b32 %r360, %r321;
mov.b32 %r361, %r321;
mov.b32 %r362, %r321;
mov.b32 %r363, %r321;
mov.b32 %r364, %r321;
mov.b32 %r365, %r321;
mov.b32 %r366, %r321;
mov.b32 %r367, %r321;
mov.b32 %r368, %r321;
mov.b32 %r369, %r321;
mov.b32 %r370, %r321;
mov.b32 %r371, %r321;
mov.b32 %r372, %r321;
mov.b32 %r373, %r321;
mov.b32 %r374, %r321;
mov.b32 %r375, %r321;
mov.b32 %r376, %r321;
mov.b32 %r377, %r321;
mov.b32 %r378, %r321;
mov.b32 %r379, %r321;
mov.b32 %r380, %r321;
mov.b32 %r381, %r321;
mov.b32 %r382, %r321;
mov.b32 %r383, %r321;
mov.b32 %r384, %r321;
@%p2 bra $L__BB0_5;
cvt.s64.s32 %rd2, %r118;
cvt.s64.s32 %rd3, %r119;
cvt.s64.s32 %rd4, %r120;
cvt.s64.s32 %rd5, %r121;
cvt.s64.s32 %rd6, %r122;
cvt.s64.s32 %rd7, %r123;
cvt.s64.s32 %rd8, %r124;
cvt.s64.s32 %rd9, %r125;
shr.s32 %r127, %r126, 31;
shr.u32 %r128, %r127, 26;
add.s32 %r129, %r126, %r128;
shr.s32 %r27, %r129, 6;
add.s32 %r30, %r27, -3;
shr.u32 %r31, %r4, 5;
add.s32 %r318, %r32, -192;
shl.b64 %rd44, %rd9, 1;
add.s64 %rd10, %rd44, 384;
shl.b64 %rd45, %rd8, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd7, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd6, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd5, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd4, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd3, 1;
add.s64 %rd16, %rd50, 384;
shl.b64 %rd51, %rd2, 1;
add.s64 %rd17, %rd51, 384;
mov.b32 %r143, 0;
mov.b32 %r321, 0f00000000;
mov.b32 %r320, 2;
mov.b32 %r319, -1;
mov.b32 %r322, %r321;
mov.b32 %r323, %r321;
mov.b32 %r324, %r321;
mov.b32 %r325, %r321;
mov.b32 %r326, %r321;
mov.b32 %r327, %r321;
mov.b32 %r328, %r321;
mov.b32 %r329, %r321;
mov.b32 %r330, %r321;
mov.b32 %r331, %r321;
mov.b32 %r332, %r321;
mov.b32 %r333, %r321;
mov.b32 %r334, %r321;
mov.b32 %r335, %r321;
mov.b32 %r336, %r321;
mov.b32 %r337, %r321;
mov.b32 %r338, %r321;
mov.b32 %r339, %r321;
mov.b32 %r340, %r321;
mov.b32 %r341, %r321;
mov.b32 %r342, %r321;
mov.b32 %r343, %r321;
mov.b32 %r344, %r321;
mov.b32 %r345, %r321;
mov.b32 %r346, %r321;
mov.b32 %r347, %r321;
mov.b32 %r348, %r321;
mov.b32 %r349, %r321;
mov.b32 %r350, %r321;
mov.b32 %r351, %r321;
mov.b32 %r352, %r321;
mov.b32 %r353, %r321;
mov.b32 %r354, %r321;
mov.b32 %r355, %r321;
mov.b32 %r356, %r321;
mov.b32 %r357, %r321;
mov.b32 %r358, %r321;
mov.b32 %r359, %r321;
mov.b32 %r360, %r321;
mov.b32 %r361, %r321;
mov.b32 %r362, %r321;
mov.b32 %r363, %r321;
mov.b32 %r364, %r321;
mov.b32 %r365, %r321;
mov.b32 %r366, %r321;
mov.b32 %r367, %r321;
mov.b32 %r368, %r321;
mov.b32 %r369, %r321;
mov.b32 %r370, %r321;
mov.b32 %r371, %r321;
mov.b32 %r372, %r321;
mov.b32 %r373, %r321;
mov.b32 %r374, %r321;
mov.b32 %r375, %r321;
mov.b32 %r376, %r321;
mov.b32 %r377, %r321;
mov.b32 %r378, %r321;
mov.b32 %r379, %r321;
mov.b32 %r380, %r321;
mov.b32 %r381, %r321;
mov.b32 %r382, %r321;
mov.b32 %r383, %r321;
mov.b32 %r384, %r321;
mov.b32 %r385, %r143;
$L__BB0_4:
setp.lt.s32 %p10, %r385, %r30;
add.s32 %r153, %r319, 1;
setp.gt.s32 %p11, %r153, 3;
selp.b32 %r319, 0, %r153, %p11;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r154, %r319, 14;
add.s32 %r155, %r133, %r154;
add.s32 %r156, %r155, 65536;
shr.u32 %r157, %r155, 4;
shfl.sync.idx.b32 %r158, %r31, 0, 31, -1;
shl.b32 %r159, %r158, 7;
and.b32 %r160, %r159, 512;
add.s32 %r161, %r160, %r157;
and.b32 %r162, %r161, 16383;
cvt.u64.u32 %rd70, %r162;
bfe.u32 %r163, %r156, 4, 14;
cvt.u64.u32 %rd71, %r163;
wgmma.fence.sync.aligned;
or.b64 %rd52, %rd70, 4611686293305294848;
or.b64 %rd53, %rd71, 4611686293305294848;
mov.pred %p9, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd52, %rd53, %p9, 1, 1, 0, 0;
add.s64 %rd54, %rd70, 4611686293305294850;
add.s64 %rd55, %rd71, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd54, %rd55, %p9, 1, 1, 0, 0;
add.s64 %rd56, %rd70, 4611686293305294852;
add.s64 %rd57, %rd71, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd56, %rd57, %p9, 1, 1, 0, 0;
add.s64 %rd58, %rd70, 4611686293305294854;
add.s64 %rd59, %rd71, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd58, %rd59, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd60, %r155;
cvt.u64.u32 %rd61, %r156;
mov.b32 %r142, %r143;
mov.b32 %r139, %r143;
mov.b32 %r140, %r143;
mov.b32 %r141, %r143;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd62, %rd88, %rd17;
add.s64 %rd63, %rd88, %rd16;
add.s64 %rd64, %rd88, %rd15;
add.s64 %rd65, %rd88, %rd14;
add.s64 %rd66, %rd88, %rd13;
add.s64 %rd67, %rd88, %rd12;
add.s64 %rd68, %rd88, %rd11;
add.s64 %rd69, %rd88, %rd10;
add.s32 %r164, %r320, 1;
setp.gt.s32 %p12, %r164, 3;
selp.b32 %r320, 0, %r164, %p12;
setp.lt.s32 %p13, %r26, %r318;
shl.b32 %r165, %r320, 14;
add.s32 %r166, %r133, %r165;
bar.sync 0;
add.s32 %r144, %r166, %r29;
selp.b32 %r167, 16, 0, %p13;
selp.b32 %r146, %r167, 0, %p10;
cp.async.cg.shared.global [ %r144 + 0 ], [ %rd62 + 0 ], 0x10, %r146;
add.s32 %r145, %r144, 4096;
cp.async.cg.shared.global [ %r145 + 0 ], [ %rd63 + 0 ], 0x10, %r146;
add.s32 %r147, %r144, 8192;
cp.async.cg.shared.global [ %r147 + 0 ], [ %rd64 + 0 ], 0x10, %r146;
add.s32 %r148, %r144, 12288;
cp.async.cg.shared.global [ %r148 + 0 ], [ %rd65 + 0 ], 0x10, %r146;
cp.async.commit_group;
add.s32 %r149, %r144, 65536;
cp.async.cg.shared.global [ %r149 + 0 ], [ %rd66 + 0 ], 0x10, %r146;
add.s32 %r150, %r144, 69632;
cp.async.cg.shared.global [ %r150 + 0 ], [ %rd67 + 0 ], 0x10, %r146;
add.s32 %r151, %r144, 73728;
cp.async.cg.shared.global [ %r151 + 0 ], [ %rd68 + 0 ], 0x10, %r146;
add.s32 %r152, %r144, 77824;
cp.async.cg.shared.global [ %r152 + 0 ], [ %rd69 + 0 ], 0x10, %r146;
cp.async.commit_group;
add.s32 %r385, %r385, 1;
add.s64 %rd88, %rd88, 128;
add.s32 %r318, %r318, -64;
setp.ne.b32 %p14, %r27, %r385;
@%p14 bra $L__BB0_4;
$L__BB0_5:
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r232, %r322, %r321;
cvt.rn.bf16x2.f32 %r233, %r324, %r323;
cvt.rn.bf16x2.f32 %r234, %r326, %r325;
cvt.rn.bf16x2.f32 %r235, %r328, %r327;
cvt.rn.bf16x2.f32 %r236, %r330, %r329;
cvt.rn.bf16x2.f32 %r237, %r332, %r331;
cvt.rn.bf16x2.f32 %r238, %r334, %r333;
cvt.rn.bf16x2.f32 %r239, %r336, %r335;
cvt.rn.bf16x2.f32 %r240, %r338, %r337;
cvt.rn.bf16x2.f32 %r241, %r340, %r339;
cvt.rn.bf16x2.f32 %r242, %r342, %r341;
cvt.rn.bf16x2.f32 %r243, %r344, %r343;
cvt.rn.bf16x2.f32 %r244, %r346, %r345;
cvt.rn.bf16x2.f32 %r245, %r348, %r347;
cvt.rn.bf16x2.f32 %r246, %r350, %r349;
cvt.rn.bf16x2.f32 %r247, %r352, %r351;
cvt.rn.bf16x2.f32 %r248, %r354, %r353;
cvt.rn.bf16x2.f32 %r249, %r356, %r355;
cvt.rn.bf16x2.f32 %r250, %r358, %r357;
cvt.rn.bf16x2.f32 %r251, %r360, %r359;
cvt.rn.bf16x2.f32 %r252, %r362, %r361;
cvt.rn.bf16x2.f32 %r253, %r364, %r363;
cvt.rn.bf16x2.f32 %r254, %r366, %r365;
cvt.rn.bf16x2.f32 %r255, %r368, %r367;
cvt.rn.bf16x2.f32 %r256, %r370, %r369;
cvt.rn.bf16x2.f32 %r257, %r372, %r371;
cvt.rn.bf16x2.f32 %r258, %r374, %r373;
cvt.rn.bf16x2.f32 %r259, %r376, %r375;
cvt.rn.bf16x2.f32 %r260, %r378, %r377;
cvt.rn.bf16x2.f32 %r261, %r380, %r379;
cvt.rn.bf16x2.f32 %r262, %r382, %r381;
cvt.rn.bf16x2.f32 %r263, %r384, %r383;
mad.lo.s32 %r264, %r7, %r36, %r24;
mad.lo.s32 %r265, %r8, %r36, %r24;
mad.lo.s32 %r266, %r9, %r36, %r24;
mad.lo.s32 %r267, %r10, %r36, %r24;
mad.lo.s32 %r268, %r11, %r36, %r24;
mad.lo.s32 %r269, %r12, %r36, %r24;
mad.lo.s32 %r270, %r13, %r36, %r24;
mad.lo.s32 %r271, %r14, %r36, %r24;
mad.wide.s32 %rd72, %r264, 2, %rd1;
mad.wide.s32 %rd73, %r265, 2, %rd1;
mad.wide.s32 %rd74, %r266, 2, %rd1;
mad.wide.s32 %rd75, %r267, 2, %rd1;
mad.wide.s32 %rd76, %r268, 2, %rd1;
mad.wide.s32 %rd77, %r269, 2, %rd1;
mad.wide.s32 %rd78, %r270, 2, %rd1;
mad.wide.s32 %rd79, %r271, 2, %rd1;
setp.lt.s32 %p31, %r7, %r37;
setp.lt.s32 %p32, %r8, %r37;
setp.lt.s32 %p33, %r9, %r37;
setp.lt.s32 %p34, %r10, %r37;
setp.lt.s32 %p35, %r11, %r37;
setp.lt.s32 %p36, %r12, %r37;
setp.lt.s32 %p37, %r13, %r37;
setp.lt.s32 %p38, %r14, %r37;
setp.lt.s32 %p39, %r24, %r37;
and.pred %p15, %p31, %p39;
and.pred %p16, %p32, %p39;
and.pred %p17, %p33, %p39;
and.pred %p18, %p34, %p39;
and.pred %p19, %p35, %p39;
and.pred %p20, %p36, %p39;
and.pred %p21, %p37, %p39;
and.pred %p22, %p38, %p39;
and.b32 %r272, %r4, 3;
shl.b32 %r273, %r272, 13;
shl.b32 %r274, %r4, 5;
and.b32 %r275, %r274, 7264;
and.b32 %r276, %r4, 24;
shl.b32 %r277, %r276, 4;
shl.b32 %r278, %r4, 2;
and.b32 %r279, %r278, 16;
or.b32 %r280, %r273, %r279;
or.b32 %r281, %r275, %r277;
or.b32 %r282, %r280, %r281;
add.s32 %r283, %r133, %r282;
st.shared.v4.b32 [%r283], {%r232, %r234, %r236, %r238};
st.shared.v4.b32 [%r283+512], {%r233, %r235, %r237, %r239};
xor.b32 %r284, %r282, 32;
add.s32 %r285, %r133, %r284;
st.shared.v4.b32 [%r285], {%r240, %r242, %r244, %r246};
st.shared.v4.b32 [%r285+512], {%r241, %r243, %r245, %r247};
xor.b32 %r286, %r282, 64;
add.s32 %r287, %r133, %r286;
st.shared.v4.b32 [%r287], {%r248, %r250, %r252, %r254};
st.shared.v4.b32 [%r287+512], {%r249, %r251, %r253, %r255};
xor.b32 %r288, %r282, 96;
add.s32 %r289, %r133, %r288;
st.shared.v4.b32 [%r289], {%r256, %r258, %r260, %r262};
st.shared.v4.b32 [%r289+512], {%r257, %r259, %r261, %r263};
bar.sync 0;
shl.b32 %r290, %r276, 10;
shl.b32 %r291, %r272, 5;
and.b32 %r292, %r278, 1008;
or.b32 %r293, %r290, %r291;
xor.b32 %r294, %r293, %r292;
add.s32 %r295, %r133, %r294;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r170, %r171}, [%r295];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r174, %r175}, [%r295+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r176, %r177, %r178, %r179}, [%r295+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r180, %r181, %r182, %r183}, [%r295+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r184, %r185, %r186, %r187}, [%r295+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r188, %r189, %r190, %r191}, [%r295+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r192, %r193, %r194, %r195}, [%r295+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r196, %r197, %r198, %r199}, [%r295+7168];
@%p15 st.global.v4.b32 [ %rd72 + 0 ], { %r168, %r169, %r170, %r171 };
@%p16 st.global.v4.b32 [ %rd73 + 0 ], { %r172, %r173, %r174, %r175 };
@%p17 st.global.v4.b32 [ %rd74 + 0 ], { %r176, %r177, %r178, %r179 };
@%p18 st.global.v4.b32 [ %rd75 + 0 ], { %r180, %r181, %r182, %r183 };
@%p19 st.global.v4.b32 [ %rd76 + 0 ], { %r184, %r185, %r186, %r187 };
@%p20 st.global.v4.b32 [ %rd77 + 0 ], { %r188, %r189, %r190, %r191 };
@%p21 st.global.v4.b32 [ %rd78 + 0 ], { %r192, %r193, %r194, %r195 };
@%p22 st.global.v4.b32 [ %rd79 + 0 ], { %r196, %r197, %r198, %r199 };
mad.lo.s32 %r296, %r16, %r36, %r15;
mad.lo.s32 %r297, %r17, %r36, %r15;
mad.lo.s32 %r298, %r18, %r36, %r15;
mad.lo.s32 %r299, %r19, %r36, %r15;
mad.lo.s32 %r300, %r20, %r36, %r15;
mad.lo.s32 %r301, %r21, %r36, %r15;
mad.lo.s32 %r302, %r22, %r36, %r15;
mad.lo.s32 %r303, %r23, %r36, %r15;
mad.wide.s32 %rd80, %r296, 2, %rd1;
mad.wide.s32 %rd81, %r297, 2, %rd1;
mad.wide.s32 %rd82, %r298, 2, %rd1;
mad.wide.s32 %rd83, %r299, 2, %rd1;
mad.wide.s32 %rd84, %r300, 2, %rd1;
mad.wide.s32 %rd85, %r301, 2, %rd1;
mad.wide.s32 %rd86, %r302, 2, %rd1;
mad.wide.s32 %rd87, %r303, 2, %rd1;
setp.lt.s32 %p40, %r16, %r37;
setp.lt.s32 %p41, %r17, %r37;
setp.lt.s32 %p42, %r18, %r37;
setp.lt.s32 %p43, %r19, %r37;
setp.lt.s32 %p44, %r20, %r37;
setp.lt.s32 %p45, %r21, %r37;
setp.lt.s32 %p46, %r22, %r37;
setp.lt.s32 %p47, %r23, %r37;
setp.lt.s32 %p48, %r15, %r37;
and.pred %p23, %p40, %p48;
and.pred %p24, %p41, %p48;
and.pred %p25, %p42, %p48;
and.pred %p26, %p43, %p48;
and.pred %p27, %p44, %p48;
and.pred %p28, %p45, %p48;
and.pred %p29, %p46, %p48;
and.pred %p30, %p47, %p48;
bar.sync 0;
shl.b32 %r304, %r25, 9;
shl.b32 %r305, %r25, 4;
and.b32 %r306, %r28, 272;
and.b32 %r307, %r4, 96;
and.b32 %r308, %r6, 128;
or.b32 %r309, %r306, %r307;
xor.b32 %r310, %r309, %r305;
add.s32 %r311, %r133, %r304;
add.s32 %r312, %r311, %r308;
add.s32 %r313, %r312, %r310;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r232, %r233, %r234, %r235};
bar.sync 0;
and.b32 %r314, %r274, 3840;
or.b32 %r315, %r314, %r305;
xor.b32 %r316, %r315, %r5;
add.s32 %r317, %r133, %r316;
ld.shared.v4.b32 {%r200, %r201, %r202, %r203}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r236, %r237, %r238, %r239};
bar.sync 0;
ld.shared.v4.b32 {%r204, %r205, %r206, %r207}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r240, %r241, %r242, %r243};
bar.sync 0;
ld.shared.v4.b32 {%r208, %r209, %r210, %r211}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r244, %r245, %r246, %r247};
bar.sync 0;
ld.shared.v4.b32 {%r212, %r213, %r214, %r215}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r248, %r249, %r250, %r251};
bar.sync 0;
ld.shared.v4.b32 {%r216, %r217, %r218, %r219}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r252, %r253, %r254, %r255};
bar.sync 0;
ld.shared.v4.b32 {%r220, %r221, %r222, %r223}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r256, %r257, %r258, %r259};
bar.sync 0;
ld.shared.v4.b32 {%r224, %r225, %r226, %r227}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r260, %r261, %r262, %r263};
bar.sync 0;
ld.shared.v4.b32 {%r228, %r229, %r230, %r231}, [%r317];
@%p23 st.global.v4.b32 [ %rd80 + 0 ], { %r200, %r201, %r202, %r203 };
@%p24 st.global.v4.b32 [ %rd81 + 0 ], { %r204, %r205, %r206, %r207 };
@%p25 st.global.v4.b32 [ %rd82 + 0 ], { %r208, %r209, %r210, %r211 };
@%p26 st.global.v4.b32 [ %rd83 + 0 ], { %r212, %r213, %r214, %r215 };
@%p27 st.global.v4.b32 [ %rd84 + 0 ], { %r216, %r217, %r218, %r219 };
@%p28 st.global.v4.b32 [ %rd85 + 0 ], { %r220, %r221, %r222, %r223 };
@%p29 st.global.v4.b32 [ %rd86 + 0 ], { %r224, %r225, %r226, %r227 };
@%p30 st.global.v4.b32 [ %rd87 + 0 ], { %r228, %r229, %r230, %r231 };
$L__BB0_1:
ret;
$L__tmp8:
$L__func_end0:
}
  "#
  | .qkGram, .compact => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XXT_kernel(
.param .u64 .ptr .global .align 1 XXT_kernel_param_0,
.param .u64 .ptr .global .align 1 XXT_kernel_param_1,
.param .u32 XXT_kernel_param_2,
.param .u32 XXT_kernel_param_3,
.param .u32 XXT_kernel_param_4,
.param .u32 XXT_kernel_param_5,
.param .u32 XXT_kernel_param_6,
.param .u32 XXT_kernel_param_7,
.param .u64 .ptr .global .align 1 XXT_kernel_param_8,
.param .u64 .ptr .global .align 1 XXT_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<48>;
.reg .b32 %r<373>;
.reg .b64 %rd<89>;
$L__func_begin0:
ld.param.b32 %r15, [XXT_kernel_param_7];
ld.param.b32 %r14, [XXT_kernel_param_6];
ld.param.b32 %r13, [XXT_kernel_param_3];
ld.param.b32 %r12, [XXT_kernel_param_2];
ld.param.b64 %rd17, [XXT_kernel_param_1];
ld.param.b64 %rd42, [XXT_kernel_param_0];
$L__tmp0:
mov.u32 %r43, %ctaid.x;
$L__tmp1:
mul.hi.u32 %r1, %r43, 1431655766;
mul.lo.s32 %r44, %r1, 3;
sub.s32 %r45, %r43, %r44;
ld.param.b32 %r46, [XXT_kernel_param_4];
mul.lo.s32 %r47, %r45, 3;
ld.param.b32 %r48, [XXT_kernel_param_5];
mov.b32 %r49, 9216;
shr.u32 %r50, %r49, %r47;
and.b32 %r2, %r50, 896;
mov.b32 %r51, 8192;
shr.u32 %r52, %r51, %r47;
and.b32 %r3, %r52, 896;
$L__tmp2:
mul.lo.s32 %r53, %r46, %r1;
mad.wide.s32 %rd88, %r53, 2, %rd42;
mov.u32 %r4, %tid.x;
shr.u32 %r54, %r4, 3;
bfe.u32 %r55, %r4, 3, 5;
or.b32 %r56, %r55, 32;
or.b32 %r57, %r55, 64;
or.b32 %r58, %r54, 96;
or.b32 %r59, %r2, %r55;
or.b32 %r60, %r2, %r56;
or.b32 %r61, %r2, %r57;
or.b32 %r62, %r2, %r58;
rem.s32 %r63, %r59, %r12;
rem.s32 %r64, %r60, %r12;
rem.s32 %r65, %r61, %r12;
rem.s32 %r66, %r62, %r12;
or.b32 %r67, %r3, %r55;
or.b32 %r68, %r3, %r56;
or.b32 %r69, %r3, %r57;
or.b32 %r70, %r3, %r58;
rem.s32 %r71, %r67, %r12;
rem.s32 %r72, %r68, %r12;
rem.s32 %r73, %r69, %r12;
rem.s32 %r74, %r70, %r12;
and.b32 %r5, %r4, 7;
shl.b32 %r6, %r5, 3;
mad.lo.s32 %r75, %r63, %r48, %r6;
mad.lo.s32 %r76, %r64, %r48, %r6;
mad.lo.s32 %r77, %r65, %r48, %r6;
mad.lo.s32 %r78, %r66, %r48, %r6;
mad.wide.s32 %rd18, %r75, 2, %rd88;
mad.wide.s32 %rd19, %r76, 2, %rd88;
mad.wide.s32 %rd20, %r77, 2, %rd88;
mad.wide.s32 %rd21, %r78, 2, %rd88;
mad.lo.s32 %r79, %r71, %r48, %r6;
mad.lo.s32 %r80, %r72, %r48, %r6;
mad.lo.s32 %r81, %r73, %r48, %r6;
mad.lo.s32 %r82, %r74, %r48, %r6;
mad.wide.s32 %rd22, %r79, 2, %rd88;
mad.wide.s32 %rd23, %r80, 2, %rd88;
mad.wide.s32 %rd24, %r81, 2, %rd88;
mad.wide.s32 %rd25, %r82, 2, %rd88;
$L__tmp3:
add.s32 %r83, %r13, 63;
$L__tmp4:
setp.lt.s32 %p1, %r83, 64;
setp.gt.s32 %p2, %r83, 63;
setp.lt.s32 %p3, %r6, %r13;
shl.b32 %r87, %r4, 4;
and.b32 %r88, %r87, 4080;
shl.b32 %r8, %r4, 1;
and.b32 %r89, %r8, 112;
xor.b32 %r9, %r88, %r89;
mov.b32 %r90, global_smem;
add.s32 %r16, %r90, %r9;
selp.b32 %r91, 16, 0, %p2;
selp.b32 %r18, %r91, 0, %p3;
cp.async.cg.shared.global [ %r16 + 0 ], [ %rd18 + 0 ], 0x10, %r18;
add.s32 %r17, %r16, 4096;
cp.async.cg.shared.global [ %r17 + 0 ], [ %rd19 + 0 ], 0x10, %r18;
add.s32 %r19, %r16, 8192;
cp.async.cg.shared.global [ %r19 + 0 ], [ %rd20 + 0 ], 0x10, %r18;
add.s32 %r20, %r16, 12288;
cp.async.cg.shared.global [ %r20 + 0 ], [ %rd21 + 0 ], 0x10, %r18;
cp.async.commit_group;
add.s32 %r21, %r16, 65536;
cp.async.cg.shared.global [ %r21 + 0 ], [ %rd22 + 0 ], 0x10, %r18;
add.s32 %r22, %r16, 69632;
cp.async.cg.shared.global [ %r22 + 0 ], [ %rd23 + 0 ], 0x10, %r18;
add.s32 %r23, %r16, 73728;
cp.async.cg.shared.global [ %r23 + 0 ], [ %rd24 + 0 ], 0x10, %r18;
add.s32 %r24, %r16, 77824;
cp.async.cg.shared.global [ %r24 + 0 ], [ %rd25 + 0 ], 0x10, %r18;
cp.async.commit_group;
setp.gt.s32 %p4, %r83, 127;
add.s64 %rd26, %rd18, 128;
add.s64 %rd27, %rd19, 128;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s32 %r92, %r13, -64;
setp.lt.s32 %p5, %r6, %r92;
bar.sync 0;
add.s32 %r25, %r16, 16384;
selp.b32 %r93, 16, 0, %p5;
selp.b32 %r27, %r93, 0, %p4;
cp.async.cg.shared.global [ %r25 + 0 ], [ %rd26 + 0 ], 0x10, %r27;
add.s32 %r26, %r16, 20480;
cp.async.cg.shared.global [ %r26 + 0 ], [ %rd27 + 0 ], 0x10, %r27;
add.s32 %r28, %r16, 24576;
cp.async.cg.shared.global [ %r28 + 0 ], [ %rd28 + 0 ], 0x10, %r27;
add.s32 %r29, %r16, 28672;
cp.async.cg.shared.global [ %r29 + 0 ], [ %rd29 + 0 ], 0x10, %r27;
cp.async.commit_group;
add.s32 %r30, %r16, 81920;
cp.async.cg.shared.global [ %r30 + 0 ], [ %rd30 + 0 ], 0x10, %r27;
add.s32 %r31, %r16, 86016;
cp.async.cg.shared.global [ %r31 + 0 ], [ %rd31 + 0 ], 0x10, %r27;
add.s32 %r32, %r16, 90112;
cp.async.cg.shared.global [ %r32 + 0 ], [ %rd32 + 0 ], 0x10, %r27;
add.s32 %r33, %r16, 94208;
cp.async.cg.shared.global [ %r33 + 0 ], [ %rd33 + 0 ], 0x10, %r27;
cp.async.commit_group;
setp.gt.s32 %p6, %r83, 191;
add.s64 %rd34, %rd18, 256;
add.s64 %rd35, %rd19, 256;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s32 %r94, %r13, -128;
setp.lt.s32 %p7, %r6, %r94;
bar.sync 0;
add.s32 %r34, %r16, 32768;
selp.b32 %r95, 16, 0, %p7;
selp.b32 %r36, %r95, 0, %p6;
cp.async.cg.shared.global [ %r34 + 0 ], [ %rd34 + 0 ], 0x10, %r36;
add.s32 %r35, %r16, 36864;
cp.async.cg.shared.global [ %r35 + 0 ], [ %rd35 + 0 ], 0x10, %r36;
add.s32 %r37, %r16, 40960;
cp.async.cg.shared.global [ %r37 + 0 ], [ %rd36 + 0 ], 0x10, %r36;
add.s32 %r38, %r16, 45056;
cp.async.cg.shared.global [ %r38 + 0 ], [ %rd37 + 0 ], 0x10, %r36;
cp.async.commit_group;
add.s32 %r39, %r16, 98304;
cp.async.cg.shared.global [ %r39 + 0 ], [ %rd38 + 0 ], 0x10, %r36;
add.s32 %r40, %r16, 102400;
cp.async.cg.shared.global [ %r40 + 0 ], [ %rd39 + 0 ], 0x10, %r36;
add.s32 %r41, %r16, 106496;
cp.async.cg.shared.global [ %r41 + 0 ], [ %rd40 + 0 ], 0x10, %r36;
add.s32 %r42, %r16, 110592;
cp.async.cg.shared.global [ %r42 + 0 ], [ %rd41 + 0 ], 0x10, %r36;
cp.async.commit_group;
mov.b32 %r308, 0f00000000;
mov.b32 %r309, %r308;
mov.b32 %r310, %r308;
mov.b32 %r311, %r308;
mov.b32 %r312, %r308;
mov.b32 %r313, %r308;
mov.b32 %r314, %r308;
mov.b32 %r315, %r308;
mov.b32 %r316, %r308;
mov.b32 %r317, %r308;
mov.b32 %r318, %r308;
mov.b32 %r319, %r308;
mov.b32 %r320, %r308;
mov.b32 %r321, %r308;
mov.b32 %r322, %r308;
mov.b32 %r323, %r308;
mov.b32 %r324, %r308;
mov.b32 %r325, %r308;
mov.b32 %r326, %r308;
mov.b32 %r327, %r308;
mov.b32 %r328, %r308;
mov.b32 %r329, %r308;
mov.b32 %r330, %r308;
mov.b32 %r331, %r308;
mov.b32 %r332, %r308;
mov.b32 %r333, %r308;
mov.b32 %r334, %r308;
mov.b32 %r335, %r308;
mov.b32 %r336, %r308;
mov.b32 %r337, %r308;
mov.b32 %r338, %r308;
mov.b32 %r339, %r308;
mov.b32 %r340, %r308;
mov.b32 %r341, %r308;
mov.b32 %r342, %r308;
mov.b32 %r343, %r308;
mov.b32 %r344, %r308;
mov.b32 %r345, %r308;
mov.b32 %r346, %r308;
mov.b32 %r347, %r308;
mov.b32 %r348, %r308;
mov.b32 %r349, %r308;
mov.b32 %r350, %r308;
mov.b32 %r351, %r308;
mov.b32 %r352, %r308;
mov.b32 %r353, %r308;
mov.b32 %r354, %r308;
mov.b32 %r355, %r308;
mov.b32 %r356, %r308;
mov.b32 %r357, %r308;
mov.b32 %r358, %r308;
mov.b32 %r359, %r308;
mov.b32 %r360, %r308;
mov.b32 %r361, %r308;
mov.b32 %r362, %r308;
mov.b32 %r363, %r308;
mov.b32 %r364, %r308;
mov.b32 %r365, %r308;
mov.b32 %r366, %r308;
mov.b32 %r367, %r308;
mov.b32 %r368, %r308;
mov.b32 %r369, %r308;
mov.b32 %r370, %r308;
mov.b32 %r371, %r308;
@%p1 bra $L__BB0_3;
cvt.s64.s32 %rd1, %r75;
cvt.s64.s32 %rd2, %r76;
cvt.s64.s32 %rd3, %r77;
cvt.s64.s32 %rd4, %r78;
cvt.s64.s32 %rd5, %r79;
cvt.s64.s32 %rd6, %r80;
cvt.s64.s32 %rd7, %r81;
cvt.s64.s32 %rd8, %r82;
shr.s32 %r84, %r83, 31;
shr.u32 %r85, %r84, 26;
add.s32 %r86, %r83, %r85;
shr.s32 %r7, %r86, 6;
add.s32 %r10, %r7, -3;
shr.u32 %r11, %r4, 5;
add.s32 %r305, %r13, -192;
shl.b64 %rd43, %rd8, 1;
add.s64 %rd9, %rd43, 384;
shl.b64 %rd44, %rd7, 1;
add.s64 %rd10, %rd44, 384;
shl.b64 %rd45, %rd6, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd5, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd4, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd3, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd2, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd1, 1;
add.s64 %rd16, %rd50, 384;
mov.b32 %r100, 0;
mov.b32 %r308, 0f00000000;
mov.b32 %r307, 2;
mov.b32 %r306, -1;
mov.b32 %r309, %r308;
mov.b32 %r310, %r308;
mov.b32 %r311, %r308;
mov.b32 %r312, %r308;
mov.b32 %r313, %r308;
mov.b32 %r314, %r308;
mov.b32 %r315, %r308;
mov.b32 %r316, %r308;
mov.b32 %r317, %r308;
mov.b32 %r318, %r308;
mov.b32 %r319, %r308;
mov.b32 %r320, %r308;
mov.b32 %r321, %r308;
mov.b32 %r322, %r308;
mov.b32 %r323, %r308;
mov.b32 %r324, %r308;
mov.b32 %r325, %r308;
mov.b32 %r326, %r308;
mov.b32 %r327, %r308;
mov.b32 %r328, %r308;
mov.b32 %r329, %r308;
mov.b32 %r330, %r308;
mov.b32 %r331, %r308;
mov.b32 %r332, %r308;
mov.b32 %r333, %r308;
mov.b32 %r334, %r308;
mov.b32 %r335, %r308;
mov.b32 %r336, %r308;
mov.b32 %r337, %r308;
mov.b32 %r338, %r308;
mov.b32 %r339, %r308;
mov.b32 %r340, %r308;
mov.b32 %r341, %r308;
mov.b32 %r342, %r308;
mov.b32 %r343, %r308;
mov.b32 %r344, %r308;
mov.b32 %r345, %r308;
mov.b32 %r346, %r308;
mov.b32 %r347, %r308;
mov.b32 %r348, %r308;
mov.b32 %r349, %r308;
mov.b32 %r350, %r308;
mov.b32 %r351, %r308;
mov.b32 %r352, %r308;
mov.b32 %r353, %r308;
mov.b32 %r354, %r308;
mov.b32 %r355, %r308;
mov.b32 %r356, %r308;
mov.b32 %r357, %r308;
mov.b32 %r358, %r308;
mov.b32 %r359, %r308;
mov.b32 %r360, %r308;
mov.b32 %r361, %r308;
mov.b32 %r362, %r308;
mov.b32 %r363, %r308;
mov.b32 %r364, %r308;
mov.b32 %r365, %r308;
mov.b32 %r366, %r308;
mov.b32 %r367, %r308;
mov.b32 %r368, %r308;
mov.b32 %r369, %r308;
mov.b32 %r370, %r308;
mov.b32 %r371, %r308;
mov.b32 %r372, %r100;
$L__BB0_2:
setp.lt.s32 %p9, %r372, %r10;
add.s32 %r110, %r306, 1;
setp.gt.s32 %p10, %r110, 3;
selp.b32 %r306, 0, %r110, %p10;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r111, %r306, 14;
add.s32 %r112, %r90, %r111;
add.s32 %r113, %r112, 65536;
shr.u32 %r114, %r112, 4;
shfl.sync.idx.b32 %r115, %r11, 0, 31, -1;
shl.b32 %r116, %r115, 7;
and.b32 %r117, %r116, 512;
add.s32 %r118, %r117, %r114;
and.b32 %r119, %r118, 16383;
cvt.u64.u32 %rd69, %r119;
bfe.u32 %r120, %r113, 4, 14;
cvt.u64.u32 %rd70, %r120;
wgmma.fence.sync.aligned;
or.b64 %rd51, %rd69, 4611686293305294848;
or.b64 %rd52, %rd70, 4611686293305294848;
mov.pred %p8, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r308,%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371}, %rd51, %rd52, %p8, 1, 1, 0, 0;
add.s64 %rd53, %rd69, 4611686293305294850;
add.s64 %rd54, %rd70, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r308,%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371}, %rd53, %rd54, %p8, 1, 1, 0, 0;
add.s64 %rd55, %rd69, 4611686293305294852;
add.s64 %rd56, %rd70, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r308,%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371}, %rd55, %rd56, %p8, 1, 1, 0, 0;
add.s64 %rd57, %rd69, 4611686293305294854;
add.s64 %rd58, %rd70, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r308,%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371}, %rd57, %rd58, %p8, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd59, %r112;
cvt.u64.u32 %rd60, %r113;
mov.b32 %r99, %r100;
mov.b32 %r96, %r100;
mov.b32 %r97, %r100;
mov.b32 %r98, %r100;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd61, %rd88, %rd16;
add.s64 %rd62, %rd88, %rd15;
add.s64 %rd63, %rd88, %rd14;
add.s64 %rd64, %rd88, %rd13;
add.s64 %rd65, %rd88, %rd12;
add.s64 %rd66, %rd88, %rd11;
add.s64 %rd67, %rd88, %rd10;
add.s64 %rd68, %rd88, %rd9;
add.s32 %r121, %r307, 1;
setp.gt.s32 %p11, %r121, 3;
selp.b32 %r307, 0, %r121, %p11;
setp.lt.s32 %p12, %r6, %r305;
shl.b32 %r122, %r307, 14;
add.s32 %r123, %r90, %r122;
bar.sync 0;
add.s32 %r101, %r123, %r9;
selp.b32 %r124, 16, 0, %p12;
selp.b32 %r103, %r124, 0, %p9;
cp.async.cg.shared.global [ %r101 + 0 ], [ %rd61 + 0 ], 0x10, %r103;
add.s32 %r102, %r101, 4096;
cp.async.cg.shared.global [ %r102 + 0 ], [ %rd62 + 0 ], 0x10, %r103;
add.s32 %r104, %r101, 8192;
cp.async.cg.shared.global [ %r104 + 0 ], [ %rd63 + 0 ], 0x10, %r103;
add.s32 %r105, %r101, 12288;
cp.async.cg.shared.global [ %r105 + 0 ], [ %rd64 + 0 ], 0x10, %r103;
cp.async.commit_group;
add.s32 %r106, %r101, 65536;
cp.async.cg.shared.global [ %r106 + 0 ], [ %rd65 + 0 ], 0x10, %r103;
add.s32 %r107, %r101, 69632;
cp.async.cg.shared.global [ %r107 + 0 ], [ %rd66 + 0 ], 0x10, %r103;
add.s32 %r108, %r101, 73728;
cp.async.cg.shared.global [ %r108 + 0 ], [ %rd67 + 0 ], 0x10, %r103;
add.s32 %r109, %r101, 77824;
cp.async.cg.shared.global [ %r109 + 0 ], [ %rd68 + 0 ], 0x10, %r103;
cp.async.commit_group;
add.s32 %r372, %r372, 1;
add.s64 %rd88, %rd88, 128;
add.s32 %r305, %r305, -64;
setp.ne.b32 %p13, %r7, %r372;
@%p13 bra $L__BB0_2;
$L__BB0_3:
shl.b32 %r189, %r4, 3;
and.b32 %r190, %r189, 120;
or.b32 %r191, %r3, %r190;
and.b32 %r192, %r4, 240;
bfe.u32 %r193, %r4, 4, 4;
or.b32 %r194, %r193, 112;
or.b32 %r195, %r3, %r194;
or.b32 %r196, %r193, 96;
or.b32 %r197, %r3, %r196;
or.b32 %r198, %r193, 80;
or.b32 %r199, %r3, %r198;
or.b32 %r200, %r193, 64;
or.b32 %r201, %r3, %r200;
or.b32 %r202, %r193, 48;
or.b32 %r203, %r3, %r202;
or.b32 %r204, %r193, 32;
or.b32 %r205, %r3, %r204;
or.b32 %r206, %r193, 16;
or.b32 %r207, %r3, %r206;
or.b32 %r208, %r3, %r193;
or.b32 %r209, %r2, %r190;
or.b32 %r210, %r2, %r194;
or.b32 %r211, %r2, %r196;
or.b32 %r212, %r2, %r198;
or.b32 %r213, %r2, %r200;
or.b32 %r214, %r2, %r202;
or.b32 %r215, %r2, %r204;
or.b32 %r216, %r2, %r206;
or.b32 %r217, %r2, %r193;
mul.lo.s32 %r218, %r14, %r1;
mad.wide.s32 %rd87, %r218, 2, %rd17;
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r219, %r309, %r308;
cvt.rn.bf16x2.f32 %r220, %r311, %r310;
cvt.rn.bf16x2.f32 %r221, %r313, %r312;
cvt.rn.bf16x2.f32 %r222, %r315, %r314;
cvt.rn.bf16x2.f32 %r223, %r317, %r316;
cvt.rn.bf16x2.f32 %r224, %r319, %r318;
cvt.rn.bf16x2.f32 %r225, %r321, %r320;
cvt.rn.bf16x2.f32 %r226, %r323, %r322;
cvt.rn.bf16x2.f32 %r227, %r325, %r324;
cvt.rn.bf16x2.f32 %r228, %r327, %r326;
cvt.rn.bf16x2.f32 %r229, %r329, %r328;
cvt.rn.bf16x2.f32 %r230, %r331, %r330;
cvt.rn.bf16x2.f32 %r231, %r333, %r332;
cvt.rn.bf16x2.f32 %r232, %r335, %r334;
cvt.rn.bf16x2.f32 %r233, %r337, %r336;
cvt.rn.bf16x2.f32 %r234, %r339, %r338;
cvt.rn.bf16x2.f32 %r235, %r341, %r340;
cvt.rn.bf16x2.f32 %r236, %r343, %r342;
cvt.rn.bf16x2.f32 %r237, %r345, %r344;
cvt.rn.bf16x2.f32 %r238, %r347, %r346;
cvt.rn.bf16x2.f32 %r239, %r349, %r348;
cvt.rn.bf16x2.f32 %r240, %r351, %r350;
cvt.rn.bf16x2.f32 %r241, %r353, %r352;
cvt.rn.bf16x2.f32 %r242, %r355, %r354;
cvt.rn.bf16x2.f32 %r243, %r357, %r356;
cvt.rn.bf16x2.f32 %r244, %r359, %r358;
cvt.rn.bf16x2.f32 %r245, %r361, %r360;
cvt.rn.bf16x2.f32 %r246, %r363, %r362;
cvt.rn.bf16x2.f32 %r247, %r365, %r364;
cvt.rn.bf16x2.f32 %r248, %r367, %r366;
cvt.rn.bf16x2.f32 %r249, %r369, %r368;
cvt.rn.bf16x2.f32 %r250, %r371, %r370;
mad.lo.s32 %r251, %r15, %r217, %r191;
mad.lo.s32 %r252, %r15, %r216, %r191;
mad.lo.s32 %r253, %r15, %r215, %r191;
mad.lo.s32 %r254, %r15, %r214, %r191;
mad.lo.s32 %r255, %r15, %r213, %r191;
mad.lo.s32 %r256, %r15, %r212, %r191;
mad.lo.s32 %r257, %r15, %r211, %r191;
mad.lo.s32 %r258, %r15, %r210, %r191;
mad.wide.s32 %rd71, %r251, 2, %rd87;
mad.wide.s32 %rd72, %r252, 2, %rd87;
mad.wide.s32 %rd73, %r253, 2, %rd87;
mad.wide.s32 %rd74, %r254, 2, %rd87;
mad.wide.s32 %rd75, %r255, 2, %rd87;
mad.wide.s32 %rd76, %r256, 2, %rd87;
mad.wide.s32 %rd77, %r257, 2, %rd87;
mad.wide.s32 %rd78, %r258, 2, %rd87;
setp.lt.s32 %p30, %r217, %r12;
setp.lt.s32 %p31, %r216, %r12;
setp.lt.s32 %p32, %r215, %r12;
setp.lt.s32 %p33, %r214, %r12;
setp.lt.s32 %p34, %r213, %r12;
setp.lt.s32 %p35, %r212, %r12;
setp.lt.s32 %p36, %r211, %r12;
setp.lt.s32 %p37, %r210, %r12;
setp.lt.s32 %p38, %r191, %r12;
and.pred %p14, %p30, %p38;
and.pred %p15, %p31, %p38;
and.pred %p16, %p32, %p38;
and.pred %p17, %p33, %p38;
and.pred %p18, %p34, %p38;
and.pred %p19, %p35, %p38;
and.pred %p20, %p36, %p38;
and.pred %p21, %p37, %p38;
and.b32 %r259, %r4, 3;
shl.b32 %r260, %r259, 13;
shl.b32 %r261, %r4, 5;
and.b32 %r262, %r261, 7264;
and.b32 %r263, %r4, 24;
shl.b32 %r264, %r263, 4;
shl.b32 %r265, %r4, 2;
and.b32 %r266, %r265, 16;
or.b32 %r267, %r260, %r266;
or.b32 %r268, %r262, %r264;
or.b32 %r269, %r267, %r268;
add.s32 %r270, %r90, %r269;
st.shared.v4.b32 [%r270], {%r219, %r221, %r223, %r225};
st.shared.v4.b32 [%r270+512], {%r220, %r222, %r224, %r226};
xor.b32 %r271, %r269, 32;
add.s32 %r272, %r90, %r271;
st.shared.v4.b32 [%r272], {%r227, %r229, %r231, %r233};
st.shared.v4.b32 [%r272+512], {%r228, %r230, %r232, %r234};
xor.b32 %r273, %r269, 64;
add.s32 %r274, %r90, %r273;
st.shared.v4.b32 [%r274], {%r235, %r237, %r239, %r241};
st.shared.v4.b32 [%r274+512], {%r236, %r238, %r240, %r242};
xor.b32 %r275, %r269, 96;
add.s32 %r276, %r90, %r275;
st.shared.v4.b32 [%r276], {%r243, %r245, %r247, %r249};
st.shared.v4.b32 [%r276+512], {%r244, %r246, %r248, %r250};
bar.sync 0;
shl.b32 %r277, %r263, 10;
shl.b32 %r278, %r259, 5;
and.b32 %r279, %r265, 1008;
or.b32 %r280, %r277, %r278;
xor.b32 %r281, %r280, %r279;
add.s32 %r282, %r90, %r281;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r282];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r282+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r282+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r137, %r138, %r139, %r140}, [%r282+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r141, %r142, %r143, %r144}, [%r282+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r147, %r148}, [%r282+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r151, %r152}, [%r282+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r282+7168];
@%p14 st.global.v4.b32 [ %rd71 + 0 ], { %r125, %r126, %r127, %r128 };
@%p15 st.global.v4.b32 [ %rd72 + 0 ], { %r129, %r130, %r131, %r132 };
@%p16 st.global.v4.b32 [ %rd73 + 0 ], { %r133, %r134, %r135, %r136 };
@%p17 st.global.v4.b32 [ %rd74 + 0 ], { %r137, %r138, %r139, %r140 };
@%p18 st.global.v4.b32 [ %rd75 + 0 ], { %r141, %r142, %r143, %r144 };
@%p19 st.global.v4.b32 [ %rd76 + 0 ], { %r145, %r146, %r147, %r148 };
@%p20 st.global.v4.b32 [ %rd77 + 0 ], { %r149, %r150, %r151, %r152 };
@%p21 st.global.v4.b32 [ %rd78 + 0 ], { %r153, %r154, %r155, %r156 };
mad.lo.s32 %r283, %r15, %r208, %r209;
mad.lo.s32 %r284, %r15, %r207, %r209;
mad.lo.s32 %r285, %r15, %r205, %r209;
mad.lo.s32 %r286, %r15, %r203, %r209;
mad.lo.s32 %r287, %r15, %r201, %r209;
mad.lo.s32 %r288, %r15, %r199, %r209;
mad.lo.s32 %r289, %r15, %r197, %r209;
mad.lo.s32 %r290, %r15, %r195, %r209;
mad.wide.s32 %rd79, %r283, 2, %rd87;
mad.wide.s32 %rd80, %r284, 2, %rd87;
mad.wide.s32 %rd81, %r285, 2, %rd87;
mad.wide.s32 %rd82, %r286, 2, %rd87;
mad.wide.s32 %rd83, %r287, 2, %rd87;
mad.wide.s32 %rd84, %r288, 2, %rd87;
mad.wide.s32 %rd85, %r289, 2, %rd87;
mad.wide.s32 %rd86, %r290, 2, %rd87;
setp.lt.s32 %p39, %r208, %r12;
setp.lt.s32 %p40, %r207, %r12;
setp.lt.s32 %p41, %r205, %r12;
setp.lt.s32 %p42, %r203, %r12;
setp.lt.s32 %p43, %r201, %r12;
setp.lt.s32 %p44, %r199, %r12;
setp.lt.s32 %p45, %r197, %r12;
setp.lt.s32 %p46, %r195, %r12;
setp.lt.s32 %p47, %r209, %r12;
and.pred %p22, %p39, %p47;
and.pred %p23, %p40, %p47;
and.pred %p24, %p41, %p47;
and.pred %p25, %p42, %p47;
and.pred %p26, %p43, %p47;
and.pred %p27, %p44, %p47;
and.pred %p28, %p45, %p47;
and.pred %p29, %p46, %p47;
bar.sync 0;
shl.b32 %r291, %r5, 9;
shl.b32 %r292, %r5, 4;
and.b32 %r293, %r8, 272;
and.b32 %r294, %r4, 96;
and.b32 %r295, %r189, 128;
or.b32 %r296, %r293, %r294;
xor.b32 %r297, %r296, %r292;
add.s32 %r298, %r90, %r291;
add.s32 %r299, %r298, %r295;
add.s32 %r300, %r299, %r297;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r219, %r220, %r221, %r222};
bar.sync 0;
and.b32 %r301, %r261, 3840;
or.b32 %r302, %r301, %r292;
xor.b32 %r303, %r302, %r192;
add.s32 %r304, %r90, %r303;
ld.shared.v4.b32 {%r157, %r158, %r159, %r160}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r223, %r224, %r225, %r226};
bar.sync 0;
ld.shared.v4.b32 {%r161, %r162, %r163, %r164}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r227, %r228, %r229, %r230};
bar.sync 0;
ld.shared.v4.b32 {%r165, %r166, %r167, %r168}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r231, %r232, %r233, %r234};
bar.sync 0;
ld.shared.v4.b32 {%r169, %r170, %r171, %r172}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r235, %r236, %r237, %r238};
bar.sync 0;
ld.shared.v4.b32 {%r173, %r174, %r175, %r176}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r239, %r240, %r241, %r242};
bar.sync 0;
ld.shared.v4.b32 {%r177, %r178, %r179, %r180}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r243, %r244, %r245, %r246};
bar.sync 0;
ld.shared.v4.b32 {%r181, %r182, %r183, %r184}, [%r304];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r300], {%r247, %r248, %r249, %r250};
bar.sync 0;
ld.shared.v4.b32 {%r185, %r186, %r187, %r188}, [%r304];
@%p22 st.global.v4.b32 [ %rd79 + 0 ], { %r157, %r158, %r159, %r160 };
@%p23 st.global.v4.b32 [ %rd80 + 0 ], { %r161, %r162, %r163, %r164 };
@%p24 st.global.v4.b32 [ %rd81 + 0 ], { %r165, %r166, %r167, %r168 };
@%p25 st.global.v4.b32 [ %rd82 + 0 ], { %r169, %r170, %r171, %r172 };
@%p26 st.global.v4.b32 [ %rd83 + 0 ], { %r173, %r174, %r175, %r176 };
@%p27 st.global.v4.b32 [ %rd84 + 0 ], { %r177, %r178, %r179, %r180 };
@%p28 st.global.v4.b32 [ %rd85 + 0 ], { %r181, %r182, %r183, %r184 };
@%p29 st.global.v4.b32 [ %rd86 + 0 ], { %r185, %r186, %r187, %r188 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .qkGram, .full => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry polar(
.param .align 64 .b8 polar_param_0[128],
.param .u32 polar_param_1,
.param .u32 polar_param_2,
.param .u64 polar_param_3,
.param .u64 polar_param_4,
.param .align 64 .b8 polar_param_5[128],
.param .u32 polar_param_6,
.param .u32 polar_param_7,
.param .u64 polar_param_8,
.param .u64 polar_param_9,
.param .u64 .ptr .global .align 1 polar_param_10,
.param .u64 .ptr .global .align 1 polar_param_11
)
.reqntid 256
{
.reg .pred %p<32>;
.reg .b32 %r<221>;
.reg .b64 %rd<41>;
$L__func_begin0:
mov.b64 %rd1, polar_param_0;
mov.b64 %rd2, polar_param_5;
$L__tmp0:
cvta.param.u64 %rd40, %rd2;
cvta.param.u64 %rd21, %rd1;
mov.u32 %r45, %ctaid.x;
mul.hi.u32 %r46, %r45, 1431655766;
$L__tmp1:
mul.lo.s32 %r47, %r46, 3;
sub.s32 %r48, %r45, %r47;
mul.lo.s32 %r49, %r48, 3;
mov.b32 %r50, 9216;
shr.u32 %r51, %r50, %r49;
and.b32 %r1, %r51, 896;
mov.b32 %r52, 8192;
shr.u32 %r53, %r52, %r49;
and.b32 %r2, %r53, 896;
$L__tmp2:
setp.eq.b32 %p2, %r1, %r2;
mov.u32 %r3, %tid.x;
and.b32 %r4, %r3, 255;
setp.eq.b32 %p1, %r4, 0;
mov.b32 %r54, global_smem;
add.s32 %r84, %r54, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r84], 1;
bar.sync 0;
add.s32 %r44, %r54, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r44], 1;
shl.b32 %r5, %r46, 8;
$L__tmp3:
add.s32 %r135, %r1, %r5;
shr.u32 %r6, %r3, 5;
setp.lt.u32 %p31, %r4, 64;
add.s32 %r70, %r2, %r5;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r84], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r87, %r6, 0, 31, -1;
elect.sync %r88|%p18, -1;
and.pred %p17, %p31, %p18;
and.b32 %r89, %r87, 1;
shl.b32 %r90, %r89, 14;
add.s32 %r85, %r54, %r90;
shl.b32 %r86, %r89, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r85], [%rd21, {%r86, %r135}], [%r84];
shl.b32 %r91, %r87, 7;
and.b32 %r9, %r91, 512;
$L__tmp6:
or.b32 %r217, %r86, 128;
mov.b32 %r218, 0;
mov.b32 %r152, 0f00000000;
mov.b32 %r153, %r152;
mov.b32 %r154, %r152;
mov.b32 %r155, %r152;
mov.b32 %r156, %r152;
mov.b32 %r157, %r152;
mov.b32 %r158, %r152;
mov.b32 %r159, %r152;
mov.b32 %r160, %r152;
mov.b32 %r161, %r152;
mov.b32 %r162, %r152;
mov.b32 %r163, %r152;
mov.b32 %r164, %r152;
mov.b32 %r165, %r152;
mov.b32 %r166, %r152;
mov.b32 %r167, %r152;
mov.b32 %r168, %r152;
mov.b32 %r169, %r152;
mov.b32 %r170, %r152;
mov.b32 %r171, %r152;
mov.b32 %r172, %r152;
mov.b32 %r173, %r152;
mov.b32 %r174, %r152;
mov.b32 %r175, %r152;
mov.b32 %r176, %r152;
mov.b32 %r177, %r152;
mov.b32 %r178, %r152;
mov.b32 %r179, %r152;
mov.b32 %r180, %r152;
mov.b32 %r181, %r152;
mov.b32 %r182, %r152;
mov.b32 %r183, %r152;
mov.b32 %r184, %r152;
mov.b32 %r185, %r152;
mov.b32 %r186, %r152;
mov.b32 %r187, %r152;
mov.b32 %r188, %r152;
mov.b32 %r189, %r152;
mov.b32 %r190, %r152;
mov.b32 %r191, %r152;
mov.b32 %r192, %r152;
mov.b32 %r193, %r152;
mov.b32 %r194, %r152;
mov.b32 %r195, %r152;
mov.b32 %r196, %r152;
mov.b32 %r197, %r152;
mov.b32 %r198, %r152;
mov.b32 %r199, %r152;
mov.b32 %r200, %r152;
mov.b32 %r201, %r152;
mov.b32 %r202, %r152;
mov.b32 %r203, %r152;
mov.b32 %r204, %r152;
mov.b32 %r205, %r152;
mov.b32 %r206, %r152;
mov.b32 %r207, %r152;
mov.b32 %r208, %r152;
mov.b32 %r209, %r152;
mov.b32 %r210, %r152;
mov.b32 %r211, %r152;
mov.b32 %r212, %r152;
mov.b32 %r213, %r152;
mov.b32 %r214, %r152;
mov.b32 %r215, %r152;
$L__BB0_4:
and.b32 %r96, %r218, 1;
bfe.u32 %r93, %r218, 1, 1;
shl.b32 %r97, %r96, 15;
add.s32 %r98, %r54, %r97;
mad.lo.s32 %r99, %r96, -32760, %r98;
add.s32 %r92, %r99, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r92], %r93;
@!complete bra.uni waitLoop;
}
shr.u32 %r100, %r98, 4;
add.s32 %r101, %r100, %r9;
and.b32 %r102, %r101, 16383;
cvt.u64.u32 %rd38, %r102;
bfe.u32 %r103, %r98, 4, 14;
cvt.u64.u32 %rd39, %r103;
wgmma.fence.sync.aligned;
or.b64 %rd22, %rd38, 4611686293305294848;
or.b64 %rd23, %rd39, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd22, %rd23, %p19, 1, 1, 0, 0;
add.s64 %rd24, %rd38, 4611686293305294850;
add.s64 %rd25, %rd39, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd24, %rd25, %p19, 1, 1, 0, 0;
add.s64 %rd26, %rd38, 4611686293305294852;
add.s64 %rd27, %rd39, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd26, %rd27, %p19, 1, 1, 0, 0;
add.s64 %rd28, %rd38, 4611686293305294854;
add.s64 %rd29, %rd39, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd28, %rd29, %p19, 1, 1, 0, 0;
add.s64 %rd30, %rd38, 4611686293305295872;
add.s64 %rd31, %rd39, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd30, %rd31, %p19, 1, 1, 0, 0;
add.s64 %rd32, %rd38, 4611686293305295874;
add.s64 %rd33, %rd39, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd32, %rd33, %p19, 1, 1, 0, 0;
add.s64 %rd34, %rd38, 4611686293305295876;
add.s64 %rd35, %rd39, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd34, %rd35, %p19, 1, 1, 0, 0;
add.s64 %rd36, %rd38, 4611686293305295878;
add.s64 %rd37, %rd39, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd36, %rd37, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r10, %r218, 1;
and.b32 %r104, %r10, 1;
setp.ne.b32 %p22, %r218, 5;
shl.b32 %r105, %r104, 3;
add.s32 %r106, %r54, %r105;
add.s32 %r94, %r106, 98304;
shl.b32 %r107, %r104, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r94], 32768;
bar.sync 0;
elect.sync %r108|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p31, %p24;
add.s32 %r95, %r85, %r107;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r95], [%rd21, {%r217, %r135}], [%r94];
$L__tmp8:
add.s32 %r217, %r217, 128;
setp.ne.b32 %p25, %r10, 6;
mov.b32 %r218, %r10;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r84], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r57, %r6, 0, 31, -1;
elect.sync %r58|%p5, -1;
and.pred %p3, %p31, %p5;
and.b32 %r59, %r57, 1;
shl.b32 %r60, %r59, 14;
add.s32 %r55, %r54, %r60;
shl.b32 %r56, %r59, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r55], [%rd21, {%r56, %r135}], [%r84];
$L__tmp10:
bar.sync 0;
elect.sync %r61|%p6, -1;
and.pred %p4, %p31, %p6;
add.s32 %r69, %r55, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r69], [%rd21, {%r56, %r70}], [%r84];
shl.b32 %r62, %r57, 7;
and.b32 %r7, %r62, 512;
add.s32 %r63, %r54, 65536;
bfe.u32 %r64, %r63, 4, 14;
cvt.u64.u32 %rd3, %r64;
or.b64 %rd5, %rd3, 4611686293305294848;
add.s64 %rd7, %rd3, 4611686293305294850;
add.s64 %rd9, %rd3, 4611686293305294852;
add.s64 %rd11, %rd3, 4611686293305294854;
add.s64 %rd13, %rd3, 4611686293305295872;
add.s64 %rd15, %rd3, 4611686293305295874;
add.s64 %rd17, %rd3, 4611686293305295876;
add.s64 %rd19, %rd3, 4611686293305295878;
$L__tmp11:
or.b32 %r151, %r56, 128;
mov.b32 %r216, 0;
mov.b32 %r152, 0f00000000;
mov.b32 %r153, %r152;
mov.b32 %r154, %r152;
mov.b32 %r155, %r152;
mov.b32 %r156, %r152;
mov.b32 %r157, %r152;
mov.b32 %r158, %r152;
mov.b32 %r159, %r152;
mov.b32 %r160, %r152;
mov.b32 %r161, %r152;
mov.b32 %r162, %r152;
mov.b32 %r163, %r152;
mov.b32 %r164, %r152;
mov.b32 %r165, %r152;
mov.b32 %r166, %r152;
mov.b32 %r167, %r152;
mov.b32 %r168, %r152;
mov.b32 %r169, %r152;
mov.b32 %r170, %r152;
mov.b32 %r171, %r152;
mov.b32 %r172, %r152;
mov.b32 %r173, %r152;
mov.b32 %r174, %r152;
mov.b32 %r175, %r152;
mov.b32 %r176, %r152;
mov.b32 %r177, %r152;
mov.b32 %r178, %r152;
mov.b32 %r179, %r152;
mov.b32 %r180, %r152;
mov.b32 %r181, %r152;
mov.b32 %r182, %r152;
mov.b32 %r183, %r152;
mov.b32 %r184, %r152;
mov.b32 %r185, %r152;
mov.b32 %r186, %r152;
mov.b32 %r187, %r152;
mov.b32 %r188, %r152;
mov.b32 %r189, %r152;
mov.b32 %r190, %r152;
mov.b32 %r191, %r152;
mov.b32 %r192, %r152;
mov.b32 %r193, %r152;
mov.b32 %r194, %r152;
mov.b32 %r195, %r152;
mov.b32 %r196, %r152;
mov.b32 %r197, %r152;
mov.b32 %r198, %r152;
mov.b32 %r199, %r152;
mov.b32 %r200, %r152;
mov.b32 %r201, %r152;
mov.b32 %r202, %r152;
mov.b32 %r203, %r152;
mov.b32 %r204, %r152;
mov.b32 %r205, %r152;
mov.b32 %r206, %r152;
mov.b32 %r207, %r152;
mov.b32 %r208, %r152;
mov.b32 %r209, %r152;
mov.b32 %r210, %r152;
mov.b32 %r211, %r152;
mov.b32 %r212, %r152;
mov.b32 %r213, %r152;
mov.b32 %r214, %r152;
mov.b32 %r215, %r152;
$L__BB0_2:
and.b32 %r71, %r216, 1;
bfe.u32 %r66, %r216, 1, 1;
shl.b32 %r72, %r71, 15;
add.s32 %r73, %r54, %r72;
mad.lo.s32 %r74, %r71, -32760, %r73;
add.s32 %r65, %r74, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r65], %r66;
@!complete bra.uni waitLoop;
}
shr.u32 %r75, %r73, 4;
add.s32 %r76, %r75, %r7;
and.b32 %r77, %r76, 16383;
cvt.u64.u32 %rd20, %r77;
wgmma.fence.sync.aligned;
or.b64 %rd4, %rd20, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd4, %rd5, %p7, 1, 1, 0, 0;
add.s64 %rd6, %rd20, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd6, %rd7, %p7, 1, 1, 0, 0;
add.s64 %rd8, %rd20, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd8, %rd9, %p7, 1, 1, 0, 0;
add.s64 %rd10, %rd20, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd10, %rd11, %p7, 1, 1, 0, 0;
add.s64 %rd12, %rd20, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd12, %rd13, %p7, 1, 1, 0, 0;
add.s64 %rd14, %rd20, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd14, %rd15, %p7, 1, 1, 0, 0;
add.s64 %rd16, %rd20, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd16, %rd17, %p7, 1, 1, 0, 0;
add.s64 %rd18, %rd20, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r152,%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215}, %rd18, %rd19, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r8, %r216, 1;
and.b32 %r78, %r8, 1;
setp.ne.b32 %p11, %r216, 5;
shl.b32 %r79, %r78, 3;
add.s32 %r80, %r54, %r79;
add.s32 %r67, %r80, 98304;
shl.b32 %r81, %r78, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r67], 65536;
bar.sync 0;
elect.sync %r82|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p31, %p13;
add.s32 %r68, %r55, %r81;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r68], [%rd21, {%r151, %r135}], [%r67];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r83|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p31, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r69], [%rd21, {%r151, %r70}], [%r67];
$L__tmp15:
add.s32 %r151, %r151, 128;
setp.eq.b32 %p16, %r8, 6;
mov.b32 %r216, %r8;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r1, %r2;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r84];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r44];
cvt.rn.bf16x2.f32 %r11, %r153, %r152;
cvt.rn.bf16x2.f32 %r12, %r155, %r154;
cvt.rn.bf16x2.f32 %r13, %r157, %r156;
cvt.rn.bf16x2.f32 %r14, %r159, %r158;
cvt.rn.bf16x2.f32 %r15, %r161, %r160;
cvt.rn.bf16x2.f32 %r16, %r163, %r162;
cvt.rn.bf16x2.f32 %r17, %r165, %r164;
cvt.rn.bf16x2.f32 %r18, %r167, %r166;
cvt.rn.bf16x2.f32 %r19, %r169, %r168;
cvt.rn.bf16x2.f32 %r20, %r171, %r170;
cvt.rn.bf16x2.f32 %r21, %r173, %r172;
cvt.rn.bf16x2.f32 %r22, %r175, %r174;
cvt.rn.bf16x2.f32 %r23, %r177, %r176;
cvt.rn.bf16x2.f32 %r24, %r179, %r178;
cvt.rn.bf16x2.f32 %r25, %r181, %r180;
cvt.rn.bf16x2.f32 %r26, %r183, %r182;
cvt.rn.bf16x2.f32 %r27, %r185, %r184;
cvt.rn.bf16x2.f32 %r28, %r187, %r186;
cvt.rn.bf16x2.f32 %r29, %r189, %r188;
cvt.rn.bf16x2.f32 %r30, %r191, %r190;
cvt.rn.bf16x2.f32 %r31, %r193, %r192;
cvt.rn.bf16x2.f32 %r32, %r195, %r194;
cvt.rn.bf16x2.f32 %r33, %r197, %r196;
cvt.rn.bf16x2.f32 %r34, %r199, %r198;
cvt.rn.bf16x2.f32 %r35, %r201, %r200;
cvt.rn.bf16x2.f32 %r36, %r203, %r202;
cvt.rn.bf16x2.f32 %r37, %r205, %r204;
cvt.rn.bf16x2.f32 %r38, %r207, %r206;
cvt.rn.bf16x2.f32 %r39, %r209, %r208;
cvt.rn.bf16x2.f32 %r40, %r211, %r210;
cvt.rn.bf16x2.f32 %r41, %r213, %r212;
cvt.rn.bf16x2.f32 %r42, %r215, %r214;
shl.b32 %r43, %r3, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r123, %r3, 4;
and.b32 %r220, %r123, 112;
shl.b32 %r219, %r3, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r109, %r43, 1920;
shl.b32 %r219, %r3, 6;
and.b32 %r110, %r219, 14336;
shl.b32 %r111, %r3, 4;
and.b32 %r220, %r111, 112;
and.b32 %r112, %r3, 16;
or.b32 %r113, %r109, %r220;
xor.b32 %r114, %r113, %r112;
or.b32 %r115, %r114, %r110;
add.s32 %r116, %r54, %r115;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r116], {%r11, %r12, %r13, %r14};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r116+16384], {%r27, %r28, %r29, %r30};
xor.b32 %r117, %r115, 32;
add.s32 %r118, %r54, %r117;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r118], {%r15, %r16, %r17, %r18};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r118+16384], {%r31, %r32, %r33, %r34};
xor.b32 %r119, %r115, 64;
add.s32 %r120, %r54, %r119;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120], {%r19, %r20, %r21, %r22};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120+16384], {%r35, %r36, %r37, %r38};
xor.b32 %r121, %r115, 96;
add.s32 %r122, %r54, %r121;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122], {%r23, %r24, %r25, %r26};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122+16384], {%r39, %r40, %r41, %r42};
$L__BB0_9:
and.b32 %r124, %r43, 17280;
and.b32 %r125, %r3, 96;
shl.b32 %r126, %r3, 1;
and.b32 %r127, %r126, 16;
and.b32 %r128, %r219, 1024;
or.b32 %r129, %r124, %r220;
or.b32 %r130, %r127, %r125;
xor.b32 %r131, %r129, %r130;
add.s32 %r132, %r54, %r128;
add.s32 %r133, %r132, %r131;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+32768], {%r11, %r12, %r13, %r14};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+34816], {%r15, %r16, %r17, %r18};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+36864], {%r19, %r20, %r21, %r22};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+38912], {%r23, %r24, %r25, %r26};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+40960], {%r27, %r28, %r29, %r30};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+43008], {%r31, %r32, %r33, %r34};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+45056], {%r35, %r36, %r37, %r38};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r133+47104], {%r39, %r40, %r41, %r42};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_11;
bar.sync 0;
elect.sync %r137|%p28, -1;
shfl.sync.idx.b32 %r138, %r6, 0, 31, -1;
and.pred %p27, %p31, %p28;
and.b32 %r139, %r138, 1;
shl.b32 %r140, %r139, 14;
add.s32 %r136, %r54, %r140;
shl.b32 %r141, %r139, 6;
or.b32 %r134, %r141, %r2;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd40, {%r134, %r135}], [%r136];
cp.async.bulk.commit_group;
$L__BB0_11:
bar.sync 0;
elect.sync %r145|%p30, -1;
shfl.sync.idx.b32 %r146, %r6, 0, 31, -1;
and.pred %p29, %p31, %p30;
and.b32 %r147, %r146, 1;
shl.b32 %r148, %r147, 14;
add.s32 %r149, %r54, %r148;
add.s32 %r143, %r149, 32768;
shl.b32 %r150, %r147, 6;
or.b32 %r142, %r150, %r1;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd40, {%r142, %r70}], [%r143];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r144, 0;
ret;
$L__tmp17:
$L__func_end0:
}
"#
  | .voGram, .baseline => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XXT_kernel(
.param .u64 .ptr .global .align 1 XXT_kernel_param_0,
.param .u64 .ptr .global .align 1 XXT_kernel_param_1,
.param .u32 XXT_kernel_param_2,
.param .u32 XXT_kernel_param_3,
.param .u32 XXT_kernel_param_4,
.param .u32 XXT_kernel_param_5,
.param .u32 XXT_kernel_param_6,
.param .u32 XXT_kernel_param_7,
.param .u64 .ptr .global .align 1 XXT_kernel_param_8,
.param .u64 .ptr .global .align 1 XXT_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<49>;
.reg .b32 %r<386>;
.reg .b64 %rd<89>;
$L__func_begin0:
ld.param.b32 %r37, [XXT_kernel_param_2];
$L__tmp0:
mov.u32 %r38, %ctaid.x;
$L__tmp1:
add.s32 %r39, %r37, 127;
shr.s32 %r40, %r39, 31;
shr.u32 %r41, %r40, 25;
add.s32 %r42, %r39, %r41;
shr.s32 %r43, %r42, 7;
$L__tmp2:
mul.lo.s32 %r44, %r43, %r43;
div.s32 %r1, %r38, %r44;
mul.lo.s32 %r45, %r1, %r44;
sub.s32 %r46, %r38, %r45;
$L__tmp3:
shl.b32 %r47, %r43, 3;
div.s32 %r48, %r46, %r47;
shl.b32 %r49, %r48, 3;
sub.s32 %r50, %r43, %r49;
min.s32 %r51, %r50, 8;
mul.lo.s32 %r52, %r48, %r47;
sub.s32 %r53, %r46, %r52;
div.s32 %r54, %r53, %r51;
mul.lo.s32 %r55, %r54, %r51;
sub.s32 %r56, %r53, %r55;
add.s32 %r57, %r56, %r49;
$L__tmp4:
shl.b32 %r2, %r57, 7;
shl.b32 %r3, %r54, 7;
$L__tmp5:
add.s32 %r58, %r2, 128;
setp.gt.s32 %p1, %r58, %r3;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b32 %r36, [XXT_kernel_param_7];
ld.param.b32 %r35, [XXT_kernel_param_6];
ld.param.b32 %r34, [XXT_kernel_param_5];
ld.param.b32 %r33, [XXT_kernel_param_4];
ld.param.b32 %r32, [XXT_kernel_param_3];
ld.param.b64 %rd19, [XXT_kernel_param_1];
ld.param.b64 %rd18, [XXT_kernel_param_0];
mul.lo.s32 %r86, %r1, %r33;
mad.wide.s32 %rd88, %r86, 2, %rd18;
mul.lo.s32 %r87, %r1, %r35;
mad.wide.s32 %rd1, %r87, 2, %rd19;
mov.u32 %r4, %tid.x;
shr.u32 %r88, %r4, 3;
bfe.u32 %r89, %r4, 3, 5;
or.b32 %r90, %r89, 32;
or.b32 %r91, %r89, 64;
or.b32 %r92, %r88, 96;
and.b32 %r5, %r4, 240;
bfe.u32 %r93, %r4, 4, 4;
or.b32 %r94, %r93, 16;
or.b32 %r95, %r93, 32;
or.b32 %r96, %r93, 48;
or.b32 %r97, %r93, 64;
or.b32 %r98, %r93, 80;
or.b32 %r99, %r93, 96;
or.b32 %r100, %r93, 112;
shl.b32 %r6, %r4, 3;
and.b32 %r101, %r6, 120;
or.b32 %r102, %r2, %r89;
or.b32 %r103, %r2, %r90;
or.b32 %r104, %r2, %r91;
or.b32 %r105, %r2, %r92;
or.b32 %r7, %r2, %r93;
or.b32 %r8, %r2, %r94;
or.b32 %r9, %r2, %r95;
or.b32 %r10, %r2, %r96;
or.b32 %r11, %r2, %r97;
or.b32 %r12, %r2, %r98;
or.b32 %r13, %r2, %r99;
or.b32 %r14, %r2, %r100;
or.b32 %r15, %r2, %r101;
rem.s32 %r106, %r102, %r37;
rem.s32 %r107, %r103, %r37;
rem.s32 %r108, %r104, %r37;
rem.s32 %r109, %r105, %r37;
or.b32 %r110, %r3, %r89;
or.b32 %r111, %r3, %r90;
or.b32 %r112, %r3, %r91;
or.b32 %r113, %r3, %r92;
or.b32 %r16, %r3, %r93;
or.b32 %r17, %r3, %r94;
or.b32 %r18, %r3, %r95;
or.b32 %r19, %r3, %r96;
or.b32 %r20, %r3, %r97;
or.b32 %r21, %r3, %r98;
or.b32 %r22, %r3, %r99;
or.b32 %r23, %r3, %r100;
or.b32 %r24, %r3, %r101;
rem.s32 %r114, %r110, %r37;
rem.s32 %r115, %r111, %r37;
rem.s32 %r116, %r112, %r37;
rem.s32 %r117, %r113, %r37;
and.b32 %r25, %r4, 7;
shl.b32 %r26, %r25, 3;
mad.lo.s32 %r118, %r106, %r34, %r26;
mad.lo.s32 %r119, %r107, %r34, %r26;
mad.lo.s32 %r120, %r108, %r34, %r26;
mad.lo.s32 %r121, %r109, %r34, %r26;
mad.wide.s32 %rd20, %r118, 2, %rd88;
mad.wide.s32 %rd21, %r119, 2, %rd88;
mad.wide.s32 %rd22, %r120, 2, %rd88;
mad.wide.s32 %rd23, %r121, 2, %rd88;
mad.lo.s32 %r122, %r114, %r34, %r26;
mad.lo.s32 %r123, %r115, %r34, %r26;
mad.lo.s32 %r124, %r116, %r34, %r26;
mad.lo.s32 %r125, %r117, %r34, %r26;
mad.wide.s32 %rd24, %r122, 2, %rd88;
mad.wide.s32 %rd25, %r123, 2, %rd88;
mad.wide.s32 %rd26, %r124, 2, %rd88;
mad.wide.s32 %rd27, %r125, 2, %rd88;
$L__tmp6:
add.s32 %r126, %r32, 63;
$L__tmp7:
setp.lt.s32 %p2, %r126, 64;
setp.gt.s32 %p3, %r126, 63;
setp.lt.s32 %p4, %r26, %r32;
shl.b32 %r130, %r4, 4;
and.b32 %r131, %r130, 4080;
shl.b32 %r28, %r4, 1;
and.b32 %r132, %r28, 112;
xor.b32 %r29, %r131, %r132;
mov.b32 %r133, global_smem;
add.s32 %r59, %r133, %r29;
selp.b32 %r134, 16, 0, %p3;
selp.b32 %r61, %r134, 0, %p4;
cp.async.cg.shared.global [ %r59 + 0 ], [ %rd20 + 0 ], 0x10, %r61;
add.s32 %r60, %r59, 4096;
cp.async.cg.shared.global [ %r60 + 0 ], [ %rd21 + 0 ], 0x10, %r61;
add.s32 %r62, %r59, 8192;
cp.async.cg.shared.global [ %r62 + 0 ], [ %rd22 + 0 ], 0x10, %r61;
add.s32 %r63, %r59, 12288;
cp.async.cg.shared.global [ %r63 + 0 ], [ %rd23 + 0 ], 0x10, %r61;
cp.async.commit_group;
add.s32 %r64, %r59, 65536;
cp.async.cg.shared.global [ %r64 + 0 ], [ %rd24 + 0 ], 0x10, %r61;
add.s32 %r65, %r59, 69632;
cp.async.cg.shared.global [ %r65 + 0 ], [ %rd25 + 0 ], 0x10, %r61;
add.s32 %r66, %r59, 73728;
cp.async.cg.shared.global [ %r66 + 0 ], [ %rd26 + 0 ], 0x10, %r61;
add.s32 %r67, %r59, 77824;
cp.async.cg.shared.global [ %r67 + 0 ], [ %rd27 + 0 ], 0x10, %r61;
cp.async.commit_group;
setp.gt.s32 %p5, %r126, 127;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s64 %rd35, %rd27, 128;
add.s32 %r135, %r32, -64;
setp.lt.s32 %p6, %r26, %r135;
bar.sync 0;
add.s32 %r68, %r59, 16384;
selp.b32 %r136, 16, 0, %p6;
selp.b32 %r70, %r136, 0, %p5;
cp.async.cg.shared.global [ %r68 + 0 ], [ %rd28 + 0 ], 0x10, %r70;
add.s32 %r69, %r59, 20480;
cp.async.cg.shared.global [ %r69 + 0 ], [ %rd29 + 0 ], 0x10, %r70;
add.s32 %r71, %r59, 24576;
cp.async.cg.shared.global [ %r71 + 0 ], [ %rd30 + 0 ], 0x10, %r70;
add.s32 %r72, %r59, 28672;
cp.async.cg.shared.global [ %r72 + 0 ], [ %rd31 + 0 ], 0x10, %r70;
cp.async.commit_group;
add.s32 %r73, %r59, 81920;
cp.async.cg.shared.global [ %r73 + 0 ], [ %rd32 + 0 ], 0x10, %r70;
add.s32 %r74, %r59, 86016;
cp.async.cg.shared.global [ %r74 + 0 ], [ %rd33 + 0 ], 0x10, %r70;
add.s32 %r75, %r59, 90112;
cp.async.cg.shared.global [ %r75 + 0 ], [ %rd34 + 0 ], 0x10, %r70;
add.s32 %r76, %r59, 94208;
cp.async.cg.shared.global [ %r76 + 0 ], [ %rd35 + 0 ], 0x10, %r70;
cp.async.commit_group;
setp.gt.s32 %p7, %r126, 191;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s64 %rd43, %rd27, 256;
add.s32 %r137, %r32, -128;
setp.lt.s32 %p8, %r26, %r137;
bar.sync 0;
add.s32 %r77, %r59, 32768;
selp.b32 %r138, 16, 0, %p8;
selp.b32 %r79, %r138, 0, %p7;
cp.async.cg.shared.global [ %r77 + 0 ], [ %rd36 + 0 ], 0x10, %r79;
add.s32 %r78, %r59, 36864;
cp.async.cg.shared.global [ %r78 + 0 ], [ %rd37 + 0 ], 0x10, %r79;
add.s32 %r80, %r59, 40960;
cp.async.cg.shared.global [ %r80 + 0 ], [ %rd38 + 0 ], 0x10, %r79;
add.s32 %r81, %r59, 45056;
cp.async.cg.shared.global [ %r81 + 0 ], [ %rd39 + 0 ], 0x10, %r79;
cp.async.commit_group;
add.s32 %r82, %r59, 98304;
cp.async.cg.shared.global [ %r82 + 0 ], [ %rd40 + 0 ], 0x10, %r79;
add.s32 %r83, %r59, 102400;
cp.async.cg.shared.global [ %r83 + 0 ], [ %rd41 + 0 ], 0x10, %r79;
add.s32 %r84, %r59, 106496;
cp.async.cg.shared.global [ %r84 + 0 ], [ %rd42 + 0 ], 0x10, %r79;
add.s32 %r85, %r59, 110592;
cp.async.cg.shared.global [ %r85 + 0 ], [ %rd43 + 0 ], 0x10, %r79;
cp.async.commit_group;
mov.b32 %r321, 0f00000000;
mov.b32 %r322, %r321;
mov.b32 %r323, %r321;
mov.b32 %r324, %r321;
mov.b32 %r325, %r321;
mov.b32 %r326, %r321;
mov.b32 %r327, %r321;
mov.b32 %r328, %r321;
mov.b32 %r329, %r321;
mov.b32 %r330, %r321;
mov.b32 %r331, %r321;
mov.b32 %r332, %r321;
mov.b32 %r333, %r321;
mov.b32 %r334, %r321;
mov.b32 %r335, %r321;
mov.b32 %r336, %r321;
mov.b32 %r337, %r321;
mov.b32 %r338, %r321;
mov.b32 %r339, %r321;
mov.b32 %r340, %r321;
mov.b32 %r341, %r321;
mov.b32 %r342, %r321;
mov.b32 %r343, %r321;
mov.b32 %r344, %r321;
mov.b32 %r345, %r321;
mov.b32 %r346, %r321;
mov.b32 %r347, %r321;
mov.b32 %r348, %r321;
mov.b32 %r349, %r321;
mov.b32 %r350, %r321;
mov.b32 %r351, %r321;
mov.b32 %r352, %r321;
mov.b32 %r353, %r321;
mov.b32 %r354, %r321;
mov.b32 %r355, %r321;
mov.b32 %r356, %r321;
mov.b32 %r357, %r321;
mov.b32 %r358, %r321;
mov.b32 %r359, %r321;
mov.b32 %r360, %r321;
mov.b32 %r361, %r321;
mov.b32 %r362, %r321;
mov.b32 %r363, %r321;
mov.b32 %r364, %r321;
mov.b32 %r365, %r321;
mov.b32 %r366, %r321;
mov.b32 %r367, %r321;
mov.b32 %r368, %r321;
mov.b32 %r369, %r321;
mov.b32 %r370, %r321;
mov.b32 %r371, %r321;
mov.b32 %r372, %r321;
mov.b32 %r373, %r321;
mov.b32 %r374, %r321;
mov.b32 %r375, %r321;
mov.b32 %r376, %r321;
mov.b32 %r377, %r321;
mov.b32 %r378, %r321;
mov.b32 %r379, %r321;
mov.b32 %r380, %r321;
mov.b32 %r381, %r321;
mov.b32 %r382, %r321;
mov.b32 %r383, %r321;
mov.b32 %r384, %r321;
@%p2 bra $L__BB0_5;
cvt.s64.s32 %rd2, %r118;
cvt.s64.s32 %rd3, %r119;
cvt.s64.s32 %rd4, %r120;
cvt.s64.s32 %rd5, %r121;
cvt.s64.s32 %rd6, %r122;
cvt.s64.s32 %rd7, %r123;
cvt.s64.s32 %rd8, %r124;
cvt.s64.s32 %rd9, %r125;
shr.s32 %r127, %r126, 31;
shr.u32 %r128, %r127, 26;
add.s32 %r129, %r126, %r128;
shr.s32 %r27, %r129, 6;
add.s32 %r30, %r27, -3;
shr.u32 %r31, %r4, 5;
add.s32 %r318, %r32, -192;
shl.b64 %rd44, %rd9, 1;
add.s64 %rd10, %rd44, 384;
shl.b64 %rd45, %rd8, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd7, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd6, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd5, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd4, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd3, 1;
add.s64 %rd16, %rd50, 384;
shl.b64 %rd51, %rd2, 1;
add.s64 %rd17, %rd51, 384;
mov.b32 %r143, 0;
mov.b32 %r321, 0f00000000;
mov.b32 %r320, 2;
mov.b32 %r319, -1;
mov.b32 %r322, %r321;
mov.b32 %r323, %r321;
mov.b32 %r324, %r321;
mov.b32 %r325, %r321;
mov.b32 %r326, %r321;
mov.b32 %r327, %r321;
mov.b32 %r328, %r321;
mov.b32 %r329, %r321;
mov.b32 %r330, %r321;
mov.b32 %r331, %r321;
mov.b32 %r332, %r321;
mov.b32 %r333, %r321;
mov.b32 %r334, %r321;
mov.b32 %r335, %r321;
mov.b32 %r336, %r321;
mov.b32 %r337, %r321;
mov.b32 %r338, %r321;
mov.b32 %r339, %r321;
mov.b32 %r340, %r321;
mov.b32 %r341, %r321;
mov.b32 %r342, %r321;
mov.b32 %r343, %r321;
mov.b32 %r344, %r321;
mov.b32 %r345, %r321;
mov.b32 %r346, %r321;
mov.b32 %r347, %r321;
mov.b32 %r348, %r321;
mov.b32 %r349, %r321;
mov.b32 %r350, %r321;
mov.b32 %r351, %r321;
mov.b32 %r352, %r321;
mov.b32 %r353, %r321;
mov.b32 %r354, %r321;
mov.b32 %r355, %r321;
mov.b32 %r356, %r321;
mov.b32 %r357, %r321;
mov.b32 %r358, %r321;
mov.b32 %r359, %r321;
mov.b32 %r360, %r321;
mov.b32 %r361, %r321;
mov.b32 %r362, %r321;
mov.b32 %r363, %r321;
mov.b32 %r364, %r321;
mov.b32 %r365, %r321;
mov.b32 %r366, %r321;
mov.b32 %r367, %r321;
mov.b32 %r368, %r321;
mov.b32 %r369, %r321;
mov.b32 %r370, %r321;
mov.b32 %r371, %r321;
mov.b32 %r372, %r321;
mov.b32 %r373, %r321;
mov.b32 %r374, %r321;
mov.b32 %r375, %r321;
mov.b32 %r376, %r321;
mov.b32 %r377, %r321;
mov.b32 %r378, %r321;
mov.b32 %r379, %r321;
mov.b32 %r380, %r321;
mov.b32 %r381, %r321;
mov.b32 %r382, %r321;
mov.b32 %r383, %r321;
mov.b32 %r384, %r321;
mov.b32 %r385, %r143;
$L__BB0_4:
setp.lt.s32 %p10, %r385, %r30;
add.s32 %r153, %r319, 1;
setp.gt.s32 %p11, %r153, 3;
selp.b32 %r319, 0, %r153, %p11;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r154, %r319, 14;
add.s32 %r155, %r133, %r154;
add.s32 %r156, %r155, 65536;
shr.u32 %r157, %r155, 4;
shfl.sync.idx.b32 %r158, %r31, 0, 31, -1;
shl.b32 %r159, %r158, 7;
and.b32 %r160, %r159, 512;
add.s32 %r161, %r160, %r157;
and.b32 %r162, %r161, 16383;
cvt.u64.u32 %rd70, %r162;
bfe.u32 %r163, %r156, 4, 14;
cvt.u64.u32 %rd71, %r163;
wgmma.fence.sync.aligned;
or.b64 %rd52, %rd70, 4611686293305294848;
or.b64 %rd53, %rd71, 4611686293305294848;
mov.pred %p9, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd52, %rd53, %p9, 1, 1, 0, 0;
add.s64 %rd54, %rd70, 4611686293305294850;
add.s64 %rd55, %rd71, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd54, %rd55, %p9, 1, 1, 0, 0;
add.s64 %rd56, %rd70, 4611686293305294852;
add.s64 %rd57, %rd71, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd56, %rd57, %p9, 1, 1, 0, 0;
add.s64 %rd58, %rd70, 4611686293305294854;
add.s64 %rd59, %rd71, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384}, %rd58, %rd59, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd60, %r155;
cvt.u64.u32 %rd61, %r156;
mov.b32 %r142, %r143;
mov.b32 %r139, %r143;
mov.b32 %r140, %r143;
mov.b32 %r141, %r143;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd62, %rd88, %rd17;
add.s64 %rd63, %rd88, %rd16;
add.s64 %rd64, %rd88, %rd15;
add.s64 %rd65, %rd88, %rd14;
add.s64 %rd66, %rd88, %rd13;
add.s64 %rd67, %rd88, %rd12;
add.s64 %rd68, %rd88, %rd11;
add.s64 %rd69, %rd88, %rd10;
add.s32 %r164, %r320, 1;
setp.gt.s32 %p12, %r164, 3;
selp.b32 %r320, 0, %r164, %p12;
setp.lt.s32 %p13, %r26, %r318;
shl.b32 %r165, %r320, 14;
add.s32 %r166, %r133, %r165;
bar.sync 0;
add.s32 %r144, %r166, %r29;
selp.b32 %r167, 16, 0, %p13;
selp.b32 %r146, %r167, 0, %p10;
cp.async.cg.shared.global [ %r144 + 0 ], [ %rd62 + 0 ], 0x10, %r146;
add.s32 %r145, %r144, 4096;
cp.async.cg.shared.global [ %r145 + 0 ], [ %rd63 + 0 ], 0x10, %r146;
add.s32 %r147, %r144, 8192;
cp.async.cg.shared.global [ %r147 + 0 ], [ %rd64 + 0 ], 0x10, %r146;
add.s32 %r148, %r144, 12288;
cp.async.cg.shared.global [ %r148 + 0 ], [ %rd65 + 0 ], 0x10, %r146;
cp.async.commit_group;
add.s32 %r149, %r144, 65536;
cp.async.cg.shared.global [ %r149 + 0 ], [ %rd66 + 0 ], 0x10, %r146;
add.s32 %r150, %r144, 69632;
cp.async.cg.shared.global [ %r150 + 0 ], [ %rd67 + 0 ], 0x10, %r146;
add.s32 %r151, %r144, 73728;
cp.async.cg.shared.global [ %r151 + 0 ], [ %rd68 + 0 ], 0x10, %r146;
add.s32 %r152, %r144, 77824;
cp.async.cg.shared.global [ %r152 + 0 ], [ %rd69 + 0 ], 0x10, %r146;
cp.async.commit_group;
add.s32 %r385, %r385, 1;
add.s64 %rd88, %rd88, 128;
add.s32 %r318, %r318, -64;
setp.ne.b32 %p14, %r27, %r385;
@%p14 bra $L__BB0_4;
$L__BB0_5:
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r232, %r322, %r321;
cvt.rn.bf16x2.f32 %r233, %r324, %r323;
cvt.rn.bf16x2.f32 %r234, %r326, %r325;
cvt.rn.bf16x2.f32 %r235, %r328, %r327;
cvt.rn.bf16x2.f32 %r236, %r330, %r329;
cvt.rn.bf16x2.f32 %r237, %r332, %r331;
cvt.rn.bf16x2.f32 %r238, %r334, %r333;
cvt.rn.bf16x2.f32 %r239, %r336, %r335;
cvt.rn.bf16x2.f32 %r240, %r338, %r337;
cvt.rn.bf16x2.f32 %r241, %r340, %r339;
cvt.rn.bf16x2.f32 %r242, %r342, %r341;
cvt.rn.bf16x2.f32 %r243, %r344, %r343;
cvt.rn.bf16x2.f32 %r244, %r346, %r345;
cvt.rn.bf16x2.f32 %r245, %r348, %r347;
cvt.rn.bf16x2.f32 %r246, %r350, %r349;
cvt.rn.bf16x2.f32 %r247, %r352, %r351;
cvt.rn.bf16x2.f32 %r248, %r354, %r353;
cvt.rn.bf16x2.f32 %r249, %r356, %r355;
cvt.rn.bf16x2.f32 %r250, %r358, %r357;
cvt.rn.bf16x2.f32 %r251, %r360, %r359;
cvt.rn.bf16x2.f32 %r252, %r362, %r361;
cvt.rn.bf16x2.f32 %r253, %r364, %r363;
cvt.rn.bf16x2.f32 %r254, %r366, %r365;
cvt.rn.bf16x2.f32 %r255, %r368, %r367;
cvt.rn.bf16x2.f32 %r256, %r370, %r369;
cvt.rn.bf16x2.f32 %r257, %r372, %r371;
cvt.rn.bf16x2.f32 %r258, %r374, %r373;
cvt.rn.bf16x2.f32 %r259, %r376, %r375;
cvt.rn.bf16x2.f32 %r260, %r378, %r377;
cvt.rn.bf16x2.f32 %r261, %r380, %r379;
cvt.rn.bf16x2.f32 %r262, %r382, %r381;
cvt.rn.bf16x2.f32 %r263, %r384, %r383;
mad.lo.s32 %r264, %r7, %r36, %r24;
mad.lo.s32 %r265, %r8, %r36, %r24;
mad.lo.s32 %r266, %r9, %r36, %r24;
mad.lo.s32 %r267, %r10, %r36, %r24;
mad.lo.s32 %r268, %r11, %r36, %r24;
mad.lo.s32 %r269, %r12, %r36, %r24;
mad.lo.s32 %r270, %r13, %r36, %r24;
mad.lo.s32 %r271, %r14, %r36, %r24;
mad.wide.s32 %rd72, %r264, 2, %rd1;
mad.wide.s32 %rd73, %r265, 2, %rd1;
mad.wide.s32 %rd74, %r266, 2, %rd1;
mad.wide.s32 %rd75, %r267, 2, %rd1;
mad.wide.s32 %rd76, %r268, 2, %rd1;
mad.wide.s32 %rd77, %r269, 2, %rd1;
mad.wide.s32 %rd78, %r270, 2, %rd1;
mad.wide.s32 %rd79, %r271, 2, %rd1;
setp.lt.s32 %p31, %r7, %r37;
setp.lt.s32 %p32, %r8, %r37;
setp.lt.s32 %p33, %r9, %r37;
setp.lt.s32 %p34, %r10, %r37;
setp.lt.s32 %p35, %r11, %r37;
setp.lt.s32 %p36, %r12, %r37;
setp.lt.s32 %p37, %r13, %r37;
setp.lt.s32 %p38, %r14, %r37;
setp.lt.s32 %p39, %r24, %r37;
and.pred %p15, %p31, %p39;
and.pred %p16, %p32, %p39;
and.pred %p17, %p33, %p39;
and.pred %p18, %p34, %p39;
and.pred %p19, %p35, %p39;
and.pred %p20, %p36, %p39;
and.pred %p21, %p37, %p39;
and.pred %p22, %p38, %p39;
and.b32 %r272, %r4, 3;
shl.b32 %r273, %r272, 13;
shl.b32 %r274, %r4, 5;
and.b32 %r275, %r274, 7264;
and.b32 %r276, %r4, 24;
shl.b32 %r277, %r276, 4;
shl.b32 %r278, %r4, 2;
and.b32 %r279, %r278, 16;
or.b32 %r280, %r273, %r279;
or.b32 %r281, %r275, %r277;
or.b32 %r282, %r280, %r281;
add.s32 %r283, %r133, %r282;
st.shared.v4.b32 [%r283], {%r232, %r234, %r236, %r238};
st.shared.v4.b32 [%r283+512], {%r233, %r235, %r237, %r239};
xor.b32 %r284, %r282, 32;
add.s32 %r285, %r133, %r284;
st.shared.v4.b32 [%r285], {%r240, %r242, %r244, %r246};
st.shared.v4.b32 [%r285+512], {%r241, %r243, %r245, %r247};
xor.b32 %r286, %r282, 64;
add.s32 %r287, %r133, %r286;
st.shared.v4.b32 [%r287], {%r248, %r250, %r252, %r254};
st.shared.v4.b32 [%r287+512], {%r249, %r251, %r253, %r255};
xor.b32 %r288, %r282, 96;
add.s32 %r289, %r133, %r288;
st.shared.v4.b32 [%r289], {%r256, %r258, %r260, %r262};
st.shared.v4.b32 [%r289+512], {%r257, %r259, %r261, %r263};
bar.sync 0;
shl.b32 %r290, %r276, 10;
shl.b32 %r291, %r272, 5;
and.b32 %r292, %r278, 1008;
or.b32 %r293, %r290, %r291;
xor.b32 %r294, %r293, %r292;
add.s32 %r295, %r133, %r294;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r170, %r171}, [%r295];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r174, %r175}, [%r295+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r176, %r177, %r178, %r179}, [%r295+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r180, %r181, %r182, %r183}, [%r295+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r184, %r185, %r186, %r187}, [%r295+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r188, %r189, %r190, %r191}, [%r295+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r192, %r193, %r194, %r195}, [%r295+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r196, %r197, %r198, %r199}, [%r295+7168];
@%p15 st.global.v4.b32 [ %rd72 + 0 ], { %r168, %r169, %r170, %r171 };
@%p16 st.global.v4.b32 [ %rd73 + 0 ], { %r172, %r173, %r174, %r175 };
@%p17 st.global.v4.b32 [ %rd74 + 0 ], { %r176, %r177, %r178, %r179 };
@%p18 st.global.v4.b32 [ %rd75 + 0 ], { %r180, %r181, %r182, %r183 };
@%p19 st.global.v4.b32 [ %rd76 + 0 ], { %r184, %r185, %r186, %r187 };
@%p20 st.global.v4.b32 [ %rd77 + 0 ], { %r188, %r189, %r190, %r191 };
@%p21 st.global.v4.b32 [ %rd78 + 0 ], { %r192, %r193, %r194, %r195 };
@%p22 st.global.v4.b32 [ %rd79 + 0 ], { %r196, %r197, %r198, %r199 };
mad.lo.s32 %r296, %r16, %r36, %r15;
mad.lo.s32 %r297, %r17, %r36, %r15;
mad.lo.s32 %r298, %r18, %r36, %r15;
mad.lo.s32 %r299, %r19, %r36, %r15;
mad.lo.s32 %r300, %r20, %r36, %r15;
mad.lo.s32 %r301, %r21, %r36, %r15;
mad.lo.s32 %r302, %r22, %r36, %r15;
mad.lo.s32 %r303, %r23, %r36, %r15;
mad.wide.s32 %rd80, %r296, 2, %rd1;
mad.wide.s32 %rd81, %r297, 2, %rd1;
mad.wide.s32 %rd82, %r298, 2, %rd1;
mad.wide.s32 %rd83, %r299, 2, %rd1;
mad.wide.s32 %rd84, %r300, 2, %rd1;
mad.wide.s32 %rd85, %r301, 2, %rd1;
mad.wide.s32 %rd86, %r302, 2, %rd1;
mad.wide.s32 %rd87, %r303, 2, %rd1;
setp.lt.s32 %p40, %r16, %r37;
setp.lt.s32 %p41, %r17, %r37;
setp.lt.s32 %p42, %r18, %r37;
setp.lt.s32 %p43, %r19, %r37;
setp.lt.s32 %p44, %r20, %r37;
setp.lt.s32 %p45, %r21, %r37;
setp.lt.s32 %p46, %r22, %r37;
setp.lt.s32 %p47, %r23, %r37;
setp.lt.s32 %p48, %r15, %r37;
and.pred %p23, %p40, %p48;
and.pred %p24, %p41, %p48;
and.pred %p25, %p42, %p48;
and.pred %p26, %p43, %p48;
and.pred %p27, %p44, %p48;
and.pred %p28, %p45, %p48;
and.pred %p29, %p46, %p48;
and.pred %p30, %p47, %p48;
bar.sync 0;
shl.b32 %r304, %r25, 9;
shl.b32 %r305, %r25, 4;
and.b32 %r306, %r28, 272;
and.b32 %r307, %r4, 96;
and.b32 %r308, %r6, 128;
or.b32 %r309, %r306, %r307;
xor.b32 %r310, %r309, %r305;
add.s32 %r311, %r133, %r304;
add.s32 %r312, %r311, %r308;
add.s32 %r313, %r312, %r310;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r232, %r233, %r234, %r235};
bar.sync 0;
and.b32 %r314, %r274, 3840;
or.b32 %r315, %r314, %r305;
xor.b32 %r316, %r315, %r5;
add.s32 %r317, %r133, %r316;
ld.shared.v4.b32 {%r200, %r201, %r202, %r203}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r236, %r237, %r238, %r239};
bar.sync 0;
ld.shared.v4.b32 {%r204, %r205, %r206, %r207}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r240, %r241, %r242, %r243};
bar.sync 0;
ld.shared.v4.b32 {%r208, %r209, %r210, %r211}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r244, %r245, %r246, %r247};
bar.sync 0;
ld.shared.v4.b32 {%r212, %r213, %r214, %r215}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r248, %r249, %r250, %r251};
bar.sync 0;
ld.shared.v4.b32 {%r216, %r217, %r218, %r219}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r252, %r253, %r254, %r255};
bar.sync 0;
ld.shared.v4.b32 {%r220, %r221, %r222, %r223}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r256, %r257, %r258, %r259};
bar.sync 0;
ld.shared.v4.b32 {%r224, %r225, %r226, %r227}, [%r317];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r313], {%r260, %r261, %r262, %r263};
bar.sync 0;
ld.shared.v4.b32 {%r228, %r229, %r230, %r231}, [%r317];
@%p23 st.global.v4.b32 [ %rd80 + 0 ], { %r200, %r201, %r202, %r203 };
@%p24 st.global.v4.b32 [ %rd81 + 0 ], { %r204, %r205, %r206, %r207 };
@%p25 st.global.v4.b32 [ %rd82 + 0 ], { %r208, %r209, %r210, %r211 };
@%p26 st.global.v4.b32 [ %rd83 + 0 ], { %r212, %r213, %r214, %r215 };
@%p27 st.global.v4.b32 [ %rd84 + 0 ], { %r216, %r217, %r218, %r219 };
@%p28 st.global.v4.b32 [ %rd85 + 0 ], { %r220, %r221, %r222, %r223 };
@%p29 st.global.v4.b32 [ %rd86 + 0 ], { %r224, %r225, %r226, %r227 };
@%p30 st.global.v4.b32 [ %rd87 + 0 ], { %r228, %r229, %r230, %r231 };
$L__BB0_1:
ret;
$L__tmp8:
$L__func_end0:
}
  "#
  | .voGram, .compact => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XXT_kernel(
.param .u64 .ptr .global .align 1 XXT_kernel_param_0,
.param .u64 .ptr .global .align 1 XXT_kernel_param_1,
.param .u32 XXT_kernel_param_2,
.param .u32 XXT_kernel_param_3,
.param .u32 XXT_kernel_param_4,
.param .u32 XXT_kernel_param_5,
.param .u32 XXT_kernel_param_6,
.param .u32 XXT_kernel_param_7,
.param .u64 .ptr .global .align 1 XXT_kernel_param_8,
.param .u64 .ptr .global .align 1 XXT_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<48>;
.reg .b32 %r<374>;
.reg .b64 %rd<93>;
$L__func_begin0:
ld.param.b32 %r15, [XXT_kernel_param_7];
ld.param.b32 %r14, [XXT_kernel_param_6];
ld.param.b32 %r13, [XXT_kernel_param_3];
ld.param.b32 %r12, [XXT_kernel_param_2];
ld.param.b64 %rd17, [XXT_kernel_param_1];
ld.param.b64 %rd42, [XXT_kernel_param_0];
$L__tmp0:
mov.u32 %r43, %ctaid.x;
$L__tmp1:
mul.hi.u32 %r44, %r43, 818089009;
shr.u32 %r1, %r44, 2;
mul.lo.s32 %r45, %r1, 21;
sub.s32 %r46, %r43, %r45;
ld.param.b32 %r47, [XXT_kernel_param_4];
mul.lo.s32 %r48, %r46, 3;
ld.param.b32 %r49, [XXT_kernel_param_5];
mov.b64 %rd43, 6569755928097638024;
shr.u64 %rd44, %rd43, %r48;
cvt.u32.u64 %r50, %rd44;
mov.b64 %rd45, 6420841298313609216;
shr.u64 %rd46, %rd45, %r48;
cvt.u32.u64 %r51, %rd46;
shl.b32 %r52, %r50, 7;
and.b32 %r2, %r52, 896;
shl.b32 %r53, %r51, 7;
and.b32 %r3, %r53, 896;
$L__tmp2:
mul.lo.s32 %r54, %r47, %r1;
mad.wide.s32 %rd92, %r54, 2, %rd42;
mov.u32 %r4, %tid.x;
shr.u32 %r55, %r4, 3;
bfe.u32 %r56, %r4, 3, 5;
or.b32 %r57, %r56, 32;
or.b32 %r58, %r56, 64;
or.b32 %r59, %r55, 96;
or.b32 %r60, %r2, %r56;
or.b32 %r61, %r2, %r57;
or.b32 %r62, %r2, %r58;
or.b32 %r63, %r2, %r59;
rem.s32 %r64, %r60, %r12;
rem.s32 %r65, %r61, %r12;
rem.s32 %r66, %r62, %r12;
rem.s32 %r67, %r63, %r12;
or.b32 %r68, %r3, %r56;
or.b32 %r69, %r3, %r57;
or.b32 %r70, %r3, %r58;
or.b32 %r71, %r3, %r59;
rem.s32 %r72, %r68, %r12;
rem.s32 %r73, %r69, %r12;
rem.s32 %r74, %r70, %r12;
rem.s32 %r75, %r71, %r12;
and.b32 %r5, %r4, 7;
shl.b32 %r6, %r5, 3;
mad.lo.s32 %r76, %r64, %r49, %r6;
mad.lo.s32 %r77, %r65, %r49, %r6;
mad.lo.s32 %r78, %r66, %r49, %r6;
mad.lo.s32 %r79, %r67, %r49, %r6;
mad.wide.s32 %rd18, %r76, 2, %rd92;
mad.wide.s32 %rd19, %r77, 2, %rd92;
mad.wide.s32 %rd20, %r78, 2, %rd92;
mad.wide.s32 %rd21, %r79, 2, %rd92;
mad.lo.s32 %r80, %r72, %r49, %r6;
mad.lo.s32 %r81, %r73, %r49, %r6;
mad.lo.s32 %r82, %r74, %r49, %r6;
mad.lo.s32 %r83, %r75, %r49, %r6;
mad.wide.s32 %rd22, %r80, 2, %rd92;
mad.wide.s32 %rd23, %r81, 2, %rd92;
mad.wide.s32 %rd24, %r82, 2, %rd92;
mad.wide.s32 %rd25, %r83, 2, %rd92;
$L__tmp3:
add.s32 %r84, %r13, 63;
$L__tmp4:
setp.lt.s32 %p1, %r84, 64;
setp.gt.s32 %p2, %r84, 63;
setp.lt.s32 %p3, %r6, %r13;
shl.b32 %r88, %r4, 4;
and.b32 %r89, %r88, 4080;
shl.b32 %r8, %r4, 1;
and.b32 %r90, %r8, 112;
xor.b32 %r9, %r89, %r90;
mov.b32 %r91, global_smem;
add.s32 %r16, %r91, %r9;
selp.b32 %r92, 16, 0, %p2;
selp.b32 %r18, %r92, 0, %p3;
cp.async.cg.shared.global [ %r16 + 0 ], [ %rd18 + 0 ], 0x10, %r18;
add.s32 %r17, %r16, 4096;
cp.async.cg.shared.global [ %r17 + 0 ], [ %rd19 + 0 ], 0x10, %r18;
add.s32 %r19, %r16, 8192;
cp.async.cg.shared.global [ %r19 + 0 ], [ %rd20 + 0 ], 0x10, %r18;
add.s32 %r20, %r16, 12288;
cp.async.cg.shared.global [ %r20 + 0 ], [ %rd21 + 0 ], 0x10, %r18;
cp.async.commit_group;
add.s32 %r21, %r16, 65536;
cp.async.cg.shared.global [ %r21 + 0 ], [ %rd22 + 0 ], 0x10, %r18;
add.s32 %r22, %r16, 69632;
cp.async.cg.shared.global [ %r22 + 0 ], [ %rd23 + 0 ], 0x10, %r18;
add.s32 %r23, %r16, 73728;
cp.async.cg.shared.global [ %r23 + 0 ], [ %rd24 + 0 ], 0x10, %r18;
add.s32 %r24, %r16, 77824;
cp.async.cg.shared.global [ %r24 + 0 ], [ %rd25 + 0 ], 0x10, %r18;
cp.async.commit_group;
setp.gt.s32 %p4, %r84, 127;
add.s64 %rd26, %rd18, 128;
add.s64 %rd27, %rd19, 128;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s32 %r93, %r13, -64;
setp.lt.s32 %p5, %r6, %r93;
bar.sync 0;
add.s32 %r25, %r16, 16384;
selp.b32 %r94, 16, 0, %p5;
selp.b32 %r27, %r94, 0, %p4;
cp.async.cg.shared.global [ %r25 + 0 ], [ %rd26 + 0 ], 0x10, %r27;
add.s32 %r26, %r16, 20480;
cp.async.cg.shared.global [ %r26 + 0 ], [ %rd27 + 0 ], 0x10, %r27;
add.s32 %r28, %r16, 24576;
cp.async.cg.shared.global [ %r28 + 0 ], [ %rd28 + 0 ], 0x10, %r27;
add.s32 %r29, %r16, 28672;
cp.async.cg.shared.global [ %r29 + 0 ], [ %rd29 + 0 ], 0x10, %r27;
cp.async.commit_group;
add.s32 %r30, %r16, 81920;
cp.async.cg.shared.global [ %r30 + 0 ], [ %rd30 + 0 ], 0x10, %r27;
add.s32 %r31, %r16, 86016;
cp.async.cg.shared.global [ %r31 + 0 ], [ %rd31 + 0 ], 0x10, %r27;
add.s32 %r32, %r16, 90112;
cp.async.cg.shared.global [ %r32 + 0 ], [ %rd32 + 0 ], 0x10, %r27;
add.s32 %r33, %r16, 94208;
cp.async.cg.shared.global [ %r33 + 0 ], [ %rd33 + 0 ], 0x10, %r27;
cp.async.commit_group;
setp.gt.s32 %p6, %r84, 191;
add.s64 %rd34, %rd18, 256;
add.s64 %rd35, %rd19, 256;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s32 %r95, %r13, -128;
setp.lt.s32 %p7, %r6, %r95;
bar.sync 0;
add.s32 %r34, %r16, 32768;
selp.b32 %r96, 16, 0, %p7;
selp.b32 %r36, %r96, 0, %p6;
cp.async.cg.shared.global [ %r34 + 0 ], [ %rd34 + 0 ], 0x10, %r36;
add.s32 %r35, %r16, 36864;
cp.async.cg.shared.global [ %r35 + 0 ], [ %rd35 + 0 ], 0x10, %r36;
add.s32 %r37, %r16, 40960;
cp.async.cg.shared.global [ %r37 + 0 ], [ %rd36 + 0 ], 0x10, %r36;
add.s32 %r38, %r16, 45056;
cp.async.cg.shared.global [ %r38 + 0 ], [ %rd37 + 0 ], 0x10, %r36;
cp.async.commit_group;
add.s32 %r39, %r16, 98304;
cp.async.cg.shared.global [ %r39 + 0 ], [ %rd38 + 0 ], 0x10, %r36;
add.s32 %r40, %r16, 102400;
cp.async.cg.shared.global [ %r40 + 0 ], [ %rd39 + 0 ], 0x10, %r36;
add.s32 %r41, %r16, 106496;
cp.async.cg.shared.global [ %r41 + 0 ], [ %rd40 + 0 ], 0x10, %r36;
add.s32 %r42, %r16, 110592;
cp.async.cg.shared.global [ %r42 + 0 ], [ %rd41 + 0 ], 0x10, %r36;
cp.async.commit_group;
mov.b32 %r309, 0f00000000;
mov.b32 %r310, %r309;
mov.b32 %r311, %r309;
mov.b32 %r312, %r309;
mov.b32 %r313, %r309;
mov.b32 %r314, %r309;
mov.b32 %r315, %r309;
mov.b32 %r316, %r309;
mov.b32 %r317, %r309;
mov.b32 %r318, %r309;
mov.b32 %r319, %r309;
mov.b32 %r320, %r309;
mov.b32 %r321, %r309;
mov.b32 %r322, %r309;
mov.b32 %r323, %r309;
mov.b32 %r324, %r309;
mov.b32 %r325, %r309;
mov.b32 %r326, %r309;
mov.b32 %r327, %r309;
mov.b32 %r328, %r309;
mov.b32 %r329, %r309;
mov.b32 %r330, %r309;
mov.b32 %r331, %r309;
mov.b32 %r332, %r309;
mov.b32 %r333, %r309;
mov.b32 %r334, %r309;
mov.b32 %r335, %r309;
mov.b32 %r336, %r309;
mov.b32 %r337, %r309;
mov.b32 %r338, %r309;
mov.b32 %r339, %r309;
mov.b32 %r340, %r309;
mov.b32 %r341, %r309;
mov.b32 %r342, %r309;
mov.b32 %r343, %r309;
mov.b32 %r344, %r309;
mov.b32 %r345, %r309;
mov.b32 %r346, %r309;
mov.b32 %r347, %r309;
mov.b32 %r348, %r309;
mov.b32 %r349, %r309;
mov.b32 %r350, %r309;
mov.b32 %r351, %r309;
mov.b32 %r352, %r309;
mov.b32 %r353, %r309;
mov.b32 %r354, %r309;
mov.b32 %r355, %r309;
mov.b32 %r356, %r309;
mov.b32 %r357, %r309;
mov.b32 %r358, %r309;
mov.b32 %r359, %r309;
mov.b32 %r360, %r309;
mov.b32 %r361, %r309;
mov.b32 %r362, %r309;
mov.b32 %r363, %r309;
mov.b32 %r364, %r309;
mov.b32 %r365, %r309;
mov.b32 %r366, %r309;
mov.b32 %r367, %r309;
mov.b32 %r368, %r309;
mov.b32 %r369, %r309;
mov.b32 %r370, %r309;
mov.b32 %r371, %r309;
mov.b32 %r372, %r309;
@%p1 bra $L__BB0_3;
cvt.s64.s32 %rd1, %r76;
cvt.s64.s32 %rd2, %r77;
cvt.s64.s32 %rd3, %r78;
cvt.s64.s32 %rd4, %r79;
cvt.s64.s32 %rd5, %r80;
cvt.s64.s32 %rd6, %r81;
cvt.s64.s32 %rd7, %r82;
cvt.s64.s32 %rd8, %r83;
shr.s32 %r85, %r84, 31;
shr.u32 %r86, %r85, 26;
add.s32 %r87, %r84, %r86;
shr.s32 %r7, %r87, 6;
add.s32 %r10, %r7, -3;
shr.u32 %r11, %r4, 5;
add.s32 %r306, %r13, -192;
shl.b64 %rd47, %rd8, 1;
add.s64 %rd9, %rd47, 384;
shl.b64 %rd48, %rd7, 1;
add.s64 %rd10, %rd48, 384;
shl.b64 %rd49, %rd6, 1;
add.s64 %rd11, %rd49, 384;
shl.b64 %rd50, %rd5, 1;
add.s64 %rd12, %rd50, 384;
shl.b64 %rd51, %rd4, 1;
add.s64 %rd13, %rd51, 384;
shl.b64 %rd52, %rd3, 1;
add.s64 %rd14, %rd52, 384;
shl.b64 %rd53, %rd2, 1;
add.s64 %rd15, %rd53, 384;
shl.b64 %rd54, %rd1, 1;
add.s64 %rd16, %rd54, 384;
mov.b32 %r101, 0;
mov.b32 %r309, 0f00000000;
mov.b32 %r308, 2;
mov.b32 %r307, -1;
mov.b32 %r310, %r309;
mov.b32 %r311, %r309;
mov.b32 %r312, %r309;
mov.b32 %r313, %r309;
mov.b32 %r314, %r309;
mov.b32 %r315, %r309;
mov.b32 %r316, %r309;
mov.b32 %r317, %r309;
mov.b32 %r318, %r309;
mov.b32 %r319, %r309;
mov.b32 %r320, %r309;
mov.b32 %r321, %r309;
mov.b32 %r322, %r309;
mov.b32 %r323, %r309;
mov.b32 %r324, %r309;
mov.b32 %r325, %r309;
mov.b32 %r326, %r309;
mov.b32 %r327, %r309;
mov.b32 %r328, %r309;
mov.b32 %r329, %r309;
mov.b32 %r330, %r309;
mov.b32 %r331, %r309;
mov.b32 %r332, %r309;
mov.b32 %r333, %r309;
mov.b32 %r334, %r309;
mov.b32 %r335, %r309;
mov.b32 %r336, %r309;
mov.b32 %r337, %r309;
mov.b32 %r338, %r309;
mov.b32 %r339, %r309;
mov.b32 %r340, %r309;
mov.b32 %r341, %r309;
mov.b32 %r342, %r309;
mov.b32 %r343, %r309;
mov.b32 %r344, %r309;
mov.b32 %r345, %r309;
mov.b32 %r346, %r309;
mov.b32 %r347, %r309;
mov.b32 %r348, %r309;
mov.b32 %r349, %r309;
mov.b32 %r350, %r309;
mov.b32 %r351, %r309;
mov.b32 %r352, %r309;
mov.b32 %r353, %r309;
mov.b32 %r354, %r309;
mov.b32 %r355, %r309;
mov.b32 %r356, %r309;
mov.b32 %r357, %r309;
mov.b32 %r358, %r309;
mov.b32 %r359, %r309;
mov.b32 %r360, %r309;
mov.b32 %r361, %r309;
mov.b32 %r362, %r309;
mov.b32 %r363, %r309;
mov.b32 %r364, %r309;
mov.b32 %r365, %r309;
mov.b32 %r366, %r309;
mov.b32 %r367, %r309;
mov.b32 %r368, %r309;
mov.b32 %r369, %r309;
mov.b32 %r370, %r309;
mov.b32 %r371, %r309;
mov.b32 %r372, %r309;
mov.b32 %r373, %r101;
$L__BB0_2:
setp.lt.s32 %p9, %r373, %r10;
add.s32 %r111, %r307, 1;
setp.gt.s32 %p10, %r111, 3;
selp.b32 %r307, 0, %r111, %p10;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r112, %r307, 14;
add.s32 %r113, %r91, %r112;
add.s32 %r114, %r113, 65536;
shr.u32 %r115, %r113, 4;
shfl.sync.idx.b32 %r116, %r11, 0, 31, -1;
shl.b32 %r117, %r116, 7;
and.b32 %r118, %r117, 512;
add.s32 %r119, %r118, %r115;
and.b32 %r120, %r119, 16383;
cvt.u64.u32 %rd73, %r120;
bfe.u32 %r121, %r114, 4, 14;
cvt.u64.u32 %rd74, %r121;
wgmma.fence.sync.aligned;
or.b64 %rd55, %rd73, 4611686293305294848;
or.b64 %rd56, %rd74, 4611686293305294848;
mov.pred %p8, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372}, %rd55, %rd56, %p8, 1, 1, 0, 0;
add.s64 %rd57, %rd73, 4611686293305294850;
add.s64 %rd58, %rd74, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372}, %rd57, %rd58, %p8, 1, 1, 0, 0;
add.s64 %rd59, %rd73, 4611686293305294852;
add.s64 %rd60, %rd74, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372}, %rd59, %rd60, %p8, 1, 1, 0, 0;
add.s64 %rd61, %rd73, 4611686293305294854;
add.s64 %rd62, %rd74, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r309,%r310,%r311,%r312,%r313,%r314,%r315,%r316,%r317,%r318,%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372}, %rd61, %rd62, %p8, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd63, %r113;
cvt.u64.u32 %rd64, %r114;
mov.b32 %r100, %r101;
mov.b32 %r97, %r101;
mov.b32 %r98, %r101;
mov.b32 %r99, %r101;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd65, %rd92, %rd16;
add.s64 %rd66, %rd92, %rd15;
add.s64 %rd67, %rd92, %rd14;
add.s64 %rd68, %rd92, %rd13;
add.s64 %rd69, %rd92, %rd12;
add.s64 %rd70, %rd92, %rd11;
add.s64 %rd71, %rd92, %rd10;
add.s64 %rd72, %rd92, %rd9;
add.s32 %r122, %r308, 1;
setp.gt.s32 %p11, %r122, 3;
selp.b32 %r308, 0, %r122, %p11;
setp.lt.s32 %p12, %r6, %r306;
shl.b32 %r123, %r308, 14;
add.s32 %r124, %r91, %r123;
bar.sync 0;
add.s32 %r102, %r124, %r9;
selp.b32 %r125, 16, 0, %p12;
selp.b32 %r104, %r125, 0, %p9;
cp.async.cg.shared.global [ %r102 + 0 ], [ %rd65 + 0 ], 0x10, %r104;
add.s32 %r103, %r102, 4096;
cp.async.cg.shared.global [ %r103 + 0 ], [ %rd66 + 0 ], 0x10, %r104;
add.s32 %r105, %r102, 8192;
cp.async.cg.shared.global [ %r105 + 0 ], [ %rd67 + 0 ], 0x10, %r104;
add.s32 %r106, %r102, 12288;
cp.async.cg.shared.global [ %r106 + 0 ], [ %rd68 + 0 ], 0x10, %r104;
cp.async.commit_group;
add.s32 %r107, %r102, 65536;
cp.async.cg.shared.global [ %r107 + 0 ], [ %rd69 + 0 ], 0x10, %r104;
add.s32 %r108, %r102, 69632;
cp.async.cg.shared.global [ %r108 + 0 ], [ %rd70 + 0 ], 0x10, %r104;
add.s32 %r109, %r102, 73728;
cp.async.cg.shared.global [ %r109 + 0 ], [ %rd71 + 0 ], 0x10, %r104;
add.s32 %r110, %r102, 77824;
cp.async.cg.shared.global [ %r110 + 0 ], [ %rd72 + 0 ], 0x10, %r104;
cp.async.commit_group;
add.s32 %r373, %r373, 1;
add.s64 %rd92, %rd92, 128;
add.s32 %r306, %r306, -64;
setp.ne.b32 %p13, %r7, %r373;
@%p13 bra $L__BB0_2;
$L__BB0_3:
shl.b32 %r190, %r4, 3;
and.b32 %r191, %r190, 120;
or.b32 %r192, %r3, %r191;
and.b32 %r193, %r4, 240;
bfe.u32 %r194, %r4, 4, 4;
or.b32 %r195, %r194, 112;
or.b32 %r196, %r3, %r195;
or.b32 %r197, %r194, 96;
or.b32 %r198, %r3, %r197;
or.b32 %r199, %r194, 80;
or.b32 %r200, %r3, %r199;
or.b32 %r201, %r194, 64;
or.b32 %r202, %r3, %r201;
or.b32 %r203, %r194, 48;
or.b32 %r204, %r3, %r203;
or.b32 %r205, %r194, 32;
or.b32 %r206, %r3, %r205;
or.b32 %r207, %r194, 16;
or.b32 %r208, %r3, %r207;
or.b32 %r209, %r3, %r194;
or.b32 %r210, %r2, %r191;
or.b32 %r211, %r2, %r195;
or.b32 %r212, %r2, %r197;
or.b32 %r213, %r2, %r199;
or.b32 %r214, %r2, %r201;
or.b32 %r215, %r2, %r203;
or.b32 %r216, %r2, %r205;
or.b32 %r217, %r2, %r207;
or.b32 %r218, %r2, %r194;
mul.lo.s32 %r219, %r14, %r1;
mad.wide.s32 %rd91, %r219, 2, %rd17;
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r220, %r310, %r309;
cvt.rn.bf16x2.f32 %r221, %r312, %r311;
cvt.rn.bf16x2.f32 %r222, %r314, %r313;
cvt.rn.bf16x2.f32 %r223, %r316, %r315;
cvt.rn.bf16x2.f32 %r224, %r318, %r317;
cvt.rn.bf16x2.f32 %r225, %r320, %r319;
cvt.rn.bf16x2.f32 %r226, %r322, %r321;
cvt.rn.bf16x2.f32 %r227, %r324, %r323;
cvt.rn.bf16x2.f32 %r228, %r326, %r325;
cvt.rn.bf16x2.f32 %r229, %r328, %r327;
cvt.rn.bf16x2.f32 %r230, %r330, %r329;
cvt.rn.bf16x2.f32 %r231, %r332, %r331;
cvt.rn.bf16x2.f32 %r232, %r334, %r333;
cvt.rn.bf16x2.f32 %r233, %r336, %r335;
cvt.rn.bf16x2.f32 %r234, %r338, %r337;
cvt.rn.bf16x2.f32 %r235, %r340, %r339;
cvt.rn.bf16x2.f32 %r236, %r342, %r341;
cvt.rn.bf16x2.f32 %r237, %r344, %r343;
cvt.rn.bf16x2.f32 %r238, %r346, %r345;
cvt.rn.bf16x2.f32 %r239, %r348, %r347;
cvt.rn.bf16x2.f32 %r240, %r350, %r349;
cvt.rn.bf16x2.f32 %r241, %r352, %r351;
cvt.rn.bf16x2.f32 %r242, %r354, %r353;
cvt.rn.bf16x2.f32 %r243, %r356, %r355;
cvt.rn.bf16x2.f32 %r244, %r358, %r357;
cvt.rn.bf16x2.f32 %r245, %r360, %r359;
cvt.rn.bf16x2.f32 %r246, %r362, %r361;
cvt.rn.bf16x2.f32 %r247, %r364, %r363;
cvt.rn.bf16x2.f32 %r248, %r366, %r365;
cvt.rn.bf16x2.f32 %r249, %r368, %r367;
cvt.rn.bf16x2.f32 %r250, %r370, %r369;
cvt.rn.bf16x2.f32 %r251, %r372, %r371;
mad.lo.s32 %r252, %r15, %r218, %r192;
mad.lo.s32 %r253, %r15, %r217, %r192;
mad.lo.s32 %r254, %r15, %r216, %r192;
mad.lo.s32 %r255, %r15, %r215, %r192;
mad.lo.s32 %r256, %r15, %r214, %r192;
mad.lo.s32 %r257, %r15, %r213, %r192;
mad.lo.s32 %r258, %r15, %r212, %r192;
mad.lo.s32 %r259, %r15, %r211, %r192;
mad.wide.s32 %rd75, %r252, 2, %rd91;
mad.wide.s32 %rd76, %r253, 2, %rd91;
mad.wide.s32 %rd77, %r254, 2, %rd91;
mad.wide.s32 %rd78, %r255, 2, %rd91;
mad.wide.s32 %rd79, %r256, 2, %rd91;
mad.wide.s32 %rd80, %r257, 2, %rd91;
mad.wide.s32 %rd81, %r258, 2, %rd91;
mad.wide.s32 %rd82, %r259, 2, %rd91;
setp.lt.s32 %p30, %r218, %r12;
setp.lt.s32 %p31, %r217, %r12;
setp.lt.s32 %p32, %r216, %r12;
setp.lt.s32 %p33, %r215, %r12;
setp.lt.s32 %p34, %r214, %r12;
setp.lt.s32 %p35, %r213, %r12;
setp.lt.s32 %p36, %r212, %r12;
setp.lt.s32 %p37, %r211, %r12;
setp.lt.s32 %p38, %r192, %r12;
and.pred %p14, %p30, %p38;
and.pred %p15, %p31, %p38;
and.pred %p16, %p32, %p38;
and.pred %p17, %p33, %p38;
and.pred %p18, %p34, %p38;
and.pred %p19, %p35, %p38;
and.pred %p20, %p36, %p38;
and.pred %p21, %p37, %p38;
and.b32 %r260, %r4, 3;
shl.b32 %r261, %r260, 13;
shl.b32 %r262, %r4, 5;
and.b32 %r263, %r262, 7264;
and.b32 %r264, %r4, 24;
shl.b32 %r265, %r264, 4;
shl.b32 %r266, %r4, 2;
and.b32 %r267, %r266, 16;
or.b32 %r268, %r261, %r267;
or.b32 %r269, %r263, %r265;
or.b32 %r270, %r268, %r269;
add.s32 %r271, %r91, %r270;
st.shared.v4.b32 [%r271], {%r220, %r222, %r224, %r226};
st.shared.v4.b32 [%r271+512], {%r221, %r223, %r225, %r227};
xor.b32 %r272, %r270, 32;
add.s32 %r273, %r91, %r272;
st.shared.v4.b32 [%r273], {%r228, %r230, %r232, %r234};
st.shared.v4.b32 [%r273+512], {%r229, %r231, %r233, %r235};
xor.b32 %r274, %r270, 64;
add.s32 %r275, %r91, %r274;
st.shared.v4.b32 [%r275], {%r236, %r238, %r240, %r242};
st.shared.v4.b32 [%r275+512], {%r237, %r239, %r241, %r243};
xor.b32 %r276, %r270, 96;
add.s32 %r277, %r91, %r276;
st.shared.v4.b32 [%r277], {%r244, %r246, %r248, %r250};
st.shared.v4.b32 [%r277+512], {%r245, %r247, %r249, %r251};
bar.sync 0;
shl.b32 %r278, %r264, 10;
shl.b32 %r279, %r260, 5;
and.b32 %r280, %r266, 1008;
or.b32 %r281, %r278, %r279;
xor.b32 %r282, %r281, %r280;
add.s32 %r283, %r91, %r282;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r128, %r129}, [%r283];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r130, %r131, %r132, %r133}, [%r283+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r136, %r137}, [%r283+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r138, %r139, %r140, %r141}, [%r283+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r142, %r143, %r144, %r145}, [%r283+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r146, %r147, %r148, %r149}, [%r283+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r150, %r151, %r152, %r153}, [%r283+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r154, %r155, %r156, %r157}, [%r283+7168];
@%p14 st.global.v4.b32 [ %rd75 + 0 ], { %r126, %r127, %r128, %r129 };
@%p15 st.global.v4.b32 [ %rd76 + 0 ], { %r130, %r131, %r132, %r133 };
@%p16 st.global.v4.b32 [ %rd77 + 0 ], { %r134, %r135, %r136, %r137 };
@%p17 st.global.v4.b32 [ %rd78 + 0 ], { %r138, %r139, %r140, %r141 };
@%p18 st.global.v4.b32 [ %rd79 + 0 ], { %r142, %r143, %r144, %r145 };
@%p19 st.global.v4.b32 [ %rd80 + 0 ], { %r146, %r147, %r148, %r149 };
@%p20 st.global.v4.b32 [ %rd81 + 0 ], { %r150, %r151, %r152, %r153 };
@%p21 st.global.v4.b32 [ %rd82 + 0 ], { %r154, %r155, %r156, %r157 };
mad.lo.s32 %r284, %r15, %r209, %r210;
mad.lo.s32 %r285, %r15, %r208, %r210;
mad.lo.s32 %r286, %r15, %r206, %r210;
mad.lo.s32 %r287, %r15, %r204, %r210;
mad.lo.s32 %r288, %r15, %r202, %r210;
mad.lo.s32 %r289, %r15, %r200, %r210;
mad.lo.s32 %r290, %r15, %r198, %r210;
mad.lo.s32 %r291, %r15, %r196, %r210;
mad.wide.s32 %rd83, %r284, 2, %rd91;
mad.wide.s32 %rd84, %r285, 2, %rd91;
mad.wide.s32 %rd85, %r286, 2, %rd91;
mad.wide.s32 %rd86, %r287, 2, %rd91;
mad.wide.s32 %rd87, %r288, 2, %rd91;
mad.wide.s32 %rd88, %r289, 2, %rd91;
mad.wide.s32 %rd89, %r290, 2, %rd91;
mad.wide.s32 %rd90, %r291, 2, %rd91;
setp.lt.s32 %p39, %r209, %r12;
setp.lt.s32 %p40, %r208, %r12;
setp.lt.s32 %p41, %r206, %r12;
setp.lt.s32 %p42, %r204, %r12;
setp.lt.s32 %p43, %r202, %r12;
setp.lt.s32 %p44, %r200, %r12;
setp.lt.s32 %p45, %r198, %r12;
setp.lt.s32 %p46, %r196, %r12;
setp.lt.s32 %p47, %r210, %r12;
and.pred %p22, %p39, %p47;
and.pred %p23, %p40, %p47;
and.pred %p24, %p41, %p47;
and.pred %p25, %p42, %p47;
and.pred %p26, %p43, %p47;
and.pred %p27, %p44, %p47;
and.pred %p28, %p45, %p47;
and.pred %p29, %p46, %p47;
bar.sync 0;
shl.b32 %r292, %r5, 9;
shl.b32 %r293, %r5, 4;
and.b32 %r294, %r8, 272;
and.b32 %r295, %r4, 96;
and.b32 %r296, %r190, 128;
or.b32 %r297, %r294, %r295;
xor.b32 %r298, %r297, %r293;
add.s32 %r299, %r91, %r292;
add.s32 %r300, %r299, %r296;
add.s32 %r301, %r300, %r298;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r220, %r221, %r222, %r223};
bar.sync 0;
and.b32 %r302, %r262, 3840;
or.b32 %r303, %r302, %r293;
xor.b32 %r304, %r303, %r193;
add.s32 %r305, %r91, %r304;
ld.shared.v4.b32 {%r158, %r159, %r160, %r161}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r224, %r225, %r226, %r227};
bar.sync 0;
ld.shared.v4.b32 {%r162, %r163, %r164, %r165}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r228, %r229, %r230, %r231};
bar.sync 0;
ld.shared.v4.b32 {%r166, %r167, %r168, %r169}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r232, %r233, %r234, %r235};
bar.sync 0;
ld.shared.v4.b32 {%r170, %r171, %r172, %r173}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r236, %r237, %r238, %r239};
bar.sync 0;
ld.shared.v4.b32 {%r174, %r175, %r176, %r177}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r240, %r241, %r242, %r243};
bar.sync 0;
ld.shared.v4.b32 {%r178, %r179, %r180, %r181}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r244, %r245, %r246, %r247};
bar.sync 0;
ld.shared.v4.b32 {%r182, %r183, %r184, %r185}, [%r305];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r301], {%r248, %r249, %r250, %r251};
bar.sync 0;
ld.shared.v4.b32 {%r186, %r187, %r188, %r189}, [%r305];
@%p22 st.global.v4.b32 [ %rd83 + 0 ], { %r158, %r159, %r160, %r161 };
@%p23 st.global.v4.b32 [ %rd84 + 0 ], { %r162, %r163, %r164, %r165 };
@%p24 st.global.v4.b32 [ %rd85 + 0 ], { %r166, %r167, %r168, %r169 };
@%p25 st.global.v4.b32 [ %rd86 + 0 ], { %r170, %r171, %r172, %r173 };
@%p26 st.global.v4.b32 [ %rd87 + 0 ], { %r174, %r175, %r176, %r177 };
@%p27 st.global.v4.b32 [ %rd88 + 0 ], { %r178, %r179, %r180, %r181 };
@%p28 st.global.v4.b32 [ %rd89 + 0 ], { %r182, %r183, %r184, %r185 };
@%p29 st.global.v4.b32 [ %rd90 + 0 ], { %r186, %r187, %r188, %r189 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .voGram, .full => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry polar(
.param .align 64 .b8 polar_param_0[128],
.param .u32 polar_param_1,
.param .u32 polar_param_2,
.param .u64 polar_param_3,
.param .u64 polar_param_4,
.param .align 64 .b8 polar_param_5[128],
.param .u32 polar_param_6,
.param .u32 polar_param_7,
.param .u64 polar_param_8,
.param .u64 polar_param_9,
.param .u64 .ptr .global .align 1 polar_param_10,
.param .u64 .ptr .global .align 1 polar_param_11
)
.reqntid 256
{
.reg .pred %p<32>;
.reg .b32 %r<222>;
.reg .b64 %rd<45>;
$L__func_begin0:
mov.b64 %rd1, polar_param_0;
mov.b64 %rd2, polar_param_5;
$L__tmp0:
cvta.param.u64 %rd44, %rd2;
cvta.param.u64 %rd25, %rd1;
mov.u32 %r45, %ctaid.x;
mul.hi.u32 %r46, %r45, 818089009;
shr.u32 %r47, %r46, 2;
$L__tmp1:
mul.lo.s32 %r48, %r47, 21;
sub.s32 %r49, %r45, %r48;
mul.lo.s32 %r50, %r49, 3;
mov.b64 %rd3, 6569755928097638024;
shr.u64 %rd4, %rd3, %r50;
cvt.u32.u64 %r51, %rd4;
shl.b32 %r52, %r51, 7;
and.b32 %r1, %r52, 896;
mov.b64 %rd5, 6420841298313609216;
shr.u64 %rd6, %rd5, %r50;
cvt.u32.u64 %r53, %rd6;
shl.b32 %r54, %r53, 7;
and.b32 %r2, %r54, 896;
$L__tmp2:
setp.eq.b32 %p2, %r1, %r2;
mov.u32 %r3, %tid.x;
and.b32 %r4, %r3, 255;
setp.eq.b32 %p1, %r4, 0;
mov.b32 %r55, global_smem;
add.s32 %r85, %r55, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r85], 1;
bar.sync 0;
add.s32 %r44, %r55, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r44], 1;
mul.lo.s32 %r5, %r47, 768;
$L__tmp3:
add.s32 %r136, %r1, %r5;
shr.u32 %r6, %r3, 5;
setp.lt.u32 %p31, %r4, 64;
add.s32 %r71, %r2, %r5;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r85], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r88, %r6, 0, 31, -1;
elect.sync %r89|%p18, -1;
and.pred %p17, %p31, %p18;
and.b32 %r90, %r88, 1;
shl.b32 %r91, %r90, 14;
add.s32 %r86, %r55, %r91;
shl.b32 %r87, %r90, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r86], [%rd25, {%r87, %r136}], [%r85];
shl.b32 %r92, %r88, 7;
and.b32 %r9, %r92, 512;
$L__tmp6:
or.b32 %r218, %r87, 128;
mov.b32 %r219, 0;
mov.b32 %r153, 0f00000000;
mov.b32 %r154, %r153;
mov.b32 %r155, %r153;
mov.b32 %r156, %r153;
mov.b32 %r157, %r153;
mov.b32 %r158, %r153;
mov.b32 %r159, %r153;
mov.b32 %r160, %r153;
mov.b32 %r161, %r153;
mov.b32 %r162, %r153;
mov.b32 %r163, %r153;
mov.b32 %r164, %r153;
mov.b32 %r165, %r153;
mov.b32 %r166, %r153;
mov.b32 %r167, %r153;
mov.b32 %r168, %r153;
mov.b32 %r169, %r153;
mov.b32 %r170, %r153;
mov.b32 %r171, %r153;
mov.b32 %r172, %r153;
mov.b32 %r173, %r153;
mov.b32 %r174, %r153;
mov.b32 %r175, %r153;
mov.b32 %r176, %r153;
mov.b32 %r177, %r153;
mov.b32 %r178, %r153;
mov.b32 %r179, %r153;
mov.b32 %r180, %r153;
mov.b32 %r181, %r153;
mov.b32 %r182, %r153;
mov.b32 %r183, %r153;
mov.b32 %r184, %r153;
mov.b32 %r185, %r153;
mov.b32 %r186, %r153;
mov.b32 %r187, %r153;
mov.b32 %r188, %r153;
mov.b32 %r189, %r153;
mov.b32 %r190, %r153;
mov.b32 %r191, %r153;
mov.b32 %r192, %r153;
mov.b32 %r193, %r153;
mov.b32 %r194, %r153;
mov.b32 %r195, %r153;
mov.b32 %r196, %r153;
mov.b32 %r197, %r153;
mov.b32 %r198, %r153;
mov.b32 %r199, %r153;
mov.b32 %r200, %r153;
mov.b32 %r201, %r153;
mov.b32 %r202, %r153;
mov.b32 %r203, %r153;
mov.b32 %r204, %r153;
mov.b32 %r205, %r153;
mov.b32 %r206, %r153;
mov.b32 %r207, %r153;
mov.b32 %r208, %r153;
mov.b32 %r209, %r153;
mov.b32 %r210, %r153;
mov.b32 %r211, %r153;
mov.b32 %r212, %r153;
mov.b32 %r213, %r153;
mov.b32 %r214, %r153;
mov.b32 %r215, %r153;
mov.b32 %r216, %r153;
$L__BB0_4:
and.b32 %r97, %r219, 1;
bfe.u32 %r94, %r219, 1, 1;
shl.b32 %r98, %r97, 15;
add.s32 %r99, %r55, %r98;
mad.lo.s32 %r100, %r97, -32760, %r99;
add.s32 %r93, %r100, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r93], %r94;
@!complete bra.uni waitLoop;
}
shr.u32 %r101, %r99, 4;
add.s32 %r102, %r101, %r9;
and.b32 %r103, %r102, 16383;
cvt.u64.u32 %rd42, %r103;
bfe.u32 %r104, %r99, 4, 14;
cvt.u64.u32 %rd43, %r104;
wgmma.fence.sync.aligned;
or.b64 %rd26, %rd42, 4611686293305294848;
or.b64 %rd27, %rd43, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd26, %rd27, %p19, 1, 1, 0, 0;
add.s64 %rd28, %rd42, 4611686293305294850;
add.s64 %rd29, %rd43, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd28, %rd29, %p19, 1, 1, 0, 0;
add.s64 %rd30, %rd42, 4611686293305294852;
add.s64 %rd31, %rd43, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd30, %rd31, %p19, 1, 1, 0, 0;
add.s64 %rd32, %rd42, 4611686293305294854;
add.s64 %rd33, %rd43, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd32, %rd33, %p19, 1, 1, 0, 0;
add.s64 %rd34, %rd42, 4611686293305295872;
add.s64 %rd35, %rd43, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd34, %rd35, %p19, 1, 1, 0, 0;
add.s64 %rd36, %rd42, 4611686293305295874;
add.s64 %rd37, %rd43, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd36, %rd37, %p19, 1, 1, 0, 0;
add.s64 %rd38, %rd42, 4611686293305295876;
add.s64 %rd39, %rd43, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd38, %rd39, %p19, 1, 1, 0, 0;
add.s64 %rd40, %rd42, 4611686293305295878;
add.s64 %rd41, %rd43, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd40, %rd41, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r10, %r219, 1;
and.b32 %r105, %r10, 1;
setp.ne.b32 %p22, %r219, 5;
shl.b32 %r106, %r105, 3;
add.s32 %r107, %r55, %r106;
add.s32 %r95, %r107, 98304;
shl.b32 %r108, %r105, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r95], 32768;
bar.sync 0;
elect.sync %r109|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p31, %p24;
add.s32 %r96, %r86, %r108;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r96], [%rd25, {%r218, %r136}], [%r95];
$L__tmp8:
add.s32 %r218, %r218, 128;
setp.ne.b32 %p25, %r10, 6;
mov.b32 %r219, %r10;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r85], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r58, %r6, 0, 31, -1;
elect.sync %r59|%p5, -1;
and.pred %p3, %p31, %p5;
and.b32 %r60, %r58, 1;
shl.b32 %r61, %r60, 14;
add.s32 %r56, %r55, %r61;
shl.b32 %r57, %r60, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r56], [%rd25, {%r57, %r136}], [%r85];
$L__tmp10:
bar.sync 0;
elect.sync %r62|%p6, -1;
and.pred %p4, %p31, %p6;
add.s32 %r70, %r56, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r70], [%rd25, {%r57, %r71}], [%r85];
shl.b32 %r63, %r58, 7;
and.b32 %r7, %r63, 512;
add.s32 %r64, %r55, 65536;
bfe.u32 %r65, %r64, 4, 14;
cvt.u64.u32 %rd7, %r65;
or.b64 %rd9, %rd7, 4611686293305294848;
add.s64 %rd11, %rd7, 4611686293305294850;
add.s64 %rd13, %rd7, 4611686293305294852;
add.s64 %rd15, %rd7, 4611686293305294854;
add.s64 %rd17, %rd7, 4611686293305295872;
add.s64 %rd19, %rd7, 4611686293305295874;
add.s64 %rd21, %rd7, 4611686293305295876;
add.s64 %rd23, %rd7, 4611686293305295878;
$L__tmp11:
or.b32 %r152, %r57, 128;
mov.b32 %r217, 0;
mov.b32 %r153, 0f00000000;
mov.b32 %r154, %r153;
mov.b32 %r155, %r153;
mov.b32 %r156, %r153;
mov.b32 %r157, %r153;
mov.b32 %r158, %r153;
mov.b32 %r159, %r153;
mov.b32 %r160, %r153;
mov.b32 %r161, %r153;
mov.b32 %r162, %r153;
mov.b32 %r163, %r153;
mov.b32 %r164, %r153;
mov.b32 %r165, %r153;
mov.b32 %r166, %r153;
mov.b32 %r167, %r153;
mov.b32 %r168, %r153;
mov.b32 %r169, %r153;
mov.b32 %r170, %r153;
mov.b32 %r171, %r153;
mov.b32 %r172, %r153;
mov.b32 %r173, %r153;
mov.b32 %r174, %r153;
mov.b32 %r175, %r153;
mov.b32 %r176, %r153;
mov.b32 %r177, %r153;
mov.b32 %r178, %r153;
mov.b32 %r179, %r153;
mov.b32 %r180, %r153;
mov.b32 %r181, %r153;
mov.b32 %r182, %r153;
mov.b32 %r183, %r153;
mov.b32 %r184, %r153;
mov.b32 %r185, %r153;
mov.b32 %r186, %r153;
mov.b32 %r187, %r153;
mov.b32 %r188, %r153;
mov.b32 %r189, %r153;
mov.b32 %r190, %r153;
mov.b32 %r191, %r153;
mov.b32 %r192, %r153;
mov.b32 %r193, %r153;
mov.b32 %r194, %r153;
mov.b32 %r195, %r153;
mov.b32 %r196, %r153;
mov.b32 %r197, %r153;
mov.b32 %r198, %r153;
mov.b32 %r199, %r153;
mov.b32 %r200, %r153;
mov.b32 %r201, %r153;
mov.b32 %r202, %r153;
mov.b32 %r203, %r153;
mov.b32 %r204, %r153;
mov.b32 %r205, %r153;
mov.b32 %r206, %r153;
mov.b32 %r207, %r153;
mov.b32 %r208, %r153;
mov.b32 %r209, %r153;
mov.b32 %r210, %r153;
mov.b32 %r211, %r153;
mov.b32 %r212, %r153;
mov.b32 %r213, %r153;
mov.b32 %r214, %r153;
mov.b32 %r215, %r153;
mov.b32 %r216, %r153;
$L__BB0_2:
and.b32 %r72, %r217, 1;
bfe.u32 %r67, %r217, 1, 1;
shl.b32 %r73, %r72, 15;
add.s32 %r74, %r55, %r73;
mad.lo.s32 %r75, %r72, -32760, %r74;
add.s32 %r66, %r75, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r66], %r67;
@!complete bra.uni waitLoop;
}
shr.u32 %r76, %r74, 4;
add.s32 %r77, %r76, %r7;
and.b32 %r78, %r77, 16383;
cvt.u64.u32 %rd24, %r78;
wgmma.fence.sync.aligned;
or.b64 %rd8, %rd24, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd8, %rd9, %p7, 1, 1, 0, 0;
add.s64 %rd10, %rd24, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd10, %rd11, %p7, 1, 1, 0, 0;
add.s64 %rd12, %rd24, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd12, %rd13, %p7, 1, 1, 0, 0;
add.s64 %rd14, %rd24, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd14, %rd15, %p7, 1, 1, 0, 0;
add.s64 %rd16, %rd24, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd16, %rd17, %p7, 1, 1, 0, 0;
add.s64 %rd18, %rd24, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd18, %rd19, %p7, 1, 1, 0, 0;
add.s64 %rd20, %rd24, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd20, %rd21, %p7, 1, 1, 0, 0;
add.s64 %rd22, %rd24, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r153,%r154,%r155,%r156,%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216}, %rd22, %rd23, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r8, %r217, 1;
and.b32 %r79, %r8, 1;
setp.ne.b32 %p11, %r217, 5;
shl.b32 %r80, %r79, 3;
add.s32 %r81, %r55, %r80;
add.s32 %r68, %r81, 98304;
shl.b32 %r82, %r79, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r68], 65536;
bar.sync 0;
elect.sync %r83|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p31, %p13;
add.s32 %r69, %r56, %r82;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r69], [%rd25, {%r152, %r136}], [%r68];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r84|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p31, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r70], [%rd25, {%r152, %r71}], [%r68];
$L__tmp15:
add.s32 %r152, %r152, 128;
setp.eq.b32 %p16, %r8, 6;
mov.b32 %r217, %r8;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r1, %r2;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r85];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r44];
cvt.rn.bf16x2.f32 %r11, %r154, %r153;
cvt.rn.bf16x2.f32 %r12, %r156, %r155;
cvt.rn.bf16x2.f32 %r13, %r158, %r157;
cvt.rn.bf16x2.f32 %r14, %r160, %r159;
cvt.rn.bf16x2.f32 %r15, %r162, %r161;
cvt.rn.bf16x2.f32 %r16, %r164, %r163;
cvt.rn.bf16x2.f32 %r17, %r166, %r165;
cvt.rn.bf16x2.f32 %r18, %r168, %r167;
cvt.rn.bf16x2.f32 %r19, %r170, %r169;
cvt.rn.bf16x2.f32 %r20, %r172, %r171;
cvt.rn.bf16x2.f32 %r21, %r174, %r173;
cvt.rn.bf16x2.f32 %r22, %r176, %r175;
cvt.rn.bf16x2.f32 %r23, %r178, %r177;
cvt.rn.bf16x2.f32 %r24, %r180, %r179;
cvt.rn.bf16x2.f32 %r25, %r182, %r181;
cvt.rn.bf16x2.f32 %r26, %r184, %r183;
cvt.rn.bf16x2.f32 %r27, %r186, %r185;
cvt.rn.bf16x2.f32 %r28, %r188, %r187;
cvt.rn.bf16x2.f32 %r29, %r190, %r189;
cvt.rn.bf16x2.f32 %r30, %r192, %r191;
cvt.rn.bf16x2.f32 %r31, %r194, %r193;
cvt.rn.bf16x2.f32 %r32, %r196, %r195;
cvt.rn.bf16x2.f32 %r33, %r198, %r197;
cvt.rn.bf16x2.f32 %r34, %r200, %r199;
cvt.rn.bf16x2.f32 %r35, %r202, %r201;
cvt.rn.bf16x2.f32 %r36, %r204, %r203;
cvt.rn.bf16x2.f32 %r37, %r206, %r205;
cvt.rn.bf16x2.f32 %r38, %r208, %r207;
cvt.rn.bf16x2.f32 %r39, %r210, %r209;
cvt.rn.bf16x2.f32 %r40, %r212, %r211;
cvt.rn.bf16x2.f32 %r41, %r214, %r213;
cvt.rn.bf16x2.f32 %r42, %r216, %r215;
shl.b32 %r43, %r3, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r124, %r3, 4;
and.b32 %r221, %r124, 112;
shl.b32 %r220, %r3, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r110, %r43, 1920;
shl.b32 %r220, %r3, 6;
and.b32 %r111, %r220, 14336;
shl.b32 %r112, %r3, 4;
and.b32 %r221, %r112, 112;
and.b32 %r113, %r3, 16;
or.b32 %r114, %r110, %r221;
xor.b32 %r115, %r114, %r113;
or.b32 %r116, %r115, %r111;
add.s32 %r117, %r55, %r116;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r117], {%r11, %r12, %r13, %r14};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r117+16384], {%r27, %r28, %r29, %r30};
xor.b32 %r118, %r116, 32;
add.s32 %r119, %r55, %r118;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r119], {%r15, %r16, %r17, %r18};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r119+16384], {%r31, %r32, %r33, %r34};
xor.b32 %r120, %r116, 64;
add.s32 %r121, %r55, %r120;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r121], {%r19, %r20, %r21, %r22};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r121+16384], {%r35, %r36, %r37, %r38};
xor.b32 %r122, %r116, 96;
add.s32 %r123, %r55, %r122;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123], {%r23, %r24, %r25, %r26};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r123+16384], {%r39, %r40, %r41, %r42};
$L__BB0_9:
and.b32 %r125, %r43, 17280;
and.b32 %r126, %r3, 96;
shl.b32 %r127, %r3, 1;
and.b32 %r128, %r127, 16;
and.b32 %r129, %r220, 1024;
or.b32 %r130, %r125, %r221;
or.b32 %r131, %r128, %r126;
xor.b32 %r132, %r130, %r131;
add.s32 %r133, %r55, %r129;
add.s32 %r134, %r133, %r132;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+32768], {%r11, %r12, %r13, %r14};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+34816], {%r15, %r16, %r17, %r18};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+36864], {%r19, %r20, %r21, %r22};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+38912], {%r23, %r24, %r25, %r26};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+40960], {%r27, %r28, %r29, %r30};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+43008], {%r31, %r32, %r33, %r34};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+45056], {%r35, %r36, %r37, %r38};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r134+47104], {%r39, %r40, %r41, %r42};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_11;
bar.sync 0;
elect.sync %r138|%p28, -1;
shfl.sync.idx.b32 %r139, %r6, 0, 31, -1;
and.pred %p27, %p31, %p28;
and.b32 %r140, %r139, 1;
shl.b32 %r141, %r140, 14;
add.s32 %r137, %r55, %r141;
shl.b32 %r142, %r140, 6;
or.b32 %r135, %r142, %r2;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd44, {%r135, %r136}], [%r137];
cp.async.bulk.commit_group;
$L__BB0_11:
bar.sync 0;
elect.sync %r146|%p30, -1;
shfl.sync.idx.b32 %r147, %r6, 0, 31, -1;
and.pred %p29, %p31, %p30;
and.b32 %r148, %r147, 1;
shl.b32 %r149, %r148, 14;
add.s32 %r150, %r55, %r149;
add.s32 %r144, %r150, 32768;
shl.b32 %r151, %r148, 6;
or.b32 %r143, %r151, %r1;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd44, {%r143, %r71}], [%r144];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r145, 0;
ret;
$L__tmp17:
$L__func_end0:
}
"#
  | .mlpGram, .baseline => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XTX_kernel(
.param .u64 .ptr .global .align 1 XTX_kernel_param_0,
.param .u64 .ptr .global .align 1 XTX_kernel_param_1,
.param .u32 XTX_kernel_param_2,
.param .u32 XTX_kernel_param_3,
.param .u32 XTX_kernel_param_4,
.param .u32 XTX_kernel_param_5,
.param .u32 XTX_kernel_param_6,
.param .u32 XTX_kernel_param_7,
.param .u64 .ptr .global .align 1 XTX_kernel_param_8,
.param .u64 .ptr .global .align 1 XTX_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<61>;
.reg .b32 %r<396>;
.reg .b64 %rd<94>;
$L__func_begin0:
ld.param.b32 %r40, [XTX_kernel_param_3];
$L__tmp0:
mov.u32 %r41, %ctaid.x;
$L__tmp1:
add.s32 %r42, %r40, 127;
shr.s32 %r43, %r42, 31;
shr.u32 %r44, %r43, 25;
add.s32 %r45, %r42, %r44;
shr.s32 %r46, %r45, 7;
$L__tmp2:
mul.lo.s32 %r47, %r46, %r46;
div.s32 %r1, %r41, %r47;
mul.lo.s32 %r48, %r1, %r47;
sub.s32 %r49, %r41, %r48;
$L__tmp3:
shl.b32 %r50, %r46, 3;
div.s32 %r51, %r49, %r50;
shl.b32 %r52, %r51, 3;
sub.s32 %r53, %r46, %r52;
min.s32 %r54, %r53, 8;
mul.lo.s32 %r55, %r51, %r50;
sub.s32 %r56, %r49, %r55;
div.s32 %r57, %r56, %r54;
mul.lo.s32 %r58, %r57, %r54;
sub.s32 %r59, %r56, %r58;
add.s32 %r60, %r59, %r52;
$L__tmp4:
shl.b32 %r2, %r60, 7;
shl.b32 %r3, %r57, 7;
$L__tmp5:
add.s32 %r61, %r2, 128;
setp.gt.s32 %p1, %r61, %r3;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b32 %r39, [XTX_kernel_param_7];
ld.param.b32 %r38, [XTX_kernel_param_6];
ld.param.b32 %r37, [XTX_kernel_param_5];
ld.param.b32 %r36, [XTX_kernel_param_4];
ld.param.b32 %r35, [XTX_kernel_param_2];
ld.param.b64 %rd22, [XTX_kernel_param_1];
ld.param.b64 %rd21, [XTX_kernel_param_0];
mul.lo.s32 %r98, %r1, %r36;
mad.wide.s32 %rd47, %r98, 2, %rd21;
mul.lo.s32 %r99, %r1, %r38;
mad.wide.s32 %rd2, %r99, 2, %rd22;
mov.u32 %r4, %tid.x;
shl.b32 %r5, %r4, 3;
and.b32 %r100, %r5, 120;
and.b32 %r6, %r4, 240;
bfe.u32 %r7, %r4, 4, 4;
or.b32 %r8, %r7, 16;
or.b32 %r9, %r7, 32;
or.b32 %r10, %r7, 48;
or.b32 %r101, %r7, 64;
or.b32 %r102, %r7, 80;
or.b32 %r103, %r7, 96;
or.b32 %r104, %r7, 112;
or.b32 %r11, %r2, %r100;
or.b32 %r12, %r2, %r7;
or.b32 %r13, %r2, %r8;
or.b32 %r14, %r2, %r9;
or.b32 %r15, %r2, %r10;
or.b32 %r16, %r2, %r101;
or.b32 %r17, %r2, %r102;
or.b32 %r18, %r2, %r103;
or.b32 %r19, %r2, %r104;
rem.s32 %r105, %r11, %r40;
or.b32 %r20, %r3, %r100;
or.b32 %r21, %r3, %r7;
or.b32 %r22, %r3, %r8;
or.b32 %r23, %r3, %r9;
or.b32 %r24, %r3, %r10;
or.b32 %r25, %r3, %r101;
or.b32 %r26, %r3, %r102;
or.b32 %r27, %r3, %r103;
or.b32 %r28, %r3, %r104;
rem.s32 %r106, %r20, %r40;
mul.lo.s32 %r107, %r37, %r7;
shl.b32 %r108, %r37, 4;
add.s32 %r109, %r107, %r108;
add.s32 %r110, %r109, %r108;
add.s32 %r111, %r110, %r108;
add.s32 %r112, %r105, %r107;
add.s32 %r113, %r105, %r109;
add.s32 %r114, %r105, %r110;
add.s32 %r115, %r105, %r111;
mad.wide.s32 %rd23, %r112, 2, %rd47;
mad.wide.s32 %rd24, %r113, 2, %rd47;
mad.wide.s32 %rd25, %r114, 2, %rd47;
mad.wide.s32 %rd26, %r115, 2, %rd47;
add.s32 %r116, %r106, %r107;
add.s32 %r117, %r106, %r109;
add.s32 %r118, %r106, %r110;
add.s32 %r119, %r106, %r111;
mad.wide.s32 %rd27, %r116, 2, %rd47;
mad.wide.s32 %rd28, %r117, 2, %rd47;
mad.wide.s32 %rd29, %r118, 2, %rd47;
mad.wide.s32 %rd30, %r119, 2, %rd47;
$L__tmp6:
add.s32 %r120, %r35, 63;
$L__tmp7:
shl.b32 %r124, %r37, 6;
setp.lt.s32 %p2, %r120, 64;
setp.gt.s32 %p3, %r120, 63;
setp.lt.s32 %p4, %r7, %r35;
setp.lt.s32 %p5, %r8, %r35;
setp.lt.s32 %p6, %r9, %r35;
setp.lt.s32 %p7, %r10, %r35;
and.b32 %r30, %r4, 7;
shl.b32 %r31, %r30, 4;
shl.b32 %r125, %r6, 3;
and.b32 %r126, %r4, 112;
shl.b32 %r127, %r4, 10;
and.b32 %r128, %r127, 8192;
or.b32 %r129, %r31, %r125;
xor.b32 %r130, %r129, %r126;
or.b32 %r32, %r130, %r128;
mov.b32 %r131, global_smem;
add.s32 %r62, %r131, %r32;
selp.b32 %r132, 16, 0, %p3;
selp.b32 %r67, %r132, 0, %p4;
cp.async.cg.shared.global [ %r62 + 0 ], [ %rd23 + 0 ], 0x10, %r67;
add.s32 %r63, %r62, 2048;
selp.b32 %r69, %r132, 0, %p5;
cp.async.cg.shared.global [ %r63 + 0 ], [ %rd24 + 0 ], 0x10, %r69;
add.s32 %r64, %r62, 4096;
selp.b32 %r71, %r132, 0, %p6;
cp.async.cg.shared.global [ %r64 + 0 ], [ %rd25 + 0 ], 0x10, %r71;
add.s32 %r65, %r62, 6144;
selp.b32 %r73, %r132, 0, %p7;
cp.async.cg.shared.global [ %r65 + 0 ], [ %rd26 + 0 ], 0x10, %r73;
cp.async.commit_group;
add.s32 %r66, %r62, 65536;
cp.async.cg.shared.global [ %r66 + 0 ], [ %rd27 + 0 ], 0x10, %r67;
add.s32 %r68, %r62, 67584;
cp.async.cg.shared.global [ %r68 + 0 ], [ %rd28 + 0 ], 0x10, %r69;
add.s32 %r70, %r62, 69632;
cp.async.cg.shared.global [ %r70 + 0 ], [ %rd29 + 0 ], 0x10, %r71;
add.s32 %r72, %r62, 71680;
cp.async.cg.shared.global [ %r72 + 0 ], [ %rd30 + 0 ], 0x10, %r73;
cp.async.commit_group;
setp.gt.s32 %p8, %r120, 127;
mul.wide.s32 %rd48, %r124, 2;
add.s64 %rd31, %rd23, %rd48;
add.s64 %rd32, %rd24, %rd48;
add.s64 %rd33, %rd25, %rd48;
add.s64 %rd34, %rd26, %rd48;
add.s64 %rd35, %rd27, %rd48;
add.s64 %rd36, %rd28, %rd48;
add.s64 %rd37, %rd29, %rd48;
add.s64 %rd38, %rd30, %rd48;
add.s32 %r133, %r35, -64;
setp.lt.s32 %p9, %r7, %r133;
setp.lt.s32 %p10, %r8, %r133;
setp.lt.s32 %p11, %r9, %r133;
setp.lt.s32 %p12, %r10, %r133;
bar.sync 0;
add.s32 %r74, %r62, 16384;
selp.b32 %r134, 16, 0, %p9;
selp.b32 %r79, %r134, 0, %p8;
cp.async.cg.shared.global [ %r74 + 0 ], [ %rd31 + 0 ], 0x10, %r79;
add.s32 %r75, %r62, 18432;
selp.b32 %r135, 16, 0, %p10;
selp.b32 %r81, %r135, 0, %p8;
cp.async.cg.shared.global [ %r75 + 0 ], [ %rd32 + 0 ], 0x10, %r81;
add.s32 %r76, %r62, 20480;
selp.b32 %r136, 16, 0, %p11;
selp.b32 %r83, %r136, 0, %p8;
cp.async.cg.shared.global [ %r76 + 0 ], [ %rd33 + 0 ], 0x10, %r83;
add.s32 %r77, %r62, 22528;
selp.b32 %r137, 16, 0, %p12;
selp.b32 %r85, %r137, 0, %p8;
cp.async.cg.shared.global [ %r77 + 0 ], [ %rd34 + 0 ], 0x10, %r85;
cp.async.commit_group;
add.s32 %r78, %r62, 81920;
cp.async.cg.shared.global [ %r78 + 0 ], [ %rd35 + 0 ], 0x10, %r79;
add.s32 %r80, %r62, 83968;
cp.async.cg.shared.global [ %r80 + 0 ], [ %rd36 + 0 ], 0x10, %r81;
add.s32 %r82, %r62, 86016;
cp.async.cg.shared.global [ %r82 + 0 ], [ %rd37 + 0 ], 0x10, %r83;
add.s32 %r84, %r62, 88064;
cp.async.cg.shared.global [ %r84 + 0 ], [ %rd38 + 0 ], 0x10, %r85;
cp.async.commit_group;
setp.gt.s32 %p13, %r120, 191;
add.s64 %rd39, %rd31, %rd48;
add.s64 %rd40, %rd32, %rd48;
add.s64 %rd41, %rd33, %rd48;
add.s64 %rd42, %rd34, %rd48;
add.s64 %rd43, %rd35, %rd48;
add.s64 %rd44, %rd36, %rd48;
add.s64 %rd45, %rd37, %rd48;
add.s64 %rd46, %rd38, %rd48;
add.s32 %r138, %r35, -128;
setp.lt.s32 %p14, %r7, %r138;
setp.lt.s32 %p15, %r8, %r138;
setp.lt.s32 %p16, %r9, %r138;
setp.lt.s32 %p17, %r10, %r138;
bar.sync 0;
add.s32 %r86, %r62, 32768;
selp.b32 %r139, 16, 0, %p14;
selp.b32 %r91, %r139, 0, %p13;
cp.async.cg.shared.global [ %r86 + 0 ], [ %rd39 + 0 ], 0x10, %r91;
add.s32 %r87, %r62, 34816;
selp.b32 %r140, 16, 0, %p15;
selp.b32 %r93, %r140, 0, %p13;
cp.async.cg.shared.global [ %r87 + 0 ], [ %rd40 + 0 ], 0x10, %r93;
add.s32 %r88, %r62, 36864;
selp.b32 %r141, 16, 0, %p16;
selp.b32 %r95, %r141, 0, %p13;
cp.async.cg.shared.global [ %r88 + 0 ], [ %rd41 + 0 ], 0x10, %r95;
add.s32 %r89, %r62, 38912;
selp.b32 %r142, 16, 0, %p17;
selp.b32 %r97, %r142, 0, %p13;
cp.async.cg.shared.global [ %r89 + 0 ], [ %rd42 + 0 ], 0x10, %r97;
cp.async.commit_group;
add.s32 %r90, %r62, 98304;
cp.async.cg.shared.global [ %r90 + 0 ], [ %rd43 + 0 ], 0x10, %r91;
add.s32 %r92, %r62, 100352;
cp.async.cg.shared.global [ %r92 + 0 ], [ %rd44 + 0 ], 0x10, %r93;
add.s32 %r94, %r62, 102400;
cp.async.cg.shared.global [ %r94 + 0 ], [ %rd45 + 0 ], 0x10, %r95;
add.s32 %r96, %r62, 104448;
cp.async.cg.shared.global [ %r96 + 0 ], [ %rd46 + 0 ], 0x10, %r97;
cp.async.commit_group;
mov.b32 %r331, 0f00000000;
mov.b32 %r332, %r331;
mov.b32 %r333, %r331;
mov.b32 %r334, %r331;
mov.b32 %r335, %r331;
mov.b32 %r336, %r331;
mov.b32 %r337, %r331;
mov.b32 %r338, %r331;
mov.b32 %r339, %r331;
mov.b32 %r340, %r331;
mov.b32 %r341, %r331;
mov.b32 %r342, %r331;
mov.b32 %r343, %r331;
mov.b32 %r344, %r331;
mov.b32 %r345, %r331;
mov.b32 %r346, %r331;
mov.b32 %r347, %r331;
mov.b32 %r348, %r331;
mov.b32 %r349, %r331;
mov.b32 %r350, %r331;
mov.b32 %r351, %r331;
mov.b32 %r352, %r331;
mov.b32 %r353, %r331;
mov.b32 %r354, %r331;
mov.b32 %r355, %r331;
mov.b32 %r356, %r331;
mov.b32 %r357, %r331;
mov.b32 %r358, %r331;
mov.b32 %r359, %r331;
mov.b32 %r360, %r331;
mov.b32 %r361, %r331;
mov.b32 %r362, %r331;
mov.b32 %r363, %r331;
mov.b32 %r364, %r331;
mov.b32 %r365, %r331;
mov.b32 %r366, %r331;
mov.b32 %r367, %r331;
mov.b32 %r368, %r331;
mov.b32 %r369, %r331;
mov.b32 %r370, %r331;
mov.b32 %r371, %r331;
mov.b32 %r372, %r331;
mov.b32 %r373, %r331;
mov.b32 %r374, %r331;
mov.b32 %r375, %r331;
mov.b32 %r376, %r331;
mov.b32 %r377, %r331;
mov.b32 %r378, %r331;
mov.b32 %r379, %r331;
mov.b32 %r380, %r331;
mov.b32 %r381, %r331;
mov.b32 %r382, %r331;
mov.b32 %r383, %r331;
mov.b32 %r384, %r331;
mov.b32 %r385, %r331;
mov.b32 %r386, %r331;
mov.b32 %r387, %r331;
mov.b32 %r388, %r331;
mov.b32 %r389, %r331;
mov.b32 %r390, %r331;
mov.b32 %r391, %r331;
mov.b32 %r392, %r331;
mov.b32 %r393, %r331;
mov.b32 %r394, %r331;
@%p2 bra $L__BB0_5;
cvt.s64.s32 %rd1, %r98;
cvt.s64.s32 %rd3, %r112;
cvt.s64.s32 %rd4, %r113;
cvt.s64.s32 %rd5, %r114;
cvt.s64.s32 %rd6, %r115;
cvt.s64.s32 %rd7, %r116;
cvt.s64.s32 %rd8, %r117;
cvt.s64.s32 %rd9, %r118;
cvt.s64.s32 %rd10, %r119;
shr.s32 %r121, %r120, 31;
shr.u32 %r122, %r121, 26;
add.s32 %r123, %r120, %r122;
shr.s32 %r29, %r123, 6;
cvt.s64.s32 %rd11, %r124;
add.s32 %r33, %r29, -3;
shr.u32 %r34, %r4, 5;
add.s32 %r328, %r35, -192;
add.s64 %rd49, %rd1, %rd10;
shl.b64 %rd12, %rd49, 1;
mad.lo.s64 %rd93, %rd11, 6, %rd21;
shl.b64 %rd13, %rd11, 1;
add.s64 %rd50, %rd1, %rd9;
shl.b64 %rd14, %rd50, 1;
add.s64 %rd51, %rd1, %rd8;
shl.b64 %rd15, %rd51, 1;
add.s64 %rd52, %rd1, %rd7;
shl.b64 %rd16, %rd52, 1;
add.s64 %rd53, %rd1, %rd6;
shl.b64 %rd17, %rd53, 1;
add.s64 %rd54, %rd1, %rd5;
shl.b64 %rd18, %rd54, 1;
add.s64 %rd55, %rd1, %rd4;
shl.b64 %rd19, %rd55, 1;
add.s64 %rd56, %rd1, %rd3;
shl.b64 %rd20, %rd56, 1;
mov.b32 %r147, 0;
mov.b32 %r331, 0f00000000;
mov.b32 %r330, 2;
mov.b32 %r329, -1;
mov.b32 %r332, %r331;
mov.b32 %r333, %r331;
mov.b32 %r334, %r331;
mov.b32 %r335, %r331;
mov.b32 %r336, %r331;
mov.b32 %r337, %r331;
mov.b32 %r338, %r331;
mov.b32 %r339, %r331;
mov.b32 %r340, %r331;
mov.b32 %r341, %r331;
mov.b32 %r342, %r331;
mov.b32 %r343, %r331;
mov.b32 %r344, %r331;
mov.b32 %r345, %r331;
mov.b32 %r346, %r331;
mov.b32 %r347, %r331;
mov.b32 %r348, %r331;
mov.b32 %r349, %r331;
mov.b32 %r350, %r331;
mov.b32 %r351, %r331;
mov.b32 %r352, %r331;
mov.b32 %r353, %r331;
mov.b32 %r354, %r331;
mov.b32 %r355, %r331;
mov.b32 %r356, %r331;
mov.b32 %r357, %r331;
mov.b32 %r358, %r331;
mov.b32 %r359, %r331;
mov.b32 %r360, %r331;
mov.b32 %r361, %r331;
mov.b32 %r362, %r331;
mov.b32 %r363, %r331;
mov.b32 %r364, %r331;
mov.b32 %r365, %r331;
mov.b32 %r366, %r331;
mov.b32 %r367, %r331;
mov.b32 %r368, %r331;
mov.b32 %r369, %r331;
mov.b32 %r370, %r331;
mov.b32 %r371, %r331;
mov.b32 %r372, %r331;
mov.b32 %r373, %r331;
mov.b32 %r374, %r331;
mov.b32 %r375, %r331;
mov.b32 %r376, %r331;
mov.b32 %r377, %r331;
mov.b32 %r378, %r331;
mov.b32 %r379, %r331;
mov.b32 %r380, %r331;
mov.b32 %r381, %r331;
mov.b32 %r382, %r331;
mov.b32 %r383, %r331;
mov.b32 %r384, %r331;
mov.b32 %r385, %r331;
mov.b32 %r386, %r331;
mov.b32 %r387, %r331;
mov.b32 %r388, %r331;
mov.b32 %r389, %r331;
mov.b32 %r390, %r331;
mov.b32 %r391, %r331;
mov.b32 %r392, %r331;
mov.b32 %r393, %r331;
mov.b32 %r394, %r331;
mov.b32 %r395, %r147;
$L__BB0_4:
setp.lt.s32 %p19, %r395, %r33;
add.s32 %r160, %r329, 1;
setp.gt.s32 %p20, %r160, 3;
selp.b32 %r329, 0, %r160, %p20;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r161, %r329, 14;
add.s32 %r162, %r131, %r161;
add.s32 %r163, %r162, 65536;
shr.u32 %r164, %r162, 4;
shfl.sync.idx.b32 %r165, %r34, 0, 31, -1;
shl.b32 %r166, %r165, 7;
and.b32 %r167, %r166, 512;
add.s32 %r168, %r167, %r164;
and.b32 %r169, %r168, 16383;
cvt.u64.u32 %rd75, %r169;
bfe.u32 %r170, %r163, 4, 14;
cvt.u64.u32 %rd76, %r170;
wgmma.fence.sync.aligned;
or.b64 %rd57, %rd75, 4611686293305294848;
or.b64 %rd58, %rd76, 4611686293338849280;
mov.pred %p18, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384,%r385,%r386,%r387,%r388,%r389,%r390,%r391,%r392,%r393,%r394}, %rd57, %rd58, %p18, 1, 1, 1, 1;
add.s64 %rd59, %rd75, 4611686293305294976;
add.s64 %rd60, %rd76, 4611686293338849408;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384,%r385,%r386,%r387,%r388,%r389,%r390,%r391,%r392,%r393,%r394}, %rd59, %rd60, %p18, 1, 1, 1, 1;
add.s64 %rd61, %rd75, 4611686293305295104;
add.s64 %rd62, %rd76, 4611686293338849536;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384,%r385,%r386,%r387,%r388,%r389,%r390,%r391,%r392,%r393,%r394}, %rd61, %rd62, %p18, 1, 1, 1, 1;
add.s64 %rd63, %rd75, 4611686293305295232;
add.s64 %rd64, %rd76, 4611686293338849664;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382,%r383,%r384,%r385,%r386,%r387,%r388,%r389,%r390,%r391,%r392,%r393,%r394}, %rd63, %rd64, %p18, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd65, %r162;
cvt.u64.u32 %rd66, %r163;
mov.b32 %r146, %r147;
mov.b32 %r143, %r147;
mov.b32 %r144, %r147;
mov.b32 %r145, %r147;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd67, %rd93, %rd20;
add.s64 %rd68, %rd93, %rd19;
add.s64 %rd69, %rd93, %rd18;
add.s64 %rd70, %rd93, %rd17;
add.s64 %rd71, %rd93, %rd16;
add.s64 %rd72, %rd93, %rd15;
add.s64 %rd73, %rd93, %rd14;
add.s64 %rd74, %rd93, %rd12;
add.s32 %r171, %r330, 1;
setp.gt.s32 %p21, %r171, 3;
selp.b32 %r330, 0, %r171, %p21;
setp.lt.s32 %p22, %r7, %r328;
setp.lt.s32 %p23, %r8, %r328;
setp.lt.s32 %p24, %r9, %r328;
setp.lt.s32 %p25, %r10, %r328;
shl.b32 %r172, %r330, 14;
add.s32 %r173, %r131, %r172;
bar.sync 0;
add.s32 %r148, %r173, %r32;
selp.b32 %r174, 16, 0, %p22;
selp.b32 %r153, %r174, 0, %p19;
cp.async.cg.shared.global [ %r148 + 0 ], [ %rd67 + 0 ], 0x10, %r153;
add.s32 %r149, %r148, 2048;
selp.b32 %r175, 16, 0, %p23;
selp.b32 %r155, %r175, 0, %p19;
cp.async.cg.shared.global [ %r149 + 0 ], [ %rd68 + 0 ], 0x10, %r155;
add.s32 %r150, %r148, 4096;
selp.b32 %r176, 16, 0, %p24;
selp.b32 %r157, %r176, 0, %p19;
cp.async.cg.shared.global [ %r150 + 0 ], [ %rd69 + 0 ], 0x10, %r157;
add.s32 %r151, %r148, 6144;
selp.b32 %r177, 16, 0, %p25;
selp.b32 %r159, %r177, 0, %p19;
cp.async.cg.shared.global [ %r151 + 0 ], [ %rd70 + 0 ], 0x10, %r159;
cp.async.commit_group;
add.s32 %r152, %r148, 65536;
cp.async.cg.shared.global [ %r152 + 0 ], [ %rd71 + 0 ], 0x10, %r153;
add.s32 %r154, %r148, 67584;
cp.async.cg.shared.global [ %r154 + 0 ], [ %rd72 + 0 ], 0x10, %r155;
add.s32 %r156, %r148, 69632;
cp.async.cg.shared.global [ %r156 + 0 ], [ %rd73 + 0 ], 0x10, %r157;
add.s32 %r158, %r148, 71680;
cp.async.cg.shared.global [ %r158 + 0 ], [ %rd74 + 0 ], 0x10, %r159;
cp.async.commit_group;
add.s32 %r395, %r395, 1;
add.s64 %rd93, %rd93, %rd13;
add.s32 %r328, %r328, -64;
setp.ne.b32 %p26, %r29, %r395;
@%p26 bra $L__BB0_4;
$L__BB0_5:
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r242, %r332, %r331;
cvt.rn.bf16x2.f32 %r243, %r334, %r333;
cvt.rn.bf16x2.f32 %r244, %r336, %r335;
cvt.rn.bf16x2.f32 %r245, %r338, %r337;
cvt.rn.bf16x2.f32 %r246, %r340, %r339;
cvt.rn.bf16x2.f32 %r247, %r342, %r341;
cvt.rn.bf16x2.f32 %r248, %r344, %r343;
cvt.rn.bf16x2.f32 %r249, %r346, %r345;
cvt.rn.bf16x2.f32 %r250, %r348, %r347;
cvt.rn.bf16x2.f32 %r251, %r350, %r349;
cvt.rn.bf16x2.f32 %r252, %r352, %r351;
cvt.rn.bf16x2.f32 %r253, %r354, %r353;
cvt.rn.bf16x2.f32 %r254, %r356, %r355;
cvt.rn.bf16x2.f32 %r255, %r358, %r357;
cvt.rn.bf16x2.f32 %r256, %r360, %r359;
cvt.rn.bf16x2.f32 %r257, %r362, %r361;
cvt.rn.bf16x2.f32 %r258, %r364, %r363;
cvt.rn.bf16x2.f32 %r259, %r366, %r365;
cvt.rn.bf16x2.f32 %r260, %r368, %r367;
cvt.rn.bf16x2.f32 %r261, %r370, %r369;
cvt.rn.bf16x2.f32 %r262, %r372, %r371;
cvt.rn.bf16x2.f32 %r263, %r374, %r373;
cvt.rn.bf16x2.f32 %r264, %r376, %r375;
cvt.rn.bf16x2.f32 %r265, %r378, %r377;
cvt.rn.bf16x2.f32 %r266, %r380, %r379;
cvt.rn.bf16x2.f32 %r267, %r382, %r381;
cvt.rn.bf16x2.f32 %r268, %r384, %r383;
cvt.rn.bf16x2.f32 %r269, %r386, %r385;
cvt.rn.bf16x2.f32 %r270, %r388, %r387;
cvt.rn.bf16x2.f32 %r271, %r390, %r389;
cvt.rn.bf16x2.f32 %r272, %r392, %r391;
cvt.rn.bf16x2.f32 %r273, %r394, %r393;
mad.lo.s32 %r274, %r12, %r39, %r20;
mad.lo.s32 %r275, %r13, %r39, %r20;
mad.lo.s32 %r276, %r14, %r39, %r20;
mad.lo.s32 %r277, %r15, %r39, %r20;
mad.lo.s32 %r278, %r16, %r39, %r20;
mad.lo.s32 %r279, %r17, %r39, %r20;
mad.lo.s32 %r280, %r18, %r39, %r20;
mad.lo.s32 %r281, %r19, %r39, %r20;
mad.wide.s32 %rd77, %r274, 2, %rd2;
mad.wide.s32 %rd78, %r275, 2, %rd2;
mad.wide.s32 %rd79, %r276, 2, %rd2;
mad.wide.s32 %rd80, %r277, 2, %rd2;
mad.wide.s32 %rd81, %r278, 2, %rd2;
mad.wide.s32 %rd82, %r279, 2, %rd2;
mad.wide.s32 %rd83, %r280, 2, %rd2;
mad.wide.s32 %rd84, %r281, 2, %rd2;
setp.lt.s32 %p43, %r12, %r40;
setp.lt.s32 %p44, %r13, %r40;
setp.lt.s32 %p45, %r14, %r40;
setp.lt.s32 %p46, %r15, %r40;
setp.lt.s32 %p47, %r16, %r40;
setp.lt.s32 %p48, %r17, %r40;
setp.lt.s32 %p49, %r18, %r40;
setp.lt.s32 %p50, %r19, %r40;
setp.lt.s32 %p51, %r20, %r40;
and.pred %p27, %p43, %p51;
and.pred %p28, %p44, %p51;
and.pred %p29, %p45, %p51;
and.pred %p30, %p46, %p51;
and.pred %p31, %p47, %p51;
and.pred %p32, %p48, %p51;
and.pred %p33, %p49, %p51;
and.pred %p34, %p50, %p51;
and.b32 %r282, %r4, 3;
shl.b32 %r283, %r282, 13;
shl.b32 %r284, %r4, 5;
and.b32 %r285, %r284, 7264;
and.b32 %r286, %r4, 24;
shl.b32 %r287, %r286, 4;
shl.b32 %r288, %r4, 2;
and.b32 %r289, %r288, 16;
or.b32 %r290, %r283, %r289;
or.b32 %r291, %r285, %r287;
or.b32 %r292, %r290, %r291;
add.s32 %r293, %r131, %r292;
st.shared.v4.b32 [%r293], {%r242, %r244, %r246, %r248};
st.shared.v4.b32 [%r293+512], {%r243, %r245, %r247, %r249};
xor.b32 %r294, %r292, 32;
add.s32 %r295, %r131, %r294;
st.shared.v4.b32 [%r295], {%r250, %r252, %r254, %r256};
st.shared.v4.b32 [%r295+512], {%r251, %r253, %r255, %r257};
xor.b32 %r296, %r292, 64;
add.s32 %r297, %r131, %r296;
st.shared.v4.b32 [%r297], {%r258, %r260, %r262, %r264};
st.shared.v4.b32 [%r297+512], {%r259, %r261, %r263, %r265};
xor.b32 %r298, %r292, 96;
add.s32 %r299, %r131, %r298;
st.shared.v4.b32 [%r299], {%r266, %r268, %r270, %r272};
st.shared.v4.b32 [%r299+512], {%r267, %r269, %r271, %r273};
bar.sync 0;
shl.b32 %r300, %r286, 10;
shl.b32 %r301, %r282, 5;
and.b32 %r302, %r288, 1008;
or.b32 %r303, %r300, %r301;
xor.b32 %r304, %r303, %r302;
add.s32 %r305, %r131, %r304;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r178, %r179, %r180, %r181}, [%r305];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r182, %r183, %r184, %r185}, [%r305+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r186, %r187, %r188, %r189}, [%r305+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r190, %r191, %r192, %r193}, [%r305+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r194, %r195, %r196, %r197}, [%r305+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r198, %r199, %r200, %r201}, [%r305+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r204, %r205}, [%r305+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r305+7168];
@%p27 st.global.v4.b32 [ %rd77 + 0 ], { %r178, %r179, %r180, %r181 };
@%p28 st.global.v4.b32 [ %rd78 + 0 ], { %r182, %r183, %r184, %r185 };
@%p29 st.global.v4.b32 [ %rd79 + 0 ], { %r186, %r187, %r188, %r189 };
@%p30 st.global.v4.b32 [ %rd80 + 0 ], { %r190, %r191, %r192, %r193 };
@%p31 st.global.v4.b32 [ %rd81 + 0 ], { %r194, %r195, %r196, %r197 };
@%p32 st.global.v4.b32 [ %rd82 + 0 ], { %r198, %r199, %r200, %r201 };
@%p33 st.global.v4.b32 [ %rd83 + 0 ], { %r202, %r203, %r204, %r205 };
@%p34 st.global.v4.b32 [ %rd84 + 0 ], { %r206, %r207, %r208, %r209 };
mad.lo.s32 %r306, %r21, %r39, %r11;
mad.lo.s32 %r307, %r22, %r39, %r11;
mad.lo.s32 %r308, %r23, %r39, %r11;
mad.lo.s32 %r309, %r24, %r39, %r11;
mad.lo.s32 %r310, %r25, %r39, %r11;
mad.lo.s32 %r311, %r26, %r39, %r11;
mad.lo.s32 %r312, %r27, %r39, %r11;
mad.lo.s32 %r313, %r28, %r39, %r11;
mad.wide.s32 %rd85, %r306, 2, %rd2;
mad.wide.s32 %rd86, %r307, 2, %rd2;
mad.wide.s32 %rd87, %r308, 2, %rd2;
mad.wide.s32 %rd88, %r309, 2, %rd2;
mad.wide.s32 %rd89, %r310, 2, %rd2;
mad.wide.s32 %rd90, %r311, 2, %rd2;
mad.wide.s32 %rd91, %r312, 2, %rd2;
mad.wide.s32 %rd92, %r313, 2, %rd2;
setp.lt.s32 %p52, %r21, %r40;
setp.lt.s32 %p53, %r22, %r40;
setp.lt.s32 %p54, %r23, %r40;
setp.lt.s32 %p55, %r24, %r40;
setp.lt.s32 %p56, %r25, %r40;
setp.lt.s32 %p57, %r26, %r40;
setp.lt.s32 %p58, %r27, %r40;
setp.lt.s32 %p59, %r28, %r40;
setp.lt.s32 %p60, %r11, %r40;
and.pred %p35, %p52, %p60;
and.pred %p36, %p53, %p60;
and.pred %p37, %p54, %p60;
and.pred %p38, %p55, %p60;
and.pred %p39, %p56, %p60;
and.pred %p40, %p57, %p60;
and.pred %p41, %p58, %p60;
and.pred %p42, %p59, %p60;
bar.sync 0;
shl.b32 %r314, %r30, 9;
shl.b32 %r315, %r4, 1;
and.b32 %r316, %r315, 272;
and.b32 %r317, %r4, 96;
and.b32 %r318, %r5, 128;
or.b32 %r319, %r316, %r317;
xor.b32 %r320, %r319, %r31;
add.s32 %r321, %r131, %r314;
add.s32 %r322, %r321, %r318;
add.s32 %r323, %r322, %r320;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r242, %r243, %r244, %r245};
bar.sync 0;
and.b32 %r324, %r284, 3840;
or.b32 %r325, %r324, %r31;
xor.b32 %r326, %r325, %r6;
add.s32 %r327, %r131, %r326;
ld.shared.v4.b32 {%r210, %r211, %r212, %r213}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r246, %r247, %r248, %r249};
bar.sync 0;
ld.shared.v4.b32 {%r214, %r215, %r216, %r217}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r250, %r251, %r252, %r253};
bar.sync 0;
ld.shared.v4.b32 {%r218, %r219, %r220, %r221}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r254, %r255, %r256, %r257};
bar.sync 0;
ld.shared.v4.b32 {%r222, %r223, %r224, %r225}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r258, %r259, %r260, %r261};
bar.sync 0;
ld.shared.v4.b32 {%r226, %r227, %r228, %r229}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r262, %r263, %r264, %r265};
bar.sync 0;
ld.shared.v4.b32 {%r230, %r231, %r232, %r233}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r266, %r267, %r268, %r269};
bar.sync 0;
ld.shared.v4.b32 {%r234, %r235, %r236, %r237}, [%r327];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r323], {%r270, %r271, %r272, %r273};
bar.sync 0;
ld.shared.v4.b32 {%r238, %r239, %r240, %r241}, [%r327];
@%p35 st.global.v4.b32 [ %rd85 + 0 ], { %r210, %r211, %r212, %r213 };
@%p36 st.global.v4.b32 [ %rd86 + 0 ], { %r214, %r215, %r216, %r217 };
@%p37 st.global.v4.b32 [ %rd87 + 0 ], { %r218, %r219, %r220, %r221 };
@%p38 st.global.v4.b32 [ %rd88 + 0 ], { %r222, %r223, %r224, %r225 };
@%p39 st.global.v4.b32 [ %rd89 + 0 ], { %r226, %r227, %r228, %r229 };
@%p40 st.global.v4.b32 [ %rd90 + 0 ], { %r230, %r231, %r232, %r233 };
@%p41 st.global.v4.b32 [ %rd91 + 0 ], { %r234, %r235, %r236, %r237 };
@%p42 st.global.v4.b32 [ %rd92 + 0 ], { %r238, %r239, %r240, %r241 };
$L__BB0_1:
ret;
$L__tmp8:
$L__func_end0:
}
  "#
  | .mlpGram, .compact => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry XTX_kernel(
.param .u64 .ptr .global .align 1 XTX_kernel_param_0,
.param .u64 .ptr .global .align 1 XTX_kernel_param_1,
.param .u32 XTX_kernel_param_2,
.param .u32 XTX_kernel_param_3,
.param .u32 XTX_kernel_param_4,
.param .u32 XTX_kernel_param_5,
.param .u32 XTX_kernel_param_6,
.param .u32 XTX_kernel_param_7,
.param .u64 .ptr .global .align 1 XTX_kernel_param_8,
.param .u64 .ptr .global .align 1 XTX_kernel_param_9
)
.reqntid 256
{
.reg .pred %p<60>;
.reg .b32 %r<384>;
.reg .b64 %rd<98>;
$L__func_begin0:
ld.param.b32 %r22, [XTX_kernel_param_7];
ld.param.b32 %r21, [XTX_kernel_param_6];
ld.param.b32 %r20, [XTX_kernel_param_3];
ld.param.b32 %r19, [XTX_kernel_param_2];
ld.param.b64 %rd21, [XTX_kernel_param_1];
ld.param.b64 %rd20, [XTX_kernel_param_0];
$L__tmp0:
mov.u32 %r59, %ctaid.x;
$L__tmp1:
mul.hi.u32 %r60, %r59, 818089009;
shr.u32 %r1, %r60, 2;
mul.lo.s32 %r61, %r1, 21;
sub.s32 %r62, %r59, %r61;
ld.param.b32 %r63, [XTX_kernel_param_4];
mul.lo.s32 %r64, %r62, 3;
ld.param.b32 %r65, [XTX_kernel_param_5];
mov.b64 %rd46, 6569755928097638024;
shr.u64 %rd47, %rd46, %r64;
cvt.u32.u64 %r66, %rd47;
mov.b64 %rd48, 6420841298313609216;
shr.u64 %rd49, %rd48, %r64;
cvt.u32.u64 %r67, %rd49;
shl.b32 %r68, %r66, 7;
and.b32 %r2, %r68, 896;
shl.b32 %r69, %r67, 7;
and.b32 %r3, %r69, 896;
$L__tmp2:
mul.lo.s32 %r70, %r63, %r1;
mad.wide.s32 %rd50, %r70, 2, %rd20;
mov.u32 %r4, %tid.x;
shl.b32 %r5, %r4, 3;
and.b32 %r71, %r5, 120;
and.b32 %r6, %r4, 240;
bfe.u32 %r7, %r4, 4, 4;
or.b32 %r8, %r7, 16;
or.b32 %r9, %r7, 32;
or.b32 %r10, %r7, 48;
or.b32 %r11, %r2, %r71;
rem.s32 %r72, %r11, %r20;
or.b32 %r12, %r3, %r71;
rem.s32 %r73, %r12, %r20;
mul.lo.s32 %r74, %r65, %r7;
shl.b32 %r75, %r65, 4;
add.s32 %r76, %r74, %r75;
add.s32 %r77, %r76, %r75;
add.s32 %r78, %r77, %r75;
add.s32 %r79, %r72, %r74;
add.s32 %r80, %r72, %r76;
add.s32 %r81, %r72, %r77;
add.s32 %r82, %r72, %r78;
mad.wide.s32 %rd22, %r79, 2, %rd50;
mad.wide.s32 %rd23, %r80, 2, %rd50;
mad.wide.s32 %rd24, %r81, 2, %rd50;
mad.wide.s32 %rd25, %r82, 2, %rd50;
add.s32 %r83, %r73, %r74;
add.s32 %r84, %r73, %r76;
add.s32 %r85, %r73, %r77;
add.s32 %r86, %r73, %r78;
mad.wide.s32 %rd26, %r83, 2, %rd50;
mad.wide.s32 %rd27, %r84, 2, %rd50;
mad.wide.s32 %rd28, %r85, 2, %rd50;
mad.wide.s32 %rd29, %r86, 2, %rd50;
$L__tmp3:
add.s32 %r87, %r19, 63;
$L__tmp4:
shl.b32 %r91, %r65, 6;
setp.lt.s32 %p1, %r87, 64;
setp.gt.s32 %p2, %r87, 63;
setp.lt.s32 %p3, %r7, %r19;
setp.lt.s32 %p4, %r8, %r19;
setp.lt.s32 %p5, %r9, %r19;
setp.lt.s32 %p6, %r10, %r19;
and.b32 %r14, %r4, 7;
shl.b32 %r15, %r14, 4;
shl.b32 %r92, %r6, 3;
and.b32 %r93, %r4, 112;
shl.b32 %r94, %r4, 10;
and.b32 %r95, %r94, 8192;
or.b32 %r96, %r15, %r92;
xor.b32 %r97, %r96, %r93;
or.b32 %r16, %r97, %r95;
mov.b32 %r98, global_smem;
add.s32 %r23, %r98, %r16;
selp.b32 %r99, 16, 0, %p2;
selp.b32 %r28, %r99, 0, %p3;
cp.async.cg.shared.global [ %r23 + 0 ], [ %rd22 + 0 ], 0x10, %r28;
add.s32 %r24, %r23, 2048;
selp.b32 %r30, %r99, 0, %p4;
cp.async.cg.shared.global [ %r24 + 0 ], [ %rd23 + 0 ], 0x10, %r30;
add.s32 %r25, %r23, 4096;
selp.b32 %r32, %r99, 0, %p5;
cp.async.cg.shared.global [ %r25 + 0 ], [ %rd24 + 0 ], 0x10, %r32;
add.s32 %r26, %r23, 6144;
selp.b32 %r34, %r99, 0, %p6;
cp.async.cg.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x10, %r34;
cp.async.commit_group;
add.s32 %r27, %r23, 65536;
cp.async.cg.shared.global [ %r27 + 0 ], [ %rd26 + 0 ], 0x10, %r28;
add.s32 %r29, %r23, 67584;
cp.async.cg.shared.global [ %r29 + 0 ], [ %rd27 + 0 ], 0x10, %r30;
add.s32 %r31, %r23, 69632;
cp.async.cg.shared.global [ %r31 + 0 ], [ %rd28 + 0 ], 0x10, %r32;
add.s32 %r33, %r23, 71680;
cp.async.cg.shared.global [ %r33 + 0 ], [ %rd29 + 0 ], 0x10, %r34;
cp.async.commit_group;
setp.gt.s32 %p7, %r87, 127;
mul.wide.s32 %rd51, %r91, 2;
add.s64 %rd30, %rd22, %rd51;
add.s64 %rd31, %rd23, %rd51;
add.s64 %rd32, %rd24, %rd51;
add.s64 %rd33, %rd25, %rd51;
add.s64 %rd34, %rd26, %rd51;
add.s64 %rd35, %rd27, %rd51;
add.s64 %rd36, %rd28, %rd51;
add.s64 %rd37, %rd29, %rd51;
add.s32 %r100, %r19, -64;
setp.lt.s32 %p8, %r7, %r100;
setp.lt.s32 %p9, %r8, %r100;
setp.lt.s32 %p10, %r9, %r100;
setp.lt.s32 %p11, %r10, %r100;
bar.sync 0;
add.s32 %r35, %r23, 16384;
selp.b32 %r101, 16, 0, %p8;
selp.b32 %r40, %r101, 0, %p7;
cp.async.cg.shared.global [ %r35 + 0 ], [ %rd30 + 0 ], 0x10, %r40;
add.s32 %r36, %r23, 18432;
selp.b32 %r102, 16, 0, %p9;
selp.b32 %r42, %r102, 0, %p7;
cp.async.cg.shared.global [ %r36 + 0 ], [ %rd31 + 0 ], 0x10, %r42;
add.s32 %r37, %r23, 20480;
selp.b32 %r103, 16, 0, %p10;
selp.b32 %r44, %r103, 0, %p7;
cp.async.cg.shared.global [ %r37 + 0 ], [ %rd32 + 0 ], 0x10, %r44;
add.s32 %r38, %r23, 22528;
selp.b32 %r104, 16, 0, %p11;
selp.b32 %r46, %r104, 0, %p7;
cp.async.cg.shared.global [ %r38 + 0 ], [ %rd33 + 0 ], 0x10, %r46;
cp.async.commit_group;
add.s32 %r39, %r23, 81920;
cp.async.cg.shared.global [ %r39 + 0 ], [ %rd34 + 0 ], 0x10, %r40;
add.s32 %r41, %r23, 83968;
cp.async.cg.shared.global [ %r41 + 0 ], [ %rd35 + 0 ], 0x10, %r42;
add.s32 %r43, %r23, 86016;
cp.async.cg.shared.global [ %r43 + 0 ], [ %rd36 + 0 ], 0x10, %r44;
add.s32 %r45, %r23, 88064;
cp.async.cg.shared.global [ %r45 + 0 ], [ %rd37 + 0 ], 0x10, %r46;
cp.async.commit_group;
setp.gt.s32 %p12, %r87, 191;
add.s64 %rd38, %rd30, %rd51;
add.s64 %rd39, %rd31, %rd51;
add.s64 %rd40, %rd32, %rd51;
add.s64 %rd41, %rd33, %rd51;
add.s64 %rd42, %rd34, %rd51;
add.s64 %rd43, %rd35, %rd51;
add.s64 %rd44, %rd36, %rd51;
add.s64 %rd45, %rd37, %rd51;
add.s32 %r105, %r19, -128;
setp.lt.s32 %p13, %r7, %r105;
setp.lt.s32 %p14, %r8, %r105;
setp.lt.s32 %p15, %r9, %r105;
setp.lt.s32 %p16, %r10, %r105;
bar.sync 0;
add.s32 %r47, %r23, 32768;
selp.b32 %r106, 16, 0, %p13;
selp.b32 %r52, %r106, 0, %p12;
cp.async.cg.shared.global [ %r47 + 0 ], [ %rd38 + 0 ], 0x10, %r52;
add.s32 %r48, %r23, 34816;
selp.b32 %r107, 16, 0, %p14;
selp.b32 %r54, %r107, 0, %p12;
cp.async.cg.shared.global [ %r48 + 0 ], [ %rd39 + 0 ], 0x10, %r54;
add.s32 %r49, %r23, 36864;
selp.b32 %r108, 16, 0, %p15;
selp.b32 %r56, %r108, 0, %p12;
cp.async.cg.shared.global [ %r49 + 0 ], [ %rd40 + 0 ], 0x10, %r56;
add.s32 %r50, %r23, 38912;
selp.b32 %r109, 16, 0, %p16;
selp.b32 %r58, %r109, 0, %p12;
cp.async.cg.shared.global [ %r50 + 0 ], [ %rd41 + 0 ], 0x10, %r58;
cp.async.commit_group;
add.s32 %r51, %r23, 98304;
cp.async.cg.shared.global [ %r51 + 0 ], [ %rd42 + 0 ], 0x10, %r52;
add.s32 %r53, %r23, 100352;
cp.async.cg.shared.global [ %r53 + 0 ], [ %rd43 + 0 ], 0x10, %r54;
add.s32 %r55, %r23, 102400;
cp.async.cg.shared.global [ %r55 + 0 ], [ %rd44 + 0 ], 0x10, %r56;
add.s32 %r57, %r23, 104448;
cp.async.cg.shared.global [ %r57 + 0 ], [ %rd45 + 0 ], 0x10, %r58;
cp.async.commit_group;
mov.b32 %r319, 0f00000000;
mov.b32 %r320, %r319;
mov.b32 %r321, %r319;
mov.b32 %r322, %r319;
mov.b32 %r323, %r319;
mov.b32 %r324, %r319;
mov.b32 %r325, %r319;
mov.b32 %r326, %r319;
mov.b32 %r327, %r319;
mov.b32 %r328, %r319;
mov.b32 %r329, %r319;
mov.b32 %r330, %r319;
mov.b32 %r331, %r319;
mov.b32 %r332, %r319;
mov.b32 %r333, %r319;
mov.b32 %r334, %r319;
mov.b32 %r335, %r319;
mov.b32 %r336, %r319;
mov.b32 %r337, %r319;
mov.b32 %r338, %r319;
mov.b32 %r339, %r319;
mov.b32 %r340, %r319;
mov.b32 %r341, %r319;
mov.b32 %r342, %r319;
mov.b32 %r343, %r319;
mov.b32 %r344, %r319;
mov.b32 %r345, %r319;
mov.b32 %r346, %r319;
mov.b32 %r347, %r319;
mov.b32 %r348, %r319;
mov.b32 %r349, %r319;
mov.b32 %r350, %r319;
mov.b32 %r351, %r319;
mov.b32 %r352, %r319;
mov.b32 %r353, %r319;
mov.b32 %r354, %r319;
mov.b32 %r355, %r319;
mov.b32 %r356, %r319;
mov.b32 %r357, %r319;
mov.b32 %r358, %r319;
mov.b32 %r359, %r319;
mov.b32 %r360, %r319;
mov.b32 %r361, %r319;
mov.b32 %r362, %r319;
mov.b32 %r363, %r319;
mov.b32 %r364, %r319;
mov.b32 %r365, %r319;
mov.b32 %r366, %r319;
mov.b32 %r367, %r319;
mov.b32 %r368, %r319;
mov.b32 %r369, %r319;
mov.b32 %r370, %r319;
mov.b32 %r371, %r319;
mov.b32 %r372, %r319;
mov.b32 %r373, %r319;
mov.b32 %r374, %r319;
mov.b32 %r375, %r319;
mov.b32 %r376, %r319;
mov.b32 %r377, %r319;
mov.b32 %r378, %r319;
mov.b32 %r379, %r319;
mov.b32 %r380, %r319;
mov.b32 %r381, %r319;
mov.b32 %r382, %r319;
@%p1 bra $L__BB0_3;
cvt.s64.s32 %rd1, %r70;
cvt.s64.s32 %rd2, %r79;
cvt.s64.s32 %rd3, %r80;
cvt.s64.s32 %rd4, %r81;
cvt.s64.s32 %rd5, %r82;
cvt.s64.s32 %rd6, %r83;
cvt.s64.s32 %rd7, %r84;
cvt.s64.s32 %rd8, %r85;
cvt.s64.s32 %rd9, %r86;
shr.s32 %r88, %r87, 31;
shr.u32 %r89, %r88, 26;
add.s32 %r90, %r87, %r89;
shr.s32 %r13, %r90, 6;
cvt.s64.s32 %rd10, %r91;
add.s32 %r17, %r13, -3;
shr.u32 %r18, %r4, 5;
add.s32 %r316, %r19, -192;
add.s64 %rd52, %rd1, %rd9;
shl.b64 %rd11, %rd52, 1;
mad.lo.s64 %rd97, %rd10, 6, %rd20;
shl.b64 %rd12, %rd10, 1;
add.s64 %rd53, %rd1, %rd8;
shl.b64 %rd13, %rd53, 1;
add.s64 %rd54, %rd1, %rd7;
shl.b64 %rd14, %rd54, 1;
add.s64 %rd55, %rd1, %rd6;
shl.b64 %rd15, %rd55, 1;
add.s64 %rd56, %rd1, %rd5;
shl.b64 %rd16, %rd56, 1;
add.s64 %rd57, %rd1, %rd4;
shl.b64 %rd17, %rd57, 1;
add.s64 %rd58, %rd1, %rd3;
shl.b64 %rd18, %rd58, 1;
add.s64 %rd59, %rd1, %rd2;
shl.b64 %rd19, %rd59, 1;
mov.b32 %r114, 0;
mov.b32 %r319, 0f00000000;
mov.b32 %r318, 2;
mov.b32 %r317, -1;
mov.b32 %r320, %r319;
mov.b32 %r321, %r319;
mov.b32 %r322, %r319;
mov.b32 %r323, %r319;
mov.b32 %r324, %r319;
mov.b32 %r325, %r319;
mov.b32 %r326, %r319;
mov.b32 %r327, %r319;
mov.b32 %r328, %r319;
mov.b32 %r329, %r319;
mov.b32 %r330, %r319;
mov.b32 %r331, %r319;
mov.b32 %r332, %r319;
mov.b32 %r333, %r319;
mov.b32 %r334, %r319;
mov.b32 %r335, %r319;
mov.b32 %r336, %r319;
mov.b32 %r337, %r319;
mov.b32 %r338, %r319;
mov.b32 %r339, %r319;
mov.b32 %r340, %r319;
mov.b32 %r341, %r319;
mov.b32 %r342, %r319;
mov.b32 %r343, %r319;
mov.b32 %r344, %r319;
mov.b32 %r345, %r319;
mov.b32 %r346, %r319;
mov.b32 %r347, %r319;
mov.b32 %r348, %r319;
mov.b32 %r349, %r319;
mov.b32 %r350, %r319;
mov.b32 %r351, %r319;
mov.b32 %r352, %r319;
mov.b32 %r353, %r319;
mov.b32 %r354, %r319;
mov.b32 %r355, %r319;
mov.b32 %r356, %r319;
mov.b32 %r357, %r319;
mov.b32 %r358, %r319;
mov.b32 %r359, %r319;
mov.b32 %r360, %r319;
mov.b32 %r361, %r319;
mov.b32 %r362, %r319;
mov.b32 %r363, %r319;
mov.b32 %r364, %r319;
mov.b32 %r365, %r319;
mov.b32 %r366, %r319;
mov.b32 %r367, %r319;
mov.b32 %r368, %r319;
mov.b32 %r369, %r319;
mov.b32 %r370, %r319;
mov.b32 %r371, %r319;
mov.b32 %r372, %r319;
mov.b32 %r373, %r319;
mov.b32 %r374, %r319;
mov.b32 %r375, %r319;
mov.b32 %r376, %r319;
mov.b32 %r377, %r319;
mov.b32 %r378, %r319;
mov.b32 %r379, %r319;
mov.b32 %r380, %r319;
mov.b32 %r381, %r319;
mov.b32 %r382, %r319;
mov.b32 %r383, %r114;
$L__BB0_2:
setp.lt.s32 %p18, %r383, %r17;
add.s32 %r127, %r317, 1;
setp.gt.s32 %p19, %r127, 3;
selp.b32 %r317, 0, %r127, %p19;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r128, %r317, 14;
add.s32 %r129, %r98, %r128;
add.s32 %r130, %r129, 65536;
shr.u32 %r131, %r129, 4;
shfl.sync.idx.b32 %r132, %r18, 0, 31, -1;
shl.b32 %r133, %r132, 7;
and.b32 %r134, %r133, 512;
add.s32 %r135, %r134, %r131;
and.b32 %r136, %r135, 16383;
cvt.u64.u32 %rd78, %r136;
bfe.u32 %r137, %r130, 4, 14;
cvt.u64.u32 %rd79, %r137;
wgmma.fence.sync.aligned;
or.b64 %rd60, %rd78, 4611686293305294848;
or.b64 %rd61, %rd79, 4611686293338849280;
mov.pred %p17, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382}, %rd60, %rd61, %p17, 1, 1, 1, 1;
add.s64 %rd62, %rd78, 4611686293305294976;
add.s64 %rd63, %rd79, 4611686293338849408;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382}, %rd62, %rd63, %p17, 1, 1, 1, 1;
add.s64 %rd64, %rd78, 4611686293305295104;
add.s64 %rd65, %rd79, 4611686293338849536;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382}, %rd64, %rd65, %p17, 1, 1, 1, 1;
add.s64 %rd66, %rd78, 4611686293305295232;
add.s64 %rd67, %rd79, 4611686293338849664;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r319,%r320,%r321,%r322,%r323,%r324,%r325,%r326,%r327,%r328,%r329,%r330,%r331,%r332,%r333,%r334,%r335,%r336,%r337,%r338,%r339,%r340,%r341,%r342,%r343,%r344,%r345,%r346,%r347,%r348,%r349,%r350,%r351,%r352,%r353,%r354,%r355,%r356,%r357,%r358,%r359,%r360,%r361,%r362,%r363,%r364,%r365,%r366,%r367,%r368,%r369,%r370,%r371,%r372,%r373,%r374,%r375,%r376,%r377,%r378,%r379,%r380,%r381,%r382}, %rd66, %rd67, %p17, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd68, %r129;
cvt.u64.u32 %rd69, %r130;
mov.b32 %r113, %r114;
mov.b32 %r110, %r114;
mov.b32 %r111, %r114;
mov.b32 %r112, %r114;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd70, %rd97, %rd19;
add.s64 %rd71, %rd97, %rd18;
add.s64 %rd72, %rd97, %rd17;
add.s64 %rd73, %rd97, %rd16;
add.s64 %rd74, %rd97, %rd15;
add.s64 %rd75, %rd97, %rd14;
add.s64 %rd76, %rd97, %rd13;
add.s64 %rd77, %rd97, %rd11;
add.s32 %r138, %r318, 1;
setp.gt.s32 %p20, %r138, 3;
selp.b32 %r318, 0, %r138, %p20;
setp.lt.s32 %p21, %r7, %r316;
setp.lt.s32 %p22, %r8, %r316;
setp.lt.s32 %p23, %r9, %r316;
setp.lt.s32 %p24, %r10, %r316;
shl.b32 %r139, %r318, 14;
add.s32 %r140, %r98, %r139;
bar.sync 0;
add.s32 %r115, %r140, %r16;
selp.b32 %r141, 16, 0, %p21;
selp.b32 %r120, %r141, 0, %p18;
cp.async.cg.shared.global [ %r115 + 0 ], [ %rd70 + 0 ], 0x10, %r120;
add.s32 %r116, %r115, 2048;
selp.b32 %r142, 16, 0, %p22;
selp.b32 %r122, %r142, 0, %p18;
cp.async.cg.shared.global [ %r116 + 0 ], [ %rd71 + 0 ], 0x10, %r122;
add.s32 %r117, %r115, 4096;
selp.b32 %r143, 16, 0, %p23;
selp.b32 %r124, %r143, 0, %p18;
cp.async.cg.shared.global [ %r117 + 0 ], [ %rd72 + 0 ], 0x10, %r124;
add.s32 %r118, %r115, 6144;
selp.b32 %r144, 16, 0, %p24;
selp.b32 %r126, %r144, 0, %p18;
cp.async.cg.shared.global [ %r118 + 0 ], [ %rd73 + 0 ], 0x10, %r126;
cp.async.commit_group;
add.s32 %r119, %r115, 65536;
cp.async.cg.shared.global [ %r119 + 0 ], [ %rd74 + 0 ], 0x10, %r120;
add.s32 %r121, %r115, 67584;
cp.async.cg.shared.global [ %r121 + 0 ], [ %rd75 + 0 ], 0x10, %r122;
add.s32 %r123, %r115, 69632;
cp.async.cg.shared.global [ %r123 + 0 ], [ %rd76 + 0 ], 0x10, %r124;
add.s32 %r125, %r115, 71680;
cp.async.cg.shared.global [ %r125 + 0 ], [ %rd77 + 0 ], 0x10, %r126;
cp.async.commit_group;
add.s32 %r383, %r383, 1;
add.s64 %rd97, %rd97, %rd12;
add.s32 %r316, %r316, -64;
setp.ne.b32 %p25, %r13, %r383;
@%p25 bra $L__BB0_2;
$L__BB0_3:
or.b32 %r209, %r7, 112;
or.b32 %r210, %r3, %r209;
or.b32 %r211, %r7, 96;
or.b32 %r212, %r3, %r211;
or.b32 %r213, %r7, 80;
or.b32 %r214, %r3, %r213;
or.b32 %r215, %r7, 64;
or.b32 %r216, %r3, %r215;
or.b32 %r217, %r3, %r10;
or.b32 %r218, %r3, %r9;
or.b32 %r219, %r3, %r8;
or.b32 %r220, %r3, %r7;
or.b32 %r221, %r2, %r209;
or.b32 %r222, %r2, %r211;
or.b32 %r223, %r2, %r213;
or.b32 %r224, %r2, %r215;
or.b32 %r225, %r2, %r10;
or.b32 %r226, %r2, %r9;
or.b32 %r227, %r2, %r8;
or.b32 %r228, %r2, %r7;
mul.lo.s32 %r229, %r21, %r1;
mad.wide.s32 %rd96, %r229, 2, %rd21;
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
cvt.rn.bf16x2.f32 %r230, %r320, %r319;
cvt.rn.bf16x2.f32 %r231, %r322, %r321;
cvt.rn.bf16x2.f32 %r232, %r324, %r323;
cvt.rn.bf16x2.f32 %r233, %r326, %r325;
cvt.rn.bf16x2.f32 %r234, %r328, %r327;
cvt.rn.bf16x2.f32 %r235, %r330, %r329;
cvt.rn.bf16x2.f32 %r236, %r332, %r331;
cvt.rn.bf16x2.f32 %r237, %r334, %r333;
cvt.rn.bf16x2.f32 %r238, %r336, %r335;
cvt.rn.bf16x2.f32 %r239, %r338, %r337;
cvt.rn.bf16x2.f32 %r240, %r340, %r339;
cvt.rn.bf16x2.f32 %r241, %r342, %r341;
cvt.rn.bf16x2.f32 %r242, %r344, %r343;
cvt.rn.bf16x2.f32 %r243, %r346, %r345;
cvt.rn.bf16x2.f32 %r244, %r348, %r347;
cvt.rn.bf16x2.f32 %r245, %r350, %r349;
cvt.rn.bf16x2.f32 %r246, %r352, %r351;
cvt.rn.bf16x2.f32 %r247, %r354, %r353;
cvt.rn.bf16x2.f32 %r248, %r356, %r355;
cvt.rn.bf16x2.f32 %r249, %r358, %r357;
cvt.rn.bf16x2.f32 %r250, %r360, %r359;
cvt.rn.bf16x2.f32 %r251, %r362, %r361;
cvt.rn.bf16x2.f32 %r252, %r364, %r363;
cvt.rn.bf16x2.f32 %r253, %r366, %r365;
cvt.rn.bf16x2.f32 %r254, %r368, %r367;
cvt.rn.bf16x2.f32 %r255, %r370, %r369;
cvt.rn.bf16x2.f32 %r256, %r372, %r371;
cvt.rn.bf16x2.f32 %r257, %r374, %r373;
cvt.rn.bf16x2.f32 %r258, %r376, %r375;
cvt.rn.bf16x2.f32 %r259, %r378, %r377;
cvt.rn.bf16x2.f32 %r260, %r380, %r379;
cvt.rn.bf16x2.f32 %r261, %r382, %r381;
mad.lo.s32 %r262, %r22, %r228, %r12;
mad.lo.s32 %r263, %r22, %r227, %r12;
mad.lo.s32 %r264, %r22, %r226, %r12;
mad.lo.s32 %r265, %r22, %r225, %r12;
mad.lo.s32 %r266, %r22, %r224, %r12;
mad.lo.s32 %r267, %r22, %r223, %r12;
mad.lo.s32 %r268, %r22, %r222, %r12;
mad.lo.s32 %r269, %r22, %r221, %r12;
mad.wide.s32 %rd80, %r262, 2, %rd96;
mad.wide.s32 %rd81, %r263, 2, %rd96;
mad.wide.s32 %rd82, %r264, 2, %rd96;
mad.wide.s32 %rd83, %r265, 2, %rd96;
mad.wide.s32 %rd84, %r266, 2, %rd96;
mad.wide.s32 %rd85, %r267, 2, %rd96;
mad.wide.s32 %rd86, %r268, 2, %rd96;
mad.wide.s32 %rd87, %r269, 2, %rd96;
setp.lt.s32 %p42, %r228, %r20;
setp.lt.s32 %p43, %r227, %r20;
setp.lt.s32 %p44, %r226, %r20;
setp.lt.s32 %p45, %r225, %r20;
setp.lt.s32 %p46, %r224, %r20;
setp.lt.s32 %p47, %r223, %r20;
setp.lt.s32 %p48, %r222, %r20;
setp.lt.s32 %p49, %r221, %r20;
setp.lt.s32 %p50, %r12, %r20;
and.pred %p26, %p42, %p50;
and.pred %p27, %p43, %p50;
and.pred %p28, %p44, %p50;
and.pred %p29, %p45, %p50;
and.pred %p30, %p46, %p50;
and.pred %p31, %p47, %p50;
and.pred %p32, %p48, %p50;
and.pred %p33, %p49, %p50;
and.b32 %r270, %r4, 3;
shl.b32 %r271, %r270, 13;
shl.b32 %r272, %r4, 5;
and.b32 %r273, %r272, 7264;
and.b32 %r274, %r4, 24;
shl.b32 %r275, %r274, 4;
shl.b32 %r276, %r4, 2;
and.b32 %r277, %r276, 16;
or.b32 %r278, %r271, %r277;
or.b32 %r279, %r273, %r275;
or.b32 %r280, %r278, %r279;
add.s32 %r281, %r98, %r280;
st.shared.v4.b32 [%r281], {%r230, %r232, %r234, %r236};
st.shared.v4.b32 [%r281+512], {%r231, %r233, %r235, %r237};
xor.b32 %r282, %r280, 32;
add.s32 %r283, %r98, %r282;
st.shared.v4.b32 [%r283], {%r238, %r240, %r242, %r244};
st.shared.v4.b32 [%r283+512], {%r239, %r241, %r243, %r245};
xor.b32 %r284, %r280, 64;
add.s32 %r285, %r98, %r284;
st.shared.v4.b32 [%r285], {%r246, %r248, %r250, %r252};
st.shared.v4.b32 [%r285+512], {%r247, %r249, %r251, %r253};
xor.b32 %r286, %r280, 96;
add.s32 %r287, %r98, %r286;
st.shared.v4.b32 [%r287], {%r254, %r256, %r258, %r260};
st.shared.v4.b32 [%r287+512], {%r255, %r257, %r259, %r261};
bar.sync 0;
shl.b32 %r288, %r274, 10;
shl.b32 %r289, %r270, 5;
and.b32 %r290, %r276, 1008;
or.b32 %r291, %r288, %r289;
xor.b32 %r292, %r291, %r290;
add.s32 %r293, %r98, %r292;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r147, %r148}, [%r293];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r151, %r152}, [%r293+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r293+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r159, %r160}, [%r293+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r293+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r167, %r168}, [%r293+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r171, %r172}, [%r293+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r175, %r176}, [%r293+7168];
@%p26 st.global.v4.b32 [ %rd80 + 0 ], { %r145, %r146, %r147, %r148 };
@%p27 st.global.v4.b32 [ %rd81 + 0 ], { %r149, %r150, %r151, %r152 };
@%p28 st.global.v4.b32 [ %rd82 + 0 ], { %r153, %r154, %r155, %r156 };
@%p29 st.global.v4.b32 [ %rd83 + 0 ], { %r157, %r158, %r159, %r160 };
@%p30 st.global.v4.b32 [ %rd84 + 0 ], { %r161, %r162, %r163, %r164 };
@%p31 st.global.v4.b32 [ %rd85 + 0 ], { %r165, %r166, %r167, %r168 };
@%p32 st.global.v4.b32 [ %rd86 + 0 ], { %r169, %r170, %r171, %r172 };
@%p33 st.global.v4.b32 [ %rd87 + 0 ], { %r173, %r174, %r175, %r176 };
mad.lo.s32 %r294, %r22, %r220, %r11;
mad.lo.s32 %r295, %r22, %r219, %r11;
mad.lo.s32 %r296, %r22, %r218, %r11;
mad.lo.s32 %r297, %r22, %r217, %r11;
mad.lo.s32 %r298, %r22, %r216, %r11;
mad.lo.s32 %r299, %r22, %r214, %r11;
mad.lo.s32 %r300, %r22, %r212, %r11;
mad.lo.s32 %r301, %r22, %r210, %r11;
mad.wide.s32 %rd88, %r294, 2, %rd96;
mad.wide.s32 %rd89, %r295, 2, %rd96;
mad.wide.s32 %rd90, %r296, 2, %rd96;
mad.wide.s32 %rd91, %r297, 2, %rd96;
mad.wide.s32 %rd92, %r298, 2, %rd96;
mad.wide.s32 %rd93, %r299, 2, %rd96;
mad.wide.s32 %rd94, %r300, 2, %rd96;
mad.wide.s32 %rd95, %r301, 2, %rd96;
setp.lt.s32 %p51, %r220, %r20;
setp.lt.s32 %p52, %r219, %r20;
setp.lt.s32 %p53, %r218, %r20;
setp.lt.s32 %p54, %r217, %r20;
setp.lt.s32 %p55, %r216, %r20;
setp.lt.s32 %p56, %r214, %r20;
setp.lt.s32 %p57, %r212, %r20;
setp.lt.s32 %p58, %r210, %r20;
setp.lt.s32 %p59, %r11, %r20;
and.pred %p34, %p51, %p59;
and.pred %p35, %p52, %p59;
and.pred %p36, %p53, %p59;
and.pred %p37, %p54, %p59;
and.pred %p38, %p55, %p59;
and.pred %p39, %p56, %p59;
and.pred %p40, %p57, %p59;
and.pred %p41, %p58, %p59;
bar.sync 0;
shl.b32 %r302, %r14, 9;
shl.b32 %r303, %r4, 1;
and.b32 %r304, %r303, 272;
and.b32 %r305, %r4, 96;
and.b32 %r306, %r5, 128;
or.b32 %r307, %r304, %r305;
xor.b32 %r308, %r307, %r15;
add.s32 %r309, %r98, %r302;
add.s32 %r310, %r309, %r306;
add.s32 %r311, %r310, %r308;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r230, %r231, %r232, %r233};
bar.sync 0;
and.b32 %r312, %r272, 3840;
or.b32 %r313, %r312, %r15;
xor.b32 %r314, %r313, %r6;
add.s32 %r315, %r98, %r314;
ld.shared.v4.b32 {%r177, %r178, %r179, %r180}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r234, %r235, %r236, %r237};
bar.sync 0;
ld.shared.v4.b32 {%r181, %r182, %r183, %r184}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r238, %r239, %r240, %r241};
bar.sync 0;
ld.shared.v4.b32 {%r185, %r186, %r187, %r188}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r242, %r243, %r244, %r245};
bar.sync 0;
ld.shared.v4.b32 {%r189, %r190, %r191, %r192}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r246, %r247, %r248, %r249};
bar.sync 0;
ld.shared.v4.b32 {%r193, %r194, %r195, %r196}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r250, %r251, %r252, %r253};
bar.sync 0;
ld.shared.v4.b32 {%r197, %r198, %r199, %r200}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r254, %r255, %r256, %r257};
bar.sync 0;
ld.shared.v4.b32 {%r201, %r202, %r203, %r204}, [%r315];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r311], {%r258, %r259, %r260, %r261};
bar.sync 0;
ld.shared.v4.b32 {%r205, %r206, %r207, %r208}, [%r315];
@%p34 st.global.v4.b32 [ %rd88 + 0 ], { %r177, %r178, %r179, %r180 };
@%p35 st.global.v4.b32 [ %rd89 + 0 ], { %r181, %r182, %r183, %r184 };
@%p36 st.global.v4.b32 [ %rd90 + 0 ], { %r185, %r186, %r187, %r188 };
@%p37 st.global.v4.b32 [ %rd91 + 0 ], { %r189, %r190, %r191, %r192 };
@%p38 st.global.v4.b32 [ %rd92 + 0 ], { %r193, %r194, %r195, %r196 };
@%p39 st.global.v4.b32 [ %rd93 + 0 ], { %r197, %r198, %r199, %r200 };
@%p40 st.global.v4.b32 [ %rd94 + 0 ], { %r201, %r202, %r203, %r204 };
@%p41 st.global.v4.b32 [ %rd95 + 0 ], { %r205, %r206, %r207, %r208 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .mlpGram, .full => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry polar(
.param .align 64 .b8 polar_param_0[128],
.param .u32 polar_param_1,
.param .u32 polar_param_2,
.param .u64 polar_param_3,
.param .u64 polar_param_4,
.param .align 64 .b8 polar_param_5[128],
.param .u32 polar_param_6,
.param .u32 polar_param_7,
.param .u64 polar_param_8,
.param .u64 polar_param_9,
.param .u64 .ptr .global .align 1 polar_param_10,
.param .u64 .ptr .global .align 1 polar_param_11
)
.reqntid 256
{
.reg .pred %p<32>;
.reg .b32 %r<226>;
.reg .b64 %rd<45>;
$L__func_begin0:
mov.b64 %rd1, polar_param_0;
mov.b64 %rd2, polar_param_5;
$L__tmp0:
cvta.param.u64 %rd44, %rd2;
cvta.param.u64 %rd25, %rd1;
mov.u32 %r47, %ctaid.x;
mul.hi.u32 %r48, %r47, 818089009;
shr.u32 %r1, %r48, 2;
$L__tmp1:
mul.lo.s32 %r49, %r1, 21;
sub.s32 %r50, %r47, %r49;
mul.lo.s32 %r51, %r50, 3;
mov.b64 %rd3, 6569755928097638024;
shr.u64 %rd4, %rd3, %r51;
cvt.u32.u64 %r52, %rd4;
shl.b32 %r53, %r52, 7;
and.b32 %r2, %r53, 896;
mov.b64 %rd5, 6420841298313609216;
shr.u64 %rd6, %rd5, %r51;
cvt.u32.u64 %r54, %rd6;
shl.b32 %r55, %r54, 7;
and.b32 %r3, %r55, 896;
$L__tmp2:
setp.eq.b32 %p2, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p1, %r5, 0;
mov.b32 %r56, global_smem;
add.s32 %r87, %r56, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r87], 1;
bar.sync 0;
add.s32 %r46, %r56, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r46], 1;
mul.lo.s32 %r6, %r1, 2816;
$L__tmp3:
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p31, %r5, 64;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r87], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r89, %r7, 0, 31, -1;
elect.sync %r90|%p18, -1;
and.pred %p17, %p31, %p18;
and.b32 %r91, %r89, 1;
shl.b32 %r92, %r91, 14;
add.s32 %r88, %r56, %r92;
shl.b32 %r93, %r91, 6;
or.b32 %r99, %r93, %r2;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r88], [%rd25, {%r99, %r6}], [%r87];
shl.b32 %r94, %r89, 8;
and.b32 %r10, %r94, 1024;
$L__tmp6:
add.s32 %r222, %r6, 128;
mov.b32 %r223, 0;
mov.b32 %r157, 0f00000000;
mov.b32 %r158, %r157;
mov.b32 %r159, %r157;
mov.b32 %r160, %r157;
mov.b32 %r161, %r157;
mov.b32 %r162, %r157;
mov.b32 %r163, %r157;
mov.b32 %r164, %r157;
mov.b32 %r165, %r157;
mov.b32 %r166, %r157;
mov.b32 %r167, %r157;
mov.b32 %r168, %r157;
mov.b32 %r169, %r157;
mov.b32 %r170, %r157;
mov.b32 %r171, %r157;
mov.b32 %r172, %r157;
mov.b32 %r173, %r157;
mov.b32 %r174, %r157;
mov.b32 %r175, %r157;
mov.b32 %r176, %r157;
mov.b32 %r177, %r157;
mov.b32 %r178, %r157;
mov.b32 %r179, %r157;
mov.b32 %r180, %r157;
mov.b32 %r181, %r157;
mov.b32 %r182, %r157;
mov.b32 %r183, %r157;
mov.b32 %r184, %r157;
mov.b32 %r185, %r157;
mov.b32 %r186, %r157;
mov.b32 %r187, %r157;
mov.b32 %r188, %r157;
mov.b32 %r189, %r157;
mov.b32 %r190, %r157;
mov.b32 %r191, %r157;
mov.b32 %r192, %r157;
mov.b32 %r193, %r157;
mov.b32 %r194, %r157;
mov.b32 %r195, %r157;
mov.b32 %r196, %r157;
mov.b32 %r197, %r157;
mov.b32 %r198, %r157;
mov.b32 %r199, %r157;
mov.b32 %r200, %r157;
mov.b32 %r201, %r157;
mov.b32 %r202, %r157;
mov.b32 %r203, %r157;
mov.b32 %r204, %r157;
mov.b32 %r205, %r157;
mov.b32 %r206, %r157;
mov.b32 %r207, %r157;
mov.b32 %r208, %r157;
mov.b32 %r209, %r157;
mov.b32 %r210, %r157;
mov.b32 %r211, %r157;
mov.b32 %r212, %r157;
mov.b32 %r213, %r157;
mov.b32 %r214, %r157;
mov.b32 %r215, %r157;
mov.b32 %r216, %r157;
mov.b32 %r217, %r157;
mov.b32 %r218, %r157;
mov.b32 %r219, %r157;
mov.b32 %r220, %r157;
$L__BB0_4:
and.b32 %r100, %r223, 1;
bfe.u32 %r96, %r223, 1, 1;
shl.b32 %r101, %r100, 15;
add.s32 %r102, %r56, %r101;
mad.lo.s32 %r103, %r100, -32760, %r102;
add.s32 %r95, %r103, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r95], %r96;
@!complete bra.uni waitLoop;
}
shr.u32 %r104, %r102, 4;
add.s32 %r105, %r104, %r10;
and.b32 %r106, %r105, 16383;
cvt.u64.u32 %rd42, %r106;
bfe.u32 %r107, %r102, 4, 14;
cvt.u64.u32 %rd43, %r107;
wgmma.fence.sync.aligned;
or.b64 %rd26, %rd42, 4611686293305294848;
or.b64 %rd27, %rd43, 4611686293372403712;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd26, %rd27, %p19, 1, 1, 1, 1;
add.s64 %rd28, %rd42, 4611686293305294976;
add.s64 %rd29, %rd43, 4611686293372403840;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd28, %rd29, %p19, 1, 1, 1, 1;
add.s64 %rd30, %rd42, 4611686293305295104;
add.s64 %rd31, %rd43, 4611686293372403968;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd30, %rd31, %p19, 1, 1, 1, 1;
add.s64 %rd32, %rd42, 4611686293305295232;
add.s64 %rd33, %rd43, 4611686293372404096;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd32, %rd33, %p19, 1, 1, 1, 1;
add.s64 %rd34, %rd42, 4611686293305295360;
add.s64 %rd35, %rd43, 4611686293372404224;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd34, %rd35, %p19, 1, 1, 1, 1;
add.s64 %rd36, %rd42, 4611686293305295488;
add.s64 %rd37, %rd43, 4611686293372404352;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd36, %rd37, %p19, 1, 1, 1, 1;
add.s64 %rd38, %rd42, 4611686293305295616;
add.s64 %rd39, %rd43, 4611686293372404480;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd38, %rd39, %p19, 1, 1, 1, 1;
add.s64 %rd40, %rd42, 4611686293305295744;
add.s64 %rd41, %rd43, 4611686293372404608;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd40, %rd41, %p19, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r11, %r223, 1;
and.b32 %r108, %r11, 1;
setp.ne.b32 %p22, %r223, 21;
shl.b32 %r109, %r108, 3;
add.s32 %r110, %r56, %r109;
add.s32 %r97, %r110, 98304;
shl.b32 %r111, %r108, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r97], 32768;
bar.sync 0;
elect.sync %r112|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p31, %p24;
add.s32 %r98, %r88, %r111;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r98], [%rd25, {%r99, %r222}], [%r97];
$L__tmp8:
add.s32 %r222, %r222, 128;
setp.ne.b32 %p25, %r11, 22;
mov.b32 %r223, %r11;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r87], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r58, %r7, 0, 31, -1;
elect.sync %r59|%p5, -1;
and.pred %p3, %p31, %p5;
and.b32 %r60, %r58, 1;
shl.b32 %r61, %r60, 14;
add.s32 %r57, %r56, %r61;
shl.b32 %r62, %r60, 6;
or.b32 %r71, %r62, %r2;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r57], [%rd25, {%r71, %r6}], [%r87];
$L__tmp10:
bar.sync 0;
elect.sync %r63|%p6, -1;
and.pred %p4, %p31, %p6;
add.s32 %r72, %r57, 65536;
or.b32 %r73, %r62, %r3;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r72], [%rd25, {%r73, %r6}], [%r87];
shl.b32 %r64, %r58, 8;
and.b32 %r8, %r64, 1024;
add.s32 %r65, %r56, 65536;
bfe.u32 %r66, %r65, 4, 14;
cvt.u64.u32 %rd7, %r66;
or.b64 %rd9, %rd7, 4611686293372403712;
add.s64 %rd11, %rd7, 4611686293372403840;
add.s64 %rd13, %rd7, 4611686293372403968;
add.s64 %rd15, %rd7, 4611686293372404096;
add.s64 %rd17, %rd7, 4611686293372404224;
add.s64 %rd19, %rd7, 4611686293372404352;
add.s64 %rd21, %rd7, 4611686293372404480;
add.s64 %rd23, %rd7, 4611686293372404608;
$L__tmp11:
add.s32 %r156, %r6, 128;
mov.b32 %r221, 0;
mov.b32 %r157, 0f00000000;
mov.b32 %r158, %r157;
mov.b32 %r159, %r157;
mov.b32 %r160, %r157;
mov.b32 %r161, %r157;
mov.b32 %r162, %r157;
mov.b32 %r163, %r157;
mov.b32 %r164, %r157;
mov.b32 %r165, %r157;
mov.b32 %r166, %r157;
mov.b32 %r167, %r157;
mov.b32 %r168, %r157;
mov.b32 %r169, %r157;
mov.b32 %r170, %r157;
mov.b32 %r171, %r157;
mov.b32 %r172, %r157;
mov.b32 %r173, %r157;
mov.b32 %r174, %r157;
mov.b32 %r175, %r157;
mov.b32 %r176, %r157;
mov.b32 %r177, %r157;
mov.b32 %r178, %r157;
mov.b32 %r179, %r157;
mov.b32 %r180, %r157;
mov.b32 %r181, %r157;
mov.b32 %r182, %r157;
mov.b32 %r183, %r157;
mov.b32 %r184, %r157;
mov.b32 %r185, %r157;
mov.b32 %r186, %r157;
mov.b32 %r187, %r157;
mov.b32 %r188, %r157;
mov.b32 %r189, %r157;
mov.b32 %r190, %r157;
mov.b32 %r191, %r157;
mov.b32 %r192, %r157;
mov.b32 %r193, %r157;
mov.b32 %r194, %r157;
mov.b32 %r195, %r157;
mov.b32 %r196, %r157;
mov.b32 %r197, %r157;
mov.b32 %r198, %r157;
mov.b32 %r199, %r157;
mov.b32 %r200, %r157;
mov.b32 %r201, %r157;
mov.b32 %r202, %r157;
mov.b32 %r203, %r157;
mov.b32 %r204, %r157;
mov.b32 %r205, %r157;
mov.b32 %r206, %r157;
mov.b32 %r207, %r157;
mov.b32 %r208, %r157;
mov.b32 %r209, %r157;
mov.b32 %r210, %r157;
mov.b32 %r211, %r157;
mov.b32 %r212, %r157;
mov.b32 %r213, %r157;
mov.b32 %r214, %r157;
mov.b32 %r215, %r157;
mov.b32 %r216, %r157;
mov.b32 %r217, %r157;
mov.b32 %r218, %r157;
mov.b32 %r219, %r157;
mov.b32 %r220, %r157;
$L__BB0_2:
and.b32 %r74, %r221, 1;
bfe.u32 %r68, %r221, 1, 1;
shl.b32 %r75, %r74, 15;
add.s32 %r76, %r56, %r75;
mad.lo.s32 %r77, %r74, -32760, %r76;
add.s32 %r67, %r77, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r67], %r68;
@!complete bra.uni waitLoop;
}
shr.u32 %r78, %r76, 4;
add.s32 %r79, %r78, %r8;
and.b32 %r80, %r79, 16383;
cvt.u64.u32 %rd24, %r80;
wgmma.fence.sync.aligned;
or.b64 %rd8, %rd24, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd8, %rd9, %p7, 1, 1, 1, 1;
add.s64 %rd10, %rd24, 4611686293305294976;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd10, %rd11, %p7, 1, 1, 1, 1;
add.s64 %rd12, %rd24, 4611686293305295104;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd12, %rd13, %p7, 1, 1, 1, 1;
add.s64 %rd14, %rd24, 4611686293305295232;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd14, %rd15, %p7, 1, 1, 1, 1;
add.s64 %rd16, %rd24, 4611686293305295360;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd16, %rd17, %p7, 1, 1, 1, 1;
add.s64 %rd18, %rd24, 4611686293305295488;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd18, %rd19, %p7, 1, 1, 1, 1;
add.s64 %rd20, %rd24, 4611686293305295616;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd20, %rd21, %p7, 1, 1, 1, 1;
add.s64 %rd22, %rd24, 4611686293305295744;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r157,%r158,%r159,%r160,%r161,%r162,%r163,%r164,%r165,%r166,%r167,%r168,%r169,%r170,%r171,%r172,%r173,%r174,%r175,%r176,%r177,%r178,%r179,%r180,%r181,%r182,%r183,%r184,%r185,%r186,%r187,%r188,%r189,%r190,%r191,%r192,%r193,%r194,%r195,%r196,%r197,%r198,%r199,%r200,%r201,%r202,%r203,%r204,%r205,%r206,%r207,%r208,%r209,%r210,%r211,%r212,%r213,%r214,%r215,%r216,%r217,%r218,%r219,%r220}, %rd22, %rd23, %p7, 1, 1, 1, 1;
wgmma.commit_group.sync.aligned;
add.s32 %r9, %r221, 1;
and.b32 %r81, %r9, 1;
setp.ne.b32 %p11, %r221, 21;
shl.b32 %r82, %r81, 3;
add.s32 %r83, %r56, %r82;
add.s32 %r69, %r83, 98304;
shl.b32 %r84, %r81, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r69], 65536;
bar.sync 0;
elect.sync %r85|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p31, %p13;
add.s32 %r70, %r57, %r84;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r70], [%rd25, {%r71, %r156}], [%r69];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r86|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p31, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r72], [%rd25, {%r73, %r156}], [%r69];
$L__tmp15:
add.s32 %r156, %r156, 128;
setp.eq.b32 %p16, %r9, 22;
mov.b32 %r221, %r9;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r2, %r3;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r87];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r46];
cvt.rn.bf16x2.f32 %r12, %r158, %r157;
cvt.rn.bf16x2.f32 %r13, %r160, %r159;
cvt.rn.bf16x2.f32 %r14, %r162, %r161;
cvt.rn.bf16x2.f32 %r15, %r164, %r163;
cvt.rn.bf16x2.f32 %r16, %r166, %r165;
cvt.rn.bf16x2.f32 %r17, %r168, %r167;
cvt.rn.bf16x2.f32 %r18, %r170, %r169;
cvt.rn.bf16x2.f32 %r19, %r172, %r171;
cvt.rn.bf16x2.f32 %r20, %r174, %r173;
cvt.rn.bf16x2.f32 %r21, %r176, %r175;
cvt.rn.bf16x2.f32 %r22, %r178, %r177;
cvt.rn.bf16x2.f32 %r23, %r180, %r179;
cvt.rn.bf16x2.f32 %r24, %r182, %r181;
cvt.rn.bf16x2.f32 %r25, %r184, %r183;
cvt.rn.bf16x2.f32 %r26, %r186, %r185;
cvt.rn.bf16x2.f32 %r27, %r188, %r187;
cvt.rn.bf16x2.f32 %r28, %r190, %r189;
cvt.rn.bf16x2.f32 %r29, %r192, %r191;
cvt.rn.bf16x2.f32 %r30, %r194, %r193;
cvt.rn.bf16x2.f32 %r31, %r196, %r195;
cvt.rn.bf16x2.f32 %r32, %r198, %r197;
cvt.rn.bf16x2.f32 %r33, %r200, %r199;
cvt.rn.bf16x2.f32 %r34, %r202, %r201;
cvt.rn.bf16x2.f32 %r35, %r204, %r203;
cvt.rn.bf16x2.f32 %r36, %r206, %r205;
cvt.rn.bf16x2.f32 %r37, %r208, %r207;
cvt.rn.bf16x2.f32 %r38, %r210, %r209;
cvt.rn.bf16x2.f32 %r39, %r212, %r211;
cvt.rn.bf16x2.f32 %r40, %r214, %r213;
cvt.rn.bf16x2.f32 %r41, %r216, %r215;
cvt.rn.bf16x2.f32 %r42, %r218, %r217;
cvt.rn.bf16x2.f32 %r43, %r220, %r219;
shl.b32 %r44, %r4, 7;
@%p26 bra $L__BB0_8;
shl.b32 %r127, %r4, 4;
and.b32 %r225, %r127, 112;
shl.b32 %r224, %r4, 6;
bra.uni $L__BB0_9;
$L__BB0_8:
and.b32 %r113, %r44, 1920;
shl.b32 %r224, %r4, 6;
and.b32 %r114, %r224, 14336;
shl.b32 %r115, %r4, 4;
and.b32 %r225, %r115, 112;
and.b32 %r116, %r4, 16;
or.b32 %r117, %r113, %r225;
xor.b32 %r118, %r117, %r116;
or.b32 %r119, %r118, %r114;
add.s32 %r120, %r56, %r119;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r120+16384], {%r28, %r29, %r30, %r31};
xor.b32 %r121, %r119, 32;
add.s32 %r122, %r56, %r121;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r122+16384], {%r32, %r33, %r34, %r35};
xor.b32 %r123, %r119, 64;
add.s32 %r124, %r56, %r123;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r124], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r124+16384], {%r36, %r37, %r38, %r39};
xor.b32 %r125, %r119, 96;
add.s32 %r126, %r56, %r125;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r126], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r126+16384], {%r40, %r41, %r42, %r43};
$L__BB0_9:
and.b32 %r128, %r44, 17280;
and.b32 %r129, %r4, 96;
shl.b32 %r130, %r4, 1;
and.b32 %r131, %r130, 16;
and.b32 %r132, %r224, 1024;
or.b32 %r133, %r128, %r225;
or.b32 %r134, %r131, %r129;
xor.b32 %r135, %r133, %r134;
add.s32 %r136, %r56, %r132;
add.s32 %r137, %r136, %r135;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+32768], {%r12, %r13, %r14, %r15};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+34816], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+36864], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+38912], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+40960], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+43008], {%r32, %r33, %r34, %r35};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+45056], {%r36, %r37, %r38, %r39};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r137+47104], {%r40, %r41, %r42, %r43};
fence.proxy.async.shared::cta;
mul.lo.s32 %r45, %r1, 768;
@%p2 bra $L__BB0_11;
add.s32 %r139, %r2, %r45;
bar.sync 0;
elect.sync %r141|%p28, -1;
shfl.sync.idx.b32 %r142, %r7, 0, 31, -1;
and.pred %p27, %p31, %p28;
and.b32 %r143, %r142, 1;
shl.b32 %r144, %r143, 14;
add.s32 %r140, %r56, %r144;
shl.b32 %r145, %r143, 6;
or.b32 %r138, %r145, %r3;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd44, {%r138, %r139}], [%r140];
cp.async.bulk.commit_group;
$L__BB0_11:
add.s32 %r147, %r3, %r45;
bar.sync 0;
elect.sync %r150|%p30, -1;
shfl.sync.idx.b32 %r151, %r7, 0, 31, -1;
and.pred %p29, %p31, %p30;
and.b32 %r152, %r151, 1;
shl.b32 %r153, %r152, 14;
add.s32 %r154, %r56, %r153;
add.s32 %r148, %r154, 32768;
shl.b32 %r155, %r152, 6;
or.b32 %r146, %r155, %r2;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd44, {%r146, %r147}], [%r148];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r149, 0;
ret;
$L__tmp17:
$L__func_end0:
}
"#
  | .qkPolynomial, .baseline => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry ba_plus_cAA_kernel(
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_0,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_1,
.param .u32 ba_plus_cAA_kernel_param_2,
.param .u32 ba_plus_cAA_kernel_param_3,
.param .u32 ba_plus_cAA_kernel_param_4,
.param .u32 ba_plus_cAA_kernel_param_5,
.param .u32 ba_plus_cAA_kernel_param_6,
.param .f32 ba_plus_cAA_kernel_param_7,
.param .f32 ba_plus_cAA_kernel_param_8,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_9,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_10
)
.reqntid 256
{
.reg .pred %p<49>;
.reg .b16 %rs<65>;
.reg .b32 %r<674>;
.reg .b64 %rd<98>;
$L__func_begin0:
ld.param.b32 %r39, [ba_plus_cAA_kernel_param_2];
$L__tmp0:
mov.u32 %r40, %ctaid.x;
$L__tmp1:
add.s32 %r41, %r39, 127;
shr.s32 %r42, %r41, 31;
shr.u32 %r43, %r42, 25;
add.s32 %r44, %r41, %r43;
shr.s32 %r45, %r44, 7;
$L__tmp2:
mul.lo.s32 %r46, %r45, %r45;
div.s32 %r1, %r40, %r46;
mul.lo.s32 %r47, %r1, %r46;
sub.s32 %r48, %r40, %r47;
$L__tmp3:
shl.b32 %r49, %r45, 3;
div.s32 %r50, %r48, %r49;
shl.b32 %r51, %r50, 3;
sub.s32 %r52, %r45, %r51;
min.s32 %r53, %r52, 8;
mul.lo.s32 %r54, %r50, %r49;
sub.s32 %r55, %r48, %r54;
div.s32 %r56, %r55, %r53;
mul.lo.s32 %r57, %r56, %r53;
sub.s32 %r58, %r55, %r57;
add.s32 %r59, %r58, %r51;
$L__tmp4:
shl.b32 %r2, %r59, 7;
shl.b32 %r3, %r56, 7;
$L__tmp5:
add.s32 %r60, %r2, 128;
setp.gt.s32 %p1, %r60, %r3;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b32 %r38, [ba_plus_cAA_kernel_param_8];
ld.param.b32 %r37, [ba_plus_cAA_kernel_param_7];
ld.param.b32 %r36, [ba_plus_cAA_kernel_param_6];
ld.param.b32 %r35, [ba_plus_cAA_kernel_param_5];
ld.param.b32 %r34, [ba_plus_cAA_kernel_param_4];
ld.param.b32 %r33, [ba_plus_cAA_kernel_param_3];
ld.param.b64 %rd20, [ba_plus_cAA_kernel_param_1];
ld.param.b64 %rd19, [ba_plus_cAA_kernel_param_0];
mul.lo.s32 %r88, %r1, %r33;
mad.wide.s32 %rd1, %r88, 2, %rd19;
mul.lo.s32 %r89, %r1, %r35;
mad.wide.s32 %rd2, %r89, 2, %rd20;
mov.u32 %r4, %tid.x;
shr.u32 %r90, %r4, 3;
bfe.u32 %r91, %r4, 3, 5;
or.b32 %r92, %r91, 32;
or.b32 %r93, %r91, 64;
or.b32 %r94, %r90, 96;
and.b32 %r5, %r4, 240;
bfe.u32 %r95, %r4, 4, 4;
or.b32 %r96, %r95, 16;
or.b32 %r97, %r95, 32;
or.b32 %r98, %r95, 48;
or.b32 %r99, %r95, 64;
or.b32 %r100, %r95, 80;
or.b32 %r101, %r95, 96;
or.b32 %r102, %r95, 112;
and.b32 %r6, %r4, 15;
shl.b32 %r103, %r6, 3;
or.b32 %r104, %r2, %r91;
or.b32 %r105, %r2, %r92;
or.b32 %r106, %r2, %r93;
or.b32 %r107, %r2, %r94;
or.b32 %r7, %r2, %r95;
or.b32 %r8, %r2, %r96;
or.b32 %r9, %r2, %r97;
or.b32 %r10, %r2, %r98;
or.b32 %r11, %r2, %r99;
or.b32 %r12, %r2, %r100;
or.b32 %r13, %r2, %r101;
or.b32 %r14, %r2, %r102;
or.b32 %r15, %r2, %r103;
rem.s32 %r108, %r104, %r39;
rem.s32 %r109, %r105, %r39;
rem.s32 %r110, %r106, %r39;
rem.s32 %r111, %r107, %r39;
or.b32 %r112, %r3, %r91;
or.b32 %r113, %r3, %r92;
or.b32 %r114, %r3, %r93;
or.b32 %r115, %r3, %r94;
or.b32 %r16, %r3, %r95;
or.b32 %r17, %r3, %r96;
or.b32 %r18, %r3, %r97;
or.b32 %r19, %r3, %r98;
or.b32 %r20, %r3, %r99;
or.b32 %r21, %r3, %r100;
or.b32 %r22, %r3, %r101;
or.b32 %r23, %r3, %r102;
or.b32 %r24, %r3, %r103;
rem.s32 %r116, %r112, %r39;
rem.s32 %r117, %r113, %r39;
rem.s32 %r118, %r114, %r39;
rem.s32 %r119, %r115, %r39;
and.b32 %r25, %r4, 7;
shl.b32 %r26, %r25, 3;
mad.lo.s32 %r120, %r108, %r34, %r26;
mad.lo.s32 %r121, %r109, %r34, %r26;
mad.lo.s32 %r122, %r110, %r34, %r26;
mad.lo.s32 %r123, %r111, %r34, %r26;
mad.wide.s32 %rd21, %r120, 2, %rd1;
mad.wide.s32 %rd22, %r121, 2, %rd1;
mad.wide.s32 %rd23, %r122, 2, %rd1;
mad.wide.s32 %rd24, %r123, 2, %rd1;
mad.lo.s32 %r124, %r116, %r34, %r26;
mad.lo.s32 %r125, %r117, %r34, %r26;
mad.lo.s32 %r126, %r118, %r34, %r26;
mad.lo.s32 %r127, %r119, %r34, %r26;
mad.wide.s32 %rd25, %r124, 2, %rd1;
mad.wide.s32 %rd26, %r125, 2, %rd1;
mad.wide.s32 %rd27, %r126, 2, %rd1;
mad.wide.s32 %rd28, %r127, 2, %rd1;
$L__tmp6:
add.s32 %r128, %r39, 63;
$L__tmp7:
setp.lt.s32 %p2, %r128, 64;
setp.gt.s32 %p3, %r128, 63;
setp.lt.s32 %p4, %r26, %r39;
shl.b32 %r28, %r4, 4;
and.b32 %r132, %r28, 4080;
shl.b32 %r29, %r4, 1;
and.b32 %r133, %r29, 112;
xor.b32 %r30, %r132, %r133;
mov.b32 %r134, global_smem;
add.s32 %r61, %r134, %r30;
selp.b32 %r135, 16, 0, %p3;
selp.b32 %r63, %r135, 0, %p4;
cp.async.cg.shared.global [ %r61 + 0 ], [ %rd21 + 0 ], 0x10, %r63;
add.s32 %r62, %r61, 4096;
cp.async.cg.shared.global [ %r62 + 0 ], [ %rd22 + 0 ], 0x10, %r63;
add.s32 %r64, %r61, 8192;
cp.async.cg.shared.global [ %r64 + 0 ], [ %rd23 + 0 ], 0x10, %r63;
add.s32 %r65, %r61, 12288;
cp.async.cg.shared.global [ %r65 + 0 ], [ %rd24 + 0 ], 0x10, %r63;
cp.async.commit_group;
add.s32 %r66, %r61, 65536;
cp.async.cg.shared.global [ %r66 + 0 ], [ %rd25 + 0 ], 0x10, %r63;
add.s32 %r67, %r61, 69632;
cp.async.cg.shared.global [ %r67 + 0 ], [ %rd26 + 0 ], 0x10, %r63;
add.s32 %r68, %r61, 73728;
cp.async.cg.shared.global [ %r68 + 0 ], [ %rd27 + 0 ], 0x10, %r63;
add.s32 %r69, %r61, 77824;
cp.async.cg.shared.global [ %r69 + 0 ], [ %rd28 + 0 ], 0x10, %r63;
cp.async.commit_group;
setp.gt.s32 %p5, %r128, 127;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s64 %rd35, %rd27, 128;
add.s64 %rd36, %rd28, 128;
add.s32 %r136, %r39, -64;
setp.lt.s32 %p6, %r26, %r136;
bar.sync 0;
add.s32 %r70, %r61, 16384;
selp.b32 %r137, 16, 0, %p6;
selp.b32 %r72, %r137, 0, %p5;
cp.async.cg.shared.global [ %r70 + 0 ], [ %rd29 + 0 ], 0x10, %r72;
add.s32 %r71, %r61, 20480;
cp.async.cg.shared.global [ %r71 + 0 ], [ %rd30 + 0 ], 0x10, %r72;
add.s32 %r73, %r61, 24576;
cp.async.cg.shared.global [ %r73 + 0 ], [ %rd31 + 0 ], 0x10, %r72;
add.s32 %r74, %r61, 28672;
cp.async.cg.shared.global [ %r74 + 0 ], [ %rd32 + 0 ], 0x10, %r72;
cp.async.commit_group;
add.s32 %r75, %r61, 81920;
cp.async.cg.shared.global [ %r75 + 0 ], [ %rd33 + 0 ], 0x10, %r72;
add.s32 %r76, %r61, 86016;
cp.async.cg.shared.global [ %r76 + 0 ], [ %rd34 + 0 ], 0x10, %r72;
add.s32 %r77, %r61, 90112;
cp.async.cg.shared.global [ %r77 + 0 ], [ %rd35 + 0 ], 0x10, %r72;
add.s32 %r78, %r61, 94208;
cp.async.cg.shared.global [ %r78 + 0 ], [ %rd36 + 0 ], 0x10, %r72;
cp.async.commit_group;
setp.gt.s32 %p7, %r128, 191;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s64 %rd43, %rd27, 256;
add.s64 %rd44, %rd28, 256;
add.s32 %r138, %r39, -128;
setp.lt.s32 %p8, %r26, %r138;
bar.sync 0;
add.s32 %r79, %r61, 32768;
selp.b32 %r139, 16, 0, %p8;
selp.b32 %r81, %r139, 0, %p7;
cp.async.cg.shared.global [ %r79 + 0 ], [ %rd37 + 0 ], 0x10, %r81;
add.s32 %r80, %r61, 36864;
cp.async.cg.shared.global [ %r80 + 0 ], [ %rd38 + 0 ], 0x10, %r81;
add.s32 %r82, %r61, 40960;
cp.async.cg.shared.global [ %r82 + 0 ], [ %rd39 + 0 ], 0x10, %r81;
add.s32 %r83, %r61, 45056;
cp.async.cg.shared.global [ %r83 + 0 ], [ %rd40 + 0 ], 0x10, %r81;
cp.async.commit_group;
add.s32 %r84, %r61, 98304;
cp.async.cg.shared.global [ %r84 + 0 ], [ %rd41 + 0 ], 0x10, %r81;
add.s32 %r85, %r61, 102400;
cp.async.cg.shared.global [ %r85 + 0 ], [ %rd42 + 0 ], 0x10, %r81;
add.s32 %r86, %r61, 106496;
cp.async.cg.shared.global [ %r86 + 0 ], [ %rd43 + 0 ], 0x10, %r81;
add.s32 %r87, %r61, 110592;
cp.async.cg.shared.global [ %r87 + 0 ], [ %rd44 + 0 ], 0x10, %r81;
cp.async.commit_group;
mov.b32 %r609, 0f00000000;
mov.b32 %r610, %r609;
mov.b32 %r611, %r609;
mov.b32 %r612, %r609;
mov.b32 %r613, %r609;
mov.b32 %r614, %r609;
mov.b32 %r615, %r609;
mov.b32 %r616, %r609;
mov.b32 %r617, %r609;
mov.b32 %r618, %r609;
mov.b32 %r619, %r609;
mov.b32 %r620, %r609;
mov.b32 %r621, %r609;
mov.b32 %r622, %r609;
mov.b32 %r623, %r609;
mov.b32 %r624, %r609;
mov.b32 %r625, %r609;
mov.b32 %r626, %r609;
mov.b32 %r627, %r609;
mov.b32 %r628, %r609;
mov.b32 %r629, %r609;
mov.b32 %r630, %r609;
mov.b32 %r631, %r609;
mov.b32 %r632, %r609;
mov.b32 %r633, %r609;
mov.b32 %r634, %r609;
mov.b32 %r635, %r609;
mov.b32 %r636, %r609;
mov.b32 %r637, %r609;
mov.b32 %r638, %r609;
mov.b32 %r639, %r609;
mov.b32 %r640, %r609;
mov.b32 %r641, %r609;
mov.b32 %r642, %r609;
mov.b32 %r643, %r609;
mov.b32 %r644, %r609;
mov.b32 %r645, %r609;
mov.b32 %r646, %r609;
mov.b32 %r647, %r609;
mov.b32 %r648, %r609;
mov.b32 %r649, %r609;
mov.b32 %r650, %r609;
mov.b32 %r651, %r609;
mov.b32 %r652, %r609;
mov.b32 %r653, %r609;
mov.b32 %r654, %r609;
mov.b32 %r655, %r609;
mov.b32 %r656, %r609;
mov.b32 %r657, %r609;
mov.b32 %r658, %r609;
mov.b32 %r659, %r609;
mov.b32 %r660, %r609;
mov.b32 %r661, %r609;
mov.b32 %r662, %r609;
mov.b32 %r663, %r609;
mov.b32 %r664, %r609;
mov.b32 %r665, %r609;
mov.b32 %r666, %r609;
mov.b32 %r667, %r609;
mov.b32 %r668, %r609;
mov.b32 %r669, %r609;
mov.b32 %r670, %r609;
mov.b32 %r671, %r609;
mov.b32 %r672, %r609;
@%p2 bra $L__BB0_5;
cvt.s64.s32 %rd3, %r120;
cvt.s64.s32 %rd4, %r121;
cvt.s64.s32 %rd5, %r122;
cvt.s64.s32 %rd6, %r123;
cvt.s64.s32 %rd7, %r124;
cvt.s64.s32 %rd8, %r125;
cvt.s64.s32 %rd9, %r126;
cvt.s64.s32 %rd10, %r127;
shr.s32 %r129, %r128, 31;
shr.u32 %r130, %r129, 26;
add.s32 %r131, %r128, %r130;
shr.s32 %r27, %r131, 6;
add.s32 %r31, %r27, -3;
shr.u32 %r32, %r4, 5;
add.s32 %r606, %r39, -192;
shl.b64 %rd45, %rd10, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd9, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd8, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd7, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd6, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd5, 1;
add.s64 %rd16, %rd50, 384;
shl.b64 %rd51, %rd4, 1;
add.s64 %rd17, %rd51, 384;
shl.b64 %rd52, %rd3, 1;
add.s64 %rd18, %rd52, 384;
mov.b32 %r144, 0;
mov.b32 %r609, 0f00000000;
mov.b32 %r608, 2;
mov.b32 %r607, -1;
mov.b64 %rd97, %rd1;
mov.b32 %r610, %r609;
mov.b32 %r611, %r609;
mov.b32 %r612, %r609;
mov.b32 %r613, %r609;
mov.b32 %r614, %r609;
mov.b32 %r615, %r609;
mov.b32 %r616, %r609;
mov.b32 %r617, %r609;
mov.b32 %r618, %r609;
mov.b32 %r619, %r609;
mov.b32 %r620, %r609;
mov.b32 %r621, %r609;
mov.b32 %r622, %r609;
mov.b32 %r623, %r609;
mov.b32 %r624, %r609;
mov.b32 %r625, %r609;
mov.b32 %r626, %r609;
mov.b32 %r627, %r609;
mov.b32 %r628, %r609;
mov.b32 %r629, %r609;
mov.b32 %r630, %r609;
mov.b32 %r631, %r609;
mov.b32 %r632, %r609;
mov.b32 %r633, %r609;
mov.b32 %r634, %r609;
mov.b32 %r635, %r609;
mov.b32 %r636, %r609;
mov.b32 %r637, %r609;
mov.b32 %r638, %r609;
mov.b32 %r639, %r609;
mov.b32 %r640, %r609;
mov.b32 %r641, %r609;
mov.b32 %r642, %r609;
mov.b32 %r643, %r609;
mov.b32 %r644, %r609;
mov.b32 %r645, %r609;
mov.b32 %r646, %r609;
mov.b32 %r647, %r609;
mov.b32 %r648, %r609;
mov.b32 %r649, %r609;
mov.b32 %r650, %r609;
mov.b32 %r651, %r609;
mov.b32 %r652, %r609;
mov.b32 %r653, %r609;
mov.b32 %r654, %r609;
mov.b32 %r655, %r609;
mov.b32 %r656, %r609;
mov.b32 %r657, %r609;
mov.b32 %r658, %r609;
mov.b32 %r659, %r609;
mov.b32 %r660, %r609;
mov.b32 %r661, %r609;
mov.b32 %r662, %r609;
mov.b32 %r663, %r609;
mov.b32 %r664, %r609;
mov.b32 %r665, %r609;
mov.b32 %r666, %r609;
mov.b32 %r667, %r609;
mov.b32 %r668, %r609;
mov.b32 %r669, %r609;
mov.b32 %r670, %r609;
mov.b32 %r671, %r609;
mov.b32 %r672, %r609;
mov.b32 %r673, %r144;
$L__BB0_4:
setp.lt.s32 %p10, %r673, %r31;
add.s32 %r154, %r607, 1;
setp.gt.s32 %p11, %r154, 3;
selp.b32 %r607, 0, %r154, %p11;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r155, %r607, 14;
add.s32 %r156, %r134, %r155;
add.s32 %r157, %r156, 65536;
shr.u32 %r158, %r156, 4;
shfl.sync.idx.b32 %r159, %r32, 0, 31, -1;
shl.b32 %r160, %r159, 7;
and.b32 %r161, %r160, 512;
add.s32 %r162, %r161, %r158;
and.b32 %r163, %r162, 16383;
cvt.u64.u32 %rd71, %r163;
bfe.u32 %r164, %r157, 4, 14;
cvt.u64.u32 %rd72, %r164;
wgmma.fence.sync.aligned;
or.b64 %rd53, %rd71, 4611686293305294848;
or.b64 %rd54, %rd72, 4611686293305294848;
mov.pred %p9, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd53, %rd54, %p9, 1, 1, 0, 0;
add.s64 %rd55, %rd71, 4611686293305294850;
add.s64 %rd56, %rd72, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd55, %rd56, %p9, 1, 1, 0, 0;
add.s64 %rd57, %rd71, 4611686293305294852;
add.s64 %rd58, %rd72, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd57, %rd58, %p9, 1, 1, 0, 0;
add.s64 %rd59, %rd71, 4611686293305294854;
add.s64 %rd60, %rd72, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd59, %rd60, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd61, %r156;
cvt.u64.u32 %rd62, %r157;
mov.b32 %r140, %r144;
mov.b32 %r141, %r144;
mov.b32 %r142, %r144;
mov.b32 %r143, %r144;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd63, %rd97, %rd18;
add.s64 %rd64, %rd97, %rd17;
add.s64 %rd65, %rd97, %rd16;
add.s64 %rd66, %rd97, %rd15;
add.s64 %rd67, %rd97, %rd14;
add.s64 %rd68, %rd97, %rd13;
add.s64 %rd69, %rd97, %rd12;
add.s64 %rd70, %rd97, %rd11;
add.s32 %r165, %r608, 1;
setp.gt.s32 %p12, %r165, 3;
selp.b32 %r608, 0, %r165, %p12;
setp.lt.s32 %p13, %r26, %r606;
shl.b32 %r166, %r608, 14;
add.s32 %r167, %r134, %r166;
bar.sync 0;
add.s32 %r145, %r167, %r30;
selp.b32 %r168, 16, 0, %p13;
selp.b32 %r147, %r168, 0, %p10;
cp.async.cg.shared.global [ %r145 + 0 ], [ %rd63 + 0 ], 0x10, %r147;
add.s32 %r146, %r145, 4096;
cp.async.cg.shared.global [ %r146 + 0 ], [ %rd64 + 0 ], 0x10, %r147;
add.s32 %r148, %r145, 8192;
cp.async.cg.shared.global [ %r148 + 0 ], [ %rd65 + 0 ], 0x10, %r147;
add.s32 %r149, %r145, 12288;
cp.async.cg.shared.global [ %r149 + 0 ], [ %rd66 + 0 ], 0x10, %r147;
cp.async.commit_group;
add.s32 %r150, %r145, 65536;
cp.async.cg.shared.global [ %r150 + 0 ], [ %rd67 + 0 ], 0x10, %r147;
add.s32 %r151, %r145, 69632;
cp.async.cg.shared.global [ %r151 + 0 ], [ %rd68 + 0 ], 0x10, %r147;
add.s32 %r152, %r145, 73728;
cp.async.cg.shared.global [ %r152 + 0 ], [ %rd69 + 0 ], 0x10, %r147;
add.s32 %r153, %r145, 77824;
cp.async.cg.shared.global [ %r153 + 0 ], [ %rd70 + 0 ], 0x10, %r147;
cp.async.commit_group;
add.s32 %r673, %r673, 1;
add.s64 %rd97, %rd97, 128;
add.s32 %r606, %r606, -64;
setp.ne.b32 %p14, %r27, %r673;
@%p14 bra $L__BB0_4;
$L__BB0_5:
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
mad.lo.s32 %r266, %r7, %r34, %r24;
mad.lo.s32 %r267, %r8, %r34, %r24;
mad.lo.s32 %r268, %r9, %r34, %r24;
mad.lo.s32 %r269, %r10, %r34, %r24;
mad.lo.s32 %r270, %r11, %r34, %r24;
mad.lo.s32 %r271, %r12, %r34, %r24;
mad.lo.s32 %r272, %r13, %r34, %r24;
mad.lo.s32 %r273, %r14, %r34, %r24;
mad.wide.s32 %rd73, %r266, 2, %rd1;
mad.wide.s32 %rd74, %r267, 2, %rd1;
mad.wide.s32 %rd75, %r268, 2, %rd1;
mad.wide.s32 %rd76, %r269, 2, %rd1;
mad.wide.s32 %rd77, %r270, 2, %rd1;
mad.wide.s32 %rd78, %r271, 2, %rd1;
mad.wide.s32 %rd79, %r272, 2, %rd1;
mad.wide.s32 %rd80, %r273, 2, %rd1;
setp.lt.s32 %p31, %r7, %r39;
setp.lt.s32 %p32, %r8, %r39;
setp.lt.s32 %p33, %r9, %r39;
setp.lt.s32 %p34, %r10, %r39;
setp.lt.s32 %p35, %r11, %r39;
setp.lt.s32 %p36, %r12, %r39;
setp.lt.s32 %p37, %r13, %r39;
setp.lt.s32 %p38, %r14, %r39;
setp.lt.s32 %p39, %r24, %r39;
and.pred %p15, %p31, %p39;
and.pred %p16, %p32, %p39;
and.pred %p17, %p33, %p39;
and.pred %p18, %p34, %p39;
and.pred %p19, %p35, %p39;
and.pred %p20, %p36, %p39;
and.pred %p21, %p37, %p39;
and.pred %p22, %p38, %p39;
mov.b32 %r173, 0;
mov.u32 %r169, %r173;
mov.u32 %r170, %r173;
mov.u32 %r171, %r173;
mov.u32 %r172, %r173;
@%p15 ld.global.v4.b32 { %r169, %r170, %r171, %r172 }, [ %rd73 + 0 ];
mov.u32 %r174, %r173;
mov.u32 %r175, %r173;
mov.u32 %r176, %r173;
mov.u32 %r177, %r173;
@%p16 ld.global.v4.b32 { %r174, %r175, %r176, %r177 }, [ %rd74 + 0 ];
mov.u32 %r178, %r173;
mov.u32 %r179, %r173;
mov.u32 %r180, %r173;
mov.u32 %r181, %r173;
@%p17 ld.global.v4.b32 { %r178, %r179, %r180, %r181 }, [ %rd75 + 0 ];
mov.u32 %r182, %r173;
mov.u32 %r183, %r173;
mov.u32 %r184, %r173;
mov.u32 %r185, %r173;
@%p18 ld.global.v4.b32 { %r182, %r183, %r184, %r185 }, [ %rd76 + 0 ];
mov.u32 %r186, %r173;
mov.u32 %r187, %r173;
mov.u32 %r188, %r173;
mov.u32 %r189, %r173;
@%p19 ld.global.v4.b32 { %r186, %r187, %r188, %r189 }, [ %rd77 + 0 ];
mov.u32 %r190, %r173;
mov.u32 %r191, %r173;
mov.u32 %r192, %r173;
mov.u32 %r193, %r173;
@%p20 ld.global.v4.b32 { %r190, %r191, %r192, %r193 }, [ %rd78 + 0 ];
mov.u32 %r194, %r173;
mov.u32 %r195, %r173;
mov.u32 %r196, %r173;
mov.u32 %r197, %r173;
@%p21 ld.global.v4.b32 { %r194, %r195, %r196, %r197 }, [ %rd79 + 0 ];
mov.u32 %r198, %r173;
mov.u32 %r199, %r173;
mov.u32 %r200, %r173;
mov.u32 %r201, %r173;
@%p22 ld.global.v4.b32 { %r198, %r199, %r200, %r201 }, [ %rd80 + 0 ];
shl.b32 %r274, %r4, 8;
and.b32 %r275, %r274, 30720;
shl.b32 %r276, %r25, 4;
or.b32 %r277, %r275, %r276;
xor.b32 %r278, %r277, %r5;
add.s32 %r279, %r134, %r278;
st.shared.v4.b32 [%r279], {%r169, %r170, %r171, %r172};
st.shared.v4.b32 [%r279+256], {%r174, %r175, %r176, %r177};
st.shared.v4.b32 [%r279+512], {%r178, %r179, %r180, %r181};
st.shared.v4.b32 [%r279+768], {%r182, %r183, %r184, %r185};
st.shared.v4.b32 [%r279+1024], {%r186, %r187, %r188, %r189};
st.shared.v4.b32 [%r279+1280], {%r190, %r191, %r192, %r193};
st.shared.v4.b32 [%r279+1536], {%r194, %r195, %r196, %r197};
st.shared.v4.b32 [%r279+1792], {%r198, %r199, %r200, %r201};
bar.sync 0;
shl.b32 %r280, %r25, 12;
shl.b32 %r281, %r6, 4;
and.b32 %r282, %r4, 224;
shl.b32 %r283, %r282, 3;
and.b32 %r284, %r4, 16;
or.b32 %r285, %r281, %r283;
xor.b32 %r286, %r285, %r284;
or.b32 %r287, %r286, %r280;
add.s32 %r288, %r134, %r287;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r291, %r292}, [%r288];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r288+2048];
xor.b32 %r297, %r287, 32;
add.s32 %r298, %r134, %r297;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r299, %r300, %r301, %r302}, [%r298];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r303, %r304, %r305, %r306}, [%r298+2048];
xor.b32 %r307, %r287, 64;
add.s32 %r308, %r134, %r307;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r311, %r312}, [%r308];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r313, %r314, %r315, %r316}, [%r308+2048];
xor.b32 %r317, %r287, 96;
add.s32 %r318, %r134, %r317;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r319, %r320, %r321, %r322}, [%r318];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r323, %r324, %r325, %r326}, [%r318+2048];
mov.b32 {%rs1, %rs2}, %r289;
cvt.f32.bf16 %r327, %rs1;
cvt.f32.bf16 %r328, %rs2;
mul.f32 %r329, %r38, %r328;
mul.f32 %r330, %r38, %r327;
fma.rn.f32 %r331, %r37, %r609, %r330;
fma.rn.f32 %r332, %r37, %r610, %r329;
cvt.rn.bf16x2.f32 %r333, %r332, %r331;
mov.b32 {%rs3, %rs4}, %r290;
cvt.f32.bf16 %r334, %rs3;
cvt.f32.bf16 %r335, %rs4;
mul.f32 %r336, %r38, %r335;
mul.f32 %r337, %r38, %r334;
fma.rn.f32 %r338, %r37, %r611, %r337;
fma.rn.f32 %r339, %r37, %r612, %r336;
cvt.rn.bf16x2.f32 %r340, %r339, %r338;
mov.b32 {%rs5, %rs6}, %r291;
cvt.f32.bf16 %r341, %rs5;
cvt.f32.bf16 %r342, %rs6;
mul.f32 %r343, %r38, %r342;
mul.f32 %r344, %r38, %r341;
fma.rn.f32 %r345, %r37, %r613, %r344;
fma.rn.f32 %r346, %r37, %r614, %r343;
cvt.rn.bf16x2.f32 %r347, %r346, %r345;
mov.b32 {%rs7, %rs8}, %r292;
cvt.f32.bf16 %r348, %rs7;
cvt.f32.bf16 %r349, %rs8;
mul.f32 %r350, %r38, %r349;
mul.f32 %r351, %r38, %r348;
fma.rn.f32 %r352, %r37, %r615, %r351;
fma.rn.f32 %r353, %r37, %r616, %r350;
cvt.rn.bf16x2.f32 %r354, %r353, %r352;
mov.b32 {%rs9, %rs10}, %r299;
cvt.f32.bf16 %r355, %rs9;
cvt.f32.bf16 %r356, %rs10;
mul.f32 %r357, %r38, %r356;
mul.f32 %r358, %r38, %r355;
fma.rn.f32 %r359, %r37, %r617, %r358;
fma.rn.f32 %r360, %r37, %r618, %r357;
cvt.rn.bf16x2.f32 %r361, %r360, %r359;
mov.b32 {%rs11, %rs12}, %r300;
cvt.f32.bf16 %r362, %rs11;
cvt.f32.bf16 %r363, %rs12;
mul.f32 %r364, %r38, %r363;
mul.f32 %r365, %r38, %r362;
fma.rn.f32 %r366, %r37, %r619, %r365;
fma.rn.f32 %r367, %r37, %r620, %r364;
cvt.rn.bf16x2.f32 %r368, %r367, %r366;
mov.b32 {%rs13, %rs14}, %r301;
cvt.f32.bf16 %r369, %rs13;
cvt.f32.bf16 %r370, %rs14;
mul.f32 %r371, %r38, %r370;
mul.f32 %r372, %r38, %r369;
fma.rn.f32 %r373, %r37, %r621, %r372;
fma.rn.f32 %r374, %r37, %r622, %r371;
cvt.rn.bf16x2.f32 %r375, %r374, %r373;
mov.b32 {%rs15, %rs16}, %r302;
cvt.f32.bf16 %r376, %rs15;
cvt.f32.bf16 %r377, %rs16;
mul.f32 %r378, %r38, %r377;
mul.f32 %r379, %r38, %r376;
fma.rn.f32 %r380, %r37, %r623, %r379;
fma.rn.f32 %r381, %r37, %r624, %r378;
cvt.rn.bf16x2.f32 %r382, %r381, %r380;
mov.b32 {%rs17, %rs18}, %r309;
cvt.f32.bf16 %r383, %rs17;
cvt.f32.bf16 %r384, %rs18;
mul.f32 %r385, %r38, %r384;
mul.f32 %r386, %r38, %r383;
fma.rn.f32 %r387, %r37, %r625, %r386;
fma.rn.f32 %r388, %r37, %r626, %r385;
cvt.rn.bf16x2.f32 %r389, %r388, %r387;
mov.b32 {%rs19, %rs20}, %r310;
cvt.f32.bf16 %r390, %rs19;
cvt.f32.bf16 %r391, %rs20;
mul.f32 %r392, %r38, %r391;
mul.f32 %r393, %r38, %r390;
fma.rn.f32 %r394, %r37, %r627, %r393;
fma.rn.f32 %r395, %r37, %r628, %r392;
cvt.rn.bf16x2.f32 %r396, %r395, %r394;
mov.b32 {%rs21, %rs22}, %r311;
cvt.f32.bf16 %r397, %rs21;
cvt.f32.bf16 %r398, %rs22;
mul.f32 %r399, %r38, %r398;
mul.f32 %r400, %r38, %r397;
fma.rn.f32 %r401, %r37, %r629, %r400;
fma.rn.f32 %r402, %r37, %r630, %r399;
cvt.rn.bf16x2.f32 %r403, %r402, %r401;
mov.b32 {%rs23, %rs24}, %r312;
cvt.f32.bf16 %r404, %rs23;
cvt.f32.bf16 %r405, %rs24;
mul.f32 %r406, %r38, %r405;
mul.f32 %r407, %r38, %r404;
fma.rn.f32 %r408, %r37, %r631, %r407;
fma.rn.f32 %r409, %r37, %r632, %r406;
cvt.rn.bf16x2.f32 %r410, %r409, %r408;
mov.b32 {%rs25, %rs26}, %r319;
cvt.f32.bf16 %r411, %rs25;
cvt.f32.bf16 %r412, %rs26;
mul.f32 %r413, %r38, %r412;
mul.f32 %r414, %r38, %r411;
fma.rn.f32 %r415, %r37, %r633, %r414;
fma.rn.f32 %r416, %r37, %r634, %r413;
cvt.rn.bf16x2.f32 %r417, %r416, %r415;
mov.b32 {%rs27, %rs28}, %r320;
cvt.f32.bf16 %r418, %rs27;
cvt.f32.bf16 %r419, %rs28;
mul.f32 %r420, %r38, %r419;
mul.f32 %r421, %r38, %r418;
fma.rn.f32 %r422, %r37, %r635, %r421;
fma.rn.f32 %r423, %r37, %r636, %r420;
cvt.rn.bf16x2.f32 %r424, %r423, %r422;
mov.b32 {%rs29, %rs30}, %r321;
cvt.f32.bf16 %r425, %rs29;
cvt.f32.bf16 %r426, %rs30;
mul.f32 %r427, %r38, %r426;
mul.f32 %r428, %r38, %r425;
fma.rn.f32 %r429, %r37, %r637, %r428;
fma.rn.f32 %r430, %r37, %r638, %r427;
cvt.rn.bf16x2.f32 %r431, %r430, %r429;
mov.b32 {%rs31, %rs32}, %r322;
cvt.f32.bf16 %r432, %rs31;
cvt.f32.bf16 %r433, %rs32;
mul.f32 %r434, %r38, %r433;
mul.f32 %r435, %r38, %r432;
fma.rn.f32 %r436, %r37, %r639, %r435;
fma.rn.f32 %r437, %r37, %r640, %r434;
cvt.rn.bf16x2.f32 %r438, %r437, %r436;
mov.b32 {%rs33, %rs34}, %r293;
cvt.f32.bf16 %r439, %rs33;
cvt.f32.bf16 %r440, %rs34;
mul.f32 %r441, %r38, %r440;
mul.f32 %r442, %r38, %r439;
fma.rn.f32 %r443, %r37, %r641, %r442;
fma.rn.f32 %r444, %r37, %r642, %r441;
cvt.rn.bf16x2.f32 %r445, %r444, %r443;
mov.b32 {%rs35, %rs36}, %r294;
cvt.f32.bf16 %r446, %rs35;
cvt.f32.bf16 %r447, %rs36;
mul.f32 %r448, %r38, %r447;
mul.f32 %r449, %r38, %r446;
fma.rn.f32 %r450, %r37, %r643, %r449;
fma.rn.f32 %r451, %r37, %r644, %r448;
cvt.rn.bf16x2.f32 %r452, %r451, %r450;
mov.b32 {%rs37, %rs38}, %r295;
cvt.f32.bf16 %r453, %rs37;
cvt.f32.bf16 %r454, %rs38;
mul.f32 %r455, %r38, %r454;
mul.f32 %r456, %r38, %r453;
fma.rn.f32 %r457, %r37, %r645, %r456;
fma.rn.f32 %r458, %r37, %r646, %r455;
cvt.rn.bf16x2.f32 %r459, %r458, %r457;
mov.b32 {%rs39, %rs40}, %r296;
cvt.f32.bf16 %r460, %rs39;
cvt.f32.bf16 %r461, %rs40;
mul.f32 %r462, %r38, %r461;
mul.f32 %r463, %r38, %r460;
fma.rn.f32 %r464, %r37, %r647, %r463;
fma.rn.f32 %r465, %r37, %r648, %r462;
cvt.rn.bf16x2.f32 %r466, %r465, %r464;
mov.b32 {%rs41, %rs42}, %r303;
cvt.f32.bf16 %r467, %rs41;
cvt.f32.bf16 %r468, %rs42;
mul.f32 %r469, %r38, %r468;
mul.f32 %r470, %r38, %r467;
fma.rn.f32 %r471, %r37, %r649, %r470;
fma.rn.f32 %r472, %r37, %r650, %r469;
cvt.rn.bf16x2.f32 %r473, %r472, %r471;
mov.b32 {%rs43, %rs44}, %r304;
cvt.f32.bf16 %r474, %rs43;
cvt.f32.bf16 %r475, %rs44;
mul.f32 %r476, %r38, %r475;
mul.f32 %r477, %r38, %r474;
fma.rn.f32 %r478, %r37, %r651, %r477;
fma.rn.f32 %r479, %r37, %r652, %r476;
cvt.rn.bf16x2.f32 %r480, %r479, %r478;
mov.b32 {%rs45, %rs46}, %r305;
cvt.f32.bf16 %r481, %rs45;
cvt.f32.bf16 %r482, %rs46;
mul.f32 %r483, %r38, %r482;
mul.f32 %r484, %r38, %r481;
fma.rn.f32 %r485, %r37, %r653, %r484;
fma.rn.f32 %r486, %r37, %r654, %r483;
cvt.rn.bf16x2.f32 %r487, %r486, %r485;
mov.b32 {%rs47, %rs48}, %r306;
cvt.f32.bf16 %r488, %rs47;
cvt.f32.bf16 %r489, %rs48;
mul.f32 %r490, %r38, %r489;
mul.f32 %r491, %r38, %r488;
fma.rn.f32 %r492, %r37, %r655, %r491;
fma.rn.f32 %r493, %r37, %r656, %r490;
cvt.rn.bf16x2.f32 %r494, %r493, %r492;
mov.b32 {%rs49, %rs50}, %r313;
cvt.f32.bf16 %r495, %rs49;
cvt.f32.bf16 %r496, %rs50;
mul.f32 %r497, %r38, %r496;
mul.f32 %r498, %r38, %r495;
fma.rn.f32 %r499, %r37, %r657, %r498;
fma.rn.f32 %r500, %r37, %r658, %r497;
cvt.rn.bf16x2.f32 %r501, %r500, %r499;
mov.b32 {%rs51, %rs52}, %r314;
cvt.f32.bf16 %r502, %rs51;
cvt.f32.bf16 %r503, %rs52;
mul.f32 %r504, %r38, %r503;
mul.f32 %r505, %r38, %r502;
fma.rn.f32 %r506, %r37, %r659, %r505;
fma.rn.f32 %r507, %r37, %r660, %r504;
cvt.rn.bf16x2.f32 %r508, %r507, %r506;
mov.b32 {%rs53, %rs54}, %r315;
cvt.f32.bf16 %r509, %rs53;
cvt.f32.bf16 %r510, %rs54;
mul.f32 %r511, %r38, %r510;
mul.f32 %r512, %r38, %r509;
fma.rn.f32 %r513, %r37, %r661, %r512;
fma.rn.f32 %r514, %r37, %r662, %r511;
cvt.rn.bf16x2.f32 %r515, %r514, %r513;
mov.b32 {%rs55, %rs56}, %r316;
cvt.f32.bf16 %r516, %rs55;
cvt.f32.bf16 %r517, %rs56;
mul.f32 %r518, %r38, %r517;
mul.f32 %r519, %r38, %r516;
fma.rn.f32 %r520, %r37, %r663, %r519;
fma.rn.f32 %r521, %r37, %r664, %r518;
cvt.rn.bf16x2.f32 %r522, %r521, %r520;
mov.b32 {%rs57, %rs58}, %r323;
cvt.f32.bf16 %r523, %rs57;
cvt.f32.bf16 %r524, %rs58;
mul.f32 %r525, %r38, %r524;
mul.f32 %r526, %r38, %r523;
fma.rn.f32 %r527, %r37, %r665, %r526;
fma.rn.f32 %r528, %r37, %r666, %r525;
cvt.rn.bf16x2.f32 %r529, %r528, %r527;
mov.b32 {%rs59, %rs60}, %r324;
cvt.f32.bf16 %r530, %rs59;
cvt.f32.bf16 %r531, %rs60;
mul.f32 %r532, %r38, %r531;
mul.f32 %r533, %r38, %r530;
fma.rn.f32 %r534, %r37, %r667, %r533;
fma.rn.f32 %r535, %r37, %r668, %r532;
cvt.rn.bf16x2.f32 %r536, %r535, %r534;
mov.b32 {%rs61, %rs62}, %r325;
cvt.f32.bf16 %r537, %rs61;
cvt.f32.bf16 %r538, %rs62;
mul.f32 %r539, %r38, %r538;
mul.f32 %r540, %r38, %r537;
fma.rn.f32 %r541, %r37, %r669, %r540;
fma.rn.f32 %r542, %r37, %r670, %r539;
cvt.rn.bf16x2.f32 %r543, %r542, %r541;
mov.b32 {%rs63, %rs64}, %r326;
cvt.f32.bf16 %r544, %rs63;
cvt.f32.bf16 %r545, %rs64;
mul.f32 %r546, %r38, %r545;
mul.f32 %r547, %r38, %r544;
fma.rn.f32 %r548, %r37, %r671, %r547;
fma.rn.f32 %r549, %r37, %r672, %r546;
cvt.rn.bf16x2.f32 %r550, %r549, %r548;
bar.sync 0;
shl.b32 %r551, %r4, 5;
and.b32 %r552, %r551, 3072;
and.b32 %r553, %r28, 368;
bfe.s32 %r554, %r4, 3, 1;
and.b32 %r555, %r554, 528;
or.b32 %r556, %r552, %r553;
or.b32 %r557, %r555, %r282;
xor.b32 %r558, %r557, %r556;
add.s32 %r559, %r134, %r558;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r333, %r340, %r347, %r354};
bar.sync 0;
shl.b32 %r560, %r25, 9;
and.b32 %r561, %r4, 112;
and.b32 %r562, %r29, 256;
xor.b32 %r563, %r281, %r561;
add.s32 %r564, %r134, %r560;
add.s32 %r565, %r564, %r562;
add.s32 %r566, %r565, %r563;
ld.shared.v4.b32 {%r234, %r235, %r236, %r237}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r361, %r368, %r375, %r382};
bar.sync 0;
ld.shared.v4.b32 {%r238, %r239, %r240, %r241}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r389, %r396, %r403, %r410};
bar.sync 0;
ld.shared.v4.b32 {%r242, %r243, %r244, %r245}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r417, %r424, %r431, %r438};
bar.sync 0;
ld.shared.v4.b32 {%r246, %r247, %r248, %r249}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r445, %r452, %r459, %r466};
bar.sync 0;
ld.shared.v4.b32 {%r250, %r251, %r252, %r253}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r473, %r480, %r487, %r494};
bar.sync 0;
ld.shared.v4.b32 {%r254, %r255, %r256, %r257}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r501, %r508, %r515, %r522};
bar.sync 0;
ld.shared.v4.b32 {%r258, %r259, %r260, %r261}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r529, %r536, %r543, %r550};
bar.sync 0;
ld.shared.v4.b32 {%r262, %r263, %r264, %r265}, [%r566];
bar.sync 0;
and.b32 %r567, %r4, 3;
shl.b32 %r568, %r567, 13;
and.b32 %r569, %r551, 7264;
and.b32 %r570, %r4, 24;
shl.b32 %r571, %r570, 4;
shl.b32 %r572, %r4, 2;
and.b32 %r573, %r572, 16;
or.b32 %r574, %r568, %r573;
or.b32 %r575, %r569, %r571;
or.b32 %r576, %r574, %r575;
add.s32 %r577, %r134, %r576;
st.shared.v4.b32 [%r577], {%r333, %r347, %r361, %r375};
st.shared.v4.b32 [%r577+512], {%r340, %r354, %r368, %r382};
xor.b32 %r578, %r576, 32;
add.s32 %r579, %r134, %r578;
st.shared.v4.b32 [%r579], {%r389, %r403, %r417, %r431};
st.shared.v4.b32 [%r579+512], {%r396, %r410, %r424, %r438};
xor.b32 %r580, %r576, 64;
add.s32 %r581, %r134, %r580;
st.shared.v4.b32 [%r581], {%r445, %r459, %r473, %r487};
st.shared.v4.b32 [%r581+512], {%r452, %r466, %r480, %r494};
xor.b32 %r582, %r576, 96;
add.s32 %r583, %r134, %r582;
st.shared.v4.b32 [%r583], {%r501, %r515, %r529, %r543};
st.shared.v4.b32 [%r583+512], {%r508, %r522, %r536, %r550};
bar.sync 0;
shl.b32 %r584, %r570, 10;
shl.b32 %r585, %r567, 5;
and.b32 %r586, %r572, 1008;
or.b32 %r587, %r584, %r585;
xor.b32 %r588, %r587, %r586;
add.s32 %r589, %r134, %r588;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r204, %r205}, [%r589];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r589+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r212, %r213}, [%r589+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r216, %r217}, [%r589+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r220, %r221}, [%r589+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r224, %r225}, [%r589+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r228, %r229}, [%r589+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r232, %r233}, [%r589+7168];
mad.lo.s32 %r590, %r7, %r36, %r24;
mad.lo.s32 %r591, %r8, %r36, %r24;
mad.lo.s32 %r592, %r9, %r36, %r24;
mad.lo.s32 %r593, %r10, %r36, %r24;
mad.lo.s32 %r594, %r11, %r36, %r24;
mad.lo.s32 %r595, %r12, %r36, %r24;
mad.lo.s32 %r596, %r13, %r36, %r24;
mad.lo.s32 %r597, %r14, %r36, %r24;
mad.wide.s32 %rd81, %r590, 2, %rd2;
mad.wide.s32 %rd82, %r591, 2, %rd2;
mad.wide.s32 %rd83, %r592, 2, %rd2;
mad.wide.s32 %rd84, %r593, 2, %rd2;
mad.wide.s32 %rd85, %r594, 2, %rd2;
mad.wide.s32 %rd86, %r595, 2, %rd2;
mad.wide.s32 %rd87, %r596, 2, %rd2;
mad.wide.s32 %rd88, %r597, 2, %rd2;
@%p15 st.global.v4.b32 [ %rd81 + 0 ], { %r202, %r203, %r204, %r205 };
@%p16 st.global.v4.b32 [ %rd82 + 0 ], { %r206, %r207, %r208, %r209 };
@%p17 st.global.v4.b32 [ %rd83 + 0 ], { %r210, %r211, %r212, %r213 };
@%p18 st.global.v4.b32 [ %rd84 + 0 ], { %r214, %r215, %r216, %r217 };
@%p19 st.global.v4.b32 [ %rd85 + 0 ], { %r218, %r219, %r220, %r221 };
@%p20 st.global.v4.b32 [ %rd86 + 0 ], { %r222, %r223, %r224, %r225 };
@%p21 st.global.v4.b32 [ %rd87 + 0 ], { %r226, %r227, %r228, %r229 };
@%p22 st.global.v4.b32 [ %rd88 + 0 ], { %r230, %r231, %r232, %r233 };
mad.lo.s32 %r598, %r16, %r36, %r15;
mad.lo.s32 %r599, %r17, %r36, %r15;
mad.lo.s32 %r600, %r18, %r36, %r15;
mad.lo.s32 %r601, %r19, %r36, %r15;
mad.lo.s32 %r602, %r20, %r36, %r15;
mad.lo.s32 %r603, %r21, %r36, %r15;
mad.lo.s32 %r604, %r22, %r36, %r15;
mad.lo.s32 %r605, %r23, %r36, %r15;
mad.wide.s32 %rd89, %r598, 2, %rd2;
mad.wide.s32 %rd90, %r599, 2, %rd2;
mad.wide.s32 %rd91, %r600, 2, %rd2;
mad.wide.s32 %rd92, %r601, 2, %rd2;
mad.wide.s32 %rd93, %r602, 2, %rd2;
mad.wide.s32 %rd94, %r603, 2, %rd2;
mad.wide.s32 %rd95, %r604, 2, %rd2;
mad.wide.s32 %rd96, %r605, 2, %rd2;
setp.lt.s32 %p40, %r16, %r39;
setp.lt.s32 %p41, %r17, %r39;
setp.lt.s32 %p42, %r18, %r39;
setp.lt.s32 %p43, %r19, %r39;
setp.lt.s32 %p44, %r20, %r39;
setp.lt.s32 %p45, %r21, %r39;
setp.lt.s32 %p46, %r22, %r39;
setp.lt.s32 %p47, %r23, %r39;
setp.lt.s32 %p48, %r15, %r39;
and.pred %p23, %p40, %p48;
and.pred %p24, %p41, %p48;
and.pred %p25, %p42, %p48;
and.pred %p26, %p43, %p48;
and.pred %p27, %p44, %p48;
and.pred %p28, %p45, %p48;
and.pred %p29, %p46, %p48;
and.pred %p30, %p47, %p48;
@%p23 st.global.v4.b32 [ %rd89 + 0 ], { %r234, %r235, %r236, %r237 };
@%p24 st.global.v4.b32 [ %rd90 + 0 ], { %r238, %r239, %r240, %r241 };
@%p25 st.global.v4.b32 [ %rd91 + 0 ], { %r242, %r243, %r244, %r245 };
@%p26 st.global.v4.b32 [ %rd92 + 0 ], { %r246, %r247, %r248, %r249 };
@%p27 st.global.v4.b32 [ %rd93 + 0 ], { %r250, %r251, %r252, %r253 };
@%p28 st.global.v4.b32 [ %rd94 + 0 ], { %r254, %r255, %r256, %r257 };
@%p29 st.global.v4.b32 [ %rd95 + 0 ], { %r258, %r259, %r260, %r261 };
@%p30 st.global.v4.b32 [ %rd96 + 0 ], { %r262, %r263, %r264, %r265 };
$L__BB0_1:
ret;
$L__tmp8:
$L__func_end0:
}
  "#
  | .qkPolynomial, .compact => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry ba_plus_cAA_kernel(
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_0,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_1,
.param .u32 ba_plus_cAA_kernel_param_2,
.param .u32 ba_plus_cAA_kernel_param_3,
.param .u32 ba_plus_cAA_kernel_param_4,
.param .u32 ba_plus_cAA_kernel_param_5,
.param .u32 ba_plus_cAA_kernel_param_6,
.param .f32 ba_plus_cAA_kernel_param_7,
.param .f32 ba_plus_cAA_kernel_param_8,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_9,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_10
)
.reqntid 256
{
.reg .pred %p<48>;
.reg .b16 %rs<65>;
.reg .b32 %r<661>;
.reg .b64 %rd<98>;
$L__func_begin0:
ld.param.b32 %r18, [ba_plus_cAA_kernel_param_8];
ld.param.b32 %r17, [ba_plus_cAA_kernel_param_7];
ld.param.b32 %r16, [ba_plus_cAA_kernel_param_6];
ld.param.b32 %r15, [ba_plus_cAA_kernel_param_5];
ld.param.b32 %r14, [ba_plus_cAA_kernel_param_4];
ld.param.b32 %r13, [ba_plus_cAA_kernel_param_2];
ld.param.b64 %rd18, [ba_plus_cAA_kernel_param_1];
ld.param.b64 %rd43, [ba_plus_cAA_kernel_param_0];
$L__tmp0:
mov.u32 %r46, %ctaid.x;
$L__tmp1:
mul.hi.u32 %r1, %r46, 1431655766;
mul.lo.s32 %r47, %r1, 3;
ld.param.b32 %r48, [ba_plus_cAA_kernel_param_3];
sub.s32 %r49, %r46, %r47;
mul.lo.s32 %r50, %r49, 3;
mov.b32 %r51, 9216;
shr.u32 %r52, %r51, %r50;
and.b32 %r2, %r52, 896;
mov.b32 %r53, 8192;
shr.u32 %r54, %r53, %r50;
and.b32 %r3, %r54, 896;
$L__tmp2:
mul.lo.s32 %r55, %r48, %r1;
mad.wide.s32 %rd1, %r55, 2, %rd43;
mov.u32 %r4, %tid.x;
shr.u32 %r56, %r4, 3;
bfe.u32 %r57, %r4, 3, 5;
or.b32 %r58, %r57, 32;
or.b32 %r59, %r57, 64;
or.b32 %r60, %r56, 96;
or.b32 %r61, %r2, %r57;
or.b32 %r62, %r2, %r58;
or.b32 %r63, %r2, %r59;
or.b32 %r64, %r2, %r60;
rem.s32 %r65, %r61, %r13;
rem.s32 %r66, %r62, %r13;
rem.s32 %r67, %r63, %r13;
rem.s32 %r68, %r64, %r13;
or.b32 %r69, %r3, %r57;
or.b32 %r70, %r3, %r58;
or.b32 %r71, %r3, %r59;
or.b32 %r72, %r3, %r60;
rem.s32 %r73, %r69, %r13;
rem.s32 %r74, %r70, %r13;
rem.s32 %r75, %r71, %r13;
rem.s32 %r76, %r72, %r13;
and.b32 %r5, %r4, 7;
shl.b32 %r6, %r5, 3;
mad.lo.s32 %r77, %r65, %r14, %r6;
mad.lo.s32 %r78, %r66, %r14, %r6;
mad.lo.s32 %r79, %r67, %r14, %r6;
mad.lo.s32 %r80, %r68, %r14, %r6;
mad.wide.s32 %rd19, %r77, 2, %rd1;
mad.wide.s32 %rd20, %r78, 2, %rd1;
mad.wide.s32 %rd21, %r79, 2, %rd1;
mad.wide.s32 %rd22, %r80, 2, %rd1;
mad.lo.s32 %r81, %r73, %r14, %r6;
mad.lo.s32 %r82, %r74, %r14, %r6;
mad.lo.s32 %r83, %r75, %r14, %r6;
mad.lo.s32 %r84, %r76, %r14, %r6;
mad.wide.s32 %rd23, %r81, 2, %rd1;
mad.wide.s32 %rd24, %r82, 2, %rd1;
mad.wide.s32 %rd25, %r83, 2, %rd1;
mad.wide.s32 %rd26, %r84, 2, %rd1;
$L__tmp3:
add.s32 %r85, %r13, 63;
$L__tmp4:
setp.lt.s32 %p1, %r85, 64;
setp.gt.s32 %p2, %r85, 63;
setp.lt.s32 %p3, %r6, %r13;
shl.b32 %r8, %r4, 4;
and.b32 %r89, %r8, 4080;
shl.b32 %r9, %r4, 1;
and.b32 %r90, %r9, 112;
xor.b32 %r10, %r89, %r90;
mov.b32 %r91, global_smem;
add.s32 %r19, %r91, %r10;
selp.b32 %r92, 16, 0, %p2;
selp.b32 %r21, %r92, 0, %p3;
cp.async.cg.shared.global [ %r19 + 0 ], [ %rd19 + 0 ], 0x10, %r21;
add.s32 %r20, %r19, 4096;
cp.async.cg.shared.global [ %r20 + 0 ], [ %rd20 + 0 ], 0x10, %r21;
add.s32 %r22, %r19, 8192;
cp.async.cg.shared.global [ %r22 + 0 ], [ %rd21 + 0 ], 0x10, %r21;
add.s32 %r23, %r19, 12288;
cp.async.cg.shared.global [ %r23 + 0 ], [ %rd22 + 0 ], 0x10, %r21;
cp.async.commit_group;
add.s32 %r24, %r19, 65536;
cp.async.cg.shared.global [ %r24 + 0 ], [ %rd23 + 0 ], 0x10, %r21;
add.s32 %r25, %r19, 69632;
cp.async.cg.shared.global [ %r25 + 0 ], [ %rd24 + 0 ], 0x10, %r21;
add.s32 %r26, %r19, 73728;
cp.async.cg.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x10, %r21;
add.s32 %r27, %r19, 77824;
cp.async.cg.shared.global [ %r27 + 0 ], [ %rd26 + 0 ], 0x10, %r21;
cp.async.commit_group;
setp.gt.s32 %p4, %r85, 127;
add.s64 %rd27, %rd19, 128;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s32 %r93, %r13, -64;
setp.lt.s32 %p5, %r6, %r93;
bar.sync 0;
add.s32 %r28, %r19, 16384;
selp.b32 %r94, 16, 0, %p5;
selp.b32 %r30, %r94, 0, %p4;
cp.async.cg.shared.global [ %r28 + 0 ], [ %rd27 + 0 ], 0x10, %r30;
add.s32 %r29, %r19, 20480;
cp.async.cg.shared.global [ %r29 + 0 ], [ %rd28 + 0 ], 0x10, %r30;
add.s32 %r31, %r19, 24576;
cp.async.cg.shared.global [ %r31 + 0 ], [ %rd29 + 0 ], 0x10, %r30;
add.s32 %r32, %r19, 28672;
cp.async.cg.shared.global [ %r32 + 0 ], [ %rd30 + 0 ], 0x10, %r30;
cp.async.commit_group;
add.s32 %r33, %r19, 81920;
cp.async.cg.shared.global [ %r33 + 0 ], [ %rd31 + 0 ], 0x10, %r30;
add.s32 %r34, %r19, 86016;
cp.async.cg.shared.global [ %r34 + 0 ], [ %rd32 + 0 ], 0x10, %r30;
add.s32 %r35, %r19, 90112;
cp.async.cg.shared.global [ %r35 + 0 ], [ %rd33 + 0 ], 0x10, %r30;
add.s32 %r36, %r19, 94208;
cp.async.cg.shared.global [ %r36 + 0 ], [ %rd34 + 0 ], 0x10, %r30;
cp.async.commit_group;
setp.gt.s32 %p6, %r85, 191;
add.s64 %rd35, %rd19, 256;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s32 %r95, %r13, -128;
setp.lt.s32 %p7, %r6, %r95;
bar.sync 0;
add.s32 %r37, %r19, 32768;
selp.b32 %r96, 16, 0, %p7;
selp.b32 %r39, %r96, 0, %p6;
cp.async.cg.shared.global [ %r37 + 0 ], [ %rd35 + 0 ], 0x10, %r39;
add.s32 %r38, %r19, 36864;
cp.async.cg.shared.global [ %r38 + 0 ], [ %rd36 + 0 ], 0x10, %r39;
add.s32 %r40, %r19, 40960;
cp.async.cg.shared.global [ %r40 + 0 ], [ %rd37 + 0 ], 0x10, %r39;
add.s32 %r41, %r19, 45056;
cp.async.cg.shared.global [ %r41 + 0 ], [ %rd38 + 0 ], 0x10, %r39;
cp.async.commit_group;
add.s32 %r42, %r19, 98304;
cp.async.cg.shared.global [ %r42 + 0 ], [ %rd39 + 0 ], 0x10, %r39;
add.s32 %r43, %r19, 102400;
cp.async.cg.shared.global [ %r43 + 0 ], [ %rd40 + 0 ], 0x10, %r39;
add.s32 %r44, %r19, 106496;
cp.async.cg.shared.global [ %r44 + 0 ], [ %rd41 + 0 ], 0x10, %r39;
add.s32 %r45, %r19, 110592;
cp.async.cg.shared.global [ %r45 + 0 ], [ %rd42 + 0 ], 0x10, %r39;
cp.async.commit_group;
mov.b32 %r596, 0f00000000;
mov.b32 %r597, %r596;
mov.b32 %r598, %r596;
mov.b32 %r599, %r596;
mov.b32 %r600, %r596;
mov.b32 %r601, %r596;
mov.b32 %r602, %r596;
mov.b32 %r603, %r596;
mov.b32 %r604, %r596;
mov.b32 %r605, %r596;
mov.b32 %r606, %r596;
mov.b32 %r607, %r596;
mov.b32 %r608, %r596;
mov.b32 %r609, %r596;
mov.b32 %r610, %r596;
mov.b32 %r611, %r596;
mov.b32 %r612, %r596;
mov.b32 %r613, %r596;
mov.b32 %r614, %r596;
mov.b32 %r615, %r596;
mov.b32 %r616, %r596;
mov.b32 %r617, %r596;
mov.b32 %r618, %r596;
mov.b32 %r619, %r596;
mov.b32 %r620, %r596;
mov.b32 %r621, %r596;
mov.b32 %r622, %r596;
mov.b32 %r623, %r596;
mov.b32 %r624, %r596;
mov.b32 %r625, %r596;
mov.b32 %r626, %r596;
mov.b32 %r627, %r596;
mov.b32 %r628, %r596;
mov.b32 %r629, %r596;
mov.b32 %r630, %r596;
mov.b32 %r631, %r596;
mov.b32 %r632, %r596;
mov.b32 %r633, %r596;
mov.b32 %r634, %r596;
mov.b32 %r635, %r596;
mov.b32 %r636, %r596;
mov.b32 %r637, %r596;
mov.b32 %r638, %r596;
mov.b32 %r639, %r596;
mov.b32 %r640, %r596;
mov.b32 %r641, %r596;
mov.b32 %r642, %r596;
mov.b32 %r643, %r596;
mov.b32 %r644, %r596;
mov.b32 %r645, %r596;
mov.b32 %r646, %r596;
mov.b32 %r647, %r596;
mov.b32 %r648, %r596;
mov.b32 %r649, %r596;
mov.b32 %r650, %r596;
mov.b32 %r651, %r596;
mov.b32 %r652, %r596;
mov.b32 %r653, %r596;
mov.b32 %r654, %r596;
mov.b32 %r655, %r596;
mov.b32 %r656, %r596;
mov.b32 %r657, %r596;
mov.b32 %r658, %r596;
mov.b32 %r659, %r596;
@%p1 bra $L__BB0_3;
cvt.s64.s32 %rd2, %r77;
cvt.s64.s32 %rd3, %r78;
cvt.s64.s32 %rd4, %r79;
cvt.s64.s32 %rd5, %r80;
cvt.s64.s32 %rd6, %r81;
cvt.s64.s32 %rd7, %r82;
cvt.s64.s32 %rd8, %r83;
cvt.s64.s32 %rd9, %r84;
shr.s32 %r86, %r85, 31;
shr.u32 %r87, %r86, 26;
add.s32 %r88, %r85, %r87;
shr.s32 %r7, %r88, 6;
add.s32 %r11, %r7, -3;
shr.u32 %r12, %r4, 5;
add.s32 %r593, %r13, -192;
shl.b64 %rd44, %rd9, 1;
add.s64 %rd10, %rd44, 384;
shl.b64 %rd45, %rd8, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd7, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd6, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd5, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd4, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd3, 1;
add.s64 %rd16, %rd50, 384;
shl.b64 %rd51, %rd2, 1;
add.s64 %rd17, %rd51, 384;
mov.b32 %r101, 0;
mov.b32 %r596, 0f00000000;
mov.b32 %r595, 2;
mov.b32 %r594, -1;
mov.b64 %rd97, %rd1;
mov.b32 %r597, %r596;
mov.b32 %r598, %r596;
mov.b32 %r599, %r596;
mov.b32 %r600, %r596;
mov.b32 %r601, %r596;
mov.b32 %r602, %r596;
mov.b32 %r603, %r596;
mov.b32 %r604, %r596;
mov.b32 %r605, %r596;
mov.b32 %r606, %r596;
mov.b32 %r607, %r596;
mov.b32 %r608, %r596;
mov.b32 %r609, %r596;
mov.b32 %r610, %r596;
mov.b32 %r611, %r596;
mov.b32 %r612, %r596;
mov.b32 %r613, %r596;
mov.b32 %r614, %r596;
mov.b32 %r615, %r596;
mov.b32 %r616, %r596;
mov.b32 %r617, %r596;
mov.b32 %r618, %r596;
mov.b32 %r619, %r596;
mov.b32 %r620, %r596;
mov.b32 %r621, %r596;
mov.b32 %r622, %r596;
mov.b32 %r623, %r596;
mov.b32 %r624, %r596;
mov.b32 %r625, %r596;
mov.b32 %r626, %r596;
mov.b32 %r627, %r596;
mov.b32 %r628, %r596;
mov.b32 %r629, %r596;
mov.b32 %r630, %r596;
mov.b32 %r631, %r596;
mov.b32 %r632, %r596;
mov.b32 %r633, %r596;
mov.b32 %r634, %r596;
mov.b32 %r635, %r596;
mov.b32 %r636, %r596;
mov.b32 %r637, %r596;
mov.b32 %r638, %r596;
mov.b32 %r639, %r596;
mov.b32 %r640, %r596;
mov.b32 %r641, %r596;
mov.b32 %r642, %r596;
mov.b32 %r643, %r596;
mov.b32 %r644, %r596;
mov.b32 %r645, %r596;
mov.b32 %r646, %r596;
mov.b32 %r647, %r596;
mov.b32 %r648, %r596;
mov.b32 %r649, %r596;
mov.b32 %r650, %r596;
mov.b32 %r651, %r596;
mov.b32 %r652, %r596;
mov.b32 %r653, %r596;
mov.b32 %r654, %r596;
mov.b32 %r655, %r596;
mov.b32 %r656, %r596;
mov.b32 %r657, %r596;
mov.b32 %r658, %r596;
mov.b32 %r659, %r596;
mov.b32 %r660, %r101;
$L__BB0_2:
setp.lt.s32 %p9, %r660, %r11;
add.s32 %r111, %r594, 1;
setp.gt.s32 %p10, %r111, 3;
selp.b32 %r594, 0, %r111, %p10;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r112, %r594, 14;
add.s32 %r113, %r91, %r112;
add.s32 %r114, %r113, 65536;
shr.u32 %r115, %r113, 4;
shfl.sync.idx.b32 %r116, %r12, 0, 31, -1;
shl.b32 %r117, %r116, 7;
and.b32 %r118, %r117, 512;
add.s32 %r119, %r118, %r115;
and.b32 %r120, %r119, 16383;
cvt.u64.u32 %rd70, %r120;
bfe.u32 %r121, %r114, 4, 14;
cvt.u64.u32 %rd71, %r121;
wgmma.fence.sync.aligned;
or.b64 %rd52, %rd70, 4611686293305294848;
or.b64 %rd53, %rd71, 4611686293305294848;
mov.pred %p8, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659}, %rd52, %rd53, %p8, 1, 1, 0, 0;
add.s64 %rd54, %rd70, 4611686293305294850;
add.s64 %rd55, %rd71, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659}, %rd54, %rd55, %p8, 1, 1, 0, 0;
add.s64 %rd56, %rd70, 4611686293305294852;
add.s64 %rd57, %rd71, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659}, %rd56, %rd57, %p8, 1, 1, 0, 0;
add.s64 %rd58, %rd70, 4611686293305294854;
add.s64 %rd59, %rd71, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r596,%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659}, %rd58, %rd59, %p8, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd60, %r113;
cvt.u64.u32 %rd61, %r114;
mov.b32 %r100, %r101;
mov.b32 %r97, %r101;
mov.b32 %r98, %r101;
mov.b32 %r99, %r101;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd62, %rd97, %rd17;
add.s64 %rd63, %rd97, %rd16;
add.s64 %rd64, %rd97, %rd15;
add.s64 %rd65, %rd97, %rd14;
add.s64 %rd66, %rd97, %rd13;
add.s64 %rd67, %rd97, %rd12;
add.s64 %rd68, %rd97, %rd11;
add.s64 %rd69, %rd97, %rd10;
add.s32 %r122, %r595, 1;
setp.gt.s32 %p11, %r122, 3;
selp.b32 %r595, 0, %r122, %p11;
setp.lt.s32 %p12, %r6, %r593;
shl.b32 %r123, %r595, 14;
add.s32 %r124, %r91, %r123;
bar.sync 0;
add.s32 %r102, %r124, %r10;
selp.b32 %r125, 16, 0, %p12;
selp.b32 %r104, %r125, 0, %p9;
cp.async.cg.shared.global [ %r102 + 0 ], [ %rd62 + 0 ], 0x10, %r104;
add.s32 %r103, %r102, 4096;
cp.async.cg.shared.global [ %r103 + 0 ], [ %rd63 + 0 ], 0x10, %r104;
add.s32 %r105, %r102, 8192;
cp.async.cg.shared.global [ %r105 + 0 ], [ %rd64 + 0 ], 0x10, %r104;
add.s32 %r106, %r102, 12288;
cp.async.cg.shared.global [ %r106 + 0 ], [ %rd65 + 0 ], 0x10, %r104;
cp.async.commit_group;
add.s32 %r107, %r102, 65536;
cp.async.cg.shared.global [ %r107 + 0 ], [ %rd66 + 0 ], 0x10, %r104;
add.s32 %r108, %r102, 69632;
cp.async.cg.shared.global [ %r108 + 0 ], [ %rd67 + 0 ], 0x10, %r104;
add.s32 %r109, %r102, 73728;
cp.async.cg.shared.global [ %r109 + 0 ], [ %rd68 + 0 ], 0x10, %r104;
add.s32 %r110, %r102, 77824;
cp.async.cg.shared.global [ %r110 + 0 ], [ %rd69 + 0 ], 0x10, %r104;
cp.async.commit_group;
add.s32 %r660, %r660, 1;
add.s64 %rd97, %rd97, 128;
add.s32 %r593, %r593, -64;
setp.ne.b32 %p13, %r7, %r660;
@%p13 bra $L__BB0_2;
$L__BB0_3:
and.b32 %r223, %r4, 15;
shl.b32 %r224, %r223, 3;
or.b32 %r225, %r3, %r224;
and.b32 %r226, %r4, 240;
bfe.u32 %r227, %r4, 4, 4;
or.b32 %r228, %r227, 112;
or.b32 %r229, %r3, %r228;
or.b32 %r230, %r227, 96;
or.b32 %r231, %r3, %r230;
or.b32 %r232, %r227, 80;
or.b32 %r233, %r3, %r232;
or.b32 %r234, %r227, 64;
or.b32 %r235, %r3, %r234;
or.b32 %r236, %r227, 48;
or.b32 %r237, %r3, %r236;
or.b32 %r238, %r227, 32;
or.b32 %r239, %r3, %r238;
or.b32 %r240, %r227, 16;
or.b32 %r241, %r3, %r240;
or.b32 %r242, %r3, %r227;
or.b32 %r243, %r2, %r224;
or.b32 %r244, %r2, %r228;
or.b32 %r245, %r2, %r230;
or.b32 %r246, %r2, %r232;
or.b32 %r247, %r2, %r234;
or.b32 %r248, %r2, %r236;
or.b32 %r249, %r2, %r238;
or.b32 %r250, %r2, %r240;
or.b32 %r251, %r2, %r227;
mul.lo.s32 %r252, %r15, %r1;
mad.wide.s32 %rd96, %r252, 2, %rd18;
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
mad.lo.s32 %r253, %r14, %r251, %r225;
mad.lo.s32 %r254, %r14, %r250, %r225;
mad.lo.s32 %r255, %r14, %r249, %r225;
mad.lo.s32 %r256, %r14, %r248, %r225;
mad.lo.s32 %r257, %r14, %r247, %r225;
mad.lo.s32 %r258, %r14, %r246, %r225;
mad.lo.s32 %r259, %r14, %r245, %r225;
mad.lo.s32 %r260, %r14, %r244, %r225;
mad.wide.s32 %rd72, %r253, 2, %rd1;
mad.wide.s32 %rd73, %r254, 2, %rd1;
mad.wide.s32 %rd74, %r255, 2, %rd1;
mad.wide.s32 %rd75, %r256, 2, %rd1;
mad.wide.s32 %rd76, %r257, 2, %rd1;
mad.wide.s32 %rd77, %r258, 2, %rd1;
mad.wide.s32 %rd78, %r259, 2, %rd1;
mad.wide.s32 %rd79, %r260, 2, %rd1;
setp.lt.s32 %p30, %r251, %r13;
setp.lt.s32 %p31, %r250, %r13;
setp.lt.s32 %p32, %r249, %r13;
setp.lt.s32 %p33, %r248, %r13;
setp.lt.s32 %p34, %r247, %r13;
setp.lt.s32 %p35, %r246, %r13;
setp.lt.s32 %p36, %r245, %r13;
setp.lt.s32 %p37, %r244, %r13;
setp.lt.s32 %p38, %r225, %r13;
and.pred %p14, %p30, %p38;
and.pred %p15, %p31, %p38;
and.pred %p16, %p32, %p38;
and.pred %p17, %p33, %p38;
and.pred %p18, %p34, %p38;
and.pred %p19, %p35, %p38;
and.pred %p20, %p36, %p38;
and.pred %p21, %p37, %p38;
mov.b32 %r130, 0;
mov.u32 %r126, %r130;
mov.u32 %r127, %r130;
mov.u32 %r128, %r130;
mov.u32 %r129, %r130;
@%p14 ld.global.v4.b32 { %r126, %r127, %r128, %r129 }, [ %rd72 + 0 ];
mov.u32 %r131, %r130;
mov.u32 %r132, %r130;
mov.u32 %r133, %r130;
mov.u32 %r134, %r130;
@%p15 ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd73 + 0 ];
mov.u32 %r135, %r130;
mov.u32 %r136, %r130;
mov.u32 %r137, %r130;
mov.u32 %r138, %r130;
@%p16 ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd74 + 0 ];
mov.u32 %r139, %r130;
mov.u32 %r140, %r130;
mov.u32 %r141, %r130;
mov.u32 %r142, %r130;
@%p17 ld.global.v4.b32 { %r139, %r140, %r141, %r142 }, [ %rd75 + 0 ];
mov.u32 %r143, %r130;
mov.u32 %r144, %r130;
mov.u32 %r145, %r130;
mov.u32 %r146, %r130;
@%p18 ld.global.v4.b32 { %r143, %r144, %r145, %r146 }, [ %rd76 + 0 ];
mov.u32 %r147, %r130;
mov.u32 %r148, %r130;
mov.u32 %r149, %r130;
mov.u32 %r150, %r130;
@%p19 ld.global.v4.b32 { %r147, %r148, %r149, %r150 }, [ %rd77 + 0 ];
mov.u32 %r151, %r130;
mov.u32 %r152, %r130;
mov.u32 %r153, %r130;
mov.u32 %r154, %r130;
@%p20 ld.global.v4.b32 { %r151, %r152, %r153, %r154 }, [ %rd78 + 0 ];
mov.u32 %r155, %r130;
mov.u32 %r156, %r130;
mov.u32 %r157, %r130;
mov.u32 %r158, %r130;
@%p21 ld.global.v4.b32 { %r155, %r156, %r157, %r158 }, [ %rd79 + 0 ];
shl.b32 %r261, %r4, 8;
and.b32 %r262, %r261, 30720;
shl.b32 %r263, %r5, 4;
or.b32 %r264, %r262, %r263;
xor.b32 %r265, %r264, %r226;
add.s32 %r266, %r91, %r265;
st.shared.v4.b32 [%r266], {%r126, %r127, %r128, %r129};
st.shared.v4.b32 [%r266+256], {%r131, %r132, %r133, %r134};
st.shared.v4.b32 [%r266+512], {%r135, %r136, %r137, %r138};
st.shared.v4.b32 [%r266+768], {%r139, %r140, %r141, %r142};
st.shared.v4.b32 [%r266+1024], {%r143, %r144, %r145, %r146};
st.shared.v4.b32 [%r266+1280], {%r147, %r148, %r149, %r150};
st.shared.v4.b32 [%r266+1536], {%r151, %r152, %r153, %r154};
st.shared.v4.b32 [%r266+1792], {%r155, %r156, %r157, %r158};
bar.sync 0;
shl.b32 %r267, %r5, 12;
shl.b32 %r268, %r223, 4;
and.b32 %r269, %r4, 224;
shl.b32 %r270, %r269, 3;
and.b32 %r271, %r4, 16;
or.b32 %r272, %r268, %r270;
xor.b32 %r273, %r272, %r271;
or.b32 %r274, %r273, %r267;
add.s32 %r275, %r91, %r274;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r276, %r277, %r278, %r279}, [%r275];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r280, %r281, %r282, %r283}, [%r275+2048];
xor.b32 %r284, %r274, 32;
add.s32 %r285, %r91, %r284;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r286, %r287, %r288, %r289}, [%r285];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r290, %r291, %r292, %r293}, [%r285+2048];
xor.b32 %r294, %r274, 64;
add.s32 %r295, %r91, %r294;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r296, %r297, %r298, %r299}, [%r295];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r300, %r301, %r302, %r303}, [%r295+2048];
xor.b32 %r304, %r274, 96;
add.s32 %r305, %r91, %r304;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r306, %r307, %r308, %r309}, [%r305];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r310, %r311, %r312, %r313}, [%r305+2048];
mov.b32 {%rs1, %rs2}, %r276;
cvt.f32.bf16 %r314, %rs1;
cvt.f32.bf16 %r315, %rs2;
mul.f32 %r316, %r18, %r315;
mul.f32 %r317, %r18, %r314;
fma.rn.f32 %r318, %r17, %r596, %r317;
fma.rn.f32 %r319, %r17, %r597, %r316;
cvt.rn.bf16x2.f32 %r320, %r319, %r318;
mov.b32 {%rs3, %rs4}, %r277;
cvt.f32.bf16 %r321, %rs3;
cvt.f32.bf16 %r322, %rs4;
mul.f32 %r323, %r18, %r322;
mul.f32 %r324, %r18, %r321;
fma.rn.f32 %r325, %r17, %r598, %r324;
fma.rn.f32 %r326, %r17, %r599, %r323;
cvt.rn.bf16x2.f32 %r327, %r326, %r325;
mov.b32 {%rs5, %rs6}, %r278;
cvt.f32.bf16 %r328, %rs5;
cvt.f32.bf16 %r329, %rs6;
mul.f32 %r330, %r18, %r329;
mul.f32 %r331, %r18, %r328;
fma.rn.f32 %r332, %r17, %r600, %r331;
fma.rn.f32 %r333, %r17, %r601, %r330;
cvt.rn.bf16x2.f32 %r334, %r333, %r332;
mov.b32 {%rs7, %rs8}, %r279;
cvt.f32.bf16 %r335, %rs7;
cvt.f32.bf16 %r336, %rs8;
mul.f32 %r337, %r18, %r336;
mul.f32 %r338, %r18, %r335;
fma.rn.f32 %r339, %r17, %r602, %r338;
fma.rn.f32 %r340, %r17, %r603, %r337;
cvt.rn.bf16x2.f32 %r341, %r340, %r339;
mov.b32 {%rs9, %rs10}, %r286;
cvt.f32.bf16 %r342, %rs9;
cvt.f32.bf16 %r343, %rs10;
mul.f32 %r344, %r18, %r343;
mul.f32 %r345, %r18, %r342;
fma.rn.f32 %r346, %r17, %r604, %r345;
fma.rn.f32 %r347, %r17, %r605, %r344;
cvt.rn.bf16x2.f32 %r348, %r347, %r346;
mov.b32 {%rs11, %rs12}, %r287;
cvt.f32.bf16 %r349, %rs11;
cvt.f32.bf16 %r350, %rs12;
mul.f32 %r351, %r18, %r350;
mul.f32 %r352, %r18, %r349;
fma.rn.f32 %r353, %r17, %r606, %r352;
fma.rn.f32 %r354, %r17, %r607, %r351;
cvt.rn.bf16x2.f32 %r355, %r354, %r353;
mov.b32 {%rs13, %rs14}, %r288;
cvt.f32.bf16 %r356, %rs13;
cvt.f32.bf16 %r357, %rs14;
mul.f32 %r358, %r18, %r357;
mul.f32 %r359, %r18, %r356;
fma.rn.f32 %r360, %r17, %r608, %r359;
fma.rn.f32 %r361, %r17, %r609, %r358;
cvt.rn.bf16x2.f32 %r362, %r361, %r360;
mov.b32 {%rs15, %rs16}, %r289;
cvt.f32.bf16 %r363, %rs15;
cvt.f32.bf16 %r364, %rs16;
mul.f32 %r365, %r18, %r364;
mul.f32 %r366, %r18, %r363;
fma.rn.f32 %r367, %r17, %r610, %r366;
fma.rn.f32 %r368, %r17, %r611, %r365;
cvt.rn.bf16x2.f32 %r369, %r368, %r367;
mov.b32 {%rs17, %rs18}, %r296;
cvt.f32.bf16 %r370, %rs17;
cvt.f32.bf16 %r371, %rs18;
mul.f32 %r372, %r18, %r371;
mul.f32 %r373, %r18, %r370;
fma.rn.f32 %r374, %r17, %r612, %r373;
fma.rn.f32 %r375, %r17, %r613, %r372;
cvt.rn.bf16x2.f32 %r376, %r375, %r374;
mov.b32 {%rs19, %rs20}, %r297;
cvt.f32.bf16 %r377, %rs19;
cvt.f32.bf16 %r378, %rs20;
mul.f32 %r379, %r18, %r378;
mul.f32 %r380, %r18, %r377;
fma.rn.f32 %r381, %r17, %r614, %r380;
fma.rn.f32 %r382, %r17, %r615, %r379;
cvt.rn.bf16x2.f32 %r383, %r382, %r381;
mov.b32 {%rs21, %rs22}, %r298;
cvt.f32.bf16 %r384, %rs21;
cvt.f32.bf16 %r385, %rs22;
mul.f32 %r386, %r18, %r385;
mul.f32 %r387, %r18, %r384;
fma.rn.f32 %r388, %r17, %r616, %r387;
fma.rn.f32 %r389, %r17, %r617, %r386;
cvt.rn.bf16x2.f32 %r390, %r389, %r388;
mov.b32 {%rs23, %rs24}, %r299;
cvt.f32.bf16 %r391, %rs23;
cvt.f32.bf16 %r392, %rs24;
mul.f32 %r393, %r18, %r392;
mul.f32 %r394, %r18, %r391;
fma.rn.f32 %r395, %r17, %r618, %r394;
fma.rn.f32 %r396, %r17, %r619, %r393;
cvt.rn.bf16x2.f32 %r397, %r396, %r395;
mov.b32 {%rs25, %rs26}, %r306;
cvt.f32.bf16 %r398, %rs25;
cvt.f32.bf16 %r399, %rs26;
mul.f32 %r400, %r18, %r399;
mul.f32 %r401, %r18, %r398;
fma.rn.f32 %r402, %r17, %r620, %r401;
fma.rn.f32 %r403, %r17, %r621, %r400;
cvt.rn.bf16x2.f32 %r404, %r403, %r402;
mov.b32 {%rs27, %rs28}, %r307;
cvt.f32.bf16 %r405, %rs27;
cvt.f32.bf16 %r406, %rs28;
mul.f32 %r407, %r18, %r406;
mul.f32 %r408, %r18, %r405;
fma.rn.f32 %r409, %r17, %r622, %r408;
fma.rn.f32 %r410, %r17, %r623, %r407;
cvt.rn.bf16x2.f32 %r411, %r410, %r409;
mov.b32 {%rs29, %rs30}, %r308;
cvt.f32.bf16 %r412, %rs29;
cvt.f32.bf16 %r413, %rs30;
mul.f32 %r414, %r18, %r413;
mul.f32 %r415, %r18, %r412;
fma.rn.f32 %r416, %r17, %r624, %r415;
fma.rn.f32 %r417, %r17, %r625, %r414;
cvt.rn.bf16x2.f32 %r418, %r417, %r416;
mov.b32 {%rs31, %rs32}, %r309;
cvt.f32.bf16 %r419, %rs31;
cvt.f32.bf16 %r420, %rs32;
mul.f32 %r421, %r18, %r420;
mul.f32 %r422, %r18, %r419;
fma.rn.f32 %r423, %r17, %r626, %r422;
fma.rn.f32 %r424, %r17, %r627, %r421;
cvt.rn.bf16x2.f32 %r425, %r424, %r423;
mov.b32 {%rs33, %rs34}, %r280;
cvt.f32.bf16 %r426, %rs33;
cvt.f32.bf16 %r427, %rs34;
mul.f32 %r428, %r18, %r427;
mul.f32 %r429, %r18, %r426;
fma.rn.f32 %r430, %r17, %r628, %r429;
fma.rn.f32 %r431, %r17, %r629, %r428;
cvt.rn.bf16x2.f32 %r432, %r431, %r430;
mov.b32 {%rs35, %rs36}, %r281;
cvt.f32.bf16 %r433, %rs35;
cvt.f32.bf16 %r434, %rs36;
mul.f32 %r435, %r18, %r434;
mul.f32 %r436, %r18, %r433;
fma.rn.f32 %r437, %r17, %r630, %r436;
fma.rn.f32 %r438, %r17, %r631, %r435;
cvt.rn.bf16x2.f32 %r439, %r438, %r437;
mov.b32 {%rs37, %rs38}, %r282;
cvt.f32.bf16 %r440, %rs37;
cvt.f32.bf16 %r441, %rs38;
mul.f32 %r442, %r18, %r441;
mul.f32 %r443, %r18, %r440;
fma.rn.f32 %r444, %r17, %r632, %r443;
fma.rn.f32 %r445, %r17, %r633, %r442;
cvt.rn.bf16x2.f32 %r446, %r445, %r444;
mov.b32 {%rs39, %rs40}, %r283;
cvt.f32.bf16 %r447, %rs39;
cvt.f32.bf16 %r448, %rs40;
mul.f32 %r449, %r18, %r448;
mul.f32 %r450, %r18, %r447;
fma.rn.f32 %r451, %r17, %r634, %r450;
fma.rn.f32 %r452, %r17, %r635, %r449;
cvt.rn.bf16x2.f32 %r453, %r452, %r451;
mov.b32 {%rs41, %rs42}, %r290;
cvt.f32.bf16 %r454, %rs41;
cvt.f32.bf16 %r455, %rs42;
mul.f32 %r456, %r18, %r455;
mul.f32 %r457, %r18, %r454;
fma.rn.f32 %r458, %r17, %r636, %r457;
fma.rn.f32 %r459, %r17, %r637, %r456;
cvt.rn.bf16x2.f32 %r460, %r459, %r458;
mov.b32 {%rs43, %rs44}, %r291;
cvt.f32.bf16 %r461, %rs43;
cvt.f32.bf16 %r462, %rs44;
mul.f32 %r463, %r18, %r462;
mul.f32 %r464, %r18, %r461;
fma.rn.f32 %r465, %r17, %r638, %r464;
fma.rn.f32 %r466, %r17, %r639, %r463;
cvt.rn.bf16x2.f32 %r467, %r466, %r465;
mov.b32 {%rs45, %rs46}, %r292;
cvt.f32.bf16 %r468, %rs45;
cvt.f32.bf16 %r469, %rs46;
mul.f32 %r470, %r18, %r469;
mul.f32 %r471, %r18, %r468;
fma.rn.f32 %r472, %r17, %r640, %r471;
fma.rn.f32 %r473, %r17, %r641, %r470;
cvt.rn.bf16x2.f32 %r474, %r473, %r472;
mov.b32 {%rs47, %rs48}, %r293;
cvt.f32.bf16 %r475, %rs47;
cvt.f32.bf16 %r476, %rs48;
mul.f32 %r477, %r18, %r476;
mul.f32 %r478, %r18, %r475;
fma.rn.f32 %r479, %r17, %r642, %r478;
fma.rn.f32 %r480, %r17, %r643, %r477;
cvt.rn.bf16x2.f32 %r481, %r480, %r479;
mov.b32 {%rs49, %rs50}, %r300;
cvt.f32.bf16 %r482, %rs49;
cvt.f32.bf16 %r483, %rs50;
mul.f32 %r484, %r18, %r483;
mul.f32 %r485, %r18, %r482;
fma.rn.f32 %r486, %r17, %r644, %r485;
fma.rn.f32 %r487, %r17, %r645, %r484;
cvt.rn.bf16x2.f32 %r488, %r487, %r486;
mov.b32 {%rs51, %rs52}, %r301;
cvt.f32.bf16 %r489, %rs51;
cvt.f32.bf16 %r490, %rs52;
mul.f32 %r491, %r18, %r490;
mul.f32 %r492, %r18, %r489;
fma.rn.f32 %r493, %r17, %r646, %r492;
fma.rn.f32 %r494, %r17, %r647, %r491;
cvt.rn.bf16x2.f32 %r495, %r494, %r493;
mov.b32 {%rs53, %rs54}, %r302;
cvt.f32.bf16 %r496, %rs53;
cvt.f32.bf16 %r497, %rs54;
mul.f32 %r498, %r18, %r497;
mul.f32 %r499, %r18, %r496;
fma.rn.f32 %r500, %r17, %r648, %r499;
fma.rn.f32 %r501, %r17, %r649, %r498;
cvt.rn.bf16x2.f32 %r502, %r501, %r500;
mov.b32 {%rs55, %rs56}, %r303;
cvt.f32.bf16 %r503, %rs55;
cvt.f32.bf16 %r504, %rs56;
mul.f32 %r505, %r18, %r504;
mul.f32 %r506, %r18, %r503;
fma.rn.f32 %r507, %r17, %r650, %r506;
fma.rn.f32 %r508, %r17, %r651, %r505;
cvt.rn.bf16x2.f32 %r509, %r508, %r507;
mov.b32 {%rs57, %rs58}, %r310;
cvt.f32.bf16 %r510, %rs57;
cvt.f32.bf16 %r511, %rs58;
mul.f32 %r512, %r18, %r511;
mul.f32 %r513, %r18, %r510;
fma.rn.f32 %r514, %r17, %r652, %r513;
fma.rn.f32 %r515, %r17, %r653, %r512;
cvt.rn.bf16x2.f32 %r516, %r515, %r514;
mov.b32 {%rs59, %rs60}, %r311;
cvt.f32.bf16 %r517, %rs59;
cvt.f32.bf16 %r518, %rs60;
mul.f32 %r519, %r18, %r518;
mul.f32 %r520, %r18, %r517;
fma.rn.f32 %r521, %r17, %r654, %r520;
fma.rn.f32 %r522, %r17, %r655, %r519;
cvt.rn.bf16x2.f32 %r523, %r522, %r521;
mov.b32 {%rs61, %rs62}, %r312;
cvt.f32.bf16 %r524, %rs61;
cvt.f32.bf16 %r525, %rs62;
mul.f32 %r526, %r18, %r525;
mul.f32 %r527, %r18, %r524;
fma.rn.f32 %r528, %r17, %r656, %r527;
fma.rn.f32 %r529, %r17, %r657, %r526;
cvt.rn.bf16x2.f32 %r530, %r529, %r528;
mov.b32 {%rs63, %rs64}, %r313;
cvt.f32.bf16 %r531, %rs63;
cvt.f32.bf16 %r532, %rs64;
mul.f32 %r533, %r18, %r532;
mul.f32 %r534, %r18, %r531;
fma.rn.f32 %r535, %r17, %r658, %r534;
fma.rn.f32 %r536, %r17, %r659, %r533;
cvt.rn.bf16x2.f32 %r537, %r536, %r535;
bar.sync 0;
shl.b32 %r538, %r4, 5;
and.b32 %r539, %r538, 3072;
and.b32 %r540, %r8, 368;
bfe.s32 %r541, %r4, 3, 1;
and.b32 %r542, %r541, 528;
or.b32 %r543, %r539, %r540;
or.b32 %r544, %r542, %r269;
xor.b32 %r545, %r544, %r543;
add.s32 %r546, %r91, %r545;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r320, %r327, %r334, %r341};
bar.sync 0;
shl.b32 %r547, %r5, 9;
and.b32 %r548, %r4, 112;
and.b32 %r549, %r9, 256;
xor.b32 %r550, %r268, %r548;
add.s32 %r551, %r91, %r547;
add.s32 %r552, %r551, %r549;
add.s32 %r553, %r552, %r550;
ld.shared.v4.b32 {%r191, %r192, %r193, %r194}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r348, %r355, %r362, %r369};
bar.sync 0;
ld.shared.v4.b32 {%r195, %r196, %r197, %r198}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r376, %r383, %r390, %r397};
bar.sync 0;
ld.shared.v4.b32 {%r199, %r200, %r201, %r202}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r404, %r411, %r418, %r425};
bar.sync 0;
ld.shared.v4.b32 {%r203, %r204, %r205, %r206}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r432, %r439, %r446, %r453};
bar.sync 0;
ld.shared.v4.b32 {%r207, %r208, %r209, %r210}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r460, %r467, %r474, %r481};
bar.sync 0;
ld.shared.v4.b32 {%r211, %r212, %r213, %r214}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r488, %r495, %r502, %r509};
bar.sync 0;
ld.shared.v4.b32 {%r215, %r216, %r217, %r218}, [%r553];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r546], {%r516, %r523, %r530, %r537};
bar.sync 0;
ld.shared.v4.b32 {%r219, %r220, %r221, %r222}, [%r553];
bar.sync 0;
and.b32 %r554, %r4, 3;
shl.b32 %r555, %r554, 13;
and.b32 %r556, %r538, 7264;
and.b32 %r557, %r4, 24;
shl.b32 %r558, %r557, 4;
shl.b32 %r559, %r4, 2;
and.b32 %r560, %r559, 16;
or.b32 %r561, %r555, %r560;
or.b32 %r562, %r556, %r558;
or.b32 %r563, %r561, %r562;
add.s32 %r564, %r91, %r563;
st.shared.v4.b32 [%r564], {%r320, %r334, %r348, %r362};
st.shared.v4.b32 [%r564+512], {%r327, %r341, %r355, %r369};
xor.b32 %r565, %r563, 32;
add.s32 %r566, %r91, %r565;
st.shared.v4.b32 [%r566], {%r376, %r390, %r404, %r418};
st.shared.v4.b32 [%r566+512], {%r383, %r397, %r411, %r425};
xor.b32 %r567, %r563, 64;
add.s32 %r568, %r91, %r567;
st.shared.v4.b32 [%r568], {%r432, %r446, %r460, %r474};
st.shared.v4.b32 [%r568+512], {%r439, %r453, %r467, %r481};
xor.b32 %r569, %r563, 96;
add.s32 %r570, %r91, %r569;
st.shared.v4.b32 [%r570], {%r488, %r502, %r516, %r530};
st.shared.v4.b32 [%r570+512], {%r495, %r509, %r523, %r537};
bar.sync 0;
shl.b32 %r571, %r557, 10;
shl.b32 %r572, %r554, 5;
and.b32 %r573, %r559, 1008;
or.b32 %r574, %r571, %r572;
xor.b32 %r575, %r574, %r573;
add.s32 %r576, %r91, %r575;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r161, %r162}, [%r576];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r165, %r166}, [%r576+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r169, %r170}, [%r576+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r171, %r172, %r173, %r174}, [%r576+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r175, %r176, %r177, %r178}, [%r576+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r179, %r180, %r181, %r182}, [%r576+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r183, %r184, %r185, %r186}, [%r576+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r187, %r188, %r189, %r190}, [%r576+7168];
mad.lo.s32 %r577, %r16, %r251, %r225;
mad.lo.s32 %r578, %r16, %r250, %r225;
mad.lo.s32 %r579, %r16, %r249, %r225;
mad.lo.s32 %r580, %r16, %r248, %r225;
mad.lo.s32 %r581, %r16, %r247, %r225;
mad.lo.s32 %r582, %r16, %r246, %r225;
mad.lo.s32 %r583, %r16, %r245, %r225;
mad.lo.s32 %r584, %r16, %r244, %r225;
mad.wide.s32 %rd80, %r577, 2, %rd96;
mad.wide.s32 %rd81, %r578, 2, %rd96;
mad.wide.s32 %rd82, %r579, 2, %rd96;
mad.wide.s32 %rd83, %r580, 2, %rd96;
mad.wide.s32 %rd84, %r581, 2, %rd96;
mad.wide.s32 %rd85, %r582, 2, %rd96;
mad.wide.s32 %rd86, %r583, 2, %rd96;
mad.wide.s32 %rd87, %r584, 2, %rd96;
@%p14 st.global.v4.b32 [ %rd80 + 0 ], { %r159, %r160, %r161, %r162 };
@%p15 st.global.v4.b32 [ %rd81 + 0 ], { %r163, %r164, %r165, %r166 };
@%p16 st.global.v4.b32 [ %rd82 + 0 ], { %r167, %r168, %r169, %r170 };
@%p17 st.global.v4.b32 [ %rd83 + 0 ], { %r171, %r172, %r173, %r174 };
@%p18 st.global.v4.b32 [ %rd84 + 0 ], { %r175, %r176, %r177, %r178 };
@%p19 st.global.v4.b32 [ %rd85 + 0 ], { %r179, %r180, %r181, %r182 };
@%p20 st.global.v4.b32 [ %rd86 + 0 ], { %r183, %r184, %r185, %r186 };
@%p21 st.global.v4.b32 [ %rd87 + 0 ], { %r187, %r188, %r189, %r190 };
mad.lo.s32 %r585, %r16, %r242, %r243;
mad.lo.s32 %r586, %r16, %r241, %r243;
mad.lo.s32 %r587, %r16, %r239, %r243;
mad.lo.s32 %r588, %r16, %r237, %r243;
mad.lo.s32 %r589, %r16, %r235, %r243;
mad.lo.s32 %r590, %r16, %r233, %r243;
mad.lo.s32 %r591, %r16, %r231, %r243;
mad.lo.s32 %r592, %r16, %r229, %r243;
mad.wide.s32 %rd88, %r585, 2, %rd96;
mad.wide.s32 %rd89, %r586, 2, %rd96;
mad.wide.s32 %rd90, %r587, 2, %rd96;
mad.wide.s32 %rd91, %r588, 2, %rd96;
mad.wide.s32 %rd92, %r589, 2, %rd96;
mad.wide.s32 %rd93, %r590, 2, %rd96;
mad.wide.s32 %rd94, %r591, 2, %rd96;
mad.wide.s32 %rd95, %r592, 2, %rd96;
setp.lt.s32 %p39, %r242, %r13;
setp.lt.s32 %p40, %r241, %r13;
setp.lt.s32 %p41, %r239, %r13;
setp.lt.s32 %p42, %r237, %r13;
setp.lt.s32 %p43, %r235, %r13;
setp.lt.s32 %p44, %r233, %r13;
setp.lt.s32 %p45, %r231, %r13;
setp.lt.s32 %p46, %r229, %r13;
setp.lt.s32 %p47, %r243, %r13;
and.pred %p22, %p39, %p47;
and.pred %p23, %p40, %p47;
and.pred %p24, %p41, %p47;
and.pred %p25, %p42, %p47;
and.pred %p26, %p43, %p47;
and.pred %p27, %p44, %p47;
and.pred %p28, %p45, %p47;
and.pred %p29, %p46, %p47;
@%p22 st.global.v4.b32 [ %rd88 + 0 ], { %r191, %r192, %r193, %r194 };
@%p23 st.global.v4.b32 [ %rd89 + 0 ], { %r195, %r196, %r197, %r198 };
@%p24 st.global.v4.b32 [ %rd90 + 0 ], { %r199, %r200, %r201, %r202 };
@%p25 st.global.v4.b32 [ %rd91 + 0 ], { %r203, %r204, %r205, %r206 };
@%p26 st.global.v4.b32 [ %rd92 + 0 ], { %r207, %r208, %r209, %r210 };
@%p27 st.global.v4.b32 [ %rd93 + 0 ], { %r211, %r212, %r213, %r214 };
@%p28 st.global.v4.b32 [ %rd94 + 0 ], { %r215, %r216, %r217, %r218 };
@%p29 st.global.v4.b32 [ %rd95 + 0 ], { %r219, %r220, %r221, %r222 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .qkPolynomial, .full => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry polar(
.param .align 64 .b8 polar_param_0[128],
.param .u32 polar_param_1,
.param .u32 polar_param_2,
.param .u64 polar_param_3,
.param .u64 polar_param_4,
.param .align 64 .b8 polar_param_5[128],
.param .u32 polar_param_6,
.param .u32 polar_param_7,
.param .u64 polar_param_8,
.param .u64 polar_param_9,
.param .u64 .ptr .global .align 1 polar_param_10,
.param .f32 polar_param_11,
.param .f32 polar_param_12,
.param .u64 .ptr .global .align 1 polar_param_13,
.param .u64 .ptr .global .align 1 polar_param_14
)
.reqntid 256
{
.reg .pred %p<32>;
.reg .b16 %rs<65>;
.reg .b32 %r<521>;
.reg .b64 %rd<51>;
$L__func_begin0:
ld.param.b32 %r47, [polar_param_12];
ld.param.b32 %r46, [polar_param_11];
ld.param.b64 %rd1, [polar_param_10];
mov.b64 %rd2, polar_param_0;
mov.b64 %rd3, polar_param_5;
$L__tmp0:
cvta.param.u64 %rd50, %rd3;
cvta.param.u64 %rd22, %rd2;
mov.u32 %r49, %ctaid.x;
mul.hi.u32 %r1, %r49, 1431655766;
$L__tmp1:
mul.lo.s32 %r50, %r1, 3;
sub.s32 %r51, %r49, %r50;
mul.lo.s32 %r52, %r51, 3;
mov.b32 %r53, 9216;
shr.u32 %r54, %r53, %r52;
and.b32 %r2, %r54, 896;
mov.b32 %r55, 8192;
shr.u32 %r56, %r55, %r52;
and.b32 %r3, %r56, 896;
$L__tmp2:
setp.eq.b32 %p4, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p3, %r5, 0;
mov.b32 %r57, global_smem;
add.s32 %r90, %r57, 98304;
@%p3 mbarrier.init.shared::cta.b64 [%r90], 1;
bar.sync 0;
add.s32 %r48, %r57, 98312;
@%p3 mbarrier.init.shared::cta.b64 [%r48], 1;
shl.b32 %r6, %r1, 8;
$L__tmp3:
add.s32 %r439, %r2, %r6;
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p29, %r5, 64;
add.s32 %r74, %r3, %r6;
$L__tmp4:
@%p4 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r90], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r94, %r7, 0, 31, -1;
elect.sync %r95|%p18, -1;
and.pred %p17, %p29, %p18;
and.b32 %r96, %r94, 1;
shl.b32 %r97, %r96, 14;
add.s32 %r91, %r57, %r97;
shl.b32 %r92, %r96, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r91], [%rd22, {%r92, %r439}], [%r90];
shl.b32 %r98, %r94, 7;
and.b32 %r9, %r98, 512;
mov.b32 %r93, 0;
mov.pred %p19, -1;
mov.b32 %r455, 0f00000000;
mov.b32 %r456, %r455;
mov.b32 %r457, %r455;
mov.b32 %r458, %r455;
mov.b32 %r459, %r455;
mov.b32 %r460, %r455;
mov.b32 %r461, %r455;
mov.b32 %r462, %r455;
mov.b32 %r463, %r455;
mov.b32 %r464, %r455;
mov.b32 %r465, %r455;
mov.b32 %r466, %r455;
mov.b32 %r467, %r455;
mov.b32 %r468, %r455;
mov.b32 %r469, %r455;
mov.b32 %r470, %r455;
mov.b32 %r471, %r455;
mov.b32 %r472, %r455;
mov.b32 %r473, %r455;
mov.b32 %r474, %r455;
mov.b32 %r475, %r455;
mov.b32 %r476, %r455;
mov.b32 %r477, %r455;
mov.b32 %r478, %r455;
mov.b32 %r479, %r455;
mov.b32 %r480, %r455;
mov.b32 %r481, %r455;
mov.b32 %r482, %r455;
mov.b32 %r483, %r455;
mov.b32 %r484, %r455;
mov.b32 %r485, %r455;
mov.b32 %r486, %r455;
mov.b32 %r487, %r455;
mov.b32 %r488, %r455;
mov.b32 %r489, %r455;
mov.b32 %r490, %r455;
mov.b32 %r491, %r455;
mov.b32 %r492, %r455;
mov.b32 %r493, %r455;
mov.b32 %r494, %r455;
mov.b32 %r495, %r455;
mov.b32 %r496, %r455;
mov.b32 %r497, %r455;
mov.b32 %r498, %r455;
mov.b32 %r499, %r455;
mov.b32 %r500, %r455;
mov.b32 %r501, %r455;
mov.b32 %r502, %r455;
mov.b32 %r503, %r455;
mov.b32 %r504, %r455;
mov.b32 %r505, %r455;
mov.b32 %r506, %r455;
mov.b32 %r507, %r455;
mov.b32 %r508, %r455;
mov.b32 %r509, %r455;
mov.b32 %r510, %r455;
mov.b32 %r511, %r455;
mov.b32 %r512, %r455;
mov.b32 %r513, %r455;
mov.b32 %r514, %r455;
mov.b32 %r515, %r455;
mov.b32 %r516, %r455;
mov.b32 %r517, %r455;
mov.b32 %r518, %r455;
mov.pred %p31, %p19;
mov.b32 %r520, %r93;
$L__tmp6:
$L__BB0_4:
mov.pred %p2, %p31;
shl.b32 %r103, %r520, 15;
add.s32 %r104, %r57, %r103;
mad.lo.s32 %r105, %r520, -32760, %r104;
add.s32 %r99, %r105, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r99], %r93;
@!complete bra.uni waitLoop;
}
shr.u32 %r106, %r104, 4;
add.s32 %r107, %r106, %r9;
and.b32 %r108, %r107, 16383;
cvt.u64.u32 %rd39, %r108;
bfe.u32 %r109, %r104, 4, 14;
cvt.u64.u32 %rd40, %r109;
wgmma.fence.sync.aligned;
or.b64 %rd23, %rd39, 4611686293305294848;
or.b64 %rd24, %rd40, 4611686293305294848;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd23, %rd24, %p19, 1, 1, 0, 0;
add.s64 %rd25, %rd39, 4611686293305294850;
add.s64 %rd26, %rd40, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd25, %rd26, %p19, 1, 1, 0, 0;
add.s64 %rd27, %rd39, 4611686293305294852;
add.s64 %rd28, %rd40, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd27, %rd28, %p19, 1, 1, 0, 0;
add.s64 %rd29, %rd39, 4611686293305294854;
add.s64 %rd30, %rd40, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd29, %rd30, %p19, 1, 1, 0, 0;
add.s64 %rd31, %rd39, 4611686293305295872;
add.s64 %rd32, %rd40, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd31, %rd32, %p19, 1, 1, 0, 0;
add.s64 %rd33, %rd39, 4611686293305295874;
add.s64 %rd34, %rd40, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd33, %rd34, %p19, 1, 1, 0, 0;
add.s64 %rd35, %rd39, 4611686293305295876;
add.s64 %rd36, %rd40, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd35, %rd36, %p19, 1, 1, 0, 0;
add.s64 %rd37, %rd39, 4611686293305295878;
add.s64 %rd38, %rd40, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd37, %rd38, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r110, %r520, 1;
and.b32 %r111, %r110, 1;
shl.b32 %r112, %r110, 7;
shl.b32 %r113, %r111, 3;
add.s32 %r114, %r57, %r113;
add.s32 %r100, %r114, 98304;
shl.b32 %r115, %r111, 15;
$L__tmp7:
bar.sync 0;
or.b32 %r116, %r520, %r5;
setp.eq.b32 %p20, %r116, 0;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r100], 32768;
bar.sync 0;
elect.sync %r117|%p22, -1;
and.pred %p23, %p2, %p22;
and.pred %p21, %p29, %p23;
add.s32 %r101, %r91, %r115;
or.b32 %r102, %r112, %r92;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r101], [%rd22, {%r102, %r439}], [%r100];
mov.b32 %r520, 1;
mov.pred %p31, 0;
$L__tmp8:
@%p2 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p3 mbarrier.arrive.expect_tx.shared.b64 _, [%r90], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r61, %r7, 0, 31, -1;
elect.sync %r62|%p7, -1;
and.pred %p5, %p29, %p7;
and.b32 %r63, %r61, 1;
shl.b32 %r64, %r63, 14;
add.s32 %r58, %r57, %r64;
shl.b32 %r59, %r63, 6;
@%p5 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r58], [%rd22, {%r59, %r439}], [%r90];
$L__tmp10:
bar.sync 0;
elect.sync %r65|%p8, -1;
and.pred %p6, %p29, %p8;
add.s32 %r73, %r58, 65536;
@%p6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r73], [%rd22, {%r59, %r74}], [%r90];
shl.b32 %r66, %r61, 7;
and.b32 %r8, %r66, 512;
add.s32 %r67, %r57, 65536;
bfe.u32 %r68, %r67, 4, 14;
cvt.u64.u32 %rd4, %r68;
or.b64 %rd6, %rd4, 4611686293305294848;
add.s64 %rd8, %rd4, 4611686293305294850;
add.s64 %rd10, %rd4, 4611686293305294852;
add.s64 %rd12, %rd4, 4611686293305294854;
add.s64 %rd14, %rd4, 4611686293305295872;
add.s64 %rd16, %rd4, 4611686293305295874;
add.s64 %rd18, %rd4, 4611686293305295876;
add.s64 %rd20, %rd4, 4611686293305295878;
mov.b32 %r60, 0;
mov.pred %p9, -1;
mov.b32 %r455, 0f00000000;
mov.b32 %r456, %r455;
mov.b32 %r457, %r455;
mov.b32 %r458, %r455;
mov.b32 %r459, %r455;
mov.b32 %r460, %r455;
mov.b32 %r461, %r455;
mov.b32 %r462, %r455;
mov.b32 %r463, %r455;
mov.b32 %r464, %r455;
mov.b32 %r465, %r455;
mov.b32 %r466, %r455;
mov.b32 %r467, %r455;
mov.b32 %r468, %r455;
mov.b32 %r469, %r455;
mov.b32 %r470, %r455;
mov.b32 %r471, %r455;
mov.b32 %r472, %r455;
mov.b32 %r473, %r455;
mov.b32 %r474, %r455;
mov.b32 %r475, %r455;
mov.b32 %r476, %r455;
mov.b32 %r477, %r455;
mov.b32 %r478, %r455;
mov.b32 %r479, %r455;
mov.b32 %r480, %r455;
mov.b32 %r481, %r455;
mov.b32 %r482, %r455;
mov.b32 %r483, %r455;
mov.b32 %r484, %r455;
mov.b32 %r485, %r455;
mov.b32 %r486, %r455;
mov.b32 %r487, %r455;
mov.b32 %r488, %r455;
mov.b32 %r489, %r455;
mov.b32 %r490, %r455;
mov.b32 %r491, %r455;
mov.b32 %r492, %r455;
mov.b32 %r493, %r455;
mov.b32 %r494, %r455;
mov.b32 %r495, %r455;
mov.b32 %r496, %r455;
mov.b32 %r497, %r455;
mov.b32 %r498, %r455;
mov.b32 %r499, %r455;
mov.b32 %r500, %r455;
mov.b32 %r501, %r455;
mov.b32 %r502, %r455;
mov.b32 %r503, %r455;
mov.b32 %r504, %r455;
mov.b32 %r505, %r455;
mov.b32 %r506, %r455;
mov.b32 %r507, %r455;
mov.b32 %r508, %r455;
mov.b32 %r509, %r455;
mov.b32 %r510, %r455;
mov.b32 %r511, %r455;
mov.b32 %r512, %r455;
mov.b32 %r513, %r455;
mov.b32 %r514, %r455;
mov.b32 %r515, %r455;
mov.b32 %r516, %r455;
mov.b32 %r517, %r455;
mov.b32 %r518, %r455;
mov.pred %p30, %p9;
mov.b32 %r519, %r60;
$L__tmp11:
$L__BB0_2:
mov.pred %p1, %p30;
shl.b32 %r75, %r519, 15;
add.s32 %r76, %r57, %r75;
mad.lo.s32 %r77, %r519, -32760, %r76;
add.s32 %r69, %r77, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r69], %r60;
@!complete bra.uni waitLoop;
}
shr.u32 %r78, %r76, 4;
add.s32 %r79, %r78, %r8;
and.b32 %r80, %r79, 16383;
cvt.u64.u32 %rd21, %r80;
wgmma.fence.sync.aligned;
or.b64 %rd5, %rd21, 4611686293305294848;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd5, %rd6, %p9, 1, 1, 0, 0;
add.s64 %rd7, %rd21, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd7, %rd8, %p9, 1, 1, 0, 0;
add.s64 %rd9, %rd21, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd9, %rd10, %p9, 1, 1, 0, 0;
add.s64 %rd11, %rd21, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd11, %rd12, %p9, 1, 1, 0, 0;
add.s64 %rd13, %rd21, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd13, %rd14, %p9, 1, 1, 0, 0;
add.s64 %rd15, %rd21, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd15, %rd16, %p9, 1, 1, 0, 0;
add.s64 %rd17, %rd21, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd17, %rd18, %p9, 1, 1, 0, 0;
add.s64 %rd19, %rd21, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515,%r516,%r517,%r518}, %rd19, %rd20, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r81, %r519, 1;
and.b32 %r82, %r81, 1;
shl.b32 %r83, %r81, 7;
shl.b32 %r84, %r82, 3;
add.s32 %r85, %r57, %r84;
add.s32 %r70, %r85, 98304;
shl.b32 %r86, %r82, 15;
$L__tmp12:
bar.sync 0;
or.b32 %r87, %r519, %r5;
setp.eq.b32 %p10, %r87, 0;
@%p10 mbarrier.arrive.expect_tx.shared.b64 _, [%r70], 65536;
bar.sync 0;
elect.sync %r88|%p13, -1;
and.pred %p14, %p1, %p13;
and.pred %p11, %p29, %p14;
add.s32 %r71, %r58, %r86;
or.b32 %r72, %r83, %r59;
@%p11 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r71], [%rd22, {%r72, %r439}], [%r70];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r89|%p15, -1;
and.pred %p16, %p1, %p15;
and.pred %p12, %p29, %p16;
@%p12 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r73], [%rd22, {%r72, %r74}], [%r70];
mov.b32 %r519, 1;
mov.pred %p30, 0;
$L__tmp15:
@%p1 bra $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p24, %r2, %r3;
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r90];
bar.sync 0;
@%p3 mbarrier.inval.shared::cta.b64 [%r48];
and.b32 %r150, %r4, 240;
bfe.u32 %r151, %r4, 4, 4;
and.b32 %r10, %r4, 15;
shl.b32 %r152, %r10, 3;
shl.b32 %r153, %r1, 16;
or.b32 %r154, %r2, %r151;
shl.b32 %r155, %r154, 8;
or.b32 %r156, %r155, 4096;
or.b32 %r157, %r155, 8192;
or.b32 %r158, %r155, 12288;
or.b32 %r159, %r155, 16384;
or.b32 %r160, %r155, 20480;
or.b32 %r161, %r155, 24576;
or.b32 %r162, %r155, 28672;
or.b32 %r163, %r3, %r152;
add.s32 %r164, %r155, %r163;
add.s32 %r165, %r156, %r163;
add.s32 %r166, %r157, %r163;
add.s32 %r167, %r158, %r163;
add.s32 %r168, %r159, %r163;
add.s32 %r169, %r160, %r163;
add.s32 %r170, %r161, %r163;
add.s32 %r171, %r162, %r163;
mad.wide.s32 %rd49, %r153, 2, %rd1;
mad.wide.u32 %rd41, %r164, 2, %rd49;
mad.wide.u32 %rd42, %r165, 2, %rd49;
mad.wide.u32 %rd43, %r166, 2, %rd49;
mad.wide.u32 %rd44, %r167, 2, %rd49;
mad.wide.u32 %rd45, %r168, 2, %rd49;
mad.wide.u32 %rd46, %r169, 2, %rd49;
mad.wide.u32 %rd47, %r170, 2, %rd49;
mad.wide.u32 %rd48, %r171, 2, %rd49;
mov.u32 %r118, 0x0;
mov.u32 %r119, 0x0;
mov.u32 %r120, 0x0;
mov.u32 %r121, 0x0;
ld.global.v4.b32 { %r118, %r119, %r120, %r121 }, [ %rd41 + 0 ];
mov.u32 %r122, 0x0;
mov.u32 %r123, 0x0;
mov.u32 %r124, 0x0;
mov.u32 %r125, 0x0;
ld.global.v4.b32 { %r122, %r123, %r124, %r125 }, [ %rd42 + 0 ];
mov.u32 %r126, 0x0;
mov.u32 %r127, 0x0;
mov.u32 %r128, 0x0;
mov.u32 %r129, 0x0;
ld.global.v4.b32 { %r126, %r127, %r128, %r129 }, [ %rd43 + 0 ];
mov.u32 %r130, 0x0;
mov.u32 %r131, 0x0;
mov.u32 %r132, 0x0;
mov.u32 %r133, 0x0;
ld.global.v4.b32 { %r130, %r131, %r132, %r133 }, [ %rd44 + 0 ];
mov.u32 %r134, 0x0;
mov.u32 %r135, 0x0;
mov.u32 %r136, 0x0;
mov.u32 %r137, 0x0;
ld.global.v4.b32 { %r134, %r135, %r136, %r137 }, [ %rd45 + 0 ];
mov.u32 %r138, 0x0;
mov.u32 %r139, 0x0;
mov.u32 %r140, 0x0;
mov.u32 %r141, 0x0;
ld.global.v4.b32 { %r138, %r139, %r140, %r141 }, [ %rd46 + 0 ];
mov.u32 %r142, 0x0;
mov.u32 %r143, 0x0;
mov.u32 %r144, 0x0;
mov.u32 %r145, 0x0;
ld.global.v4.b32 { %r142, %r143, %r144, %r145 }, [ %rd47 + 0 ];
mov.u32 %r146, 0x0;
mov.u32 %r147, 0x0;
mov.u32 %r148, 0x0;
mov.u32 %r149, 0x0;
ld.global.v4.b32 { %r146, %r147, %r148, %r149 }, [ %rd48 + 0 ];
shl.b32 %r172, %r4, 8;
and.b32 %r173, %r172, 30720;
and.b32 %r174, %r4, 7;
shl.b32 %r11, %r174, 4;
or.b32 %r175, %r173, %r11;
xor.b32 %r176, %r175, %r150;
add.s32 %r177, %r57, %r176;
st.shared.v4.b32 [%r177], {%r118, %r119, %r120, %r121};
st.shared.v4.b32 [%r177+256], {%r122, %r123, %r124, %r125};
st.shared.v4.b32 [%r177+512], {%r126, %r127, %r128, %r129};
st.shared.v4.b32 [%r177+768], {%r130, %r131, %r132, %r133};
st.shared.v4.b32 [%r177+1024], {%r134, %r135, %r136, %r137};
st.shared.v4.b32 [%r177+1280], {%r138, %r139, %r140, %r141};
st.shared.v4.b32 [%r177+1536], {%r142, %r143, %r144, %r145};
st.shared.v4.b32 [%r177+1792], {%r146, %r147, %r148, %r149};
bar.sync 0;
shl.b32 %r178, %r174, 12;
shl.b32 %r179, %r10, 4;
and.b32 %r12, %r4, 224;
shl.b32 %r180, %r12, 3;
and.b32 %r13, %r4, 16;
or.b32 %r181, %r179, %r180;
xor.b32 %r182, %r181, %r13;
or.b32 %r183, %r182, %r178;
add.s32 %r184, %r57, %r183;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r185, %r186, %r187, %r188}, [%r184];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r184+2048];
xor.b32 %r193, %r183, 32;
add.s32 %r194, %r57, %r193;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r195, %r196, %r197, %r198}, [%r194];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r201, %r202}, [%r194+2048];
xor.b32 %r203, %r183, 64;
add.s32 %r204, %r57, %r203;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r205, %r206, %r207, %r208}, [%r204];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r209, %r210, %r211, %r212}, [%r204+2048];
xor.b32 %r213, %r183, 96;
add.s32 %r214, %r57, %r213;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r215, %r216, %r217, %r218}, [%r214];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r219, %r220, %r221, %r222}, [%r214+2048];
mov.b32 {%rs1, %rs2}, %r185;
cvt.f32.bf16 %r223, %rs1;
cvt.f32.bf16 %r224, %rs2;
mul.f32 %r225, %r47, %r224;
mul.f32 %r226, %r47, %r223;
fma.rn.f32 %r227, %r46, %r455, %r226;
fma.rn.f32 %r228, %r46, %r456, %r225;
mov.b32 {%rs3, %rs4}, %r186;
cvt.f32.bf16 %r229, %rs3;
cvt.f32.bf16 %r230, %rs4;
mul.f32 %r231, %r47, %r230;
mul.f32 %r232, %r47, %r229;
fma.rn.f32 %r233, %r46, %r457, %r232;
fma.rn.f32 %r234, %r46, %r458, %r231;
mov.b32 {%rs5, %rs6}, %r187;
cvt.f32.bf16 %r235, %rs5;
cvt.f32.bf16 %r236, %rs6;
mul.f32 %r237, %r47, %r236;
mul.f32 %r238, %r47, %r235;
fma.rn.f32 %r239, %r46, %r459, %r238;
fma.rn.f32 %r240, %r46, %r460, %r237;
mov.b32 {%rs7, %rs8}, %r188;
cvt.f32.bf16 %r241, %rs7;
cvt.f32.bf16 %r242, %rs8;
mul.f32 %r243, %r47, %r242;
mul.f32 %r244, %r47, %r241;
fma.rn.f32 %r245, %r46, %r461, %r244;
fma.rn.f32 %r246, %r46, %r462, %r243;
mov.b32 {%rs9, %rs10}, %r195;
cvt.f32.bf16 %r247, %rs9;
cvt.f32.bf16 %r248, %rs10;
mul.f32 %r249, %r47, %r248;
mul.f32 %r250, %r47, %r247;
fma.rn.f32 %r251, %r46, %r463, %r250;
fma.rn.f32 %r252, %r46, %r464, %r249;
mov.b32 {%rs11, %rs12}, %r196;
cvt.f32.bf16 %r253, %rs11;
cvt.f32.bf16 %r254, %rs12;
mul.f32 %r255, %r47, %r254;
mul.f32 %r256, %r47, %r253;
fma.rn.f32 %r257, %r46, %r465, %r256;
fma.rn.f32 %r258, %r46, %r466, %r255;
mov.b32 {%rs13, %rs14}, %r197;
cvt.f32.bf16 %r259, %rs13;
cvt.f32.bf16 %r260, %rs14;
mul.f32 %r261, %r47, %r260;
mul.f32 %r262, %r47, %r259;
fma.rn.f32 %r263, %r46, %r467, %r262;
fma.rn.f32 %r264, %r46, %r468, %r261;
mov.b32 {%rs15, %rs16}, %r198;
cvt.f32.bf16 %r265, %rs15;
cvt.f32.bf16 %r266, %rs16;
mul.f32 %r267, %r47, %r266;
mul.f32 %r268, %r47, %r265;
fma.rn.f32 %r269, %r46, %r469, %r268;
fma.rn.f32 %r270, %r46, %r470, %r267;
mov.b32 {%rs17, %rs18}, %r205;
cvt.f32.bf16 %r271, %rs17;
cvt.f32.bf16 %r272, %rs18;
mul.f32 %r273, %r47, %r272;
mul.f32 %r274, %r47, %r271;
fma.rn.f32 %r275, %r46, %r471, %r274;
fma.rn.f32 %r276, %r46, %r472, %r273;
mov.b32 {%rs19, %rs20}, %r206;
cvt.f32.bf16 %r277, %rs19;
cvt.f32.bf16 %r278, %rs20;
mul.f32 %r279, %r47, %r278;
mul.f32 %r280, %r47, %r277;
fma.rn.f32 %r281, %r46, %r473, %r280;
fma.rn.f32 %r282, %r46, %r474, %r279;
mov.b32 {%rs21, %rs22}, %r207;
cvt.f32.bf16 %r283, %rs21;
cvt.f32.bf16 %r284, %rs22;
mul.f32 %r285, %r47, %r284;
mul.f32 %r286, %r47, %r283;
fma.rn.f32 %r287, %r46, %r475, %r286;
fma.rn.f32 %r288, %r46, %r476, %r285;
mov.b32 {%rs23, %rs24}, %r208;
cvt.f32.bf16 %r289, %rs23;
cvt.f32.bf16 %r290, %rs24;
mul.f32 %r291, %r47, %r290;
mul.f32 %r292, %r47, %r289;
fma.rn.f32 %r293, %r46, %r477, %r292;
fma.rn.f32 %r294, %r46, %r478, %r291;
mov.b32 {%rs25, %rs26}, %r215;
cvt.f32.bf16 %r295, %rs25;
cvt.f32.bf16 %r296, %rs26;
mul.f32 %r297, %r47, %r296;
mul.f32 %r298, %r47, %r295;
fma.rn.f32 %r299, %r46, %r479, %r298;
fma.rn.f32 %r300, %r46, %r480, %r297;
mov.b32 {%rs27, %rs28}, %r216;
cvt.f32.bf16 %r301, %rs27;
cvt.f32.bf16 %r302, %rs28;
mul.f32 %r303, %r47, %r302;
mul.f32 %r304, %r47, %r301;
fma.rn.f32 %r305, %r46, %r481, %r304;
fma.rn.f32 %r306, %r46, %r482, %r303;
mov.b32 {%rs29, %rs30}, %r217;
cvt.f32.bf16 %r307, %rs29;
cvt.f32.bf16 %r308, %rs30;
mul.f32 %r309, %r47, %r308;
mul.f32 %r310, %r47, %r307;
fma.rn.f32 %r311, %r46, %r483, %r310;
fma.rn.f32 %r312, %r46, %r484, %r309;
mov.b32 {%rs31, %rs32}, %r218;
cvt.f32.bf16 %r313, %rs31;
cvt.f32.bf16 %r314, %rs32;
mul.f32 %r315, %r47, %r314;
mul.f32 %r316, %r47, %r313;
fma.rn.f32 %r317, %r46, %r485, %r316;
fma.rn.f32 %r318, %r46, %r486, %r315;
mov.b32 {%rs33, %rs34}, %r189;
cvt.f32.bf16 %r319, %rs33;
cvt.f32.bf16 %r320, %rs34;
mul.f32 %r321, %r47, %r320;
mul.f32 %r322, %r47, %r319;
fma.rn.f32 %r323, %r46, %r487, %r322;
fma.rn.f32 %r324, %r46, %r488, %r321;
mov.b32 {%rs35, %rs36}, %r190;
cvt.f32.bf16 %r325, %rs35;
cvt.f32.bf16 %r326, %rs36;
mul.f32 %r327, %r47, %r326;
mul.f32 %r328, %r47, %r325;
fma.rn.f32 %r329, %r46, %r489, %r328;
fma.rn.f32 %r330, %r46, %r490, %r327;
mov.b32 {%rs37, %rs38}, %r191;
cvt.f32.bf16 %r331, %rs37;
cvt.f32.bf16 %r332, %rs38;
mul.f32 %r333, %r47, %r332;
mul.f32 %r334, %r47, %r331;
fma.rn.f32 %r335, %r46, %r491, %r334;
fma.rn.f32 %r336, %r46, %r492, %r333;
mov.b32 {%rs39, %rs40}, %r192;
cvt.f32.bf16 %r337, %rs39;
cvt.f32.bf16 %r338, %rs40;
mul.f32 %r339, %r47, %r338;
mul.f32 %r340, %r47, %r337;
fma.rn.f32 %r341, %r46, %r493, %r340;
fma.rn.f32 %r342, %r46, %r494, %r339;
mov.b32 {%rs41, %rs42}, %r199;
cvt.f32.bf16 %r343, %rs41;
cvt.f32.bf16 %r344, %rs42;
mul.f32 %r345, %r47, %r344;
mul.f32 %r346, %r47, %r343;
fma.rn.f32 %r347, %r46, %r495, %r346;
fma.rn.f32 %r348, %r46, %r496, %r345;
mov.b32 {%rs43, %rs44}, %r200;
cvt.f32.bf16 %r349, %rs43;
cvt.f32.bf16 %r350, %rs44;
mul.f32 %r351, %r47, %r350;
mul.f32 %r352, %r47, %r349;
fma.rn.f32 %r353, %r46, %r497, %r352;
fma.rn.f32 %r354, %r46, %r498, %r351;
mov.b32 {%rs45, %rs46}, %r201;
cvt.f32.bf16 %r355, %rs45;
cvt.f32.bf16 %r356, %rs46;
mul.f32 %r357, %r47, %r356;
mul.f32 %r358, %r47, %r355;
fma.rn.f32 %r359, %r46, %r499, %r358;
fma.rn.f32 %r360, %r46, %r500, %r357;
mov.b32 {%rs47, %rs48}, %r202;
cvt.f32.bf16 %r361, %rs47;
cvt.f32.bf16 %r362, %rs48;
mul.f32 %r363, %r47, %r362;
mul.f32 %r364, %r47, %r361;
fma.rn.f32 %r365, %r46, %r501, %r364;
fma.rn.f32 %r366, %r46, %r502, %r363;
mov.b32 {%rs49, %rs50}, %r209;
cvt.f32.bf16 %r367, %rs49;
cvt.f32.bf16 %r368, %rs50;
mul.f32 %r369, %r47, %r368;
mul.f32 %r370, %r47, %r367;
fma.rn.f32 %r371, %r46, %r503, %r370;
fma.rn.f32 %r372, %r46, %r504, %r369;
mov.b32 {%rs51, %rs52}, %r210;
cvt.f32.bf16 %r373, %rs51;
cvt.f32.bf16 %r374, %rs52;
mul.f32 %r375, %r47, %r374;
mul.f32 %r376, %r47, %r373;
fma.rn.f32 %r377, %r46, %r505, %r376;
fma.rn.f32 %r378, %r46, %r506, %r375;
mov.b32 {%rs53, %rs54}, %r211;
cvt.f32.bf16 %r379, %rs53;
cvt.f32.bf16 %r380, %rs54;
mul.f32 %r381, %r47, %r380;
mul.f32 %r382, %r47, %r379;
fma.rn.f32 %r383, %r46, %r507, %r382;
fma.rn.f32 %r384, %r46, %r508, %r381;
mov.b32 {%rs55, %rs56}, %r212;
cvt.f32.bf16 %r385, %rs55;
cvt.f32.bf16 %r386, %rs56;
mul.f32 %r387, %r47, %r386;
mul.f32 %r388, %r47, %r385;
fma.rn.f32 %r389, %r46, %r509, %r388;
fma.rn.f32 %r390, %r46, %r510, %r387;
mov.b32 {%rs57, %rs58}, %r219;
cvt.f32.bf16 %r391, %rs57;
cvt.f32.bf16 %r392, %rs58;
mul.f32 %r393, %r47, %r392;
mul.f32 %r394, %r47, %r391;
fma.rn.f32 %r395, %r46, %r511, %r394;
fma.rn.f32 %r396, %r46, %r512, %r393;
mov.b32 {%rs59, %rs60}, %r220;
cvt.f32.bf16 %r397, %rs59;
cvt.f32.bf16 %r398, %rs60;
mul.f32 %r399, %r47, %r398;
mul.f32 %r400, %r47, %r397;
fma.rn.f32 %r401, %r46, %r513, %r400;
fma.rn.f32 %r402, %r46, %r514, %r399;
mov.b32 {%rs61, %rs62}, %r221;
cvt.f32.bf16 %r403, %rs61;
cvt.f32.bf16 %r404, %rs62;
mul.f32 %r405, %r47, %r404;
mul.f32 %r406, %r47, %r403;
fma.rn.f32 %r407, %r46, %r515, %r406;
fma.rn.f32 %r408, %r46, %r516, %r405;
mov.b32 {%rs63, %rs64}, %r222;
cvt.f32.bf16 %r409, %rs63;
cvt.f32.bf16 %r410, %rs64;
mul.f32 %r411, %r47, %r410;
mul.f32 %r412, %r47, %r409;
fma.rn.f32 %r413, %r46, %r517, %r412;
fma.rn.f32 %r414, %r46, %r518, %r411;
cvt.rn.bf16x2.f32 %r14, %r228, %r227;
cvt.rn.bf16x2.f32 %r15, %r234, %r233;
cvt.rn.bf16x2.f32 %r16, %r240, %r239;
cvt.rn.bf16x2.f32 %r17, %r246, %r245;
cvt.rn.bf16x2.f32 %r18, %r252, %r251;
cvt.rn.bf16x2.f32 %r19, %r258, %r257;
cvt.rn.bf16x2.f32 %r20, %r264, %r263;
cvt.rn.bf16x2.f32 %r21, %r270, %r269;
cvt.rn.bf16x2.f32 %r22, %r276, %r275;
cvt.rn.bf16x2.f32 %r23, %r282, %r281;
cvt.rn.bf16x2.f32 %r24, %r288, %r287;
cvt.rn.bf16x2.f32 %r25, %r294, %r293;
cvt.rn.bf16x2.f32 %r26, %r300, %r299;
cvt.rn.bf16x2.f32 %r27, %r306, %r305;
cvt.rn.bf16x2.f32 %r28, %r312, %r311;
cvt.rn.bf16x2.f32 %r29, %r318, %r317;
cvt.rn.bf16x2.f32 %r30, %r324, %r323;
cvt.rn.bf16x2.f32 %r31, %r330, %r329;
cvt.rn.bf16x2.f32 %r32, %r336, %r335;
cvt.rn.bf16x2.f32 %r33, %r342, %r341;
cvt.rn.bf16x2.f32 %r34, %r348, %r347;
cvt.rn.bf16x2.f32 %r35, %r354, %r353;
cvt.rn.bf16x2.f32 %r36, %r360, %r359;
cvt.rn.bf16x2.f32 %r37, %r366, %r365;
cvt.rn.bf16x2.f32 %r38, %r372, %r371;
cvt.rn.bf16x2.f32 %r39, %r378, %r377;
cvt.rn.bf16x2.f32 %r40, %r384, %r383;
cvt.rn.bf16x2.f32 %r41, %r390, %r389;
cvt.rn.bf16x2.f32 %r42, %r396, %r395;
cvt.rn.bf16x2.f32 %r43, %r402, %r401;
cvt.rn.bf16x2.f32 %r44, %r408, %r407;
cvt.rn.bf16x2.f32 %r45, %r414, %r413;
@%p24 bra $L__BB0_7;
bra.uni $L__BB0_8;
$L__BB0_7:
bar.sync 0;
shl.b32 %r415, %r10, 7;
shl.b32 %r416, %r12, 6;
or.b32 %r417, %r415, %r11;
xor.b32 %r418, %r417, %r13;
or.b32 %r419, %r418, %r416;
add.s32 %r420, %r57, %r419;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r420], {%r14, %r15, %r16, %r17};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r420+16384], {%r30, %r31, %r32, %r33};
xor.b32 %r421, %r419, 32;
add.s32 %r422, %r57, %r421;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r422], {%r18, %r19, %r20, %r21};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r422+16384], {%r34, %r35, %r36, %r37};
xor.b32 %r423, %r419, 64;
add.s32 %r424, %r57, %r423;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r424], {%r22, %r23, %r24, %r25};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r424+16384], {%r38, %r39, %r40, %r41};
xor.b32 %r425, %r419, 96;
add.s32 %r426, %r57, %r425;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r426], {%r26, %r27, %r28, %r29};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r426+16384], {%r42, %r43, %r44, %r45};
$L__BB0_8:
shl.b32 %r427, %r4, 7;
and.b32 %r428, %r427, 17280;
and.b32 %r429, %r4, 96;
shl.b32 %r430, %r4, 1;
and.b32 %r431, %r430, 16;
shl.b32 %r432, %r13, 6;
or.b32 %r433, %r428, %r11;
or.b32 %r434, %r431, %r429;
xor.b32 %r435, %r433, %r434;
add.s32 %r436, %r57, %r432;
add.s32 %r437, %r436, %r435;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+32768], {%r14, %r15, %r16, %r17};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+34816], {%r18, %r19, %r20, %r21};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+36864], {%r22, %r23, %r24, %r25};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+38912], {%r26, %r27, %r28, %r29};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+40960], {%r30, %r31, %r32, %r33};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+43008], {%r34, %r35, %r36, %r37};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+45056], {%r38, %r39, %r40, %r41};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r437+47104], {%r42, %r43, %r44, %r45};
fence.proxy.async.shared::cta;
@%p4 bra $L__BB0_10;
bar.sync 0;
elect.sync %r441|%p26, -1;
shfl.sync.idx.b32 %r442, %r7, 0, 31, -1;
and.pred %p25, %p29, %p26;
and.b32 %r443, %r442, 1;
shl.b32 %r444, %r443, 14;
add.s32 %r440, %r57, %r444;
shl.b32 %r445, %r443, 6;
or.b32 %r438, %r445, %r3;
@%p25 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd50, {%r438, %r439}], [%r440];
cp.async.bulk.commit_group;
$L__BB0_10:
bar.sync 0;
elect.sync %r449|%p28, -1;
shfl.sync.idx.b32 %r450, %r7, 0, 31, -1;
and.pred %p27, %p29, %p28;
and.b32 %r451, %r450, 1;
shl.b32 %r452, %r451, 14;
add.s32 %r453, %r57, %r452;
add.s32 %r447, %r453, 32768;
shl.b32 %r454, %r451, 6;
or.b32 %r446, %r454, %r2;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd50, {%r446, %r74}], [%r447];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r448, 0;
ret;
$L__tmp17:
$L__func_end0:
}
"#
  | .widePolynomial, .baseline => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry ba_plus_cAA_kernel(
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_0,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_1,
.param .u32 ba_plus_cAA_kernel_param_2,
.param .u32 ba_plus_cAA_kernel_param_3,
.param .u32 ba_plus_cAA_kernel_param_4,
.param .u32 ba_plus_cAA_kernel_param_5,
.param .u32 ba_plus_cAA_kernel_param_6,
.param .f32 ba_plus_cAA_kernel_param_7,
.param .f32 ba_plus_cAA_kernel_param_8,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_9,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_10
)
.reqntid 256
{
.reg .pred %p<49>;
.reg .b16 %rs<65>;
.reg .b32 %r<674>;
.reg .b64 %rd<98>;
$L__func_begin0:
ld.param.b32 %r39, [ba_plus_cAA_kernel_param_2];
$L__tmp0:
mov.u32 %r40, %ctaid.x;
$L__tmp1:
add.s32 %r41, %r39, 127;
shr.s32 %r42, %r41, 31;
shr.u32 %r43, %r42, 25;
add.s32 %r44, %r41, %r43;
shr.s32 %r45, %r44, 7;
$L__tmp2:
mul.lo.s32 %r46, %r45, %r45;
div.s32 %r1, %r40, %r46;
mul.lo.s32 %r47, %r1, %r46;
sub.s32 %r48, %r40, %r47;
$L__tmp3:
shl.b32 %r49, %r45, 3;
div.s32 %r50, %r48, %r49;
shl.b32 %r51, %r50, 3;
sub.s32 %r52, %r45, %r51;
min.s32 %r53, %r52, 8;
mul.lo.s32 %r54, %r50, %r49;
sub.s32 %r55, %r48, %r54;
div.s32 %r56, %r55, %r53;
mul.lo.s32 %r57, %r56, %r53;
sub.s32 %r58, %r55, %r57;
add.s32 %r59, %r58, %r51;
$L__tmp4:
shl.b32 %r2, %r59, 7;
shl.b32 %r3, %r56, 7;
$L__tmp5:
add.s32 %r60, %r2, 128;
setp.gt.s32 %p1, %r60, %r3;
@%p1 bra $L__BB0_2;
bra.uni $L__BB0_1;
$L__BB0_2:
ld.param.b32 %r38, [ba_plus_cAA_kernel_param_8];
ld.param.b32 %r37, [ba_plus_cAA_kernel_param_7];
ld.param.b32 %r36, [ba_plus_cAA_kernel_param_6];
ld.param.b32 %r35, [ba_plus_cAA_kernel_param_5];
ld.param.b32 %r34, [ba_plus_cAA_kernel_param_4];
ld.param.b32 %r33, [ba_plus_cAA_kernel_param_3];
ld.param.b64 %rd20, [ba_plus_cAA_kernel_param_1];
ld.param.b64 %rd19, [ba_plus_cAA_kernel_param_0];
mul.lo.s32 %r88, %r1, %r33;
mad.wide.s32 %rd1, %r88, 2, %rd19;
mul.lo.s32 %r89, %r1, %r35;
mad.wide.s32 %rd2, %r89, 2, %rd20;
mov.u32 %r4, %tid.x;
shr.u32 %r90, %r4, 3;
bfe.u32 %r91, %r4, 3, 5;
or.b32 %r92, %r91, 32;
or.b32 %r93, %r91, 64;
or.b32 %r94, %r90, 96;
and.b32 %r5, %r4, 240;
bfe.u32 %r95, %r4, 4, 4;
or.b32 %r96, %r95, 16;
or.b32 %r97, %r95, 32;
or.b32 %r98, %r95, 48;
or.b32 %r99, %r95, 64;
or.b32 %r100, %r95, 80;
or.b32 %r101, %r95, 96;
or.b32 %r102, %r95, 112;
and.b32 %r6, %r4, 15;
shl.b32 %r103, %r6, 3;
or.b32 %r104, %r2, %r91;
or.b32 %r105, %r2, %r92;
or.b32 %r106, %r2, %r93;
or.b32 %r107, %r2, %r94;
or.b32 %r7, %r2, %r95;
or.b32 %r8, %r2, %r96;
or.b32 %r9, %r2, %r97;
or.b32 %r10, %r2, %r98;
or.b32 %r11, %r2, %r99;
or.b32 %r12, %r2, %r100;
or.b32 %r13, %r2, %r101;
or.b32 %r14, %r2, %r102;
or.b32 %r15, %r2, %r103;
rem.s32 %r108, %r104, %r39;
rem.s32 %r109, %r105, %r39;
rem.s32 %r110, %r106, %r39;
rem.s32 %r111, %r107, %r39;
or.b32 %r112, %r3, %r91;
or.b32 %r113, %r3, %r92;
or.b32 %r114, %r3, %r93;
or.b32 %r115, %r3, %r94;
or.b32 %r16, %r3, %r95;
or.b32 %r17, %r3, %r96;
or.b32 %r18, %r3, %r97;
or.b32 %r19, %r3, %r98;
or.b32 %r20, %r3, %r99;
or.b32 %r21, %r3, %r100;
or.b32 %r22, %r3, %r101;
or.b32 %r23, %r3, %r102;
or.b32 %r24, %r3, %r103;
rem.s32 %r116, %r112, %r39;
rem.s32 %r117, %r113, %r39;
rem.s32 %r118, %r114, %r39;
rem.s32 %r119, %r115, %r39;
and.b32 %r25, %r4, 7;
shl.b32 %r26, %r25, 3;
mad.lo.s32 %r120, %r108, %r34, %r26;
mad.lo.s32 %r121, %r109, %r34, %r26;
mad.lo.s32 %r122, %r110, %r34, %r26;
mad.lo.s32 %r123, %r111, %r34, %r26;
mad.wide.s32 %rd21, %r120, 2, %rd1;
mad.wide.s32 %rd22, %r121, 2, %rd1;
mad.wide.s32 %rd23, %r122, 2, %rd1;
mad.wide.s32 %rd24, %r123, 2, %rd1;
mad.lo.s32 %r124, %r116, %r34, %r26;
mad.lo.s32 %r125, %r117, %r34, %r26;
mad.lo.s32 %r126, %r118, %r34, %r26;
mad.lo.s32 %r127, %r119, %r34, %r26;
mad.wide.s32 %rd25, %r124, 2, %rd1;
mad.wide.s32 %rd26, %r125, 2, %rd1;
mad.wide.s32 %rd27, %r126, 2, %rd1;
mad.wide.s32 %rd28, %r127, 2, %rd1;
$L__tmp6:
add.s32 %r128, %r39, 63;
$L__tmp7:
setp.lt.s32 %p2, %r128, 64;
setp.gt.s32 %p3, %r128, 63;
setp.lt.s32 %p4, %r26, %r39;
shl.b32 %r28, %r4, 4;
and.b32 %r132, %r28, 4080;
shl.b32 %r29, %r4, 1;
and.b32 %r133, %r29, 112;
xor.b32 %r30, %r132, %r133;
mov.b32 %r134, global_smem;
add.s32 %r61, %r134, %r30;
selp.b32 %r135, 16, 0, %p3;
selp.b32 %r63, %r135, 0, %p4;
cp.async.cg.shared.global [ %r61 + 0 ], [ %rd21 + 0 ], 0x10, %r63;
add.s32 %r62, %r61, 4096;
cp.async.cg.shared.global [ %r62 + 0 ], [ %rd22 + 0 ], 0x10, %r63;
add.s32 %r64, %r61, 8192;
cp.async.cg.shared.global [ %r64 + 0 ], [ %rd23 + 0 ], 0x10, %r63;
add.s32 %r65, %r61, 12288;
cp.async.cg.shared.global [ %r65 + 0 ], [ %rd24 + 0 ], 0x10, %r63;
cp.async.commit_group;
add.s32 %r66, %r61, 65536;
cp.async.cg.shared.global [ %r66 + 0 ], [ %rd25 + 0 ], 0x10, %r63;
add.s32 %r67, %r61, 69632;
cp.async.cg.shared.global [ %r67 + 0 ], [ %rd26 + 0 ], 0x10, %r63;
add.s32 %r68, %r61, 73728;
cp.async.cg.shared.global [ %r68 + 0 ], [ %rd27 + 0 ], 0x10, %r63;
add.s32 %r69, %r61, 77824;
cp.async.cg.shared.global [ %r69 + 0 ], [ %rd28 + 0 ], 0x10, %r63;
cp.async.commit_group;
setp.gt.s32 %p5, %r128, 127;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s64 %rd35, %rd27, 128;
add.s64 %rd36, %rd28, 128;
add.s32 %r136, %r39, -64;
setp.lt.s32 %p6, %r26, %r136;
bar.sync 0;
add.s32 %r70, %r61, 16384;
selp.b32 %r137, 16, 0, %p6;
selp.b32 %r72, %r137, 0, %p5;
cp.async.cg.shared.global [ %r70 + 0 ], [ %rd29 + 0 ], 0x10, %r72;
add.s32 %r71, %r61, 20480;
cp.async.cg.shared.global [ %r71 + 0 ], [ %rd30 + 0 ], 0x10, %r72;
add.s32 %r73, %r61, 24576;
cp.async.cg.shared.global [ %r73 + 0 ], [ %rd31 + 0 ], 0x10, %r72;
add.s32 %r74, %r61, 28672;
cp.async.cg.shared.global [ %r74 + 0 ], [ %rd32 + 0 ], 0x10, %r72;
cp.async.commit_group;
add.s32 %r75, %r61, 81920;
cp.async.cg.shared.global [ %r75 + 0 ], [ %rd33 + 0 ], 0x10, %r72;
add.s32 %r76, %r61, 86016;
cp.async.cg.shared.global [ %r76 + 0 ], [ %rd34 + 0 ], 0x10, %r72;
add.s32 %r77, %r61, 90112;
cp.async.cg.shared.global [ %r77 + 0 ], [ %rd35 + 0 ], 0x10, %r72;
add.s32 %r78, %r61, 94208;
cp.async.cg.shared.global [ %r78 + 0 ], [ %rd36 + 0 ], 0x10, %r72;
cp.async.commit_group;
setp.gt.s32 %p7, %r128, 191;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s64 %rd43, %rd27, 256;
add.s64 %rd44, %rd28, 256;
add.s32 %r138, %r39, -128;
setp.lt.s32 %p8, %r26, %r138;
bar.sync 0;
add.s32 %r79, %r61, 32768;
selp.b32 %r139, 16, 0, %p8;
selp.b32 %r81, %r139, 0, %p7;
cp.async.cg.shared.global [ %r79 + 0 ], [ %rd37 + 0 ], 0x10, %r81;
add.s32 %r80, %r61, 36864;
cp.async.cg.shared.global [ %r80 + 0 ], [ %rd38 + 0 ], 0x10, %r81;
add.s32 %r82, %r61, 40960;
cp.async.cg.shared.global [ %r82 + 0 ], [ %rd39 + 0 ], 0x10, %r81;
add.s32 %r83, %r61, 45056;
cp.async.cg.shared.global [ %r83 + 0 ], [ %rd40 + 0 ], 0x10, %r81;
cp.async.commit_group;
add.s32 %r84, %r61, 98304;
cp.async.cg.shared.global [ %r84 + 0 ], [ %rd41 + 0 ], 0x10, %r81;
add.s32 %r85, %r61, 102400;
cp.async.cg.shared.global [ %r85 + 0 ], [ %rd42 + 0 ], 0x10, %r81;
add.s32 %r86, %r61, 106496;
cp.async.cg.shared.global [ %r86 + 0 ], [ %rd43 + 0 ], 0x10, %r81;
add.s32 %r87, %r61, 110592;
cp.async.cg.shared.global [ %r87 + 0 ], [ %rd44 + 0 ], 0x10, %r81;
cp.async.commit_group;
mov.b32 %r609, 0f00000000;
mov.b32 %r610, %r609;
mov.b32 %r611, %r609;
mov.b32 %r612, %r609;
mov.b32 %r613, %r609;
mov.b32 %r614, %r609;
mov.b32 %r615, %r609;
mov.b32 %r616, %r609;
mov.b32 %r617, %r609;
mov.b32 %r618, %r609;
mov.b32 %r619, %r609;
mov.b32 %r620, %r609;
mov.b32 %r621, %r609;
mov.b32 %r622, %r609;
mov.b32 %r623, %r609;
mov.b32 %r624, %r609;
mov.b32 %r625, %r609;
mov.b32 %r626, %r609;
mov.b32 %r627, %r609;
mov.b32 %r628, %r609;
mov.b32 %r629, %r609;
mov.b32 %r630, %r609;
mov.b32 %r631, %r609;
mov.b32 %r632, %r609;
mov.b32 %r633, %r609;
mov.b32 %r634, %r609;
mov.b32 %r635, %r609;
mov.b32 %r636, %r609;
mov.b32 %r637, %r609;
mov.b32 %r638, %r609;
mov.b32 %r639, %r609;
mov.b32 %r640, %r609;
mov.b32 %r641, %r609;
mov.b32 %r642, %r609;
mov.b32 %r643, %r609;
mov.b32 %r644, %r609;
mov.b32 %r645, %r609;
mov.b32 %r646, %r609;
mov.b32 %r647, %r609;
mov.b32 %r648, %r609;
mov.b32 %r649, %r609;
mov.b32 %r650, %r609;
mov.b32 %r651, %r609;
mov.b32 %r652, %r609;
mov.b32 %r653, %r609;
mov.b32 %r654, %r609;
mov.b32 %r655, %r609;
mov.b32 %r656, %r609;
mov.b32 %r657, %r609;
mov.b32 %r658, %r609;
mov.b32 %r659, %r609;
mov.b32 %r660, %r609;
mov.b32 %r661, %r609;
mov.b32 %r662, %r609;
mov.b32 %r663, %r609;
mov.b32 %r664, %r609;
mov.b32 %r665, %r609;
mov.b32 %r666, %r609;
mov.b32 %r667, %r609;
mov.b32 %r668, %r609;
mov.b32 %r669, %r609;
mov.b32 %r670, %r609;
mov.b32 %r671, %r609;
mov.b32 %r672, %r609;
@%p2 bra $L__BB0_5;
cvt.s64.s32 %rd3, %r120;
cvt.s64.s32 %rd4, %r121;
cvt.s64.s32 %rd5, %r122;
cvt.s64.s32 %rd6, %r123;
cvt.s64.s32 %rd7, %r124;
cvt.s64.s32 %rd8, %r125;
cvt.s64.s32 %rd9, %r126;
cvt.s64.s32 %rd10, %r127;
shr.s32 %r129, %r128, 31;
shr.u32 %r130, %r129, 26;
add.s32 %r131, %r128, %r130;
shr.s32 %r27, %r131, 6;
add.s32 %r31, %r27, -3;
shr.u32 %r32, %r4, 5;
add.s32 %r606, %r39, -192;
shl.b64 %rd45, %rd10, 1;
add.s64 %rd11, %rd45, 384;
shl.b64 %rd46, %rd9, 1;
add.s64 %rd12, %rd46, 384;
shl.b64 %rd47, %rd8, 1;
add.s64 %rd13, %rd47, 384;
shl.b64 %rd48, %rd7, 1;
add.s64 %rd14, %rd48, 384;
shl.b64 %rd49, %rd6, 1;
add.s64 %rd15, %rd49, 384;
shl.b64 %rd50, %rd5, 1;
add.s64 %rd16, %rd50, 384;
shl.b64 %rd51, %rd4, 1;
add.s64 %rd17, %rd51, 384;
shl.b64 %rd52, %rd3, 1;
add.s64 %rd18, %rd52, 384;
mov.b32 %r144, 0;
mov.b32 %r609, 0f00000000;
mov.b32 %r608, 2;
mov.b32 %r607, -1;
mov.b64 %rd97, %rd1;
mov.b32 %r610, %r609;
mov.b32 %r611, %r609;
mov.b32 %r612, %r609;
mov.b32 %r613, %r609;
mov.b32 %r614, %r609;
mov.b32 %r615, %r609;
mov.b32 %r616, %r609;
mov.b32 %r617, %r609;
mov.b32 %r618, %r609;
mov.b32 %r619, %r609;
mov.b32 %r620, %r609;
mov.b32 %r621, %r609;
mov.b32 %r622, %r609;
mov.b32 %r623, %r609;
mov.b32 %r624, %r609;
mov.b32 %r625, %r609;
mov.b32 %r626, %r609;
mov.b32 %r627, %r609;
mov.b32 %r628, %r609;
mov.b32 %r629, %r609;
mov.b32 %r630, %r609;
mov.b32 %r631, %r609;
mov.b32 %r632, %r609;
mov.b32 %r633, %r609;
mov.b32 %r634, %r609;
mov.b32 %r635, %r609;
mov.b32 %r636, %r609;
mov.b32 %r637, %r609;
mov.b32 %r638, %r609;
mov.b32 %r639, %r609;
mov.b32 %r640, %r609;
mov.b32 %r641, %r609;
mov.b32 %r642, %r609;
mov.b32 %r643, %r609;
mov.b32 %r644, %r609;
mov.b32 %r645, %r609;
mov.b32 %r646, %r609;
mov.b32 %r647, %r609;
mov.b32 %r648, %r609;
mov.b32 %r649, %r609;
mov.b32 %r650, %r609;
mov.b32 %r651, %r609;
mov.b32 %r652, %r609;
mov.b32 %r653, %r609;
mov.b32 %r654, %r609;
mov.b32 %r655, %r609;
mov.b32 %r656, %r609;
mov.b32 %r657, %r609;
mov.b32 %r658, %r609;
mov.b32 %r659, %r609;
mov.b32 %r660, %r609;
mov.b32 %r661, %r609;
mov.b32 %r662, %r609;
mov.b32 %r663, %r609;
mov.b32 %r664, %r609;
mov.b32 %r665, %r609;
mov.b32 %r666, %r609;
mov.b32 %r667, %r609;
mov.b32 %r668, %r609;
mov.b32 %r669, %r609;
mov.b32 %r670, %r609;
mov.b32 %r671, %r609;
mov.b32 %r672, %r609;
mov.b32 %r673, %r144;
$L__BB0_4:
setp.lt.s32 %p10, %r673, %r31;
add.s32 %r154, %r607, 1;
setp.gt.s32 %p11, %r154, 3;
selp.b32 %r607, 0, %r154, %p11;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r155, %r607, 14;
add.s32 %r156, %r134, %r155;
add.s32 %r157, %r156, 65536;
shr.u32 %r158, %r156, 4;
shfl.sync.idx.b32 %r159, %r32, 0, 31, -1;
shl.b32 %r160, %r159, 7;
and.b32 %r161, %r160, 512;
add.s32 %r162, %r161, %r158;
and.b32 %r163, %r162, 16383;
cvt.u64.u32 %rd71, %r163;
bfe.u32 %r164, %r157, 4, 14;
cvt.u64.u32 %rd72, %r164;
wgmma.fence.sync.aligned;
or.b64 %rd53, %rd71, 4611686293305294848;
or.b64 %rd54, %rd72, 4611686293305294848;
mov.pred %p9, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd53, %rd54, %p9, 1, 1, 0, 0;
add.s64 %rd55, %rd71, 4611686293305294850;
add.s64 %rd56, %rd72, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd55, %rd56, %p9, 1, 1, 0, 0;
add.s64 %rd57, %rd71, 4611686293305294852;
add.s64 %rd58, %rd72, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd57, %rd58, %p9, 1, 1, 0, 0;
add.s64 %rd59, %rd71, 4611686293305294854;
add.s64 %rd60, %rd72, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660,%r661,%r662,%r663,%r664,%r665,%r666,%r667,%r668,%r669,%r670,%r671,%r672}, %rd59, %rd60, %p9, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd61, %r156;
cvt.u64.u32 %rd62, %r157;
mov.b32 %r140, %r144;
mov.b32 %r141, %r144;
mov.b32 %r142, %r144;
mov.b32 %r143, %r144;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd63, %rd97, %rd18;
add.s64 %rd64, %rd97, %rd17;
add.s64 %rd65, %rd97, %rd16;
add.s64 %rd66, %rd97, %rd15;
add.s64 %rd67, %rd97, %rd14;
add.s64 %rd68, %rd97, %rd13;
add.s64 %rd69, %rd97, %rd12;
add.s64 %rd70, %rd97, %rd11;
add.s32 %r165, %r608, 1;
setp.gt.s32 %p12, %r165, 3;
selp.b32 %r608, 0, %r165, %p12;
setp.lt.s32 %p13, %r26, %r606;
shl.b32 %r166, %r608, 14;
add.s32 %r167, %r134, %r166;
bar.sync 0;
add.s32 %r145, %r167, %r30;
selp.b32 %r168, 16, 0, %p13;
selp.b32 %r147, %r168, 0, %p10;
cp.async.cg.shared.global [ %r145 + 0 ], [ %rd63 + 0 ], 0x10, %r147;
add.s32 %r146, %r145, 4096;
cp.async.cg.shared.global [ %r146 + 0 ], [ %rd64 + 0 ], 0x10, %r147;
add.s32 %r148, %r145, 8192;
cp.async.cg.shared.global [ %r148 + 0 ], [ %rd65 + 0 ], 0x10, %r147;
add.s32 %r149, %r145, 12288;
cp.async.cg.shared.global [ %r149 + 0 ], [ %rd66 + 0 ], 0x10, %r147;
cp.async.commit_group;
add.s32 %r150, %r145, 65536;
cp.async.cg.shared.global [ %r150 + 0 ], [ %rd67 + 0 ], 0x10, %r147;
add.s32 %r151, %r145, 69632;
cp.async.cg.shared.global [ %r151 + 0 ], [ %rd68 + 0 ], 0x10, %r147;
add.s32 %r152, %r145, 73728;
cp.async.cg.shared.global [ %r152 + 0 ], [ %rd69 + 0 ], 0x10, %r147;
add.s32 %r153, %r145, 77824;
cp.async.cg.shared.global [ %r153 + 0 ], [ %rd70 + 0 ], 0x10, %r147;
cp.async.commit_group;
add.s32 %r673, %r673, 1;
add.s64 %rd97, %rd97, 128;
add.s32 %r606, %r606, -64;
setp.ne.b32 %p14, %r27, %r673;
@%p14 bra $L__BB0_4;
$L__BB0_5:
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
mad.lo.s32 %r266, %r7, %r34, %r24;
mad.lo.s32 %r267, %r8, %r34, %r24;
mad.lo.s32 %r268, %r9, %r34, %r24;
mad.lo.s32 %r269, %r10, %r34, %r24;
mad.lo.s32 %r270, %r11, %r34, %r24;
mad.lo.s32 %r271, %r12, %r34, %r24;
mad.lo.s32 %r272, %r13, %r34, %r24;
mad.lo.s32 %r273, %r14, %r34, %r24;
mad.wide.s32 %rd73, %r266, 2, %rd1;
mad.wide.s32 %rd74, %r267, 2, %rd1;
mad.wide.s32 %rd75, %r268, 2, %rd1;
mad.wide.s32 %rd76, %r269, 2, %rd1;
mad.wide.s32 %rd77, %r270, 2, %rd1;
mad.wide.s32 %rd78, %r271, 2, %rd1;
mad.wide.s32 %rd79, %r272, 2, %rd1;
mad.wide.s32 %rd80, %r273, 2, %rd1;
setp.lt.s32 %p31, %r7, %r39;
setp.lt.s32 %p32, %r8, %r39;
setp.lt.s32 %p33, %r9, %r39;
setp.lt.s32 %p34, %r10, %r39;
setp.lt.s32 %p35, %r11, %r39;
setp.lt.s32 %p36, %r12, %r39;
setp.lt.s32 %p37, %r13, %r39;
setp.lt.s32 %p38, %r14, %r39;
setp.lt.s32 %p39, %r24, %r39;
and.pred %p15, %p31, %p39;
and.pred %p16, %p32, %p39;
and.pred %p17, %p33, %p39;
and.pred %p18, %p34, %p39;
and.pred %p19, %p35, %p39;
and.pred %p20, %p36, %p39;
and.pred %p21, %p37, %p39;
and.pred %p22, %p38, %p39;
mov.b32 %r173, 0;
mov.u32 %r169, %r173;
mov.u32 %r170, %r173;
mov.u32 %r171, %r173;
mov.u32 %r172, %r173;
@%p15 ld.global.v4.b32 { %r169, %r170, %r171, %r172 }, [ %rd73 + 0 ];
mov.u32 %r174, %r173;
mov.u32 %r175, %r173;
mov.u32 %r176, %r173;
mov.u32 %r177, %r173;
@%p16 ld.global.v4.b32 { %r174, %r175, %r176, %r177 }, [ %rd74 + 0 ];
mov.u32 %r178, %r173;
mov.u32 %r179, %r173;
mov.u32 %r180, %r173;
mov.u32 %r181, %r173;
@%p17 ld.global.v4.b32 { %r178, %r179, %r180, %r181 }, [ %rd75 + 0 ];
mov.u32 %r182, %r173;
mov.u32 %r183, %r173;
mov.u32 %r184, %r173;
mov.u32 %r185, %r173;
@%p18 ld.global.v4.b32 { %r182, %r183, %r184, %r185 }, [ %rd76 + 0 ];
mov.u32 %r186, %r173;
mov.u32 %r187, %r173;
mov.u32 %r188, %r173;
mov.u32 %r189, %r173;
@%p19 ld.global.v4.b32 { %r186, %r187, %r188, %r189 }, [ %rd77 + 0 ];
mov.u32 %r190, %r173;
mov.u32 %r191, %r173;
mov.u32 %r192, %r173;
mov.u32 %r193, %r173;
@%p20 ld.global.v4.b32 { %r190, %r191, %r192, %r193 }, [ %rd78 + 0 ];
mov.u32 %r194, %r173;
mov.u32 %r195, %r173;
mov.u32 %r196, %r173;
mov.u32 %r197, %r173;
@%p21 ld.global.v4.b32 { %r194, %r195, %r196, %r197 }, [ %rd79 + 0 ];
mov.u32 %r198, %r173;
mov.u32 %r199, %r173;
mov.u32 %r200, %r173;
mov.u32 %r201, %r173;
@%p22 ld.global.v4.b32 { %r198, %r199, %r200, %r201 }, [ %rd80 + 0 ];
shl.b32 %r274, %r4, 8;
and.b32 %r275, %r274, 30720;
shl.b32 %r276, %r25, 4;
or.b32 %r277, %r275, %r276;
xor.b32 %r278, %r277, %r5;
add.s32 %r279, %r134, %r278;
st.shared.v4.b32 [%r279], {%r169, %r170, %r171, %r172};
st.shared.v4.b32 [%r279+256], {%r174, %r175, %r176, %r177};
st.shared.v4.b32 [%r279+512], {%r178, %r179, %r180, %r181};
st.shared.v4.b32 [%r279+768], {%r182, %r183, %r184, %r185};
st.shared.v4.b32 [%r279+1024], {%r186, %r187, %r188, %r189};
st.shared.v4.b32 [%r279+1280], {%r190, %r191, %r192, %r193};
st.shared.v4.b32 [%r279+1536], {%r194, %r195, %r196, %r197};
st.shared.v4.b32 [%r279+1792], {%r198, %r199, %r200, %r201};
bar.sync 0;
shl.b32 %r280, %r25, 12;
shl.b32 %r281, %r6, 4;
and.b32 %r282, %r4, 224;
shl.b32 %r283, %r282, 3;
and.b32 %r284, %r4, 16;
or.b32 %r285, %r281, %r283;
xor.b32 %r286, %r285, %r284;
or.b32 %r287, %r286, %r280;
add.s32 %r288, %r134, %r287;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r291, %r292}, [%r288];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r288+2048];
xor.b32 %r297, %r287, 32;
add.s32 %r298, %r134, %r297;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r299, %r300, %r301, %r302}, [%r298];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r303, %r304, %r305, %r306}, [%r298+2048];
xor.b32 %r307, %r287, 64;
add.s32 %r308, %r134, %r307;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r311, %r312}, [%r308];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r313, %r314, %r315, %r316}, [%r308+2048];
xor.b32 %r317, %r287, 96;
add.s32 %r318, %r134, %r317;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r319, %r320, %r321, %r322}, [%r318];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r323, %r324, %r325, %r326}, [%r318+2048];
mov.b32 {%rs1, %rs2}, %r289;
cvt.f32.bf16 %r327, %rs1;
cvt.f32.bf16 %r328, %rs2;
mul.f32 %r329, %r38, %r328;
mul.f32 %r330, %r38, %r327;
fma.rn.f32 %r331, %r37, %r609, %r330;
fma.rn.f32 %r332, %r37, %r610, %r329;
cvt.rn.bf16x2.f32 %r333, %r332, %r331;
mov.b32 {%rs3, %rs4}, %r290;
cvt.f32.bf16 %r334, %rs3;
cvt.f32.bf16 %r335, %rs4;
mul.f32 %r336, %r38, %r335;
mul.f32 %r337, %r38, %r334;
fma.rn.f32 %r338, %r37, %r611, %r337;
fma.rn.f32 %r339, %r37, %r612, %r336;
cvt.rn.bf16x2.f32 %r340, %r339, %r338;
mov.b32 {%rs5, %rs6}, %r291;
cvt.f32.bf16 %r341, %rs5;
cvt.f32.bf16 %r342, %rs6;
mul.f32 %r343, %r38, %r342;
mul.f32 %r344, %r38, %r341;
fma.rn.f32 %r345, %r37, %r613, %r344;
fma.rn.f32 %r346, %r37, %r614, %r343;
cvt.rn.bf16x2.f32 %r347, %r346, %r345;
mov.b32 {%rs7, %rs8}, %r292;
cvt.f32.bf16 %r348, %rs7;
cvt.f32.bf16 %r349, %rs8;
mul.f32 %r350, %r38, %r349;
mul.f32 %r351, %r38, %r348;
fma.rn.f32 %r352, %r37, %r615, %r351;
fma.rn.f32 %r353, %r37, %r616, %r350;
cvt.rn.bf16x2.f32 %r354, %r353, %r352;
mov.b32 {%rs9, %rs10}, %r299;
cvt.f32.bf16 %r355, %rs9;
cvt.f32.bf16 %r356, %rs10;
mul.f32 %r357, %r38, %r356;
mul.f32 %r358, %r38, %r355;
fma.rn.f32 %r359, %r37, %r617, %r358;
fma.rn.f32 %r360, %r37, %r618, %r357;
cvt.rn.bf16x2.f32 %r361, %r360, %r359;
mov.b32 {%rs11, %rs12}, %r300;
cvt.f32.bf16 %r362, %rs11;
cvt.f32.bf16 %r363, %rs12;
mul.f32 %r364, %r38, %r363;
mul.f32 %r365, %r38, %r362;
fma.rn.f32 %r366, %r37, %r619, %r365;
fma.rn.f32 %r367, %r37, %r620, %r364;
cvt.rn.bf16x2.f32 %r368, %r367, %r366;
mov.b32 {%rs13, %rs14}, %r301;
cvt.f32.bf16 %r369, %rs13;
cvt.f32.bf16 %r370, %rs14;
mul.f32 %r371, %r38, %r370;
mul.f32 %r372, %r38, %r369;
fma.rn.f32 %r373, %r37, %r621, %r372;
fma.rn.f32 %r374, %r37, %r622, %r371;
cvt.rn.bf16x2.f32 %r375, %r374, %r373;
mov.b32 {%rs15, %rs16}, %r302;
cvt.f32.bf16 %r376, %rs15;
cvt.f32.bf16 %r377, %rs16;
mul.f32 %r378, %r38, %r377;
mul.f32 %r379, %r38, %r376;
fma.rn.f32 %r380, %r37, %r623, %r379;
fma.rn.f32 %r381, %r37, %r624, %r378;
cvt.rn.bf16x2.f32 %r382, %r381, %r380;
mov.b32 {%rs17, %rs18}, %r309;
cvt.f32.bf16 %r383, %rs17;
cvt.f32.bf16 %r384, %rs18;
mul.f32 %r385, %r38, %r384;
mul.f32 %r386, %r38, %r383;
fma.rn.f32 %r387, %r37, %r625, %r386;
fma.rn.f32 %r388, %r37, %r626, %r385;
cvt.rn.bf16x2.f32 %r389, %r388, %r387;
mov.b32 {%rs19, %rs20}, %r310;
cvt.f32.bf16 %r390, %rs19;
cvt.f32.bf16 %r391, %rs20;
mul.f32 %r392, %r38, %r391;
mul.f32 %r393, %r38, %r390;
fma.rn.f32 %r394, %r37, %r627, %r393;
fma.rn.f32 %r395, %r37, %r628, %r392;
cvt.rn.bf16x2.f32 %r396, %r395, %r394;
mov.b32 {%rs21, %rs22}, %r311;
cvt.f32.bf16 %r397, %rs21;
cvt.f32.bf16 %r398, %rs22;
mul.f32 %r399, %r38, %r398;
mul.f32 %r400, %r38, %r397;
fma.rn.f32 %r401, %r37, %r629, %r400;
fma.rn.f32 %r402, %r37, %r630, %r399;
cvt.rn.bf16x2.f32 %r403, %r402, %r401;
mov.b32 {%rs23, %rs24}, %r312;
cvt.f32.bf16 %r404, %rs23;
cvt.f32.bf16 %r405, %rs24;
mul.f32 %r406, %r38, %r405;
mul.f32 %r407, %r38, %r404;
fma.rn.f32 %r408, %r37, %r631, %r407;
fma.rn.f32 %r409, %r37, %r632, %r406;
cvt.rn.bf16x2.f32 %r410, %r409, %r408;
mov.b32 {%rs25, %rs26}, %r319;
cvt.f32.bf16 %r411, %rs25;
cvt.f32.bf16 %r412, %rs26;
mul.f32 %r413, %r38, %r412;
mul.f32 %r414, %r38, %r411;
fma.rn.f32 %r415, %r37, %r633, %r414;
fma.rn.f32 %r416, %r37, %r634, %r413;
cvt.rn.bf16x2.f32 %r417, %r416, %r415;
mov.b32 {%rs27, %rs28}, %r320;
cvt.f32.bf16 %r418, %rs27;
cvt.f32.bf16 %r419, %rs28;
mul.f32 %r420, %r38, %r419;
mul.f32 %r421, %r38, %r418;
fma.rn.f32 %r422, %r37, %r635, %r421;
fma.rn.f32 %r423, %r37, %r636, %r420;
cvt.rn.bf16x2.f32 %r424, %r423, %r422;
mov.b32 {%rs29, %rs30}, %r321;
cvt.f32.bf16 %r425, %rs29;
cvt.f32.bf16 %r426, %rs30;
mul.f32 %r427, %r38, %r426;
mul.f32 %r428, %r38, %r425;
fma.rn.f32 %r429, %r37, %r637, %r428;
fma.rn.f32 %r430, %r37, %r638, %r427;
cvt.rn.bf16x2.f32 %r431, %r430, %r429;
mov.b32 {%rs31, %rs32}, %r322;
cvt.f32.bf16 %r432, %rs31;
cvt.f32.bf16 %r433, %rs32;
mul.f32 %r434, %r38, %r433;
mul.f32 %r435, %r38, %r432;
fma.rn.f32 %r436, %r37, %r639, %r435;
fma.rn.f32 %r437, %r37, %r640, %r434;
cvt.rn.bf16x2.f32 %r438, %r437, %r436;
mov.b32 {%rs33, %rs34}, %r293;
cvt.f32.bf16 %r439, %rs33;
cvt.f32.bf16 %r440, %rs34;
mul.f32 %r441, %r38, %r440;
mul.f32 %r442, %r38, %r439;
fma.rn.f32 %r443, %r37, %r641, %r442;
fma.rn.f32 %r444, %r37, %r642, %r441;
cvt.rn.bf16x2.f32 %r445, %r444, %r443;
mov.b32 {%rs35, %rs36}, %r294;
cvt.f32.bf16 %r446, %rs35;
cvt.f32.bf16 %r447, %rs36;
mul.f32 %r448, %r38, %r447;
mul.f32 %r449, %r38, %r446;
fma.rn.f32 %r450, %r37, %r643, %r449;
fma.rn.f32 %r451, %r37, %r644, %r448;
cvt.rn.bf16x2.f32 %r452, %r451, %r450;
mov.b32 {%rs37, %rs38}, %r295;
cvt.f32.bf16 %r453, %rs37;
cvt.f32.bf16 %r454, %rs38;
mul.f32 %r455, %r38, %r454;
mul.f32 %r456, %r38, %r453;
fma.rn.f32 %r457, %r37, %r645, %r456;
fma.rn.f32 %r458, %r37, %r646, %r455;
cvt.rn.bf16x2.f32 %r459, %r458, %r457;
mov.b32 {%rs39, %rs40}, %r296;
cvt.f32.bf16 %r460, %rs39;
cvt.f32.bf16 %r461, %rs40;
mul.f32 %r462, %r38, %r461;
mul.f32 %r463, %r38, %r460;
fma.rn.f32 %r464, %r37, %r647, %r463;
fma.rn.f32 %r465, %r37, %r648, %r462;
cvt.rn.bf16x2.f32 %r466, %r465, %r464;
mov.b32 {%rs41, %rs42}, %r303;
cvt.f32.bf16 %r467, %rs41;
cvt.f32.bf16 %r468, %rs42;
mul.f32 %r469, %r38, %r468;
mul.f32 %r470, %r38, %r467;
fma.rn.f32 %r471, %r37, %r649, %r470;
fma.rn.f32 %r472, %r37, %r650, %r469;
cvt.rn.bf16x2.f32 %r473, %r472, %r471;
mov.b32 {%rs43, %rs44}, %r304;
cvt.f32.bf16 %r474, %rs43;
cvt.f32.bf16 %r475, %rs44;
mul.f32 %r476, %r38, %r475;
mul.f32 %r477, %r38, %r474;
fma.rn.f32 %r478, %r37, %r651, %r477;
fma.rn.f32 %r479, %r37, %r652, %r476;
cvt.rn.bf16x2.f32 %r480, %r479, %r478;
mov.b32 {%rs45, %rs46}, %r305;
cvt.f32.bf16 %r481, %rs45;
cvt.f32.bf16 %r482, %rs46;
mul.f32 %r483, %r38, %r482;
mul.f32 %r484, %r38, %r481;
fma.rn.f32 %r485, %r37, %r653, %r484;
fma.rn.f32 %r486, %r37, %r654, %r483;
cvt.rn.bf16x2.f32 %r487, %r486, %r485;
mov.b32 {%rs47, %rs48}, %r306;
cvt.f32.bf16 %r488, %rs47;
cvt.f32.bf16 %r489, %rs48;
mul.f32 %r490, %r38, %r489;
mul.f32 %r491, %r38, %r488;
fma.rn.f32 %r492, %r37, %r655, %r491;
fma.rn.f32 %r493, %r37, %r656, %r490;
cvt.rn.bf16x2.f32 %r494, %r493, %r492;
mov.b32 {%rs49, %rs50}, %r313;
cvt.f32.bf16 %r495, %rs49;
cvt.f32.bf16 %r496, %rs50;
mul.f32 %r497, %r38, %r496;
mul.f32 %r498, %r38, %r495;
fma.rn.f32 %r499, %r37, %r657, %r498;
fma.rn.f32 %r500, %r37, %r658, %r497;
cvt.rn.bf16x2.f32 %r501, %r500, %r499;
mov.b32 {%rs51, %rs52}, %r314;
cvt.f32.bf16 %r502, %rs51;
cvt.f32.bf16 %r503, %rs52;
mul.f32 %r504, %r38, %r503;
mul.f32 %r505, %r38, %r502;
fma.rn.f32 %r506, %r37, %r659, %r505;
fma.rn.f32 %r507, %r37, %r660, %r504;
cvt.rn.bf16x2.f32 %r508, %r507, %r506;
mov.b32 {%rs53, %rs54}, %r315;
cvt.f32.bf16 %r509, %rs53;
cvt.f32.bf16 %r510, %rs54;
mul.f32 %r511, %r38, %r510;
mul.f32 %r512, %r38, %r509;
fma.rn.f32 %r513, %r37, %r661, %r512;
fma.rn.f32 %r514, %r37, %r662, %r511;
cvt.rn.bf16x2.f32 %r515, %r514, %r513;
mov.b32 {%rs55, %rs56}, %r316;
cvt.f32.bf16 %r516, %rs55;
cvt.f32.bf16 %r517, %rs56;
mul.f32 %r518, %r38, %r517;
mul.f32 %r519, %r38, %r516;
fma.rn.f32 %r520, %r37, %r663, %r519;
fma.rn.f32 %r521, %r37, %r664, %r518;
cvt.rn.bf16x2.f32 %r522, %r521, %r520;
mov.b32 {%rs57, %rs58}, %r323;
cvt.f32.bf16 %r523, %rs57;
cvt.f32.bf16 %r524, %rs58;
mul.f32 %r525, %r38, %r524;
mul.f32 %r526, %r38, %r523;
fma.rn.f32 %r527, %r37, %r665, %r526;
fma.rn.f32 %r528, %r37, %r666, %r525;
cvt.rn.bf16x2.f32 %r529, %r528, %r527;
mov.b32 {%rs59, %rs60}, %r324;
cvt.f32.bf16 %r530, %rs59;
cvt.f32.bf16 %r531, %rs60;
mul.f32 %r532, %r38, %r531;
mul.f32 %r533, %r38, %r530;
fma.rn.f32 %r534, %r37, %r667, %r533;
fma.rn.f32 %r535, %r37, %r668, %r532;
cvt.rn.bf16x2.f32 %r536, %r535, %r534;
mov.b32 {%rs61, %rs62}, %r325;
cvt.f32.bf16 %r537, %rs61;
cvt.f32.bf16 %r538, %rs62;
mul.f32 %r539, %r38, %r538;
mul.f32 %r540, %r38, %r537;
fma.rn.f32 %r541, %r37, %r669, %r540;
fma.rn.f32 %r542, %r37, %r670, %r539;
cvt.rn.bf16x2.f32 %r543, %r542, %r541;
mov.b32 {%rs63, %rs64}, %r326;
cvt.f32.bf16 %r544, %rs63;
cvt.f32.bf16 %r545, %rs64;
mul.f32 %r546, %r38, %r545;
mul.f32 %r547, %r38, %r544;
fma.rn.f32 %r548, %r37, %r671, %r547;
fma.rn.f32 %r549, %r37, %r672, %r546;
cvt.rn.bf16x2.f32 %r550, %r549, %r548;
bar.sync 0;
shl.b32 %r551, %r4, 5;
and.b32 %r552, %r551, 3072;
and.b32 %r553, %r28, 368;
bfe.s32 %r554, %r4, 3, 1;
and.b32 %r555, %r554, 528;
or.b32 %r556, %r552, %r553;
or.b32 %r557, %r555, %r282;
xor.b32 %r558, %r557, %r556;
add.s32 %r559, %r134, %r558;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r333, %r340, %r347, %r354};
bar.sync 0;
shl.b32 %r560, %r25, 9;
and.b32 %r561, %r4, 112;
and.b32 %r562, %r29, 256;
xor.b32 %r563, %r281, %r561;
add.s32 %r564, %r134, %r560;
add.s32 %r565, %r564, %r562;
add.s32 %r566, %r565, %r563;
ld.shared.v4.b32 {%r234, %r235, %r236, %r237}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r361, %r368, %r375, %r382};
bar.sync 0;
ld.shared.v4.b32 {%r238, %r239, %r240, %r241}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r389, %r396, %r403, %r410};
bar.sync 0;
ld.shared.v4.b32 {%r242, %r243, %r244, %r245}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r417, %r424, %r431, %r438};
bar.sync 0;
ld.shared.v4.b32 {%r246, %r247, %r248, %r249}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r445, %r452, %r459, %r466};
bar.sync 0;
ld.shared.v4.b32 {%r250, %r251, %r252, %r253}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r473, %r480, %r487, %r494};
bar.sync 0;
ld.shared.v4.b32 {%r254, %r255, %r256, %r257}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r501, %r508, %r515, %r522};
bar.sync 0;
ld.shared.v4.b32 {%r258, %r259, %r260, %r261}, [%r566];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r559], {%r529, %r536, %r543, %r550};
bar.sync 0;
ld.shared.v4.b32 {%r262, %r263, %r264, %r265}, [%r566];
bar.sync 0;
and.b32 %r567, %r4, 3;
shl.b32 %r568, %r567, 13;
and.b32 %r569, %r551, 7264;
and.b32 %r570, %r4, 24;
shl.b32 %r571, %r570, 4;
shl.b32 %r572, %r4, 2;
and.b32 %r573, %r572, 16;
or.b32 %r574, %r568, %r573;
or.b32 %r575, %r569, %r571;
or.b32 %r576, %r574, %r575;
add.s32 %r577, %r134, %r576;
st.shared.v4.b32 [%r577], {%r333, %r347, %r361, %r375};
st.shared.v4.b32 [%r577+512], {%r340, %r354, %r368, %r382};
xor.b32 %r578, %r576, 32;
add.s32 %r579, %r134, %r578;
st.shared.v4.b32 [%r579], {%r389, %r403, %r417, %r431};
st.shared.v4.b32 [%r579+512], {%r396, %r410, %r424, %r438};
xor.b32 %r580, %r576, 64;
add.s32 %r581, %r134, %r580;
st.shared.v4.b32 [%r581], {%r445, %r459, %r473, %r487};
st.shared.v4.b32 [%r581+512], {%r452, %r466, %r480, %r494};
xor.b32 %r582, %r576, 96;
add.s32 %r583, %r134, %r582;
st.shared.v4.b32 [%r583], {%r501, %r515, %r529, %r543};
st.shared.v4.b32 [%r583+512], {%r508, %r522, %r536, %r550};
bar.sync 0;
shl.b32 %r584, %r570, 10;
shl.b32 %r585, %r567, 5;
and.b32 %r586, %r572, 1008;
or.b32 %r587, %r584, %r585;
xor.b32 %r588, %r587, %r586;
add.s32 %r589, %r134, %r588;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r204, %r205}, [%r589];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r589+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r212, %r213}, [%r589+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r216, %r217}, [%r589+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r220, %r221}, [%r589+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r224, %r225}, [%r589+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r228, %r229}, [%r589+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r232, %r233}, [%r589+7168];
mad.lo.s32 %r590, %r7, %r36, %r24;
mad.lo.s32 %r591, %r8, %r36, %r24;
mad.lo.s32 %r592, %r9, %r36, %r24;
mad.lo.s32 %r593, %r10, %r36, %r24;
mad.lo.s32 %r594, %r11, %r36, %r24;
mad.lo.s32 %r595, %r12, %r36, %r24;
mad.lo.s32 %r596, %r13, %r36, %r24;
mad.lo.s32 %r597, %r14, %r36, %r24;
mad.wide.s32 %rd81, %r590, 2, %rd2;
mad.wide.s32 %rd82, %r591, 2, %rd2;
mad.wide.s32 %rd83, %r592, 2, %rd2;
mad.wide.s32 %rd84, %r593, 2, %rd2;
mad.wide.s32 %rd85, %r594, 2, %rd2;
mad.wide.s32 %rd86, %r595, 2, %rd2;
mad.wide.s32 %rd87, %r596, 2, %rd2;
mad.wide.s32 %rd88, %r597, 2, %rd2;
@%p15 st.global.v4.b32 [ %rd81 + 0 ], { %r202, %r203, %r204, %r205 };
@%p16 st.global.v4.b32 [ %rd82 + 0 ], { %r206, %r207, %r208, %r209 };
@%p17 st.global.v4.b32 [ %rd83 + 0 ], { %r210, %r211, %r212, %r213 };
@%p18 st.global.v4.b32 [ %rd84 + 0 ], { %r214, %r215, %r216, %r217 };
@%p19 st.global.v4.b32 [ %rd85 + 0 ], { %r218, %r219, %r220, %r221 };
@%p20 st.global.v4.b32 [ %rd86 + 0 ], { %r222, %r223, %r224, %r225 };
@%p21 st.global.v4.b32 [ %rd87 + 0 ], { %r226, %r227, %r228, %r229 };
@%p22 st.global.v4.b32 [ %rd88 + 0 ], { %r230, %r231, %r232, %r233 };
mad.lo.s32 %r598, %r16, %r36, %r15;
mad.lo.s32 %r599, %r17, %r36, %r15;
mad.lo.s32 %r600, %r18, %r36, %r15;
mad.lo.s32 %r601, %r19, %r36, %r15;
mad.lo.s32 %r602, %r20, %r36, %r15;
mad.lo.s32 %r603, %r21, %r36, %r15;
mad.lo.s32 %r604, %r22, %r36, %r15;
mad.lo.s32 %r605, %r23, %r36, %r15;
mad.wide.s32 %rd89, %r598, 2, %rd2;
mad.wide.s32 %rd90, %r599, 2, %rd2;
mad.wide.s32 %rd91, %r600, 2, %rd2;
mad.wide.s32 %rd92, %r601, 2, %rd2;
mad.wide.s32 %rd93, %r602, 2, %rd2;
mad.wide.s32 %rd94, %r603, 2, %rd2;
mad.wide.s32 %rd95, %r604, 2, %rd2;
mad.wide.s32 %rd96, %r605, 2, %rd2;
setp.lt.s32 %p40, %r16, %r39;
setp.lt.s32 %p41, %r17, %r39;
setp.lt.s32 %p42, %r18, %r39;
setp.lt.s32 %p43, %r19, %r39;
setp.lt.s32 %p44, %r20, %r39;
setp.lt.s32 %p45, %r21, %r39;
setp.lt.s32 %p46, %r22, %r39;
setp.lt.s32 %p47, %r23, %r39;
setp.lt.s32 %p48, %r15, %r39;
and.pred %p23, %p40, %p48;
and.pred %p24, %p41, %p48;
and.pred %p25, %p42, %p48;
and.pred %p26, %p43, %p48;
and.pred %p27, %p44, %p48;
and.pred %p28, %p45, %p48;
and.pred %p29, %p46, %p48;
and.pred %p30, %p47, %p48;
@%p23 st.global.v4.b32 [ %rd89 + 0 ], { %r234, %r235, %r236, %r237 };
@%p24 st.global.v4.b32 [ %rd90 + 0 ], { %r238, %r239, %r240, %r241 };
@%p25 st.global.v4.b32 [ %rd91 + 0 ], { %r242, %r243, %r244, %r245 };
@%p26 st.global.v4.b32 [ %rd92 + 0 ], { %r246, %r247, %r248, %r249 };
@%p27 st.global.v4.b32 [ %rd93 + 0 ], { %r250, %r251, %r252, %r253 };
@%p28 st.global.v4.b32 [ %rd94 + 0 ], { %r254, %r255, %r256, %r257 };
@%p29 st.global.v4.b32 [ %rd95 + 0 ], { %r258, %r259, %r260, %r261 };
@%p30 st.global.v4.b32 [ %rd96 + 0 ], { %r262, %r263, %r264, %r265 };
$L__BB0_1:
ret;
$L__tmp8:
$L__func_end0:
}
  "#
  | .widePolynomial, .compact => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry ba_plus_cAA_kernel(
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_0,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_1,
.param .u32 ba_plus_cAA_kernel_param_2,
.param .u32 ba_plus_cAA_kernel_param_3,
.param .u32 ba_plus_cAA_kernel_param_4,
.param .u32 ba_plus_cAA_kernel_param_5,
.param .u32 ba_plus_cAA_kernel_param_6,
.param .f32 ba_plus_cAA_kernel_param_7,
.param .f32 ba_plus_cAA_kernel_param_8,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_9,
.param .u64 .ptr .global .align 1 ba_plus_cAA_kernel_param_10
)
.reqntid 256
{
.reg .pred %p<48>;
.reg .b16 %rs<65>;
.reg .b32 %r<662>;
.reg .b64 %rd<102>;
$L__func_begin0:
ld.param.b32 %r18, [ba_plus_cAA_kernel_param_8];
ld.param.b32 %r17, [ba_plus_cAA_kernel_param_7];
ld.param.b32 %r16, [ba_plus_cAA_kernel_param_6];
ld.param.b32 %r15, [ba_plus_cAA_kernel_param_5];
ld.param.b32 %r14, [ba_plus_cAA_kernel_param_4];
ld.param.b32 %r13, [ba_plus_cAA_kernel_param_2];
ld.param.b64 %rd18, [ba_plus_cAA_kernel_param_1];
ld.param.b64 %rd43, [ba_plus_cAA_kernel_param_0];
$L__tmp0:
mov.u32 %r46, %ctaid.x;
$L__tmp1:
mul.hi.u32 %r47, %r46, 818089009;
shr.u32 %r1, %r47, 2;
mul.lo.s32 %r48, %r1, 21;
ld.param.b32 %r49, [ba_plus_cAA_kernel_param_3];
sub.s32 %r50, %r46, %r48;
mul.lo.s32 %r51, %r50, 3;
mov.b64 %rd44, 6569755928097638024;
shr.u64 %rd45, %rd44, %r51;
cvt.u32.u64 %r52, %rd45;
mov.b64 %rd46, 6420841298313609216;
shr.u64 %rd47, %rd46, %r51;
cvt.u32.u64 %r53, %rd47;
shl.b32 %r54, %r52, 7;
and.b32 %r2, %r54, 896;
shl.b32 %r55, %r53, 7;
and.b32 %r3, %r55, 896;
$L__tmp2:
mul.lo.s32 %r56, %r49, %r1;
mad.wide.s32 %rd1, %r56, 2, %rd43;
mov.u32 %r4, %tid.x;
shr.u32 %r57, %r4, 3;
bfe.u32 %r58, %r4, 3, 5;
or.b32 %r59, %r58, 32;
or.b32 %r60, %r58, 64;
or.b32 %r61, %r57, 96;
or.b32 %r62, %r2, %r58;
or.b32 %r63, %r2, %r59;
or.b32 %r64, %r2, %r60;
or.b32 %r65, %r2, %r61;
rem.s32 %r66, %r62, %r13;
rem.s32 %r67, %r63, %r13;
rem.s32 %r68, %r64, %r13;
rem.s32 %r69, %r65, %r13;
or.b32 %r70, %r3, %r58;
or.b32 %r71, %r3, %r59;
or.b32 %r72, %r3, %r60;
or.b32 %r73, %r3, %r61;
rem.s32 %r74, %r70, %r13;
rem.s32 %r75, %r71, %r13;
rem.s32 %r76, %r72, %r13;
rem.s32 %r77, %r73, %r13;
and.b32 %r5, %r4, 7;
shl.b32 %r6, %r5, 3;
mad.lo.s32 %r78, %r66, %r14, %r6;
mad.lo.s32 %r79, %r67, %r14, %r6;
mad.lo.s32 %r80, %r68, %r14, %r6;
mad.lo.s32 %r81, %r69, %r14, %r6;
mad.wide.s32 %rd19, %r78, 2, %rd1;
mad.wide.s32 %rd20, %r79, 2, %rd1;
mad.wide.s32 %rd21, %r80, 2, %rd1;
mad.wide.s32 %rd22, %r81, 2, %rd1;
mad.lo.s32 %r82, %r74, %r14, %r6;
mad.lo.s32 %r83, %r75, %r14, %r6;
mad.lo.s32 %r84, %r76, %r14, %r6;
mad.lo.s32 %r85, %r77, %r14, %r6;
mad.wide.s32 %rd23, %r82, 2, %rd1;
mad.wide.s32 %rd24, %r83, 2, %rd1;
mad.wide.s32 %rd25, %r84, 2, %rd1;
mad.wide.s32 %rd26, %r85, 2, %rd1;
$L__tmp3:
add.s32 %r86, %r13, 63;
$L__tmp4:
setp.lt.s32 %p1, %r86, 64;
setp.gt.s32 %p2, %r86, 63;
setp.lt.s32 %p3, %r6, %r13;
shl.b32 %r8, %r4, 4;
and.b32 %r90, %r8, 4080;
shl.b32 %r9, %r4, 1;
and.b32 %r91, %r9, 112;
xor.b32 %r10, %r90, %r91;
mov.b32 %r92, global_smem;
add.s32 %r19, %r92, %r10;
selp.b32 %r93, 16, 0, %p2;
selp.b32 %r21, %r93, 0, %p3;
cp.async.cg.shared.global [ %r19 + 0 ], [ %rd19 + 0 ], 0x10, %r21;
add.s32 %r20, %r19, 4096;
cp.async.cg.shared.global [ %r20 + 0 ], [ %rd20 + 0 ], 0x10, %r21;
add.s32 %r22, %r19, 8192;
cp.async.cg.shared.global [ %r22 + 0 ], [ %rd21 + 0 ], 0x10, %r21;
add.s32 %r23, %r19, 12288;
cp.async.cg.shared.global [ %r23 + 0 ], [ %rd22 + 0 ], 0x10, %r21;
cp.async.commit_group;
add.s32 %r24, %r19, 65536;
cp.async.cg.shared.global [ %r24 + 0 ], [ %rd23 + 0 ], 0x10, %r21;
add.s32 %r25, %r19, 69632;
cp.async.cg.shared.global [ %r25 + 0 ], [ %rd24 + 0 ], 0x10, %r21;
add.s32 %r26, %r19, 73728;
cp.async.cg.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x10, %r21;
add.s32 %r27, %r19, 77824;
cp.async.cg.shared.global [ %r27 + 0 ], [ %rd26 + 0 ], 0x10, %r21;
cp.async.commit_group;
setp.gt.s32 %p4, %r86, 127;
add.s64 %rd27, %rd19, 128;
add.s64 %rd28, %rd20, 128;
add.s64 %rd29, %rd21, 128;
add.s64 %rd30, %rd22, 128;
add.s64 %rd31, %rd23, 128;
add.s64 %rd32, %rd24, 128;
add.s64 %rd33, %rd25, 128;
add.s64 %rd34, %rd26, 128;
add.s32 %r94, %r13, -64;
setp.lt.s32 %p5, %r6, %r94;
bar.sync 0;
add.s32 %r28, %r19, 16384;
selp.b32 %r95, 16, 0, %p5;
selp.b32 %r30, %r95, 0, %p4;
cp.async.cg.shared.global [ %r28 + 0 ], [ %rd27 + 0 ], 0x10, %r30;
add.s32 %r29, %r19, 20480;
cp.async.cg.shared.global [ %r29 + 0 ], [ %rd28 + 0 ], 0x10, %r30;
add.s32 %r31, %r19, 24576;
cp.async.cg.shared.global [ %r31 + 0 ], [ %rd29 + 0 ], 0x10, %r30;
add.s32 %r32, %r19, 28672;
cp.async.cg.shared.global [ %r32 + 0 ], [ %rd30 + 0 ], 0x10, %r30;
cp.async.commit_group;
add.s32 %r33, %r19, 81920;
cp.async.cg.shared.global [ %r33 + 0 ], [ %rd31 + 0 ], 0x10, %r30;
add.s32 %r34, %r19, 86016;
cp.async.cg.shared.global [ %r34 + 0 ], [ %rd32 + 0 ], 0x10, %r30;
add.s32 %r35, %r19, 90112;
cp.async.cg.shared.global [ %r35 + 0 ], [ %rd33 + 0 ], 0x10, %r30;
add.s32 %r36, %r19, 94208;
cp.async.cg.shared.global [ %r36 + 0 ], [ %rd34 + 0 ], 0x10, %r30;
cp.async.commit_group;
setp.gt.s32 %p6, %r86, 191;
add.s64 %rd35, %rd19, 256;
add.s64 %rd36, %rd20, 256;
add.s64 %rd37, %rd21, 256;
add.s64 %rd38, %rd22, 256;
add.s64 %rd39, %rd23, 256;
add.s64 %rd40, %rd24, 256;
add.s64 %rd41, %rd25, 256;
add.s64 %rd42, %rd26, 256;
add.s32 %r96, %r13, -128;
setp.lt.s32 %p7, %r6, %r96;
bar.sync 0;
add.s32 %r37, %r19, 32768;
selp.b32 %r97, 16, 0, %p7;
selp.b32 %r39, %r97, 0, %p6;
cp.async.cg.shared.global [ %r37 + 0 ], [ %rd35 + 0 ], 0x10, %r39;
add.s32 %r38, %r19, 36864;
cp.async.cg.shared.global [ %r38 + 0 ], [ %rd36 + 0 ], 0x10, %r39;
add.s32 %r40, %r19, 40960;
cp.async.cg.shared.global [ %r40 + 0 ], [ %rd37 + 0 ], 0x10, %r39;
add.s32 %r41, %r19, 45056;
cp.async.cg.shared.global [ %r41 + 0 ], [ %rd38 + 0 ], 0x10, %r39;
cp.async.commit_group;
add.s32 %r42, %r19, 98304;
cp.async.cg.shared.global [ %r42 + 0 ], [ %rd39 + 0 ], 0x10, %r39;
add.s32 %r43, %r19, 102400;
cp.async.cg.shared.global [ %r43 + 0 ], [ %rd40 + 0 ], 0x10, %r39;
add.s32 %r44, %r19, 106496;
cp.async.cg.shared.global [ %r44 + 0 ], [ %rd41 + 0 ], 0x10, %r39;
add.s32 %r45, %r19, 110592;
cp.async.cg.shared.global [ %r45 + 0 ], [ %rd42 + 0 ], 0x10, %r39;
cp.async.commit_group;
mov.b32 %r597, 0f00000000;
mov.b32 %r598, %r597;
mov.b32 %r599, %r597;
mov.b32 %r600, %r597;
mov.b32 %r601, %r597;
mov.b32 %r602, %r597;
mov.b32 %r603, %r597;
mov.b32 %r604, %r597;
mov.b32 %r605, %r597;
mov.b32 %r606, %r597;
mov.b32 %r607, %r597;
mov.b32 %r608, %r597;
mov.b32 %r609, %r597;
mov.b32 %r610, %r597;
mov.b32 %r611, %r597;
mov.b32 %r612, %r597;
mov.b32 %r613, %r597;
mov.b32 %r614, %r597;
mov.b32 %r615, %r597;
mov.b32 %r616, %r597;
mov.b32 %r617, %r597;
mov.b32 %r618, %r597;
mov.b32 %r619, %r597;
mov.b32 %r620, %r597;
mov.b32 %r621, %r597;
mov.b32 %r622, %r597;
mov.b32 %r623, %r597;
mov.b32 %r624, %r597;
mov.b32 %r625, %r597;
mov.b32 %r626, %r597;
mov.b32 %r627, %r597;
mov.b32 %r628, %r597;
mov.b32 %r629, %r597;
mov.b32 %r630, %r597;
mov.b32 %r631, %r597;
mov.b32 %r632, %r597;
mov.b32 %r633, %r597;
mov.b32 %r634, %r597;
mov.b32 %r635, %r597;
mov.b32 %r636, %r597;
mov.b32 %r637, %r597;
mov.b32 %r638, %r597;
mov.b32 %r639, %r597;
mov.b32 %r640, %r597;
mov.b32 %r641, %r597;
mov.b32 %r642, %r597;
mov.b32 %r643, %r597;
mov.b32 %r644, %r597;
mov.b32 %r645, %r597;
mov.b32 %r646, %r597;
mov.b32 %r647, %r597;
mov.b32 %r648, %r597;
mov.b32 %r649, %r597;
mov.b32 %r650, %r597;
mov.b32 %r651, %r597;
mov.b32 %r652, %r597;
mov.b32 %r653, %r597;
mov.b32 %r654, %r597;
mov.b32 %r655, %r597;
mov.b32 %r656, %r597;
mov.b32 %r657, %r597;
mov.b32 %r658, %r597;
mov.b32 %r659, %r597;
mov.b32 %r660, %r597;
@%p1 bra $L__BB0_3;
cvt.s64.s32 %rd2, %r78;
cvt.s64.s32 %rd3, %r79;
cvt.s64.s32 %rd4, %r80;
cvt.s64.s32 %rd5, %r81;
cvt.s64.s32 %rd6, %r82;
cvt.s64.s32 %rd7, %r83;
cvt.s64.s32 %rd8, %r84;
cvt.s64.s32 %rd9, %r85;
shr.s32 %r87, %r86, 31;
shr.u32 %r88, %r87, 26;
add.s32 %r89, %r86, %r88;
shr.s32 %r7, %r89, 6;
add.s32 %r11, %r7, -3;
shr.u32 %r12, %r4, 5;
add.s32 %r594, %r13, -192;
shl.b64 %rd48, %rd9, 1;
add.s64 %rd10, %rd48, 384;
shl.b64 %rd49, %rd8, 1;
add.s64 %rd11, %rd49, 384;
shl.b64 %rd50, %rd7, 1;
add.s64 %rd12, %rd50, 384;
shl.b64 %rd51, %rd6, 1;
add.s64 %rd13, %rd51, 384;
shl.b64 %rd52, %rd5, 1;
add.s64 %rd14, %rd52, 384;
shl.b64 %rd53, %rd4, 1;
add.s64 %rd15, %rd53, 384;
shl.b64 %rd54, %rd3, 1;
add.s64 %rd16, %rd54, 384;
shl.b64 %rd55, %rd2, 1;
add.s64 %rd17, %rd55, 384;
mov.b32 %r102, 0;
mov.b32 %r597, 0f00000000;
mov.b32 %r596, 2;
mov.b32 %r595, -1;
mov.b64 %rd101, %rd1;
mov.b32 %r598, %r597;
mov.b32 %r599, %r597;
mov.b32 %r600, %r597;
mov.b32 %r601, %r597;
mov.b32 %r602, %r597;
mov.b32 %r603, %r597;
mov.b32 %r604, %r597;
mov.b32 %r605, %r597;
mov.b32 %r606, %r597;
mov.b32 %r607, %r597;
mov.b32 %r608, %r597;
mov.b32 %r609, %r597;
mov.b32 %r610, %r597;
mov.b32 %r611, %r597;
mov.b32 %r612, %r597;
mov.b32 %r613, %r597;
mov.b32 %r614, %r597;
mov.b32 %r615, %r597;
mov.b32 %r616, %r597;
mov.b32 %r617, %r597;
mov.b32 %r618, %r597;
mov.b32 %r619, %r597;
mov.b32 %r620, %r597;
mov.b32 %r621, %r597;
mov.b32 %r622, %r597;
mov.b32 %r623, %r597;
mov.b32 %r624, %r597;
mov.b32 %r625, %r597;
mov.b32 %r626, %r597;
mov.b32 %r627, %r597;
mov.b32 %r628, %r597;
mov.b32 %r629, %r597;
mov.b32 %r630, %r597;
mov.b32 %r631, %r597;
mov.b32 %r632, %r597;
mov.b32 %r633, %r597;
mov.b32 %r634, %r597;
mov.b32 %r635, %r597;
mov.b32 %r636, %r597;
mov.b32 %r637, %r597;
mov.b32 %r638, %r597;
mov.b32 %r639, %r597;
mov.b32 %r640, %r597;
mov.b32 %r641, %r597;
mov.b32 %r642, %r597;
mov.b32 %r643, %r597;
mov.b32 %r644, %r597;
mov.b32 %r645, %r597;
mov.b32 %r646, %r597;
mov.b32 %r647, %r597;
mov.b32 %r648, %r597;
mov.b32 %r649, %r597;
mov.b32 %r650, %r597;
mov.b32 %r651, %r597;
mov.b32 %r652, %r597;
mov.b32 %r653, %r597;
mov.b32 %r654, %r597;
mov.b32 %r655, %r597;
mov.b32 %r656, %r597;
mov.b32 %r657, %r597;
mov.b32 %r658, %r597;
mov.b32 %r659, %r597;
mov.b32 %r660, %r597;
mov.b32 %r661, %r102;
$L__BB0_2:
setp.lt.s32 %p9, %r661, %r11;
add.s32 %r112, %r595, 1;
setp.gt.s32 %p10, %r112, 3;
selp.b32 %r595, 0, %r112, %p10;
cp.async.wait_group 4;
bar.sync 0;
shl.b32 %r113, %r595, 14;
add.s32 %r114, %r92, %r113;
add.s32 %r115, %r114, 65536;
shr.u32 %r116, %r114, 4;
shfl.sync.idx.b32 %r117, %r12, 0, 31, -1;
shl.b32 %r118, %r117, 7;
and.b32 %r119, %r118, 512;
add.s32 %r120, %r119, %r116;
and.b32 %r121, %r120, 16383;
cvt.u64.u32 %rd74, %r121;
bfe.u32 %r122, %r115, 4, 14;
cvt.u64.u32 %rd75, %r122;
wgmma.fence.sync.aligned;
or.b64 %rd56, %rd74, 4611686293305294848;
or.b64 %rd57, %rd75, 4611686293305294848;
mov.pred %p8, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660}, %rd56, %rd57, %p8, 1, 1, 0, 0;
add.s64 %rd58, %rd74, 4611686293305294850;
add.s64 %rd59, %rd75, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660}, %rd58, %rd59, %p8, 1, 1, 0, 0;
add.s64 %rd60, %rd74, 4611686293305294852;
add.s64 %rd61, %rd75, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660}, %rd60, %rd61, %p8, 1, 1, 0, 0;
add.s64 %rd62, %rd74, 4611686293305294854;
add.s64 %rd63, %rd75, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r597,%r598,%r599,%r600,%r601,%r602,%r603,%r604,%r605,%r606,%r607,%r608,%r609,%r610,%r611,%r612,%r613,%r614,%r615,%r616,%r617,%r618,%r619,%r620,%r621,%r622,%r623,%r624,%r625,%r626,%r627,%r628,%r629,%r630,%r631,%r632,%r633,%r634,%r635,%r636,%r637,%r638,%r639,%r640,%r641,%r642,%r643,%r644,%r645,%r646,%r647,%r648,%r649,%r650,%r651,%r652,%r653,%r654,%r655,%r656,%r657,%r658,%r659,%r660}, %rd62, %rd63, %p8, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
cvt.u64.u32 %rd64, %r114;
cvt.u64.u32 %rd65, %r115;
mov.b32 %r101, %r102;
mov.b32 %r98, %r102;
mov.b32 %r99, %r102;
mov.b32 %r100, %r102;
wgmma.wait_group.sync.aligned 1;
add.s64 %rd66, %rd101, %rd17;
add.s64 %rd67, %rd101, %rd16;
add.s64 %rd68, %rd101, %rd15;
add.s64 %rd69, %rd101, %rd14;
add.s64 %rd70, %rd101, %rd13;
add.s64 %rd71, %rd101, %rd12;
add.s64 %rd72, %rd101, %rd11;
add.s64 %rd73, %rd101, %rd10;
add.s32 %r123, %r596, 1;
setp.gt.s32 %p11, %r123, 3;
selp.b32 %r596, 0, %r123, %p11;
setp.lt.s32 %p12, %r6, %r594;
shl.b32 %r124, %r596, 14;
add.s32 %r125, %r92, %r124;
bar.sync 0;
add.s32 %r103, %r125, %r10;
selp.b32 %r126, 16, 0, %p12;
selp.b32 %r105, %r126, 0, %p9;
cp.async.cg.shared.global [ %r103 + 0 ], [ %rd66 + 0 ], 0x10, %r105;
add.s32 %r104, %r103, 4096;
cp.async.cg.shared.global [ %r104 + 0 ], [ %rd67 + 0 ], 0x10, %r105;
add.s32 %r106, %r103, 8192;
cp.async.cg.shared.global [ %r106 + 0 ], [ %rd68 + 0 ], 0x10, %r105;
add.s32 %r107, %r103, 12288;
cp.async.cg.shared.global [ %r107 + 0 ], [ %rd69 + 0 ], 0x10, %r105;
cp.async.commit_group;
add.s32 %r108, %r103, 65536;
cp.async.cg.shared.global [ %r108 + 0 ], [ %rd70 + 0 ], 0x10, %r105;
add.s32 %r109, %r103, 69632;
cp.async.cg.shared.global [ %r109 + 0 ], [ %rd71 + 0 ], 0x10, %r105;
add.s32 %r110, %r103, 73728;
cp.async.cg.shared.global [ %r110 + 0 ], [ %rd72 + 0 ], 0x10, %r105;
add.s32 %r111, %r103, 77824;
cp.async.cg.shared.global [ %r111 + 0 ], [ %rd73 + 0 ], 0x10, %r105;
cp.async.commit_group;
add.s32 %r661, %r661, 1;
add.s64 %rd101, %rd101, 128;
add.s32 %r594, %r594, -64;
setp.ne.b32 %p13, %r7, %r661;
@%p13 bra $L__BB0_2;
$L__BB0_3:
and.b32 %r224, %r4, 15;
shl.b32 %r225, %r224, 3;
or.b32 %r226, %r3, %r225;
and.b32 %r227, %r4, 240;
bfe.u32 %r228, %r4, 4, 4;
or.b32 %r229, %r228, 112;
or.b32 %r230, %r3, %r229;
or.b32 %r231, %r228, 96;
or.b32 %r232, %r3, %r231;
or.b32 %r233, %r228, 80;
or.b32 %r234, %r3, %r233;
or.b32 %r235, %r228, 64;
or.b32 %r236, %r3, %r235;
or.b32 %r237, %r228, 48;
or.b32 %r238, %r3, %r237;
or.b32 %r239, %r228, 32;
or.b32 %r240, %r3, %r239;
or.b32 %r241, %r228, 16;
or.b32 %r242, %r3, %r241;
or.b32 %r243, %r3, %r228;
or.b32 %r244, %r2, %r225;
or.b32 %r245, %r2, %r229;
or.b32 %r246, %r2, %r231;
or.b32 %r247, %r2, %r233;
or.b32 %r248, %r2, %r235;
or.b32 %r249, %r2, %r237;
or.b32 %r250, %r2, %r239;
or.b32 %r251, %r2, %r241;
or.b32 %r252, %r2, %r228;
mul.lo.s32 %r253, %r15, %r1;
mad.wide.s32 %rd100, %r253, 2, %rd18;
wgmma.wait_group.sync.aligned 0;
cp.async.wait_group 0;
bar.sync 0;
mad.lo.s32 %r254, %r252, %r14, %r226;
mad.lo.s32 %r255, %r251, %r14, %r226;
mad.lo.s32 %r256, %r250, %r14, %r226;
mad.lo.s32 %r257, %r249, %r14, %r226;
mad.lo.s32 %r258, %r248, %r14, %r226;
mad.lo.s32 %r259, %r247, %r14, %r226;
mad.lo.s32 %r260, %r246, %r14, %r226;
mad.lo.s32 %r261, %r245, %r14, %r226;
mad.wide.s32 %rd76, %r254, 2, %rd1;
mad.wide.s32 %rd77, %r255, 2, %rd1;
mad.wide.s32 %rd78, %r256, 2, %rd1;
mad.wide.s32 %rd79, %r257, 2, %rd1;
mad.wide.s32 %rd80, %r258, 2, %rd1;
mad.wide.s32 %rd81, %r259, 2, %rd1;
mad.wide.s32 %rd82, %r260, 2, %rd1;
mad.wide.s32 %rd83, %r261, 2, %rd1;
setp.lt.s32 %p30, %r252, %r13;
setp.lt.s32 %p31, %r251, %r13;
setp.lt.s32 %p32, %r250, %r13;
setp.lt.s32 %p33, %r249, %r13;
setp.lt.s32 %p34, %r248, %r13;
setp.lt.s32 %p35, %r247, %r13;
setp.lt.s32 %p36, %r246, %r13;
setp.lt.s32 %p37, %r245, %r13;
setp.lt.s32 %p38, %r226, %r13;
and.pred %p14, %p30, %p38;
and.pred %p15, %p31, %p38;
and.pred %p16, %p32, %p38;
and.pred %p17, %p33, %p38;
and.pred %p18, %p34, %p38;
and.pred %p19, %p35, %p38;
and.pred %p20, %p36, %p38;
and.pred %p21, %p37, %p38;
mov.b32 %r131, 0;
mov.u32 %r127, %r131;
mov.u32 %r128, %r131;
mov.u32 %r129, %r131;
mov.u32 %r130, %r131;
@%p14 ld.global.v4.b32 { %r127, %r128, %r129, %r130 }, [ %rd76 + 0 ];
mov.u32 %r132, %r131;
mov.u32 %r133, %r131;
mov.u32 %r134, %r131;
mov.u32 %r135, %r131;
@%p15 ld.global.v4.b32 { %r132, %r133, %r134, %r135 }, [ %rd77 + 0 ];
mov.u32 %r136, %r131;
mov.u32 %r137, %r131;
mov.u32 %r138, %r131;
mov.u32 %r139, %r131;
@%p16 ld.global.v4.b32 { %r136, %r137, %r138, %r139 }, [ %rd78 + 0 ];
mov.u32 %r140, %r131;
mov.u32 %r141, %r131;
mov.u32 %r142, %r131;
mov.u32 %r143, %r131;
@%p17 ld.global.v4.b32 { %r140, %r141, %r142, %r143 }, [ %rd79 + 0 ];
mov.u32 %r144, %r131;
mov.u32 %r145, %r131;
mov.u32 %r146, %r131;
mov.u32 %r147, %r131;
@%p18 ld.global.v4.b32 { %r144, %r145, %r146, %r147 }, [ %rd80 + 0 ];
mov.u32 %r148, %r131;
mov.u32 %r149, %r131;
mov.u32 %r150, %r131;
mov.u32 %r151, %r131;
@%p19 ld.global.v4.b32 { %r148, %r149, %r150, %r151 }, [ %rd81 + 0 ];
mov.u32 %r152, %r131;
mov.u32 %r153, %r131;
mov.u32 %r154, %r131;
mov.u32 %r155, %r131;
@%p20 ld.global.v4.b32 { %r152, %r153, %r154, %r155 }, [ %rd82 + 0 ];
mov.u32 %r156, %r131;
mov.u32 %r157, %r131;
mov.u32 %r158, %r131;
mov.u32 %r159, %r131;
@%p21 ld.global.v4.b32 { %r156, %r157, %r158, %r159 }, [ %rd83 + 0 ];
shl.b32 %r262, %r4, 8;
and.b32 %r263, %r262, 30720;
shl.b32 %r264, %r5, 4;
or.b32 %r265, %r263, %r264;
xor.b32 %r266, %r265, %r227;
add.s32 %r267, %r92, %r266;
st.shared.v4.b32 [%r267], {%r127, %r128, %r129, %r130};
st.shared.v4.b32 [%r267+256], {%r132, %r133, %r134, %r135};
st.shared.v4.b32 [%r267+512], {%r136, %r137, %r138, %r139};
st.shared.v4.b32 [%r267+768], {%r140, %r141, %r142, %r143};
st.shared.v4.b32 [%r267+1024], {%r144, %r145, %r146, %r147};
st.shared.v4.b32 [%r267+1280], {%r148, %r149, %r150, %r151};
st.shared.v4.b32 [%r267+1536], {%r152, %r153, %r154, %r155};
st.shared.v4.b32 [%r267+1792], {%r156, %r157, %r158, %r159};
bar.sync 0;
shl.b32 %r268, %r5, 12;
shl.b32 %r269, %r224, 4;
and.b32 %r270, %r4, 224;
shl.b32 %r271, %r270, 3;
and.b32 %r272, %r4, 16;
or.b32 %r273, %r269, %r271;
xor.b32 %r274, %r273, %r272;
or.b32 %r275, %r274, %r268;
add.s32 %r276, %r92, %r275;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r277, %r278, %r279, %r280}, [%r276];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r281, %r282, %r283, %r284}, [%r276+2048];
xor.b32 %r285, %r275, 32;
add.s32 %r286, %r92, %r285;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r287, %r288, %r289, %r290}, [%r286];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r291, %r292, %r293, %r294}, [%r286+2048];
xor.b32 %r295, %r275, 64;
add.s32 %r296, %r92, %r295;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r297, %r298, %r299, %r300}, [%r296];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r301, %r302, %r303, %r304}, [%r296+2048];
xor.b32 %r305, %r275, 96;
add.s32 %r306, %r92, %r305;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r307, %r308, %r309, %r310}, [%r306];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r311, %r312, %r313, %r314}, [%r306+2048];
mov.b32 {%rs1, %rs2}, %r277;
cvt.f32.bf16 %r315, %rs1;
cvt.f32.bf16 %r316, %rs2;
mul.f32 %r317, %r18, %r316;
mul.f32 %r318, %r18, %r315;
fma.rn.f32 %r319, %r17, %r597, %r318;
fma.rn.f32 %r320, %r17, %r598, %r317;
cvt.rn.bf16x2.f32 %r321, %r320, %r319;
mov.b32 {%rs3, %rs4}, %r278;
cvt.f32.bf16 %r322, %rs3;
cvt.f32.bf16 %r323, %rs4;
mul.f32 %r324, %r18, %r323;
mul.f32 %r325, %r18, %r322;
fma.rn.f32 %r326, %r17, %r599, %r325;
fma.rn.f32 %r327, %r17, %r600, %r324;
cvt.rn.bf16x2.f32 %r328, %r327, %r326;
mov.b32 {%rs5, %rs6}, %r279;
cvt.f32.bf16 %r329, %rs5;
cvt.f32.bf16 %r330, %rs6;
mul.f32 %r331, %r18, %r330;
mul.f32 %r332, %r18, %r329;
fma.rn.f32 %r333, %r17, %r601, %r332;
fma.rn.f32 %r334, %r17, %r602, %r331;
cvt.rn.bf16x2.f32 %r335, %r334, %r333;
mov.b32 {%rs7, %rs8}, %r280;
cvt.f32.bf16 %r336, %rs7;
cvt.f32.bf16 %r337, %rs8;
mul.f32 %r338, %r18, %r337;
mul.f32 %r339, %r18, %r336;
fma.rn.f32 %r340, %r17, %r603, %r339;
fma.rn.f32 %r341, %r17, %r604, %r338;
cvt.rn.bf16x2.f32 %r342, %r341, %r340;
mov.b32 {%rs9, %rs10}, %r287;
cvt.f32.bf16 %r343, %rs9;
cvt.f32.bf16 %r344, %rs10;
mul.f32 %r345, %r18, %r344;
mul.f32 %r346, %r18, %r343;
fma.rn.f32 %r347, %r17, %r605, %r346;
fma.rn.f32 %r348, %r17, %r606, %r345;
cvt.rn.bf16x2.f32 %r349, %r348, %r347;
mov.b32 {%rs11, %rs12}, %r288;
cvt.f32.bf16 %r350, %rs11;
cvt.f32.bf16 %r351, %rs12;
mul.f32 %r352, %r18, %r351;
mul.f32 %r353, %r18, %r350;
fma.rn.f32 %r354, %r17, %r607, %r353;
fma.rn.f32 %r355, %r17, %r608, %r352;
cvt.rn.bf16x2.f32 %r356, %r355, %r354;
mov.b32 {%rs13, %rs14}, %r289;
cvt.f32.bf16 %r357, %rs13;
cvt.f32.bf16 %r358, %rs14;
mul.f32 %r359, %r18, %r358;
mul.f32 %r360, %r18, %r357;
fma.rn.f32 %r361, %r17, %r609, %r360;
fma.rn.f32 %r362, %r17, %r610, %r359;
cvt.rn.bf16x2.f32 %r363, %r362, %r361;
mov.b32 {%rs15, %rs16}, %r290;
cvt.f32.bf16 %r364, %rs15;
cvt.f32.bf16 %r365, %rs16;
mul.f32 %r366, %r18, %r365;
mul.f32 %r367, %r18, %r364;
fma.rn.f32 %r368, %r17, %r611, %r367;
fma.rn.f32 %r369, %r17, %r612, %r366;
cvt.rn.bf16x2.f32 %r370, %r369, %r368;
mov.b32 {%rs17, %rs18}, %r297;
cvt.f32.bf16 %r371, %rs17;
cvt.f32.bf16 %r372, %rs18;
mul.f32 %r373, %r18, %r372;
mul.f32 %r374, %r18, %r371;
fma.rn.f32 %r375, %r17, %r613, %r374;
fma.rn.f32 %r376, %r17, %r614, %r373;
cvt.rn.bf16x2.f32 %r377, %r376, %r375;
mov.b32 {%rs19, %rs20}, %r298;
cvt.f32.bf16 %r378, %rs19;
cvt.f32.bf16 %r379, %rs20;
mul.f32 %r380, %r18, %r379;
mul.f32 %r381, %r18, %r378;
fma.rn.f32 %r382, %r17, %r615, %r381;
fma.rn.f32 %r383, %r17, %r616, %r380;
cvt.rn.bf16x2.f32 %r384, %r383, %r382;
mov.b32 {%rs21, %rs22}, %r299;
cvt.f32.bf16 %r385, %rs21;
cvt.f32.bf16 %r386, %rs22;
mul.f32 %r387, %r18, %r386;
mul.f32 %r388, %r18, %r385;
fma.rn.f32 %r389, %r17, %r617, %r388;
fma.rn.f32 %r390, %r17, %r618, %r387;
cvt.rn.bf16x2.f32 %r391, %r390, %r389;
mov.b32 {%rs23, %rs24}, %r300;
cvt.f32.bf16 %r392, %rs23;
cvt.f32.bf16 %r393, %rs24;
mul.f32 %r394, %r18, %r393;
mul.f32 %r395, %r18, %r392;
fma.rn.f32 %r396, %r17, %r619, %r395;
fma.rn.f32 %r397, %r17, %r620, %r394;
cvt.rn.bf16x2.f32 %r398, %r397, %r396;
mov.b32 {%rs25, %rs26}, %r307;
cvt.f32.bf16 %r399, %rs25;
cvt.f32.bf16 %r400, %rs26;
mul.f32 %r401, %r18, %r400;
mul.f32 %r402, %r18, %r399;
fma.rn.f32 %r403, %r17, %r621, %r402;
fma.rn.f32 %r404, %r17, %r622, %r401;
cvt.rn.bf16x2.f32 %r405, %r404, %r403;
mov.b32 {%rs27, %rs28}, %r308;
cvt.f32.bf16 %r406, %rs27;
cvt.f32.bf16 %r407, %rs28;
mul.f32 %r408, %r18, %r407;
mul.f32 %r409, %r18, %r406;
fma.rn.f32 %r410, %r17, %r623, %r409;
fma.rn.f32 %r411, %r17, %r624, %r408;
cvt.rn.bf16x2.f32 %r412, %r411, %r410;
mov.b32 {%rs29, %rs30}, %r309;
cvt.f32.bf16 %r413, %rs29;
cvt.f32.bf16 %r414, %rs30;
mul.f32 %r415, %r18, %r414;
mul.f32 %r416, %r18, %r413;
fma.rn.f32 %r417, %r17, %r625, %r416;
fma.rn.f32 %r418, %r17, %r626, %r415;
cvt.rn.bf16x2.f32 %r419, %r418, %r417;
mov.b32 {%rs31, %rs32}, %r310;
cvt.f32.bf16 %r420, %rs31;
cvt.f32.bf16 %r421, %rs32;
mul.f32 %r422, %r18, %r421;
mul.f32 %r423, %r18, %r420;
fma.rn.f32 %r424, %r17, %r627, %r423;
fma.rn.f32 %r425, %r17, %r628, %r422;
cvt.rn.bf16x2.f32 %r426, %r425, %r424;
mov.b32 {%rs33, %rs34}, %r281;
cvt.f32.bf16 %r427, %rs33;
cvt.f32.bf16 %r428, %rs34;
mul.f32 %r429, %r18, %r428;
mul.f32 %r430, %r18, %r427;
fma.rn.f32 %r431, %r17, %r629, %r430;
fma.rn.f32 %r432, %r17, %r630, %r429;
cvt.rn.bf16x2.f32 %r433, %r432, %r431;
mov.b32 {%rs35, %rs36}, %r282;
cvt.f32.bf16 %r434, %rs35;
cvt.f32.bf16 %r435, %rs36;
mul.f32 %r436, %r18, %r435;
mul.f32 %r437, %r18, %r434;
fma.rn.f32 %r438, %r17, %r631, %r437;
fma.rn.f32 %r439, %r17, %r632, %r436;
cvt.rn.bf16x2.f32 %r440, %r439, %r438;
mov.b32 {%rs37, %rs38}, %r283;
cvt.f32.bf16 %r441, %rs37;
cvt.f32.bf16 %r442, %rs38;
mul.f32 %r443, %r18, %r442;
mul.f32 %r444, %r18, %r441;
fma.rn.f32 %r445, %r17, %r633, %r444;
fma.rn.f32 %r446, %r17, %r634, %r443;
cvt.rn.bf16x2.f32 %r447, %r446, %r445;
mov.b32 {%rs39, %rs40}, %r284;
cvt.f32.bf16 %r448, %rs39;
cvt.f32.bf16 %r449, %rs40;
mul.f32 %r450, %r18, %r449;
mul.f32 %r451, %r18, %r448;
fma.rn.f32 %r452, %r17, %r635, %r451;
fma.rn.f32 %r453, %r17, %r636, %r450;
cvt.rn.bf16x2.f32 %r454, %r453, %r452;
mov.b32 {%rs41, %rs42}, %r291;
cvt.f32.bf16 %r455, %rs41;
cvt.f32.bf16 %r456, %rs42;
mul.f32 %r457, %r18, %r456;
mul.f32 %r458, %r18, %r455;
fma.rn.f32 %r459, %r17, %r637, %r458;
fma.rn.f32 %r460, %r17, %r638, %r457;
cvt.rn.bf16x2.f32 %r461, %r460, %r459;
mov.b32 {%rs43, %rs44}, %r292;
cvt.f32.bf16 %r462, %rs43;
cvt.f32.bf16 %r463, %rs44;
mul.f32 %r464, %r18, %r463;
mul.f32 %r465, %r18, %r462;
fma.rn.f32 %r466, %r17, %r639, %r465;
fma.rn.f32 %r467, %r17, %r640, %r464;
cvt.rn.bf16x2.f32 %r468, %r467, %r466;
mov.b32 {%rs45, %rs46}, %r293;
cvt.f32.bf16 %r469, %rs45;
cvt.f32.bf16 %r470, %rs46;
mul.f32 %r471, %r18, %r470;
mul.f32 %r472, %r18, %r469;
fma.rn.f32 %r473, %r17, %r641, %r472;
fma.rn.f32 %r474, %r17, %r642, %r471;
cvt.rn.bf16x2.f32 %r475, %r474, %r473;
mov.b32 {%rs47, %rs48}, %r294;
cvt.f32.bf16 %r476, %rs47;
cvt.f32.bf16 %r477, %rs48;
mul.f32 %r478, %r18, %r477;
mul.f32 %r479, %r18, %r476;
fma.rn.f32 %r480, %r17, %r643, %r479;
fma.rn.f32 %r481, %r17, %r644, %r478;
cvt.rn.bf16x2.f32 %r482, %r481, %r480;
mov.b32 {%rs49, %rs50}, %r301;
cvt.f32.bf16 %r483, %rs49;
cvt.f32.bf16 %r484, %rs50;
mul.f32 %r485, %r18, %r484;
mul.f32 %r486, %r18, %r483;
fma.rn.f32 %r487, %r17, %r645, %r486;
fma.rn.f32 %r488, %r17, %r646, %r485;
cvt.rn.bf16x2.f32 %r489, %r488, %r487;
mov.b32 {%rs51, %rs52}, %r302;
cvt.f32.bf16 %r490, %rs51;
cvt.f32.bf16 %r491, %rs52;
mul.f32 %r492, %r18, %r491;
mul.f32 %r493, %r18, %r490;
fma.rn.f32 %r494, %r17, %r647, %r493;
fma.rn.f32 %r495, %r17, %r648, %r492;
cvt.rn.bf16x2.f32 %r496, %r495, %r494;
mov.b32 {%rs53, %rs54}, %r303;
cvt.f32.bf16 %r497, %rs53;
cvt.f32.bf16 %r498, %rs54;
mul.f32 %r499, %r18, %r498;
mul.f32 %r500, %r18, %r497;
fma.rn.f32 %r501, %r17, %r649, %r500;
fma.rn.f32 %r502, %r17, %r650, %r499;
cvt.rn.bf16x2.f32 %r503, %r502, %r501;
mov.b32 {%rs55, %rs56}, %r304;
cvt.f32.bf16 %r504, %rs55;
cvt.f32.bf16 %r505, %rs56;
mul.f32 %r506, %r18, %r505;
mul.f32 %r507, %r18, %r504;
fma.rn.f32 %r508, %r17, %r651, %r507;
fma.rn.f32 %r509, %r17, %r652, %r506;
cvt.rn.bf16x2.f32 %r510, %r509, %r508;
mov.b32 {%rs57, %rs58}, %r311;
cvt.f32.bf16 %r511, %rs57;
cvt.f32.bf16 %r512, %rs58;
mul.f32 %r513, %r18, %r512;
mul.f32 %r514, %r18, %r511;
fma.rn.f32 %r515, %r17, %r653, %r514;
fma.rn.f32 %r516, %r17, %r654, %r513;
cvt.rn.bf16x2.f32 %r517, %r516, %r515;
mov.b32 {%rs59, %rs60}, %r312;
cvt.f32.bf16 %r518, %rs59;
cvt.f32.bf16 %r519, %rs60;
mul.f32 %r520, %r18, %r519;
mul.f32 %r521, %r18, %r518;
fma.rn.f32 %r522, %r17, %r655, %r521;
fma.rn.f32 %r523, %r17, %r656, %r520;
cvt.rn.bf16x2.f32 %r524, %r523, %r522;
mov.b32 {%rs61, %rs62}, %r313;
cvt.f32.bf16 %r525, %rs61;
cvt.f32.bf16 %r526, %rs62;
mul.f32 %r527, %r18, %r526;
mul.f32 %r528, %r18, %r525;
fma.rn.f32 %r529, %r17, %r657, %r528;
fma.rn.f32 %r530, %r17, %r658, %r527;
cvt.rn.bf16x2.f32 %r531, %r530, %r529;
mov.b32 {%rs63, %rs64}, %r314;
cvt.f32.bf16 %r532, %rs63;
cvt.f32.bf16 %r533, %rs64;
mul.f32 %r534, %r18, %r533;
mul.f32 %r535, %r18, %r532;
fma.rn.f32 %r536, %r17, %r659, %r535;
fma.rn.f32 %r537, %r17, %r660, %r534;
cvt.rn.bf16x2.f32 %r538, %r537, %r536;
bar.sync 0;
shl.b32 %r539, %r4, 5;
and.b32 %r540, %r539, 3072;
and.b32 %r541, %r8, 368;
bfe.s32 %r542, %r4, 3, 1;
and.b32 %r543, %r542, 528;
or.b32 %r544, %r540, %r541;
or.b32 %r545, %r543, %r270;
xor.b32 %r546, %r545, %r544;
add.s32 %r547, %r92, %r546;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r321, %r328, %r335, %r342};
bar.sync 0;
shl.b32 %r548, %r5, 9;
and.b32 %r549, %r4, 112;
and.b32 %r550, %r9, 256;
xor.b32 %r551, %r269, %r549;
add.s32 %r552, %r92, %r548;
add.s32 %r553, %r552, %r550;
add.s32 %r554, %r553, %r551;
ld.shared.v4.b32 {%r192, %r193, %r194, %r195}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r349, %r356, %r363, %r370};
bar.sync 0;
ld.shared.v4.b32 {%r196, %r197, %r198, %r199}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r377, %r384, %r391, %r398};
bar.sync 0;
ld.shared.v4.b32 {%r200, %r201, %r202, %r203}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r405, %r412, %r419, %r426};
bar.sync 0;
ld.shared.v4.b32 {%r204, %r205, %r206, %r207}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r433, %r440, %r447, %r454};
bar.sync 0;
ld.shared.v4.b32 {%r208, %r209, %r210, %r211}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r461, %r468, %r475, %r482};
bar.sync 0;
ld.shared.v4.b32 {%r212, %r213, %r214, %r215}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r489, %r496, %r503, %r510};
bar.sync 0;
ld.shared.v4.b32 {%r216, %r217, %r218, %r219}, [%r554];
bar.sync 0;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r547], {%r517, %r524, %r531, %r538};
bar.sync 0;
ld.shared.v4.b32 {%r220, %r221, %r222, %r223}, [%r554];
bar.sync 0;
and.b32 %r555, %r4, 3;
shl.b32 %r556, %r555, 13;
and.b32 %r557, %r539, 7264;
and.b32 %r558, %r4, 24;
shl.b32 %r559, %r558, 4;
shl.b32 %r560, %r4, 2;
and.b32 %r561, %r560, 16;
or.b32 %r562, %r556, %r561;
or.b32 %r563, %r557, %r559;
or.b32 %r564, %r562, %r563;
add.s32 %r565, %r92, %r564;
st.shared.v4.b32 [%r565], {%r321, %r335, %r349, %r363};
st.shared.v4.b32 [%r565+512], {%r328, %r342, %r356, %r370};
xor.b32 %r566, %r564, 32;
add.s32 %r567, %r92, %r566;
st.shared.v4.b32 [%r567], {%r377, %r391, %r405, %r419};
st.shared.v4.b32 [%r567+512], {%r384, %r398, %r412, %r426};
xor.b32 %r568, %r564, 64;
add.s32 %r569, %r92, %r568;
st.shared.v4.b32 [%r569], {%r433, %r447, %r461, %r475};
st.shared.v4.b32 [%r569+512], {%r440, %r454, %r468, %r482};
xor.b32 %r570, %r564, 96;
add.s32 %r571, %r92, %r570;
st.shared.v4.b32 [%r571], {%r489, %r503, %r517, %r531};
st.shared.v4.b32 [%r571+512], {%r496, %r510, %r524, %r538};
bar.sync 0;
shl.b32 %r572, %r558, 10;
shl.b32 %r573, %r555, 5;
and.b32 %r574, %r560, 1008;
or.b32 %r575, %r572, %r573;
xor.b32 %r576, %r575, %r574;
add.s32 %r577, %r92, %r576;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r577];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r166, %r167}, [%r577+1024];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r170, %r171}, [%r577+2048];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r174, %r175}, [%r577+3072];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r176, %r177, %r178, %r179}, [%r577+4096];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r180, %r181, %r182, %r183}, [%r577+5120];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r184, %r185, %r186, %r187}, [%r577+6144];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r188, %r189, %r190, %r191}, [%r577+7168];
mad.lo.s32 %r578, %r252, %r16, %r226;
mad.lo.s32 %r579, %r251, %r16, %r226;
mad.lo.s32 %r580, %r250, %r16, %r226;
mad.lo.s32 %r581, %r249, %r16, %r226;
mad.lo.s32 %r582, %r248, %r16, %r226;
mad.lo.s32 %r583, %r247, %r16, %r226;
mad.lo.s32 %r584, %r246, %r16, %r226;
mad.lo.s32 %r585, %r245, %r16, %r226;
mad.wide.s32 %rd84, %r578, 2, %rd100;
mad.wide.s32 %rd85, %r579, 2, %rd100;
mad.wide.s32 %rd86, %r580, 2, %rd100;
mad.wide.s32 %rd87, %r581, 2, %rd100;
mad.wide.s32 %rd88, %r582, 2, %rd100;
mad.wide.s32 %rd89, %r583, 2, %rd100;
mad.wide.s32 %rd90, %r584, 2, %rd100;
mad.wide.s32 %rd91, %r585, 2, %rd100;
@%p14 st.global.v4.b32 [ %rd84 + 0 ], { %r160, %r161, %r162, %r163 };
@%p15 st.global.v4.b32 [ %rd85 + 0 ], { %r164, %r165, %r166, %r167 };
@%p16 st.global.v4.b32 [ %rd86 + 0 ], { %r168, %r169, %r170, %r171 };
@%p17 st.global.v4.b32 [ %rd87 + 0 ], { %r172, %r173, %r174, %r175 };
@%p18 st.global.v4.b32 [ %rd88 + 0 ], { %r176, %r177, %r178, %r179 };
@%p19 st.global.v4.b32 [ %rd89 + 0 ], { %r180, %r181, %r182, %r183 };
@%p20 st.global.v4.b32 [ %rd90 + 0 ], { %r184, %r185, %r186, %r187 };
@%p21 st.global.v4.b32 [ %rd91 + 0 ], { %r188, %r189, %r190, %r191 };
mad.lo.s32 %r586, %r243, %r16, %r244;
mad.lo.s32 %r587, %r242, %r16, %r244;
mad.lo.s32 %r588, %r240, %r16, %r244;
mad.lo.s32 %r589, %r238, %r16, %r244;
mad.lo.s32 %r590, %r236, %r16, %r244;
mad.lo.s32 %r591, %r234, %r16, %r244;
mad.lo.s32 %r592, %r232, %r16, %r244;
mad.lo.s32 %r593, %r230, %r16, %r244;
mad.wide.s32 %rd92, %r586, 2, %rd100;
mad.wide.s32 %rd93, %r587, 2, %rd100;
mad.wide.s32 %rd94, %r588, 2, %rd100;
mad.wide.s32 %rd95, %r589, 2, %rd100;
mad.wide.s32 %rd96, %r590, 2, %rd100;
mad.wide.s32 %rd97, %r591, 2, %rd100;
mad.wide.s32 %rd98, %r592, 2, %rd100;
mad.wide.s32 %rd99, %r593, 2, %rd100;
setp.lt.s32 %p39, %r243, %r13;
setp.lt.s32 %p40, %r242, %r13;
setp.lt.s32 %p41, %r240, %r13;
setp.lt.s32 %p42, %r238, %r13;
setp.lt.s32 %p43, %r236, %r13;
setp.lt.s32 %p44, %r234, %r13;
setp.lt.s32 %p45, %r232, %r13;
setp.lt.s32 %p46, %r230, %r13;
setp.lt.s32 %p47, %r244, %r13;
and.pred %p22, %p39, %p47;
and.pred %p23, %p40, %p47;
and.pred %p24, %p41, %p47;
and.pred %p25, %p42, %p47;
and.pred %p26, %p43, %p47;
and.pred %p27, %p44, %p47;
and.pred %p28, %p45, %p47;
and.pred %p29, %p46, %p47;
@%p22 st.global.v4.b32 [ %rd92 + 0 ], { %r192, %r193, %r194, %r195 };
@%p23 st.global.v4.b32 [ %rd93 + 0 ], { %r196, %r197, %r198, %r199 };
@%p24 st.global.v4.b32 [ %rd94 + 0 ], { %r200, %r201, %r202, %r203 };
@%p25 st.global.v4.b32 [ %rd95 + 0 ], { %r204, %r205, %r206, %r207 };
@%p26 st.global.v4.b32 [ %rd96 + 0 ], { %r208, %r209, %r210, %r211 };
@%p27 st.global.v4.b32 [ %rd97 + 0 ], { %r212, %r213, %r214, %r215 };
@%p28 st.global.v4.b32 [ %rd98 + 0 ], { %r216, %r217, %r218, %r219 };
@%p29 st.global.v4.b32 [ %rd99 + 0 ], { %r220, %r221, %r222, %r223 };
ret;
$L__tmp5:
$L__func_end0:
}
  "#
  | .widePolynomial, .full => r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry polar(
.param .align 64 .b8 polar_param_0[128],
.param .u32 polar_param_1,
.param .u32 polar_param_2,
.param .u64 polar_param_3,
.param .u64 polar_param_4,
.param .align 64 .b8 polar_param_5[128],
.param .u32 polar_param_6,
.param .u32 polar_param_7,
.param .u64 polar_param_8,
.param .u64 polar_param_9,
.param .u64 .ptr .global .align 1 polar_param_10,
.param .f32 polar_param_11,
.param .f32 polar_param_12,
.param .u64 .ptr .global .align 1 polar_param_13,
.param .u64 .ptr .global .align 1 polar_param_14
)
.reqntid 256
{
.reg .pred %p<32>;
.reg .b16 %rs<65>;
.reg .b32 %r<519>;
.reg .b64 %rd<55>;
$L__func_begin0:
ld.param.b32 %r49, [polar_param_12];
ld.param.b32 %r48, [polar_param_11];
ld.param.b64 %rd1, [polar_param_10];
mov.b64 %rd2, polar_param_0;
mov.b64 %rd3, polar_param_5;
$L__tmp0:
cvta.param.u64 %rd54, %rd3;
cvta.param.u64 %rd26, %rd2;
mov.u32 %r51, %ctaid.x;
mul.hi.u32 %r52, %r51, 818089009;
shr.u32 %r1, %r52, 2;
$L__tmp1:
mul.lo.s32 %r53, %r1, 21;
sub.s32 %r54, %r51, %r53;
mul.lo.s32 %r55, %r54, 3;
mov.b64 %rd4, 6569755928097638024;
shr.u64 %rd5, %rd4, %r55;
cvt.u32.u64 %r56, %rd5;
shl.b32 %r57, %r56, 7;
and.b32 %r2, %r57, 896;
mov.b64 %rd6, 6420841298313609216;
shr.u64 %rd7, %rd6, %r55;
cvt.u32.u64 %r58, %rd7;
shl.b32 %r59, %r58, 7;
and.b32 %r3, %r59, 896;
$L__tmp2:
setp.eq.b32 %p2, %r2, %r3;
mov.u32 %r4, %tid.x;
and.b32 %r5, %r4, 255;
setp.eq.b32 %p1, %r5, 0;
mov.b32 %r60, global_smem;
add.s32 %r90, %r60, 98304;
@%p1 mbarrier.init.shared::cta.b64 [%r90], 1;
bar.sync 0;
add.s32 %r50, %r60, 98312;
@%p1 mbarrier.init.shared::cta.b64 [%r50], 1;
mul.lo.s32 %r6, %r1, 768;
$L__tmp3:
add.s32 %r435, %r2, %r6;
shr.u32 %r7, %r4, 5;
setp.lt.u32 %p31, %r5, 64;
add.s32 %r76, %r3, %r6;
$L__tmp4:
@%p2 bra $L__BB0_3;
bra.uni $L__BB0_1;
$L__BB0_3:
$L__tmp5:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r90], 32768;
bar.sync 0;
shfl.sync.idx.b32 %r93, %r7, 0, 31, -1;
elect.sync %r94|%p18, -1;
and.pred %p17, %p31, %p18;
and.b32 %r95, %r93, 1;
shl.b32 %r96, %r95, 14;
add.s32 %r91, %r60, %r96;
shl.b32 %r92, %r95, 6;
@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r91], [%rd26, {%r92, %r435}], [%r90];
shl.b32 %r97, %r93, 7;
and.b32 %r10, %r97, 512;
$L__tmp6:
or.b32 %r517, %r92, 128;
mov.b32 %r518, 0;
mov.b32 %r452, 0f00000000;
mov.b32 %r453, %r452;
mov.b32 %r454, %r452;
mov.b32 %r455, %r452;
mov.b32 %r456, %r452;
mov.b32 %r457, %r452;
mov.b32 %r458, %r452;
mov.b32 %r459, %r452;
mov.b32 %r460, %r452;
mov.b32 %r461, %r452;
mov.b32 %r462, %r452;
mov.b32 %r463, %r452;
mov.b32 %r464, %r452;
mov.b32 %r465, %r452;
mov.b32 %r466, %r452;
mov.b32 %r467, %r452;
mov.b32 %r468, %r452;
mov.b32 %r469, %r452;
mov.b32 %r470, %r452;
mov.b32 %r471, %r452;
mov.b32 %r472, %r452;
mov.b32 %r473, %r452;
mov.b32 %r474, %r452;
mov.b32 %r475, %r452;
mov.b32 %r476, %r452;
mov.b32 %r477, %r452;
mov.b32 %r478, %r452;
mov.b32 %r479, %r452;
mov.b32 %r480, %r452;
mov.b32 %r481, %r452;
mov.b32 %r482, %r452;
mov.b32 %r483, %r452;
mov.b32 %r484, %r452;
mov.b32 %r485, %r452;
mov.b32 %r486, %r452;
mov.b32 %r487, %r452;
mov.b32 %r488, %r452;
mov.b32 %r489, %r452;
mov.b32 %r490, %r452;
mov.b32 %r491, %r452;
mov.b32 %r492, %r452;
mov.b32 %r493, %r452;
mov.b32 %r494, %r452;
mov.b32 %r495, %r452;
mov.b32 %r496, %r452;
mov.b32 %r497, %r452;
mov.b32 %r498, %r452;
mov.b32 %r499, %r452;
mov.b32 %r500, %r452;
mov.b32 %r501, %r452;
mov.b32 %r502, %r452;
mov.b32 %r503, %r452;
mov.b32 %r504, %r452;
mov.b32 %r505, %r452;
mov.b32 %r506, %r452;
mov.b32 %r507, %r452;
mov.b32 %r508, %r452;
mov.b32 %r509, %r452;
mov.b32 %r510, %r452;
mov.b32 %r511, %r452;
mov.b32 %r512, %r452;
mov.b32 %r513, %r452;
mov.b32 %r514, %r452;
mov.b32 %r515, %r452;
$L__BB0_4:
and.b32 %r102, %r518, 1;
bfe.u32 %r99, %r518, 1, 1;
shl.b32 %r103, %r102, 15;
add.s32 %r104, %r60, %r103;
mad.lo.s32 %r105, %r102, -32760, %r104;
add.s32 %r98, %r105, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r98], %r99;
@!complete bra.uni waitLoop;
}
shr.u32 %r106, %r104, 4;
add.s32 %r107, %r106, %r10;
and.b32 %r108, %r107, 16383;
cvt.u64.u32 %rd43, %r108;
bfe.u32 %r109, %r104, 4, 14;
cvt.u64.u32 %rd44, %r109;
wgmma.fence.sync.aligned;
or.b64 %rd27, %rd43, 4611686293305294848;
or.b64 %rd28, %rd44, 4611686293305294848;
mov.pred %p19, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd27, %rd28, %p19, 1, 1, 0, 0;
add.s64 %rd29, %rd43, 4611686293305294850;
add.s64 %rd30, %rd44, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd29, %rd30, %p19, 1, 1, 0, 0;
add.s64 %rd31, %rd43, 4611686293305294852;
add.s64 %rd32, %rd44, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd31, %rd32, %p19, 1, 1, 0, 0;
add.s64 %rd33, %rd43, 4611686293305294854;
add.s64 %rd34, %rd44, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd33, %rd34, %p19, 1, 1, 0, 0;
add.s64 %rd35, %rd43, 4611686293305295872;
add.s64 %rd36, %rd44, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd35, %rd36, %p19, 1, 1, 0, 0;
add.s64 %rd37, %rd43, 4611686293305295874;
add.s64 %rd38, %rd44, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd37, %rd38, %p19, 1, 1, 0, 0;
add.s64 %rd39, %rd43, 4611686293305295876;
add.s64 %rd40, %rd44, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd39, %rd40, %p19, 1, 1, 0, 0;
add.s64 %rd41, %rd43, 4611686293305295878;
add.s64 %rd42, %rd44, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd41, %rd42, %p19, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 1;
add.s32 %r11, %r518, 1;
and.b32 %r110, %r11, 1;
setp.ne.b32 %p22, %r518, 5;
shl.b32 %r111, %r110, 3;
add.s32 %r112, %r60, %r111;
add.s32 %r100, %r112, 98304;
shl.b32 %r113, %r110, 15;
$L__tmp7:
bar.sync 0;
and.pred %p20, %p1, %p22;
@%p20 mbarrier.arrive.expect_tx.shared.b64 _, [%r100], 32768;
bar.sync 0;
elect.sync %r114|%p23, -1;
and.pred %p24, %p22, %p23;
and.pred %p21, %p31, %p24;
add.s32 %r101, %r91, %r113;
@%p21 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r101], [%rd26, {%r517, %r435}], [%r100];
$L__tmp8:
add.s32 %r517, %r517, 128;
setp.ne.b32 %p25, %r11, 6;
mov.b32 %r518, %r11;
@%p25 bra $L__BB0_4;
wgmma.wait_group.sync.aligned 0;
bra.uni $L__BB0_6;
$L__tmp9:
$L__BB0_1:
bar.sync 0;
@%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r90], 65536;
bar.sync 0;
shfl.sync.idx.b32 %r63, %r7, 0, 31, -1;
elect.sync %r64|%p5, -1;
and.pred %p3, %p31, %p5;
and.b32 %r65, %r63, 1;
shl.b32 %r66, %r65, 14;
add.s32 %r61, %r60, %r66;
shl.b32 %r62, %r65, 6;
@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r61], [%rd26, {%r62, %r435}], [%r90];
$L__tmp10:
bar.sync 0;
elect.sync %r67|%p6, -1;
and.pred %p4, %p31, %p6;
add.s32 %r75, %r61, 65536;
@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r75], [%rd26, {%r62, %r76}], [%r90];
shl.b32 %r68, %r63, 7;
and.b32 %r8, %r68, 512;
add.s32 %r69, %r60, 65536;
bfe.u32 %r70, %r69, 4, 14;
cvt.u64.u32 %rd8, %r70;
or.b64 %rd10, %rd8, 4611686293305294848;
add.s64 %rd12, %rd8, 4611686293305294850;
add.s64 %rd14, %rd8, 4611686293305294852;
add.s64 %rd16, %rd8, 4611686293305294854;
add.s64 %rd18, %rd8, 4611686293305295872;
add.s64 %rd20, %rd8, 4611686293305295874;
add.s64 %rd22, %rd8, 4611686293305295876;
add.s64 %rd24, %rd8, 4611686293305295878;
$L__tmp11:
or.b32 %r451, %r62, 128;
mov.b32 %r516, 0;
mov.b32 %r452, 0f00000000;
mov.b32 %r453, %r452;
mov.b32 %r454, %r452;
mov.b32 %r455, %r452;
mov.b32 %r456, %r452;
mov.b32 %r457, %r452;
mov.b32 %r458, %r452;
mov.b32 %r459, %r452;
mov.b32 %r460, %r452;
mov.b32 %r461, %r452;
mov.b32 %r462, %r452;
mov.b32 %r463, %r452;
mov.b32 %r464, %r452;
mov.b32 %r465, %r452;
mov.b32 %r466, %r452;
mov.b32 %r467, %r452;
mov.b32 %r468, %r452;
mov.b32 %r469, %r452;
mov.b32 %r470, %r452;
mov.b32 %r471, %r452;
mov.b32 %r472, %r452;
mov.b32 %r473, %r452;
mov.b32 %r474, %r452;
mov.b32 %r475, %r452;
mov.b32 %r476, %r452;
mov.b32 %r477, %r452;
mov.b32 %r478, %r452;
mov.b32 %r479, %r452;
mov.b32 %r480, %r452;
mov.b32 %r481, %r452;
mov.b32 %r482, %r452;
mov.b32 %r483, %r452;
mov.b32 %r484, %r452;
mov.b32 %r485, %r452;
mov.b32 %r486, %r452;
mov.b32 %r487, %r452;
mov.b32 %r488, %r452;
mov.b32 %r489, %r452;
mov.b32 %r490, %r452;
mov.b32 %r491, %r452;
mov.b32 %r492, %r452;
mov.b32 %r493, %r452;
mov.b32 %r494, %r452;
mov.b32 %r495, %r452;
mov.b32 %r496, %r452;
mov.b32 %r497, %r452;
mov.b32 %r498, %r452;
mov.b32 %r499, %r452;
mov.b32 %r500, %r452;
mov.b32 %r501, %r452;
mov.b32 %r502, %r452;
mov.b32 %r503, %r452;
mov.b32 %r504, %r452;
mov.b32 %r505, %r452;
mov.b32 %r506, %r452;
mov.b32 %r507, %r452;
mov.b32 %r508, %r452;
mov.b32 %r509, %r452;
mov.b32 %r510, %r452;
mov.b32 %r511, %r452;
mov.b32 %r512, %r452;
mov.b32 %r513, %r452;
mov.b32 %r514, %r452;
mov.b32 %r515, %r452;
$L__BB0_2:
and.b32 %r77, %r516, 1;
bfe.u32 %r72, %r516, 1, 1;
shl.b32 %r78, %r77, 15;
add.s32 %r79, %r60, %r78;
mad.lo.s32 %r80, %r77, -32760, %r79;
add.s32 %r71, %r80, 98304;
bar.sync 0;
{
.reg .pred complete;
waitLoop:
mbarrier.try_wait.parity.shared.b64 complete, [%r71], %r72;
@!complete bra.uni waitLoop;
}
shr.u32 %r81, %r79, 4;
add.s32 %r82, %r81, %r8;
and.b32 %r83, %r82, 16383;
cvt.u64.u32 %rd25, %r83;
wgmma.fence.sync.aligned;
or.b64 %rd9, %rd25, 4611686293305294848;
mov.pred %p7, -1;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd9, %rd10, %p7, 1, 1, 0, 0;
add.s64 %rd11, %rd25, 4611686293305294850;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd11, %rd12, %p7, 1, 1, 0, 0;
add.s64 %rd13, %rd25, 4611686293305294852;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd13, %rd14, %p7, 1, 1, 0, 0;
add.s64 %rd15, %rd25, 4611686293305294854;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd15, %rd16, %p7, 1, 1, 0, 0;
add.s64 %rd17, %rd25, 4611686293305295872;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd17, %rd18, %p7, 1, 1, 0, 0;
add.s64 %rd19, %rd25, 4611686293305295874;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd19, %rd20, %p7, 1, 1, 0, 0;
add.s64 %rd21, %rd25, 4611686293305295876;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd21, %rd22, %p7, 1, 1, 0, 0;
add.s64 %rd23, %rd25, 4611686293305295878;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r452,%r453,%r454,%r455,%r456,%r457,%r458,%r459,%r460,%r461,%r462,%r463,%r464,%r465,%r466,%r467,%r468,%r469,%r470,%r471,%r472,%r473,%r474,%r475,%r476,%r477,%r478,%r479,%r480,%r481,%r482,%r483,%r484,%r485,%r486,%r487,%r488,%r489,%r490,%r491,%r492,%r493,%r494,%r495,%r496,%r497,%r498,%r499,%r500,%r501,%r502,%r503,%r504,%r505,%r506,%r507,%r508,%r509,%r510,%r511,%r512,%r513,%r514,%r515}, %rd23, %rd24, %p7, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;
add.s32 %r9, %r516, 1;
and.b32 %r84, %r9, 1;
setp.ne.b32 %p11, %r516, 5;
shl.b32 %r85, %r84, 3;
add.s32 %r86, %r60, %r85;
add.s32 %r73, %r86, 98304;
shl.b32 %r87, %r84, 15;
$L__tmp12:
bar.sync 0;
and.pred %p8, %p1, %p11;
@%p8 mbarrier.arrive.expect_tx.shared.b64 _, [%r73], 65536;
bar.sync 0;
elect.sync %r88|%p12, -1;
and.pred %p13, %p11, %p12;
and.pred %p9, %p31, %p13;
add.s32 %r74, %r61, %r87;
@%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r74], [%rd26, {%r451, %r435}], [%r73];
$L__tmp13:
wgmma.wait_group.sync.aligned 0;
$L__tmp14:
bar.sync 0;
elect.sync %r89|%p14, -1;
and.pred %p15, %p11, %p14;
and.pred %p10, %p31, %p15;
@%p10 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r75], [%rd26, {%r451, %r76}], [%r73];
$L__tmp15:
add.s32 %r451, %r451, 128;
setp.eq.b32 %p16, %r9, 6;
mov.b32 %r516, %r9;
@%p16 bra $L__BB0_6;
bra.uni $L__BB0_2;
$L__tmp16:
$L__BB0_6:
setp.ne.b32 %p26, %r2, %r3;
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r90];
bar.sync 0;
@%p1 mbarrier.inval.shared::cta.b64 [%r50];
and.b32 %r147, %r4, 240;
bfe.u32 %r148, %r4, 4, 4;
and.b32 %r12, %r4, 15;
shl.b32 %r149, %r12, 3;
mul.lo.s32 %r150, %r1, 589824;
or.b32 %r151, %r2, %r148;
or.b32 %r152, %r151, 16;
or.b32 %r153, %r151, 32;
or.b32 %r154, %r151, 48;
or.b32 %r155, %r151, 64;
or.b32 %r156, %r151, 80;
or.b32 %r157, %r151, 96;
or.b32 %r158, %r151, 112;
or.b32 %r159, %r3, %r149;
mad.lo.s32 %r160, %r151, 768, %r159;
mad.lo.s32 %r161, %r152, 768, %r159;
mad.lo.s32 %r162, %r153, 768, %r159;
mad.lo.s32 %r163, %r154, 768, %r159;
mad.lo.s32 %r164, %r155, 768, %r159;
mad.lo.s32 %r165, %r156, 768, %r159;
mad.lo.s32 %r166, %r157, 768, %r159;
mad.lo.s32 %r167, %r158, 768, %r159;
mad.wide.s32 %rd53, %r150, 2, %rd1;
mad.wide.u32 %rd45, %r160, 2, %rd53;
mad.wide.u32 %rd46, %r161, 2, %rd53;
mad.wide.u32 %rd47, %r162, 2, %rd53;
mad.wide.u32 %rd48, %r163, 2, %rd53;
mad.wide.u32 %rd49, %r164, 2, %rd53;
mad.wide.u32 %rd50, %r165, 2, %rd53;
mad.wide.u32 %rd51, %r166, 2, %rd53;
mad.wide.u32 %rd52, %r167, 2, %rd53;
mov.u32 %r115, 0x0;
mov.u32 %r116, 0x0;
mov.u32 %r117, 0x0;
mov.u32 %r118, 0x0;
ld.global.v4.b32 { %r115, %r116, %r117, %r118 }, [ %rd45 + 0 ];
mov.u32 %r119, 0x0;
mov.u32 %r120, 0x0;
mov.u32 %r121, 0x0;
mov.u32 %r122, 0x0;
ld.global.v4.b32 { %r119, %r120, %r121, %r122 }, [ %rd46 + 0 ];
mov.u32 %r123, 0x0;
mov.u32 %r124, 0x0;
mov.u32 %r125, 0x0;
mov.u32 %r126, 0x0;
ld.global.v4.b32 { %r123, %r124, %r125, %r126 }, [ %rd47 + 0 ];
mov.u32 %r127, 0x0;
mov.u32 %r128, 0x0;
mov.u32 %r129, 0x0;
mov.u32 %r130, 0x0;
ld.global.v4.b32 { %r127, %r128, %r129, %r130 }, [ %rd48 + 0 ];
mov.u32 %r131, 0x0;
mov.u32 %r132, 0x0;
mov.u32 %r133, 0x0;
mov.u32 %r134, 0x0;
ld.global.v4.b32 { %r131, %r132, %r133, %r134 }, [ %rd49 + 0 ];
mov.u32 %r135, 0x0;
mov.u32 %r136, 0x0;
mov.u32 %r137, 0x0;
mov.u32 %r138, 0x0;
ld.global.v4.b32 { %r135, %r136, %r137, %r138 }, [ %rd50 + 0 ];
mov.u32 %r139, 0x0;
mov.u32 %r140, 0x0;
mov.u32 %r141, 0x0;
mov.u32 %r142, 0x0;
ld.global.v4.b32 { %r139, %r140, %r141, %r142 }, [ %rd51 + 0 ];
mov.u32 %r143, 0x0;
mov.u32 %r144, 0x0;
mov.u32 %r145, 0x0;
mov.u32 %r146, 0x0;
ld.global.v4.b32 { %r143, %r144, %r145, %r146 }, [ %rd52 + 0 ];
shl.b32 %r168, %r4, 8;
and.b32 %r169, %r168, 30720;
and.b32 %r170, %r4, 7;
shl.b32 %r13, %r170, 4;
or.b32 %r171, %r169, %r13;
xor.b32 %r172, %r171, %r147;
add.s32 %r173, %r60, %r172;
st.shared.v4.b32 [%r173], {%r115, %r116, %r117, %r118};
st.shared.v4.b32 [%r173+256], {%r119, %r120, %r121, %r122};
st.shared.v4.b32 [%r173+512], {%r123, %r124, %r125, %r126};
st.shared.v4.b32 [%r173+768], {%r127, %r128, %r129, %r130};
st.shared.v4.b32 [%r173+1024], {%r131, %r132, %r133, %r134};
st.shared.v4.b32 [%r173+1280], {%r135, %r136, %r137, %r138};
st.shared.v4.b32 [%r173+1536], {%r139, %r140, %r141, %r142};
st.shared.v4.b32 [%r173+1792], {%r143, %r144, %r145, %r146};
bar.sync 0;
shl.b32 %r174, %r170, 12;
shl.b32 %r175, %r12, 4;
and.b32 %r14, %r4, 224;
shl.b32 %r176, %r14, 3;
and.b32 %r15, %r4, 16;
or.b32 %r177, %r175, %r176;
xor.b32 %r178, %r177, %r15;
or.b32 %r179, %r178, %r174;
add.s32 %r180, %r60, %r179;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r180];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r185, %r186, %r187, %r188}, [%r180+2048];
xor.b32 %r189, %r179, 32;
add.s32 %r190, %r60, %r189;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r191, %r192, %r193, %r194}, [%r190];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r195, %r196, %r197, %r198}, [%r190+2048];
xor.b32 %r199, %r179, 64;
add.s32 %r200, %r60, %r199;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r201, %r202, %r203, %r204}, [%r200];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r205, %r206, %r207, %r208}, [%r200+2048];
xor.b32 %r209, %r179, 96;
add.s32 %r210, %r60, %r209;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r211, %r212, %r213, %r214}, [%r210];
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r215, %r216, %r217, %r218}, [%r210+2048];
mov.b32 {%rs1, %rs2}, %r181;
cvt.f32.bf16 %r219, %rs1;
cvt.f32.bf16 %r220, %rs2;
mul.f32 %r221, %r49, %r220;
mul.f32 %r222, %r49, %r219;
fma.rn.f32 %r223, %r48, %r452, %r222;
fma.rn.f32 %r224, %r48, %r453, %r221;
mov.b32 {%rs3, %rs4}, %r182;
cvt.f32.bf16 %r225, %rs3;
cvt.f32.bf16 %r226, %rs4;
mul.f32 %r227, %r49, %r226;
mul.f32 %r228, %r49, %r225;
fma.rn.f32 %r229, %r48, %r454, %r228;
fma.rn.f32 %r230, %r48, %r455, %r227;
mov.b32 {%rs5, %rs6}, %r183;
cvt.f32.bf16 %r231, %rs5;
cvt.f32.bf16 %r232, %rs6;
mul.f32 %r233, %r49, %r232;
mul.f32 %r234, %r49, %r231;
fma.rn.f32 %r235, %r48, %r456, %r234;
fma.rn.f32 %r236, %r48, %r457, %r233;
mov.b32 {%rs7, %rs8}, %r184;
cvt.f32.bf16 %r237, %rs7;
cvt.f32.bf16 %r238, %rs8;
mul.f32 %r239, %r49, %r238;
mul.f32 %r240, %r49, %r237;
fma.rn.f32 %r241, %r48, %r458, %r240;
fma.rn.f32 %r242, %r48, %r459, %r239;
mov.b32 {%rs9, %rs10}, %r191;
cvt.f32.bf16 %r243, %rs9;
cvt.f32.bf16 %r244, %rs10;
mul.f32 %r245, %r49, %r244;
mul.f32 %r246, %r49, %r243;
fma.rn.f32 %r247, %r48, %r460, %r246;
fma.rn.f32 %r248, %r48, %r461, %r245;
mov.b32 {%rs11, %rs12}, %r192;
cvt.f32.bf16 %r249, %rs11;
cvt.f32.bf16 %r250, %rs12;
mul.f32 %r251, %r49, %r250;
mul.f32 %r252, %r49, %r249;
fma.rn.f32 %r253, %r48, %r462, %r252;
fma.rn.f32 %r254, %r48, %r463, %r251;
mov.b32 {%rs13, %rs14}, %r193;
cvt.f32.bf16 %r255, %rs13;
cvt.f32.bf16 %r256, %rs14;
mul.f32 %r257, %r49, %r256;
mul.f32 %r258, %r49, %r255;
fma.rn.f32 %r259, %r48, %r464, %r258;
fma.rn.f32 %r260, %r48, %r465, %r257;
mov.b32 {%rs15, %rs16}, %r194;
cvt.f32.bf16 %r261, %rs15;
cvt.f32.bf16 %r262, %rs16;
mul.f32 %r263, %r49, %r262;
mul.f32 %r264, %r49, %r261;
fma.rn.f32 %r265, %r48, %r466, %r264;
fma.rn.f32 %r266, %r48, %r467, %r263;
mov.b32 {%rs17, %rs18}, %r201;
cvt.f32.bf16 %r267, %rs17;
cvt.f32.bf16 %r268, %rs18;
mul.f32 %r269, %r49, %r268;
mul.f32 %r270, %r49, %r267;
fma.rn.f32 %r271, %r48, %r468, %r270;
fma.rn.f32 %r272, %r48, %r469, %r269;
mov.b32 {%rs19, %rs20}, %r202;
cvt.f32.bf16 %r273, %rs19;
cvt.f32.bf16 %r274, %rs20;
mul.f32 %r275, %r49, %r274;
mul.f32 %r276, %r49, %r273;
fma.rn.f32 %r277, %r48, %r470, %r276;
fma.rn.f32 %r278, %r48, %r471, %r275;
mov.b32 {%rs21, %rs22}, %r203;
cvt.f32.bf16 %r279, %rs21;
cvt.f32.bf16 %r280, %rs22;
mul.f32 %r281, %r49, %r280;
mul.f32 %r282, %r49, %r279;
fma.rn.f32 %r283, %r48, %r472, %r282;
fma.rn.f32 %r284, %r48, %r473, %r281;
mov.b32 {%rs23, %rs24}, %r204;
cvt.f32.bf16 %r285, %rs23;
cvt.f32.bf16 %r286, %rs24;
mul.f32 %r287, %r49, %r286;
mul.f32 %r288, %r49, %r285;
fma.rn.f32 %r289, %r48, %r474, %r288;
fma.rn.f32 %r290, %r48, %r475, %r287;
mov.b32 {%rs25, %rs26}, %r211;
cvt.f32.bf16 %r291, %rs25;
cvt.f32.bf16 %r292, %rs26;
mul.f32 %r293, %r49, %r292;
mul.f32 %r294, %r49, %r291;
fma.rn.f32 %r295, %r48, %r476, %r294;
fma.rn.f32 %r296, %r48, %r477, %r293;
mov.b32 {%rs27, %rs28}, %r212;
cvt.f32.bf16 %r297, %rs27;
cvt.f32.bf16 %r298, %rs28;
mul.f32 %r299, %r49, %r298;
mul.f32 %r300, %r49, %r297;
fma.rn.f32 %r301, %r48, %r478, %r300;
fma.rn.f32 %r302, %r48, %r479, %r299;
mov.b32 {%rs29, %rs30}, %r213;
cvt.f32.bf16 %r303, %rs29;
cvt.f32.bf16 %r304, %rs30;
mul.f32 %r305, %r49, %r304;
mul.f32 %r306, %r49, %r303;
fma.rn.f32 %r307, %r48, %r480, %r306;
fma.rn.f32 %r308, %r48, %r481, %r305;
mov.b32 {%rs31, %rs32}, %r214;
cvt.f32.bf16 %r309, %rs31;
cvt.f32.bf16 %r310, %rs32;
mul.f32 %r311, %r49, %r310;
mul.f32 %r312, %r49, %r309;
fma.rn.f32 %r313, %r48, %r482, %r312;
fma.rn.f32 %r314, %r48, %r483, %r311;
mov.b32 {%rs33, %rs34}, %r185;
cvt.f32.bf16 %r315, %rs33;
cvt.f32.bf16 %r316, %rs34;
mul.f32 %r317, %r49, %r316;
mul.f32 %r318, %r49, %r315;
fma.rn.f32 %r319, %r48, %r484, %r318;
fma.rn.f32 %r320, %r48, %r485, %r317;
mov.b32 {%rs35, %rs36}, %r186;
cvt.f32.bf16 %r321, %rs35;
cvt.f32.bf16 %r322, %rs36;
mul.f32 %r323, %r49, %r322;
mul.f32 %r324, %r49, %r321;
fma.rn.f32 %r325, %r48, %r486, %r324;
fma.rn.f32 %r326, %r48, %r487, %r323;
mov.b32 {%rs37, %rs38}, %r187;
cvt.f32.bf16 %r327, %rs37;
cvt.f32.bf16 %r328, %rs38;
mul.f32 %r329, %r49, %r328;
mul.f32 %r330, %r49, %r327;
fma.rn.f32 %r331, %r48, %r488, %r330;
fma.rn.f32 %r332, %r48, %r489, %r329;
mov.b32 {%rs39, %rs40}, %r188;
cvt.f32.bf16 %r333, %rs39;
cvt.f32.bf16 %r334, %rs40;
mul.f32 %r335, %r49, %r334;
mul.f32 %r336, %r49, %r333;
fma.rn.f32 %r337, %r48, %r490, %r336;
fma.rn.f32 %r338, %r48, %r491, %r335;
mov.b32 {%rs41, %rs42}, %r195;
cvt.f32.bf16 %r339, %rs41;
cvt.f32.bf16 %r340, %rs42;
mul.f32 %r341, %r49, %r340;
mul.f32 %r342, %r49, %r339;
fma.rn.f32 %r343, %r48, %r492, %r342;
fma.rn.f32 %r344, %r48, %r493, %r341;
mov.b32 {%rs43, %rs44}, %r196;
cvt.f32.bf16 %r345, %rs43;
cvt.f32.bf16 %r346, %rs44;
mul.f32 %r347, %r49, %r346;
mul.f32 %r348, %r49, %r345;
fma.rn.f32 %r349, %r48, %r494, %r348;
fma.rn.f32 %r350, %r48, %r495, %r347;
mov.b32 {%rs45, %rs46}, %r197;
cvt.f32.bf16 %r351, %rs45;
cvt.f32.bf16 %r352, %rs46;
mul.f32 %r353, %r49, %r352;
mul.f32 %r354, %r49, %r351;
fma.rn.f32 %r355, %r48, %r496, %r354;
fma.rn.f32 %r356, %r48, %r497, %r353;
mov.b32 {%rs47, %rs48}, %r198;
cvt.f32.bf16 %r357, %rs47;
cvt.f32.bf16 %r358, %rs48;
mul.f32 %r359, %r49, %r358;
mul.f32 %r360, %r49, %r357;
fma.rn.f32 %r361, %r48, %r498, %r360;
fma.rn.f32 %r362, %r48, %r499, %r359;
mov.b32 {%rs49, %rs50}, %r205;
cvt.f32.bf16 %r363, %rs49;
cvt.f32.bf16 %r364, %rs50;
mul.f32 %r365, %r49, %r364;
mul.f32 %r366, %r49, %r363;
fma.rn.f32 %r367, %r48, %r500, %r366;
fma.rn.f32 %r368, %r48, %r501, %r365;
mov.b32 {%rs51, %rs52}, %r206;
cvt.f32.bf16 %r369, %rs51;
cvt.f32.bf16 %r370, %rs52;
mul.f32 %r371, %r49, %r370;
mul.f32 %r372, %r49, %r369;
fma.rn.f32 %r373, %r48, %r502, %r372;
fma.rn.f32 %r374, %r48, %r503, %r371;
mov.b32 {%rs53, %rs54}, %r207;
cvt.f32.bf16 %r375, %rs53;
cvt.f32.bf16 %r376, %rs54;
mul.f32 %r377, %r49, %r376;
mul.f32 %r378, %r49, %r375;
fma.rn.f32 %r379, %r48, %r504, %r378;
fma.rn.f32 %r380, %r48, %r505, %r377;
mov.b32 {%rs55, %rs56}, %r208;
cvt.f32.bf16 %r381, %rs55;
cvt.f32.bf16 %r382, %rs56;
mul.f32 %r383, %r49, %r382;
mul.f32 %r384, %r49, %r381;
fma.rn.f32 %r385, %r48, %r506, %r384;
fma.rn.f32 %r386, %r48, %r507, %r383;
mov.b32 {%rs57, %rs58}, %r215;
cvt.f32.bf16 %r387, %rs57;
cvt.f32.bf16 %r388, %rs58;
mul.f32 %r389, %r49, %r388;
mul.f32 %r390, %r49, %r387;
fma.rn.f32 %r391, %r48, %r508, %r390;
fma.rn.f32 %r392, %r48, %r509, %r389;
mov.b32 {%rs59, %rs60}, %r216;
cvt.f32.bf16 %r393, %rs59;
cvt.f32.bf16 %r394, %rs60;
mul.f32 %r395, %r49, %r394;
mul.f32 %r396, %r49, %r393;
fma.rn.f32 %r397, %r48, %r510, %r396;
fma.rn.f32 %r398, %r48, %r511, %r395;
mov.b32 {%rs61, %rs62}, %r217;
cvt.f32.bf16 %r399, %rs61;
cvt.f32.bf16 %r400, %rs62;
mul.f32 %r401, %r49, %r400;
mul.f32 %r402, %r49, %r399;
fma.rn.f32 %r403, %r48, %r512, %r402;
fma.rn.f32 %r404, %r48, %r513, %r401;
mov.b32 {%rs63, %rs64}, %r218;
cvt.f32.bf16 %r405, %rs63;
cvt.f32.bf16 %r406, %rs64;
mul.f32 %r407, %r49, %r406;
mul.f32 %r408, %r49, %r405;
fma.rn.f32 %r409, %r48, %r514, %r408;
fma.rn.f32 %r410, %r48, %r515, %r407;
cvt.rn.bf16x2.f32 %r16, %r224, %r223;
cvt.rn.bf16x2.f32 %r17, %r230, %r229;
cvt.rn.bf16x2.f32 %r18, %r236, %r235;
cvt.rn.bf16x2.f32 %r19, %r242, %r241;
cvt.rn.bf16x2.f32 %r20, %r248, %r247;
cvt.rn.bf16x2.f32 %r21, %r254, %r253;
cvt.rn.bf16x2.f32 %r22, %r260, %r259;
cvt.rn.bf16x2.f32 %r23, %r266, %r265;
cvt.rn.bf16x2.f32 %r24, %r272, %r271;
cvt.rn.bf16x2.f32 %r25, %r278, %r277;
cvt.rn.bf16x2.f32 %r26, %r284, %r283;
cvt.rn.bf16x2.f32 %r27, %r290, %r289;
cvt.rn.bf16x2.f32 %r28, %r296, %r295;
cvt.rn.bf16x2.f32 %r29, %r302, %r301;
cvt.rn.bf16x2.f32 %r30, %r308, %r307;
cvt.rn.bf16x2.f32 %r31, %r314, %r313;
cvt.rn.bf16x2.f32 %r32, %r320, %r319;
cvt.rn.bf16x2.f32 %r33, %r326, %r325;
cvt.rn.bf16x2.f32 %r34, %r332, %r331;
cvt.rn.bf16x2.f32 %r35, %r338, %r337;
cvt.rn.bf16x2.f32 %r36, %r344, %r343;
cvt.rn.bf16x2.f32 %r37, %r350, %r349;
cvt.rn.bf16x2.f32 %r38, %r356, %r355;
cvt.rn.bf16x2.f32 %r39, %r362, %r361;
cvt.rn.bf16x2.f32 %r40, %r368, %r367;
cvt.rn.bf16x2.f32 %r41, %r374, %r373;
cvt.rn.bf16x2.f32 %r42, %r380, %r379;
cvt.rn.bf16x2.f32 %r43, %r386, %r385;
cvt.rn.bf16x2.f32 %r44, %r392, %r391;
cvt.rn.bf16x2.f32 %r45, %r398, %r397;
cvt.rn.bf16x2.f32 %r46, %r404, %r403;
cvt.rn.bf16x2.f32 %r47, %r410, %r409;
@%p26 bra $L__BB0_7;
bra.uni $L__BB0_8;
$L__BB0_7:
bar.sync 0;
shl.b32 %r411, %r12, 7;
shl.b32 %r412, %r14, 6;
or.b32 %r413, %r411, %r13;
xor.b32 %r414, %r413, %r15;
or.b32 %r415, %r414, %r412;
add.s32 %r416, %r60, %r415;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r416], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r416+16384], {%r32, %r33, %r34, %r35};
xor.b32 %r417, %r415, 32;
add.s32 %r418, %r60, %r417;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r418], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r418+16384], {%r36, %r37, %r38, %r39};
xor.b32 %r419, %r415, 64;
add.s32 %r420, %r60, %r419;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r420], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r420+16384], {%r40, %r41, %r42, %r43};
xor.b32 %r421, %r415, 96;
add.s32 %r422, %r60, %r421;
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r422], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r422+16384], {%r44, %r45, %r46, %r47};
$L__BB0_8:
shl.b32 %r423, %r4, 7;
and.b32 %r424, %r423, 17280;
and.b32 %r425, %r4, 96;
shl.b32 %r426, %r4, 1;
and.b32 %r427, %r426, 16;
shl.b32 %r428, %r15, 6;
or.b32 %r429, %r424, %r13;
or.b32 %r430, %r427, %r425;
xor.b32 %r431, %r429, %r430;
add.s32 %r432, %r60, %r428;
add.s32 %r433, %r432, %r431;
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+32768], {%r16, %r17, %r18, %r19};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+34816], {%r20, %r21, %r22, %r23};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+36864], {%r24, %r25, %r26, %r27};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+38912], {%r28, %r29, %r30, %r31};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+40960], {%r32, %r33, %r34, %r35};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+43008], {%r36, %r37, %r38, %r39};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+45056], {%r40, %r41, %r42, %r43};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%r433+47104], {%r44, %r45, %r46, %r47};
fence.proxy.async.shared::cta;
@%p2 bra $L__BB0_10;
bar.sync 0;
elect.sync %r437|%p28, -1;
shfl.sync.idx.b32 %r438, %r7, 0, 31, -1;
and.pred %p27, %p31, %p28;
and.b32 %r439, %r438, 1;
shl.b32 %r440, %r439, 14;
add.s32 %r436, %r60, %r440;
shl.b32 %r441, %r439, 6;
or.b32 %r434, %r441, %r3;
@%p27 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd54, {%r434, %r435}], [%r436];
cp.async.bulk.commit_group;
$L__BB0_10:
bar.sync 0;
elect.sync %r445|%p30, -1;
shfl.sync.idx.b32 %r446, %r7, 0, 31, -1;
and.pred %p29, %p31, %p30;
and.b32 %r447, %r446, 1;
shl.b32 %r448, %r447, 14;
add.s32 %r449, %r60, %r448;
add.s32 %r443, %r449, 32768;
shl.b32 %r450, %r447, 6;
or.b32 %r442, %r450, %r2;
@%p29 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd54, {%r442, %r76}], [%r443];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0; mov.u32 %r444, 0;
ret;
$L__tmp17:
$L__func_end0:
}
"#


private def lines (s : String) : List String :=
  (s.splitOn "\n").map String.trimAscii |>.filterMap fun line =>
    if line.isEmpty then none else some line.toString

def matchesCode (shape : Shape) (kind : Kind) (code : Code) : Bool :=
  QKVTTIR.normalizeTTIR code.ttgir == some (lines (graph shape kind)) &&
    QKVTTIR.normalizePTX code.ptx == lines (assembly shape kind)

structure Artifacts where
  baseline : Code
  compact : Code
  fullTile : Code
  deriving BEq

def matchesArtifacts (shape : Shape) (a : Artifacts) : Bool :=
  matchesCode shape .baseline a.baseline && matchesCode shape .compact a.compact &&
    matchesCode shape .full a.fullTile

structure Program (shape : Shape) where
  artifacts : Artifacts
  checked : matchesArtifacts shape artifacts = true

def Program.check (shape : Shape) (a : Artifacts) : Option (Program shape) :=
  if h : matchesArtifacts shape a = true then some ⟨a, h⟩ else none

abbrev Programs := (shape : Shape) → Program shape

def Programs.run (programs : Programs) : Kernel := fun bank op input =>
  let shape := Shape.of bank op
  if matchesArtifacts shape (programs shape).artifacts then
    match input.launchMode with
    | .dense => reference bank op input
    | .triangular => triangular bank op input
    | .fullTiles => TMA.run bank op input
  else reference bank op input

theorem Programs.run_eq (programs : Programs) : programs.run = reference := by
  funext bank op input
  simp only [Programs.run, (programs (Shape.of bank op)).checked, if_true]
  cases input.launchMode <;> simp only [triangular_equal, TMA.run_eq, stores_equal]

#axiomsClean Programs.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram
