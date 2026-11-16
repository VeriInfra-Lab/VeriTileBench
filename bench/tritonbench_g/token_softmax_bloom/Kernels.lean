import bench.tritonbench_g.token_softmax_llama.Kernels

/-! Both complete Python bodies are identical to Llama's checked quotations.
The actual implementations are shared at that canonical source entry. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom
abbrev indexCommutedSource := TokenSoftmaxLlama.KernelSource._fwd_kernel_token_softmax_candidateKernel
end VeriTile.Bench.Repairs.TokenSoftmaxBloom
