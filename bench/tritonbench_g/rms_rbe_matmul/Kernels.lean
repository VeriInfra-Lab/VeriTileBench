import bench.tritonbench_g.rms_matmul_rbe.Kernels
import VeriTile.Triton
import bench.tritonbench_g.rbe_triton_transform.Proofs.Sources.Kernels
import VeriTile.Triton.DSL.IntegralJitFunctions

/-! Parameters for the literal-parameter RMS caller. -/
namespace VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource
open VeriTile.Triton
structure Arguments where
  (X W RMS OUT : RegionName)
  (M N K sxb sxm sxk swk swn srms sob som son : Nat)
  (BM BN BK headDim : Nat)
  start : Int
  rotary : Bool
  (theta epsilon : ℝ)

end VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource

/-! The repaired RMS callee is shared verbatim with rms_matmul_rbe. This
interface realizes EPS/THETA through their original FP32 literal ports.
The separate rotary source and its helper below are complete Python bodies. -/
namespace VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
open VeriTile.Triton
abbrev Arguments := VeriTile.Bench.Repairs.RmsMatmulRbe.TypedSource.Arguments
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

def rms_matmul_rbeKernel (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  instantiate_source_procedure
    ({RmsMatmulRbeScalarDomain.TypedSource.rms_matmul_rbeSource with materialized:=[]} : SourceProcedure)
    signature (RmsMatmulRbeScalarDomain.TypedSource.rmsSignature
      (if fp8 then .byte else .half) (if fp8 then .e5 else .half))
    procedure_args [x_ptr := p.X, w_ptr := p.W, rms_w_ptr := p.RMS, out_ptr := p.OUT, M := p.M, N := p.N, K := p.K, stride_x_batch := p.sxb, stride_x_m := p.sxm, stride_x_k := p.sxk, stride_w_k := p.swk, stride_w_n := p.swn, stride_rms_w := p.srms, stride_out_batch := p.sob, stride_out_m := p.som, stride_out_n := p.son, start_token_position := p.start, USE_FP8 := fp8, RBE_EPILOGUE := p.rotary, THETA := p.theta, EPS := p.epsilon, HEAD_DIM := p.headDim, BLOCK_SIZE_M := p.BM, BLOCK_SIZE_N := p.BN, BLOCK_SIZE_K := p.BK]


def rms_matmul_rbe_candidateSource : SourceProcedure :=
  with_triton_params (x_ptr, w_ptr, rms_w_ptr, out_ptr, M, N, K, stride_x_batch, stride_x_m, stride_x_k, stride_w_k, stride_w_n, stride_rms_w, stride_out_batch, stride_out_m, stride_out_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    constexpr_args [USE_FP8, RBE_EPILOGUE, THETA, EPS, HEAD_DIM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
    fp32_constants [THETA, EPS] in triton {
    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_k = tl.cast(stride_x_k, tl.int64)
    stride_w_k = tl.cast(stride_w_k, tl.int64)
    stride_w_n = tl.cast(stride_w_n, tl.int64)
    stride_rms_w = tl.cast(stride_rms_w, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))
    pid_n = pid % (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for step in range(0, (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        w = w.to(x.dtype)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    # Rotate normalized FP32 pairs before the source's sole output cast.
    if RBE_EPILOGUE:
        real, imag = tl.split(tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
        pair_n = pid_n * BLOCK_SIZE_N + 2 * tl.arange(0, BLOCK_SIZE_N // 2).to(tl.int64)
        row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        exponent = (pair_n % HEAD_DIM).to(tl.float32) / HEAD_DIM
        frequency = tl.extra.cuda.libdevice.pow(THETA, exponent)
        position = tl.cast(start_token_position, tl.int64) + row_m
        angle = position.to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)
}

def rms_matmul_rbe_candidateKernel (fp8 : Bool) (p : Arguments) : ComputeKernel :=
  instantiate_source_procedure
    ({rms_matmul_rbe_candidateSource with materialized:=[]} : SourceProcedure)
    signature (RmsMatmulRbeScalarDomain.TypedSource.rmsSignature
      (if fp8 then .byte else .half) (if fp8 then .e5 else .half))
    procedure_args [x_ptr := p.X, w_ptr := p.W, rms_w_ptr := p.RMS, out_ptr := p.OUT, M := p.M, N := p.N, K := p.K, stride_x_batch := p.sxb, stride_x_m := p.sxm, stride_x_k := p.sxk, stride_w_k := p.swk, stride_w_n := p.swn, stride_rms_w := p.srms, stride_out_batch := p.sob, stride_out_m := p.som, stride_out_n := p.son, start_token_position := p.start, USE_FP8 := fp8, RBE_EPILOGUE := p.rotary, THETA := p.theta, EPS := p.epsilon, HEAD_DIM := p.headDim, BLOCK_SIZE_M := p.BM, BLOCK_SIZE_N := p.BN, BLOCK_SIZE_K := p.BK]

def kernel (fp8 : Bool) (p : Arguments) : ComputeKernel := rms_matmul_rbeKernel fp8 p

/-- Complete original helper; DIM=128 is defined by its Python source. -/
def get_freq_multi_tokensSource : StatementJitFunction :=
  with_triton_params (offs_cn, starting_idx, theta, NB_TOKENS) constexpr_args [theta, NB_TOKENS] in triton {
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS).to(tl.int64) + tl.cast(starting_idx, tl.int64))[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)
  }

def rbeSignature (dtype : FloatDType) : CallSignature where
  floatingPointers := [("x_ptr",dtype),("out_ptr",dtype)]
  scalars := [("M",.integer),("K",.integer),("stride_x_batch",.integer),
    ("stride_x_m",.integer),("stride_x_n",.integer),("stride_out_batch",.integer),
    ("stride_out_m",.integer),("stride_out_n",.integer),("start_token_position",.integer)]
  constexpr := ["THETA","BLOCK_SIZE_M","BLOCK_SIZE_K"]
  readPorts := ["x_ptr"]
  writePorts := ["out_ptr"]

def rbe_tritonKernel (dtype : FloatDType) (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n : Nat)
    (start_token_position : Int) (THETA : ℝ) (BLOCK_SIZE_M BLOCK_SIZE_K : Nat) : ComputeKernel :=
  with_integral_operands in
  with_triton_call (rbeSignature dtype) call_args [M:=M,K:=K,
    stride_x_batch:=stride_x_batch,stride_x_m:=stride_x_m,stride_x_n:=stride_x_n,
    stride_out_batch:=stride_out_batch,stride_out_m:=stride_out_m,stride_out_n:=stride_out_n,
    start_token_position:=start_token_position] jit_functions [get_freq_multi_tokens:=get_freq_multi_tokensSource] in triton {
    M = tl.cast(M, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    start_token_position = tl.cast(start_token_position, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))
    pid_n = pid % (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64) * 2
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)
  }

end
end VeriTile.Bench.Repairs.RmsRbeMatmul.TypedSource
