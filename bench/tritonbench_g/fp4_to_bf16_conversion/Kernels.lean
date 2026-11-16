import VeriTile.Triton
import VeriTile.Triton.DSL.SourceProcedures

namespace VeriTile.Bench.Repairs.Fp4Scaled.Kernels
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000

def _fp4_packed_to_bf16Source : StatementJitFunction :=
  with_triton_params (x_packed, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1, ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32, f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32)
    constexpr_args [] in triton {
  """
  Input: a tensor of packed fp4 values
  Output: a tensor of bfloat16 values
  """

  # low-bits: original location 0:3
  # high-bits: original location 4:7
  x_low_bits = x_packed >> 4
  x_high_bits = x_packed & 0xF
  x = tl.interleave(x_low_bits, x_high_bits)

  # cast logic below
  # output = x_unpacked.to(tl.float32)

  # save the sign
  sign_f4 = x & sign_mask_f4

  # set everything to positive, will add sign back at the end
  x_pos = x ^ sign_f4

  # Special case zero
  zero_mask = x_pos == 0

  # There is only one denormal value in fp4: s001, which is 0.5 in f32
  # Special case it.
  # TODO(later): will it be faster to repeat this for all 8 positive
  # values instead of the bit manipulations?
  denormal_mask = x_pos == 1

  # calculate the new exponent and shift it to bits 2:9 of the result
  exp_biased_f4 = x_pos >> mbits_f4_e2m1
  exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
  exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

  # shift the mantissa to bits 10:32 of the result
  mantissa_f4 = x_pos & mantissa_mask_f4
  mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
  output = mantissa_f32

  # combine the pieces
  result = exp_biased_f32 | mantissa_f32
  # result[zero_mask] = ZERO_BITS_F32
  result = tl.where(zero_mask, zero_bits_f32, result)
  # result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32
  result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

  # add sign back
  sign_f32 = sign_f4.to(tl.int32) << (
      mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
  )
  result = result | sign_f32

  # The bit shifting above is for float32, so for now we
  # bitcast to float32 and then regular cast to bfloat16
  # TODO(later): it should be pretty easy to cast directly to bf16, just
  # need to adjust the mbits/ebits/special values. Perf impact is likely
  # to be small as we would not be chaning memory access patterns.
  output = result.to(tl.float32, bitcast=True)
  output = output.to(tl.bfloat16)
  return output
  }


def triton_f4_to_scaled_bf16_kernelSource : SourceProcedure :=
  with_triton_params (x_ptr, s_ptr, output_ptr, n_elements_in,
      mx_block_size, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1,
      ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32,
      f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32, e8m0_exponent_bias,
      e8m0_exponent_nan_val, BLOCK_SIZE_IN)
    constexpr_args [mx_block_size, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1, ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32, f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32, e8m0_exponent_bias, e8m0_exponent_nan_val, BLOCK_SIZE_IN]
    jit_functions [_fp4_packed_to_bf16 := _fp4_packed_to_bf16Source] in triton {
    pid = tl.program_id(axis=0).to(tl.int64)
    n_elements_in = tl.cast(n_elements_in, tl.int64)
    n_elements_out = n_elements_in * 2
    n_elements_s = (n_elements_out + 31) // 32

    BLOCK_SIZE_S: tl.constexpr = BLOCK_SIZE_IN // 16
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

    block_start_in = pid * BLOCK_SIZE_IN
    offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN).to(tl.int64)
    mask_in = offsets_in < n_elements_in
    # packed uint8
    x_packed = tl.load(x_ptr + offsets_in, mask=mask_in, other=0)
    output = _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    )

    # load scale
    block_start_s = pid * BLOCK_SIZE_S
    offsets_s = block_start_s + tl.arange(0, BLOCK_SIZE_S).to(tl.int64)
    mask_s = offsets_s < n_elements_s
    s = tl.load(s_ptr + offsets_s, mask=mask_s, other=0)

    # create the scale in bf16
    s_offset = s.to(tl.int16) - e8m0_exponent_bias
    s_fp = tl.extra.cuda.libdevice.pow(2.0, s_offset.to(tl.int32)).to(tl.bfloat16)
    nan_bits = tl.cast(0x7FC00000, tl.int32)
    nan_fp32 = nan_bits.to(tl.float32, bitcast=True)
    nan_bf16 = nan_fp32.to(tl.bfloat16)
    s_fp = tl.where(s != e8m0_exponent_nan_val, s_fp, nan_bf16)

    # multiply output by scale
    # TODO(later): see if manipulating the exponent instead of fp
    # multiplication is going to give a significant speedup
    output = tl.reshape(
        output, [BLOCK_SIZE_OUT // mx_block_size, mx_block_size]
    )  # noqa: E501
    s_fp = tl.reshape(s_fp, [BLOCK_SIZE_S // 1, 1])
    output = output * s_fp
    output = tl.reshape(output, [BLOCK_SIZE_OUT])

    # set up output offsets
    block_start_out = pid * BLOCK_SIZE_OUT
    offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT).to(tl.int64)
    mask_out = offsets_out < n_elements_out

    tl.store(output_ptr + offsets_out, output, mask=mask_out)
  }

def callSignature : CallSignature where
  pointers := [("x_ptr",.u8),("s_ptr",.u8),("output_ptr",.bf16)]
  scalars := [("n_elements_in",.integer)]
  constexpr := ["mx_block_size", "sign_mask_f4", "mantissa_mask_f4", "mbits_f4_e2m1", "ebits_f4_e2m1", "f4_e2m1_exp_bias", "mbits_f32", "ebits_f32", "f32_exp_bias", "zero_bits_f32", "zero_point_five_bits_f32", "e8m0_exponent_bias", "e8m0_exponent_nan_val", "BLOCK_SIZE_IN"]
  readPorts := ["x_ptr","s_ptr"]
  writePorts := ["output_ptr"]

/-- The Python launch's E2M1/E8M0 format specialization. -/
def triton_f4_to_scaled_bf16_kernel (x_ptr s_ptr output_ptr : RegionName)
    (n_elements_in BLOCK_SIZE_IN : Nat) : ComputeKernel :=
  instantiate_source_procedure triton_f4_to_scaled_bf16_kernelSource signature callSignature
    procedure_args [x_ptr := x_ptr, s_ptr := s_ptr, output_ptr := output_ptr,
      n_elements_in := n_elements_in, mx_block_size := (32 : Nat),
      sign_mask_f4 := (8 : Nat), mantissa_mask_f4 := (1 : Nat),
      mbits_f4_e2m1 := (1 : Nat), ebits_f4_e2m1 := (2 : Nat), f4_e2m1_exp_bias := (1 : Nat),
      mbits_f32 := (23 : Nat), ebits_f32 := (8 : Nat), f32_exp_bias := (127 : Nat),
      zero_bits_f32 := (0 : Nat), zero_point_five_bits_f32 := (0x3f000000 : Nat),
      e8m0_exponent_bias := (127 : Int), e8m0_exponent_nan_val := (255 : Nat),
      BLOCK_SIZE_IN := BLOCK_SIZE_IN]

end
end VeriTile.Bench.Repairs.Fp4Scaled.Kernels
