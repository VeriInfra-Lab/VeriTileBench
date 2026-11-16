# Scaled E2M1 FP4 to BF16

`fp4_to_bf16_conversion.py` packs two E2M1 nibbles per uint8 input byte and
applies one E8M0 scale to each 32 decoded values. The high nibble is emitted
first. Scale bytes 0–254 encode `2^(s-127)`; byte 255 selects a canonical NaN.
This is the scaled conversion task, separate from `fp4_to_bf16`.

The repaired source retains the conversion helper and the five original tile
sizes (128, 256, 512, 1024, 2048). It uses an int32 exponent for `libdevice.pow`,
a ceiling count for the last partial scale group, explicit inactive-load zeros,
wide addresses and an explicit NaN bitcast. Both complete function bodies are
pasted into `Kernels.lean`; source correspondence is discovered automatically.
`_fp4_packed_to_bf16Source : StatementJitFunction` retains the helper return value.
`triton_f4_to_scaled_bf16_kernelSource : SourceProcedure` contains the complete caller body and binds
that helper with `jit_functions`. `triton_f4_to_scaled_bf16_kernel : ComputeKernel` supplies only
the launch signature and arguments through `instantiate_source_procedure`.
The five tile choices and their launch model live in `Proofs/Model.lean`.
The source was recovered from the task's upstream Python snapshot retained in
VeriTile commit `9327d49ad`, under
`bench/review-checkpoints/2026-10-14/next-value-repair/fp4-upstream/fp4_to_bf16_conversion.py`.

`Contract.lean` describes the two byte inputs, scale length, finite CUDA launch
bounds and numerical domains. `IO.lean` defines the complete memory interface,
logical arrays and CTA schedules. The output allocation is separate from both
inputs; every complete permutation of the grid is supported, including empty
input and partial final tiles.

`RealCorrect.lean` proves the independent E2M1 value times `2^(s-127)` in Real
semantics, with rounding erased and NaN scale byte 255 excluded. `FPEquiv.lean`
compares complete CTA schedules of the repaired source in the explicit software
FP model. It retains signed integer conversions, FP32 bitcasts, BF16 narrowing,
NaNs, signed zeros, underflow and overflow. This structural comparison admits no
numerical rewrite and uses no two-gates atom. It does not compare against the
uncompilable upstream caller or assert hardware/compiler conformance.

The proof follows all 49 lowered source statements. Integer decoding reuses the
shared finite-format lemmas from the unscaled conversion. Source-slice equalities
only decompose that actual source; they do not construct a separate kernel.
Execution, masked reads and stores, every logical output, complete grid coverage,
and the full exterior memory frame are proved in `Proofs/`.

The reproducible GPU check and validation receipts are under
`validation/tasks/fp4_to_bf16_conversion/`. GPU observations, source correspondence,
Lean compilation and independent comparator replay are recorded separately.
