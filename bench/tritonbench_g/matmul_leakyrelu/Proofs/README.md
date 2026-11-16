# matmul_leakyrelu

[The repaired Python source](../matmul_leakyrelu.py) and its activation helper are copied into [Kernels.lean](../Kernels.lean). Each complete kernel body has 24 top-level Python statements, including its docstring; each helper has one. Caller dtypes, scalar arguments and constexpr values remain outside the quotations. Source statements, two-argument dot followed by addition, pointer updates, masks, activation call and FP16 conversion are retained.

The comparison candidate passes the live accumulator to dot and multiplies
the activation slope on the right. Its helper is copied separately. The local
`DOT-ADD-TO-DOTACC` relation uses the shared typed template and the selected
experiment record. Its numerical outcome is REJECT; it is not promoted to an
available assumption when either gate or the recorded scope rejects it.
See [numerical validation](../../../../validation/tasks/matmul_leakyrelu/NUMERICAL_VALIDATION.md).

[RealCorrect.lean](../RealCorrect.lean) connects the complete selected grid to the independent matrix-product and Leaky ReLU formula. [FPEquiv.lean](../FPEquiv.lean) compares the two complete programs under the explicit rule. Execution, output, frame, both source safety judgments and all five retained input profiles have been rechecked locally. Proof replay and numerical admission are recorded separately; see [validation status](../../../../validation/body-migration-progress.json).

[Contract](../Contract.lean) · [IO](../IO.lean) · [Proof helpers](.) · [Typed call and mathematical adapters](Sources).

## Source-model disproof

`FPEquiv.source_pair_disproved` uses `refutation` to negate the universal
source-equivalence claim written after the colon, under the current FP32
`DotAdmission.all` table, which is empty. The original
`source_pair_equiv` conditional theorem keeps its existing `Rules`, model,
region-separation and IO premises. Real mathematical correctness is unchanged.

[Unconditional.lean](Unconditional.lean) constructs the interpretation and a
complete source counterexample. Dot truncates FP32 inputs to TF32 and folds
FP32 fused products toward zero; the independent scalar add and the final
binary16 conversion round to nearest even. Every arithmetic operation and
conversion used by this witness is explicit. GPU conformance is outside the
claim's scope.

The input matrices have shapes 1×96 and 96×1. All omitted entries are zero:

| k | A[0,k] | B[k,0] |
| ---: | ---: | ---: |
| 0 | 65536 | 1 |
| 1 | 48 | 2^-11 |
| 32 | 65536 | 1 |
| 64 | -131072 | 1 |

Every input is exactly representable in TF32. Activation is disabled, as
allowed by the source contract. The first K32 block creates `65536+3/128`.
The second block exposes the different rounding boundaries. The third removes
the common large component. The complete original and candidate executions
store normal FP16 values **1/32** and **1/64**, respectively. Legal indices,
input bindings, allocation bounds and both source traces' safety are proved.
The final argument covers every derivation constructor in the public source
calculus, so it establishes a negation of the stated goal.

FP32 numerical rejection and this source-model counterexample are separate
evidence. An FP64 experiment cannot supply the FP32 atom or serve as the
reason for this disproof.
