# Numerical validation

Rule: `DOT-ADD-TO-DOTACC`.
Outcome: **REJECT**. Bias gate `FAIL`; magnitude gate `PASS`.

The selected experiment uses NVIDIA GeForce RTX 4090 (SM89),
Triton 3.5.1, FP32 operands/accumulator/output, TF32 dot input precision,
K=32 and a 32×32 output tile. Four warps, three stages and FP fusion disabled.
It ran once to 4,294,967,296 independent whole input groups, using quantized
Normal(1,1) entries and unit weights. The 1024 output lanes share operands;
standard error is computed across whole groups.

| Metric | Recorded value | Criterion |
| --- | ---: | ---: |
| Mean signed ULP shift | -0.12170168115225977 | |
| Standard error | 0.0058806878334038111 | |
| abs(mean) + 5 SE | 0.15110512031927881 | ≤ 0.05 ULP |
| Reference absolute oracle-error peak | 0.11381051382235796 | |
| Candidate absolute oracle-error peak | 0.11381051382235796 | |
| Peak ratio | 1.0 | ≤ 10 |

The reference is an FP32 dot followed by a separate FP32 add to the live
accumulator. The candidate seeds dot with that accumulator. Reference execution
uses an FP32 intermediate between two launches to prevent Triton's dot/add
combiner from turning both sides into the same calculation. Actual PTX and the
ordered FP64 oracle were audited. The final FP16 store and activation remain
outside this local numerical rule.

The [shared template](https://github.com/VeriInfra-Lab/VeriTile/blob/6d520c6f715d3c9224440df33472652e6a94f859/VeriTile/Triton/Float/DotContraction.lean),
[evidence table](https://github.com/VeriInfra-Lab/VeriTile/blob/6d520c6f715d3c9224440df33472652e6a94f859/VeriTile/Triton/Float/DotAdmission.lean) and
[receipt](https://github.com/VeriInfra-Lab/VeriTile/blob/6d520c6f715d3c9224440df33472652e6a94f859/experiments/floating_point/dot/report/receipt.json) are pinned by
the repository's VeriTile dependency. The shared guide describes the complete
fixed-budget protocol and raw evidence checks. SM89 evidence does not validate
an SM90 compiler profile.

`SourceIO.lean` proves that both shared fragments are exactly the statements of
the actual Python/Lean bodies, including the carrier conversion. The source
pair, full schedules, safety, outputs, frames and all-word formal domain remain
unchanged. Real correctness and the conditional FP derivation are separate from
numerical admission. `Rules` still requires an explicit accepted atom and the
model interpretation; the recorded outcome supplies no missing premise.

Special-value diagnostics recorded 0 classification mismatches
and 0 NaN-payload differences. They are separate from the Normal
gate statistics and do not establish exhaustive all-word hardware conformance.
Finite numerical acceptance would not prove IEEE word equality or a kernel
error bound. A rejected local atom alone does not refute a complete kernel pair.

## Source-model verdict

`FPEquiv.source_pair_disproved` is declared with `refutation`: the universal
equivalence claim appears after the colon, and the exported theorem negates it
under the current FP32 dot admission table. Its independent countermodel is in
`Proofs/Unconditional.lean`: TF32 input truncation, sequential toward-zero
FP32 fused accumulation, nearest-even scalar addition and nearest-even binary16
conversion. Legal finite 1×96 and 96×1 inputs lead through the complete actual
source bodies to distinct normal FP16 outputs, 1/32 and 1/64. Input bindings,
index/allocation bounds, source-trace safety and every public proof-calculus
constructor are checked. This theorem concerns the declared source-model
judgment. GPU conformance is outside its scope.

The original conditional FP theorem and its premises are retained. Real
correctness remains proved. Numerical rejection is recorded independently of
the constructive source-model disproof.


## FP64 control

The FP64 input/accumulator/output profile ran once to **2^32 = 4,294,967,296**
independent whole matrix groups. Distribution and thresholds are unchanged.
Its **Normal two-gates decision is ACCEPT**; no groups were excluded.

| Metric | Recorded value | Criterion |
| --- | ---: | ---: |
| Mean signed ULP shift | 0.0018087388154981454 | |
| Standard error | 0.0018492411269843512 | |
| abs(mean) + 5 SE | 0.011054944450419902 | ≤ 0.05 ULP |
| Reference absolute oracle-error peak | 8.9621036488053036e-14 | |
| Candidate absolute oracle-error peak | 8.9621036488053036e-14 | |
| Peak ratio | 1 | ≤ 10 |

Server and independent local replay agree. All 524,288 execution-part statistics
and 16 mandatory raw audits were checked; 524,288 raw output lanes were checked
against exact rational dot products. In all 512 saved input groups, converting
FP64 inputs to FP32 reproduces the FP32 experiment's saved operands exactly.
GPU measurement elapsed 8031.028 seconds.

The separate special-value diagnostic has **1 input group / 1024 output lanes**
with A=-0, B=+0 and incoming accumulator=-0. The reference returns +0 and the
candidate -0. Consequently **all-word admission remains disabled**, even though
the Normal statistical gates pass. Zero signs are retained in the evidence.
FP64 evidence supplies no FP32 assumption and is independent of the constructive
FP32 source-model disproof above.

The retained artifact is VeriTile's
`experiments/floating_point/dot/fp64/report/receipt.json`, with source-manifest
SHA256 `477bdb15ca5b0c3d6952f1dcf7e76c1da0eb217d53105869cf111a19fb00d972`. This numerical control has its own frozen source
snapshot; the FP32 proof's VeriTile dependency remains pinned separately.
