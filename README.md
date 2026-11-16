# VeriTileBench

Triton benchmarks and verification evidence for [VeriTile](https://github.com/VeriInfra-Lab/VeriTile).
The corpus was extracted from VeriTile commit `84e769e4c1de259841925d867b5b3bec916eb77c`.
The original history and earlier proof snapshots remain available in that repository.

| Collection | Contents |
| --- | --- |
| [TritonBench-G](bench/tritonbench_g) | 184 selected repairs with Python, Lean source, proof entries, and recorded proof scopes |
| [Bug detection](bench/bug_detection) | 65 independent challenges |
| [Pretraining kernels](bench/pretrain_kernels) | 269 independent challenges |
| [Worked examples](bench/examples) | 19 examples with Python sources and public Real/FP proof entries |

## Verify

`lakefile.toml` and `lake-manifest.json` lock the exact VeriTile commit and its
transitive dependencies. Install the Lean version in `lean-toolchain`, then run:

```bash
lake exe cache get
bash scripts/bench/audit_tritonbench_g.sh --source-only
lake build bench.tritonbench_g.matrix_transpose.RealCorrect bench.tritonbench_g.matrix_transpose.FPEquiv
```

PRs and main pushes run source/tool checks, then build and independently replay
changed cases and affected proof dependencies. Daily and full manual audits replay
every retained TritonBench-G/example module and regression. The 334 pending
challenge interfaces have a separate workflow and retain their pending status.
See the [CI checks and local commands](documents/bench/ci.md),
[source acceptance guide](documents/bench/current_source_acceptance.md)
and [validation records](validation). Pending proof or backend obligations retain
their recorded status. Record both Git commit SHAs when reproducing experiments.
