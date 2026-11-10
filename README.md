# VeriTileBench

Triton benchmarks and verification evidence for [VeriTile](https://github.com/VeriInfra-Lab/VeriTile).
The corpus was extracted from VeriTile commit `84e769e4c1de259841925d867b5b3bec916eb77c`.
The original history and earlier proof snapshots remain available in that repository.

| Collection | Contents |
| --- | --- |
| [TritonBench-G](bench/tritonbench_g) | 183 selected repairs with Python, Lean source, proof entries, and recorded proof scopes |
| [Bug detection](bench/bug_detection) | 65 independent challenges |
| [Pretraining kernels](bench/pretrain_kernels) | 269 independent challenges |
| [Benchmark tutorial](bench/examples/TritonBenchVectorAddition) | Vector addition backed by the imported benchmark kernel |

## Verify

`lakefile.toml` and `lake-manifest.json` lock the exact VeriTile commit and its
transitive dependencies. Install the Lean version in `lean-toolchain`, then run:

```bash
lake exe cache get
bash scripts/bench/audit_tritonbench_g.sh --source-only
lake build bench.tritonbench_g.matrix_transpose.RealCorrect bench.tritonbench_g.matrix_transpose.FPEquiv
```

The daily or manually dispatched Bench audit builds the imported challenges and
builds and independently replays the retained TritonBench-G proof modules.
See the [source acceptance guide](documents/bench/current_source_acceptance.md)
and [validation records](validation). Pending proof or backend obligations retain
their recorded status. Record both Git commit SHAs when reproducing experiments.
