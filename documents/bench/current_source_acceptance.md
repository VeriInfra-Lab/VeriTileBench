# Final selected corpus

`validation/task_progress.json` selects the repaired source and public entries for all 183 tasks. The inventory tool checks these entries against the current directories and imports. The specification sheets read the selected Lean declarations directly. CI generates a temporary inventory and checks the source hashes again after compilation and during independent proof replay when requested.

The [archived contract review](../../validation/archive/contracts.json) preserves mathematical targets, FP program pairs, required conditions and numerical dependencies from the earlier review. The [proof replay receipts](../../validation/body_receipts/benchmark-layout-replay.json) record the benchmark layout validation.

Run the final checks from the repository root:

```sh
python3 -B -m unittest discover -s tests/python -p 'test_*.py'
python3 -B scripts/bench/final_inventory.py --check
python3 -B -m scripts.bench.check_body_corpus --check
python3 -B scripts/bench/check_imported_kernel_review.py
lake build TritonBenchGFinal BenchmarkRegressions BenchmarkExamples BugDetection PretrainKernels
scripts/bench/audit_trust.sh --final-inventory
```

The proof comparator requires the pinned tools and their systemd/Landlock isolation; `.github/actions/setup-comparator` installs them in CI. Every retained corpus module must pass the axiom/specification audit and independent proof replay. The complete gate fails if any worker is missing, skipped, cancelled or fails.

PR and main-push CI run the source/regression checks and build the matrix-transpose
proofs against the locked VeriTile package. Complete corpus builds and all 334
draft interfaces run daily at 04:00 Asia/Singapore (20:00 UTC). Sunday's run adds
per-module independent comparator replay; it replaces that day's build-only
run. Manual dispatch defaults to `mode=build`; use `mode=replay` for the complete
independent audit or `mode=smoke` for the PR checks. A build-only pass is recorded
separately and does not establish independent replay or resolve pending proofs.
Run the complete applicable proof checks locally before merging proof changes.

Source correspondence is tracked independently in `validation/source-correspondence.json`. Pending Python-to-Lean correspondence remains pending. The 65 bug-detection and 269 pretraining benchmark tasks are explicit unsolved challenges; their successful compilation does not complete their proof or backend obligations. Native execution and compiler conformance are not established by these checks.

The validation workflow reads the current source tree and records. The earlier static inventory is available in the [validation archive](../../validation/archive/source.json).

The selected source is each task's repaired implementation. Current comparison
candidates remain only where the FP entry compares a real transformation.
Proof modules derive checked program views from that source; retired complete
kernel quotations are kept in Git history rather than as another executable
implementation in the live tree. Mathematical specifications and finite-index
refinements remain independent proof objects.

The verbatim-body gate and the Lean proof gate are separate. A checked repaired
implementation does not make an uncovered Python JIT function a body-copy pass.
The correspondence inventory continues to report those pending functions and
worked-example Python references explicitly.
