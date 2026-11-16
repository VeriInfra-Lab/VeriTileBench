# Benchmark CI

VeriTile checks the library and three small worked examples (`VectorAdd`,
`RowWiseSum`, `LogExp`), each with Real and FP proofs. VeriTileBench owns the
184 TritonBench-G tasks and 19 example groups. The 334 pending challenges
(65 bug-detection and 269 pretraining tasks) have a separate interface workflow.

| Check | PR and main push | Daily or full manual audit |
| --- | --- | --- |
| Source correspondence, bidirectional naming/comment coverage, public verdicts and tool regressions | Yes | Yes |
| Build and independent replay of changed cases and affected dependencies | According to Git diff and import graphs | Covered by full audit |
| Official proof-auditor rejection regressions | When tools, workflows or build dependencies change | Yes |
| Every retained TritonBench-G/example module and regression theorem | When the change affects the whole corpus | Yes, 64 shards |

`.github/workflows/bench-audit.yml` runs at **04:00 Asia/Singapore** daily and
on manual full-audit dispatch. PRs and main pushes use the actual base/head Git
revisions. Lean and Python imports from both revisions find reverse dependents,
including removed or renamed shared sources and Python source-reference aliases.
A changed case selects every retained Lean file in that case, including helpers
that public entries do not import. Dependent cases and regressions are selected,
and their owned Lean dependencies are also replayed. Changing the locked library,
Lake configuration or toolchain selects all verified cases and regressions.

Documentation changes need only the fast checks. If Git base history is missing,
selection falls back to every verified case and regression; it never treats an
unknown diff as empty. Manual dispatch without `full_audit` has no event base and
uses this conservative fallback. A full manual dispatch uses the 64 daily shards. Large PR selections also split
across up to 64 workers. Consecutive sorted module paths keep related proof
files together and limit repeated dependency builds. All selected modules and
public goals must be covered once across the successful shards.

The plan artifact records changed paths, selected cases, public theorem names,
module paths and source fingerprints. The proof job recomputes the plan and
checks the snapshot before building, before replay and after replay. Every
selected module must pass axiom/specification gates and the official comparator;
public Real/FP goals and required refutations must occur in the replay inventory.
A successful build alone does not accept a case. Auxiliary, private and generated
theorems remain in the inventory.

All proof replays use bounded batches, 30-minute file/batch limits and 30-second
progress messages. Compilation groups of up to 32 modules have a separate
60-minute budget, including cold dependencies. The cached Lean setup builds the
shard's exact targets before saving artifacts; the replay stage rechecks those
targets against the same plan. Compiled artifacts never replace proof replay.
All jobs that save the shared Lean cache include the locked library build, so a
small audit bootstrap cannot populate an incomplete dependency cache first.
Coverage incompatibility can fall back to complete individual
replay; a comparator rejection or timeout fails the job. Failed proof jobs retain
diagnostics for seven days. The required `TritonBench-G audit complete` status
fails on a failed, cancelled or unexpectedly skipped planned job.

Fork PRs receive the same read-only checks. The fast source-check job fetches
only the exact VeriTile Git source revision locked in `lake-manifest.json`;
it needs neither Lean nor the comparator. Proof builds/replay run separately.
VeriTile dependency modules are audited by VeriTile's own CI.

## Pending challenges

`.github/workflows/pending-interfaces.yml` runs when pending collections, their
review record/checker or build dependencies change, and on manual dispatch.
Both PRs and main pushes use these path filters.
It builds `BugDetection` and `PretrainKernels` and verifies their recorded
interfaces. This preserves **pending** mathematical/backend obligations; it does
not certify proofs. It is separate from the verified-corpus completion status
and is not part of its daily full replay. The fast corpus checks still reject
unreviewed changes to the pending findings record.

## Local commands

```sh
python3 -B scripts/bench/ci.py checks
# Use full Git SHAs; the checkout must be at HEAD_SHA.
python3 -B scripts/bench/ci.py affected --base "$BASE_SHA" --head "$HEAD_SHA" \
  --write /tmp/affected-plan.json
# Requires Lean, the official comparator and its isolation tools:
python3 -B scripts/bench/ci.py verify-affected --plan /tmp/affected-plan.json \
  --workspace /tmp/affected-replay --results /tmp/affected-result.json
# For a multi-shard plan, run every index with separate workspaces/results and
# --shard-index INDEX. Its shard_count is recorded in the plan.
python3 -B scripts/bench/ci.py auditor
# Complete local corpus build/replay (excludes the 334 pending interfaces):
bash scripts/bench/audit_tritonbench_g.sh
```

The body-migration manifest defines the selected cases and public goals.
`DISPROVED` verdicts require the declared `refutation`, including its negative
Lean type. CI does not rerun GPU two-gates experiments or alter numerical
assumptions. Existing source-bound proof receipts retain their recorded scope;
new CI validations do not rewrite old proof evidence.
