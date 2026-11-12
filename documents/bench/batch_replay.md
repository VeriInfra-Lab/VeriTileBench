# Task batch replay pilot

The opt-in batch driver replays every theorem in a selected TritonBench-G task
through the pinned official comparator. The regular per-file audit remains the
default. The initial performance trial uses `rms_matmul_rbe`.

```sh
python3 scripts/check_comparator_batch.py --task rms_matmul_rbe
```

Use the existing [judge setup](../../scripts/README.md). The command requires
the same exporter, comparator, allowed axioms, and systemd/Landlock isolation as
the per-file driver. It does not change kernel sources, specifications, FP
rules, or theorem premises.

## Coverage and acceptance

The driver freezes an independent source/build-cache copy, then recursively
selects **all** Lean modules in the task, including modules not imported by
`RealCorrect.lean` or `FPEquiv.lean`.

1. Compile the same audited source copies used by the per-file driver. Retain
   each module's axiom/specification audit and complete theorem inventory. Check
   the standard-axiom boundary for **every** theorem occurrence before merging;
   a bad proof cannot hide behind another proof of a same-named theorem.
2. Import all original task modules into one generated module. Resolve the union
   of the per-file inventories in the actual kernel environment and supplement
   it with all theorems owned by the original task modules. Dependencies outside
   the task remain part of recursive proof checking.
3. Compare the old and new per-file inventories exactly, including zero-theorem
   modules. Normalize only the private-name module prefix introduced by the old
   source-copy mechanism; retain private counters and complete names. Lean's
   import compatibility checks reject same-named declarations with conflicting
   types. Deduplicate replay targets only after all per-file axiom audits pass.
4. Pass every discovered target to one official comparator invocation. The
   exporter deduplicates shared dependencies within each export. Record batch
   acceptance only after official replay and the final source-hash check pass.

Equation theorems are generated lazily and may be attributed to different
consumer modules under combined imports. Filtering targets solely by their
final module attribution would omit some old targets. The driver therefore
preserves each file's original inventory and checks its union. Original-module
theorems not already in that union are also replayed and reported separately.
Reports distinguish per-file theorem occurrences from unique replay targets.

The source snapshot is checked before and after replay. Missing/unexpected targets,
failed per-module audits, missing modules, import conflicts and comparator
rejections fail the command. Definition-only tasks retain the standard primitive
checks while reporting zero original theorem targets. Sources using Lean's new
`module` header and source-copy hygienic macro names currently require the
existing individual replay path. The pilot rejects these cases explicitly
rather than erasing scopes or dropping declarations. Private names, quoted names,
generated matcher equations and imported equation theorems are covered by the
pilot regressions.

## Reproduce the performance comparison

Use a frozen old checkout and the pilot checkout with copies of the same built
dependency cache. Keep the official tools fixed, set `AUDIT_TRUST_JOBS=1` for the
baseline, and run the two trials sequentially on the same host. Each trial
creates and cleans its own independent judge workspace.

```sh
trial_dir="$(mktemp -d)"
export AUDIT_TRUST_JOBS=1
python3 scripts/bench/measure_replay.py \
  --cwd /path/to/frozen-baseline --workspace "$trial_dir/old-tmp" \
  --output "$trial_dir/old-metrics.json" --log "$trial_dir/old.log" \
  -- bash scripts/bench/audit_trust.sh rms_matmul_rbe
python3 scripts/bench/measure_replay.py \
  --cwd "$PWD" --workspace "$trial_dir/new-tmp" \
  --output "$trial_dir/new-metrics.json" --log "$trial_dir/new.log" \
  -- python3 scripts/check_comparator_batch.py --task rms_matmul_rbe
```

Wall time includes snapshot preparation and cleanup. Memory is sampled as the
sum of process RSS, with a 100 ms wait between scans; the actual maximum sampling
gap is recorded in each receipt. The sampler follows both driver descendants
and processes whose working directory belongs to the private judge workspace,
so comparator services launched separately by systemd remain included. RSS can
count shared pages repeatedly; the sampled peak is not an exact physical-memory
high-water mark or a prediction of GitHub runner capacity.

`scripts/bench/compare_replay.py` compares the old receipt directory and the new
batch receipt. It requires identical frozen proof inputs, identical comparator
binary hashes, every old/new acceptance result, exact canonical theorem lists,
and matching per-module trust-audit summaries. It writes a compact JSON summary
and a deterministic compressed theorem inventory. Raw logs stay outside Git.

## Measured `rms_matmul_rbe` result

The [generated comparison summary](../../validation/replay_batch/rms_matmul_rbe.json)
and [compressed complete theorem inventory](../../validation/replay_batch/rms_matmul_rbe.theorems.json.gz)
record one sequential local-host trial pair:

| Measurement | Existing per-file replay | Task batch replay |
| --- | ---: | ---: |
| Wall time, including preparation and cleanup | 6,897.6 s (114m 58s) | 1,639.7 s (27m 20s) |
| Sampled aggregate RSS peak | 11.62 GiB | 11.65 GiB |
| Official comparator invocations | 100 | 1 |
| Distinct theorem targets | 1,890 | 1,890 |
| Private targets included above | 374 | 374 |
| Official acceptance | All 100 files pass | Entire batch passes |

Wall time fell by **4.21 times**, while sampled aggregate RSS remained nearly
unchanged. In the batch run, snapshot preparation plus per-module compilation,
audits and inventories took 1,444.5 s; combined inventory generation took 21.5 s;
the comparator phase, including the final source checks, took 158.6 s. The
remaining time covers report writing and cleanup. The preparation/audit stage
is now the main cost.

All **100** module inventories match exactly: **2,030** per-file theorem entries
reduce to **1,890** unique targets, including the **374** private targets. There
are zero missing or additional batch targets. Every per-file trust-audit summary
matches, and the new full-theorem axiom checks pass for every module. All files
under `Proofs/` are selected, including optional helpers; selection does not rely
on whether a declaration is referenced by a public theorem.

The baseline is commit `9e4e6ed1ece1661073ff5de36195d44bff710c64`; the measured
implementation is `417dc425de567b0654db2dda82586e78aca2b755`. Their 10,741 frozen
proof inputs and comparator binary hashes match. Both runs use Lean 4.29.0, the
same pinned VeriTile dependency and copies of the same built dependency cache.
The JSON binds these commits, source/tool hashes, the complete inventory digest
and the measured driver hashes. No task Lean or Python source changed; public
Real/FP theorem conditions and the existing numerical scope are preserved.

Validation also passed all four source-correspondence pairs, 742 Python tests,
101 audit regressions and 16 official-comparator integration tests. These checks
include rejection of unused private proofs with `sorry`/extra axioms, same-named
duplicate proofs carrying forbidden axioms, and dropped helper targets.

These are single-pair local measurements with 96 CPUs available to the process,
not predictions for GitHub runners. The largest sampling gaps were 170.3 ms
(old) and 165.5 ms (new); short memory spikes may be missed. The pilot is opt-in
and does not replace the full-corpus CI driver. Wider adoption requires checking
batch compatibility, memory limits and runtime on representative tasks.
