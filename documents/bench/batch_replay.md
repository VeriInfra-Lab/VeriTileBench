# Batch proof audit

`scripts/bench/audit_trust.sh` uses bounded batch replay for the selected corpus.
CI freezes its complete inventory, selects the shard, builds the selected
modules and dependencies, and passes that exact file list to the audit driver.
Selection includes retained auxiliary and private proofs, even when the public
Real/FP entries do not import them.

```sh
AUDIT_TRUST_SHARD_COUNT=64 AUDIT_TRUST_SHARD_INDEX=41 \
  scripts/bench/audit_trust.sh --final-inventory
```

For a selected task or an explicit file list:

```sh
scripts/bench/audit_trust.sh rms_matmul_rbe
python3 scripts/check_comparator_batch.py --task rms_matmul_rbe
python3 scripts/check_comparator_batch.py --files-from /path/to/selected-files.txt
```

## Scheduling and limits

| Setting | Default | Meaning |
| --- | ---: | --- |
| `AUDIT_TRUST_AUDIT_JOBS` | `2` | Concurrent audited-file compilers |
| `AUDIT_TRUST_BATCH_SIZE` | `32` | Maximum files in one comparator batch |
| `AUDIT_TRUST_FILE_TIMEOUT` | `1800` seconds | Active compilation and replay budget per file |
| `AUDIT_TRUST_BATCH_TIMEOUT` | `1800` seconds | Combined inventory/replay budget per batch |
| `AUDIT_TRUST_HEARTBEAT_SECONDS` | `30` seconds | Live progress interval |

Batches keep neighboring files from the same task together and never add files
outside the selected shard. Each source receives its own audited compilation,
compiler trace and complete theorem inventory. At most the configured number
of audit compilers run concurrently; each has an independent deadline.

The batch deadline is also capped by its members' remaining file budgets.
Individual replay uses the remaining file budget. Snapshot copying, queueing
and final source-identity checks are global preparation/verification work,
rather than active work attributed to one file. Timeouts fail the audit; they
never skip a file or authorize a partial result. The task-level CI limit is
six hours.

Each subprocess emits flushed begin, heartbeat and end records naming its
source, phase, elapsed time, timeout and log path. Batch records list their
members. A timed-out compiler's process group is killed. A timed-out comparator
also stops its uniquely named systemd service, including exporter workers
outside the launcher's process group.

`AUDIT_TRUST_REPLAY=individual` selects the independent per-file path for diagnosis;
it uses the same deadlines and live logs. `AUDIT_TRUST_JOBS` controls concurrency
in that mode. Individual checks also expose `--timeout` and `--heartbeat`.
The batch CLI exposes `--audit-batch-size`, `--batch-size`, `--file-timeout`,
`--batch-timeout` and `--heartbeat`. `--task` selects one complete task;
`--files-from` uses exactly the nonempty, unique paths in its input file.

## Coverage and acceptance

Every selected module must retain its complete per-file theorem inventory,
including zero-theorem modules, unused helpers, quoted names and generated
matcher equations. Ordinary modules also receive a standard-axiom check for
every theorem before batches are combined. Compiler messages come from the
module's own Lake trace; a dependency's successful audit cannot satisfy a
missing audit in the selected source.

The combined inventory imports original modules, resolves the union of their
per-file targets and adds every theorem owned by the selected original modules.
Only the copied module prefix of private names is normalized; private counters
and macro scopes are not erased. Lazy equation-theorem ownership can move under
combined imports, so the complete target union is checked independently of
final module attribution. Missing, duplicate or unexpected inventories/targets
fail the audit.

Each accepted group passes the pinned official comparator with the unchanged
allowed axioms and systemd/Landlock isolation. The exporter recursively checks
proof dependencies. Zero-theorem modules retain the standard primitive checks.
The source and judge fingerprints are checked before and after the selection;
the final inventory snapshot must also match the build-time snapshot.

Lean `module` files and source-copy hygienic theorem names receive independent
replay. A combined-import compilation conflict also triggers explicit individual
replay of every affected file. These paths retain full target enumeration and
require official acceptance of every file; they do not authorize a batch.
A theorem-coverage mismatch, failed trust audit, timeout or comparator rejection
is a failure, not a reason to ignore a declaration. The individual path is
important for `module` imports: hidden imported theorem bodies must not be
misclassified as additional axioms by an elaborator-side traversal.

The selection result accounts for every batch member and individual fallback
exactly once. It requires every replay to succeed before reporting the shard
as accepted. All shards and the interface/source gates must succeed for a full
CI acceptance. Kernel sources, public theorem conditions and FP numerical
admission remain separate from audit scheduling.

## Evidence

`Logs/comparator-check-*` contains source/tool fingerprints, per-file inventories
and trust logs, batch membership/coverage, official comparator output and
selection results. CI retains these diagnostics for seven days when a shard
fails. No measurement logs are committed to Git.

Recorded performance evidence is under
[`validation/replay_batch/`](../../validation/replay_batch/). Each receipt binds
its measured source, tools, theorem inventories, elapsed time and sampled
aggregate RSS; it is not a runtime or physical-memory guarantee for another
host or selection. `scripts/bench/measure_replay.py` measures an explicit command,
and `scripts/bench/compare_replay.py` requires matching proof inputs, tools,
per-file theorem lists and trust summaries before accepting a comparison.
