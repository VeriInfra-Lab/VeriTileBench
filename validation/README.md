# Validation records

- `task_progress.json` selects the 183 current repaired tasks.
- `source-correspondence.json` records pasted-body coverage and pending translations.
- [`body_receipts/source-authoring-replay.json`](body_receipts/source-authoring-replay.json)
  records the full comparator replay of all 183 TritonBench-G tasks and 19 examples
  at source commit `98d4130aa`: 8,468 case modules and 293 shared checks passed,
  with 226,321 theorem targets replayed. It binds the results to the unchanged
  source/tool hashes and the archived per-file evidence.
- `body_receipts/benchmark-layout-replay.json` preserves the earlier independent proof replay for the standardized task layout.
- `source-authoring.json` preserves the earlier local source-authoring validation;
  its comparator availability entry describes that earlier attempt. The complete
  follow-up replay is recorded above.
- [`archive/contracts.json`](archive/contracts.json) preserves the earlier review of specifications,
  conditions and numerical scope. [`archive/source.json`](archive/source.json) preserves the earlier
  source and import inventory. Generate a current inventory with
  `python3 scripts/bench/final_inventory.py --write PATH`.
- `imported_kernel_review.json` preserves findings for the 334 imported challenges.
- `bench-layout.json` records the path migration and unchanged Lean/JIT code.
- [`examples-copy.json`](examples-copy.json) records the 18 worked examples copied
  from a fixed VeriTile main revision, unchanged Lean/Python bytes, and local
  source-correspondence/build checks. The existing TritonBenchVectorAddition
  example remains alongside them.
- [`replay_batch/rms_matmul_rbe.json`](replay_batch/rms_matmul_rbe.json) records the
  opt-in batch-replay pilot: exact old/new inventories, official acceptance,
  runtime and sampled aggregate RSS. The adjacent compressed inventory preserves
  every per-file theorem name and the batch target union. See the
  [measurement method and limits](../documents/bench/batch_replay.md).
- The follow-up receipts [`replay_batch/rms_matmul_rbe.audit-waves-2.json`](replay_batch/rms_matmul_rbe.audit-waves-2.json)
  and [`replay_batch/rms_matmul_rbe.audit-waves-4.json`](replay_batch/rms_matmul_rbe.audit-waves-4.json)
  compare bounded audit-build waves against both original replay paths. Their
  complete theorem inventory is identical to the initial pilot's inventory.

Task-specific compiler and numerical receipts are under `tasks/`. Their JSON
bytes, source hashes, outcomes and original paths are preserved. Resolve those
original paths against [the source snapshot before the layout migration](https://github.com/VeriInfra-Lab/VeriTile/tree/4edf710ba663205d9bbc16ed1efe12ea1a72a803); they are historical evidence,
not claims that an experiment or compiler probe was repeated after moving files.
Current proof/source bindings are the top-level records listed above.

Prior runnable Python implementations and retired task validators remain in Git
history. The live benchmark tree exposes only each task's current repaired program
and the comparison candidates/components required by its present FP interface.
