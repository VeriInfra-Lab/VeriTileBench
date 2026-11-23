"""Run interleaved Track 1 baselines and candidates on the same eight H100s."""

import argparse
import json
import os
from pathlib import Path
import re
from statistics import mean
import subprocess

from verify_candidate import check_checkout


CURRENT_RECORD_SECONDS = 39.914
LOSS_TARGET = 3.28
VALIDATION = re.compile(r"^step:(\d+)/(\d+) val_loss:(\S+) train_time:(\d+)ms\b")
RESULT_FIELDS = ("train_seconds", "validation_loss", "steps")


def final_result(log):
    results = []
    for line in log.splitlines():
        match = VALIDATION.match(line)
        if match and match[1] == match[2]:
            results.append(dict(zip(RESULT_FIELDS, (int(match[4]) / 1000, float(match[3]), int(match[2])))))
    if len(results) != 1:
        raise ValueError("Expected one final validation line from the upstream trainer")
    result = results[0]
    if not 0.0 < result["validation_loss"] < float("inf") or not 0.0 < result["train_seconds"]:
        raise ValueError("Run reported invalid final values")
    return result


def run_pair(baseline, candidate, data_root, output, pairs):
    check_checkout(baseline, candidate)
    devices = subprocess.check_output(
        ("nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"), text=True,
    ).splitlines()
    if len(devices) != 8 or any("H100" not in device for device in devices):
        raise ValueError("Track 1 timings require a node with eight H100s")
    data = data_root / "data/fineweb10B"
    if not list(data.glob("fineweb_train_*.bin")) or not list(data.glob("fineweb_val_*.bin")):
        raise ValueError("Point data_root at a directory containing the upstream data/fineweb10B files")
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "DATA_PATH": str(data_root.resolve())}
    env.pop("NUM_SCHEDULED_ITERATIONS", None)
    state = {"devices": devices, "data_path": env["DATA_PATH"], "runs": []}
    for pair in range(pairs):
        variants = [("baseline", baseline), ("candidate", candidate)]
        for name, checkout in variants if pair % 2 == 0 else reversed(variants):
            path = output / f"{pair + 1}-{name}.log"
            with path.open("w") as log:
                subprocess.run(("bash", "run.sh"), cwd=checkout, env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
            result = final_result(path.read_text())
            state["runs"].append({"pair": pair + 1, "variant": name, **result})
            (output / "results.json").write_text(json.dumps(state, indent=2) + "\n")
            print(name, result, flush=True)
    baseline_rows = [run for run in state["runs"] if run["variant"] == "baseline"]
    candidate_rows = [run for run in state["runs"] if run["variant"] == "candidate"]
    baseline_seconds = mean(run["train_seconds"] for run in baseline_rows)
    candidate_seconds = mean(run["train_seconds"] for run in candidate_rows)
    loss = mean(run["validation_loss"] for run in candidate_rows)
    state["summary"] = {
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "paired_mean_seconds_saved": baseline_seconds - candidate_seconds,
        "candidate_mean_loss": loss,
        "below_published_record_and_local_baseline": (
            candidate_seconds < min(baseline_seconds, CURRENT_RECORD_SECONDS) and loss <= LOSS_TARGET
        ),
    }
    (output / "results.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    run_pair(args.baseline, args.candidate, args.data_root, args.output, args.pairs)
