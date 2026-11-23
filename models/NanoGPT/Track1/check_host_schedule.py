"""Compare the installed host schedule, upload layouts and optimizer settings."""

import argparse
import ast
import json
from pathlib import Path
import runpy
import subprocess
import timeit
from types import SimpleNamespace
from unittest.mock import patch

import torch

from install_candidate import ANVIL, CONFIG, SCHEDULE
from verify_candidate import check_checkout


BASE_COUNTS = (1122, 600, 1121, 1194)


def load_schedule(root):
    config = runpy.run_path(str(root / CONFIG))
    table = ast.parse((root / SCHEDULE).read_text())
    table.body = [node for node in table.body if not isinstance(node, ast.ImportFrom)
                  or node.module != "track_1_short.config"]
    namespace = {**config, "torch": torch}
    exec(compile(table, str(root / SCHEDULE), "exec"), namespace)
    return namespace


def construct(namespace, base):
    return namespace["TrainingSchedule"](namespace["TRAINING_STAGES"], base, 20, torch.device("cpu"),
                                         namespace["LR_COOLDOWN_FRAC"], namespace["SPLIT_EMBED_STAGE"],
                                         namespace["WS_POST_YARN_EXT"])


def load_optimizer(root, common):
    path = root / "track_1_short/training.py"
    tree = ast.parse(path.read_text())
    manager, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainingManager")
    initialize, = (node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = {"step_optimizers", "is_adam_step"}
    methods = [node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name in names]
    value_constants = [node for node in tree.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id.startswith("VALUE_EMBED_")]
    value_function, = (node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "value_embed_betas_and_wd_mul")
    rails, = (node for node in ast.parse((root / ANVIL).read_text()).body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
              and any(isinstance(x, ast.Name) and x.id == "RAIL_ENGAGE_STEP" for x in node.targets[0].elts))
    event_period, = (node for node in ast.parse((root / "track_1_short/ngram_table.py").read_text()).body
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "NGRAM_ADAM_PERIOD4_START")
    defaults, = (node.value for node in initialize.body if isinstance(node, ast.Assign)
                 and any(ast.unparse(t) == "self.adam_defaults" for t in node.targets))
    adam = {item.arg: ast.literal_eval(item.value) for item in defaults.keywords}
    namespace = dict(torch=torch, RowPrefetch=object, get_rail_beta=common["get_rail_beta"])
    code = ast.Module(body=[*value_constants, value_function, event_period, rails, *methods], type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)
    return namespace, adam


def observe_optimizer(namespace, adam, schedule, step):
    groups = {
        "anvil": SimpleNamespace(initial_lr=0.023, optim="anvil", label="fixture"),
        "value": SimpleNamespace(initial_lr=0.56, optim="adam", label="value_embeds"),
        "other": SimpleNamespace(initial_lr=0.008, optim="adam", label="fixture"),
    }
    state = {}
    optimizer = SimpleNamespace(param_cfgs=groups, set_rails=lambda **kw: state.update(kw),
                                step=lambda **kw: state.update(kw),
                                copy_lm_state_to_embed=lambda: state.update(split=True))
    prefetch = SimpleNamespace(optimizer_event=lambda _, **kw: kw)
    manager = SimpleNamespace(schedule=schedule, optimizer=optimizer, adam_defaults=adam,
                              split_step=schedule.split_step)
    manager.is_adam_step = lambda n: namespace["is_adam_step"](manager, n)
    namespace["step_optimizers"](manager, step, prefetch, frozenset())
    sparse = state["sparse_update"]
    anvil, value, other = (groups[k] for k in ("anvil", "value", "other"))
    words = [[anvil.lr, anvil.momentum], [value.lr, *value.adam_betas, value.wd_mul], [other.lr]]
    return ([[list(v.as_integer_ratio()) for v in group] for group in words],
            *(list(state[k].as_integer_ratio()) for k in ("fast_beta", "fast_weight")),
            *(list(sparse[k].as_integer_ratio()) for k in ("lr", "weight_decay", "eps")),
            state["do_adam"], state.get("split", False))


def check(baseline, candidate, benchmark):
    check_checkout(baseline, candidate)
    before, after = (load_schedule(path) for path in (baseline, candidate))
    optimizer, adam = load_optimizer(baseline, before)
    lean_file = Path(__file__).with_name("CheckHostSchedule.lean")
    run = subprocess.run(["lake", "env", "lean", "--run", str(lean_file)], text=True, stdout=subprocess.PIPE, check=True)
    modeled = json.loads(run.stdout)
    assert [row[0] for row in modeled] == list(BASE_COUNTS)
    checked = 0
    for base, limits, split, rows in modeled:
        with patch.object(torch, "tensor", wraps=torch.tensor) as tensor:
            original = construct(before, base)
            original_allocations = tensor.call_count
            tensor.reset_mock()
            optimized = construct(after, base)
            packed_allocations = tensor.call_count
        assert original_allocations == 2 * (original.total_steps + 1) and packed_allocations == 2
        assert limits == [list(pair) for pair in original.boundaries] == [list(pair) for pair in optimized.boundaries]
        assert split == original.split_step == optimized.split_step
        assert original.total_steps + 1 == optimized.total_steps + 1 == len(rows)
        for i, modeled_row in enumerate(rows):
            stage, progress = original.lookup(i)
            opt_stage, opt_progress = optimized.lookup(i)
            assert opt_progress == progress and vars_from(opt_stage) == vars_from(stage)
            mtp, prefix_weight = (original.mtp_weights[i], original.prefix_weights[i])
            for x, y in ((mtp, optimized.mtp_weights[i]), (prefix_weight, optimized.prefix_weights[i])):
                assert x.dtype == y.dtype == torch.float32 and x.shape == y.shape
                assert y.is_contiguous() and y.data_ptr() % 16 == 0 and y.storage_offset() == 4 * i
                assert torch.equal(x.view(torch.int32), y.view(torch.int32))
            actual_row = [stage.batch_size, stage.train_max_seq_len, *stage.window_sizes, list(progress.as_integer_ratio()),
                          list(original.get_lr(i).as_integer_ratio()),
                          list(before["get_rail_beta"](i, original.total_steps).as_integer_ratio()),
                          mtp.view(torch.int32).tolist(), int(prefix_weight.view(torch.int32)[0]),
                          *observe_optimizer(optimizer, adam, original, i)]
            assert actual_row == modeled_row, (base, i, actual_row, modeled_row)
            assert optimized.get_lr(i) == original.get_lr(i)
        checked += len(rows)
    print(f"Matched {checked} schedule rows and optimizer settings. Default constructor tensor calls 2390 -> 2.", flush=True)
    if benchmark:
        a, b = [min(timeit.repeat(lambda: construct(v, 1122), number=12, repeat=5)) * 1000 / 12
                for v in (before, after)]
        print(f"CPU constructor {a:.2f} ms -> {b:.2f} ms", flush=True)


def vars_from(stage):
    return [getattr(stage, name) for name in stage.__slots__]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.benchmark)
