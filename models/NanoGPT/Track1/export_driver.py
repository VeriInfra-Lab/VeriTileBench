"""Extract the control flow of the pinned trainer for the Lean execution model."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from install_candidate import BASELINE, DRIVER, git, replace_driver


THIS = Path(__file__).resolve().parent
ARTIFACT = THIS / "DriverSource.lean"

# Runtime bookkeeping has no numerical result. The input to the numerical
# model includes device/event scheduling, which the wall clock can influence.
DIAGNOSTIC_CALLS = {
    "print0", "flush_log", "log_environment", "gc.collect", "gc.freeze",
    "gc.enable", "gc.disable",
}
DIAGNOSTIC_TARGETS = {"code", "training_time_ms", "t0", "approx_training_time_ms"}
VALIDATION_STEPS = {
    "assert args.val_tokens % args.val_batch_size == 0": "checkBatchCount",
    "val_steps = args.val_tokens // args.val_batch_size": "bindBatchCount",
    "val_batches = [next(val_loader_iter) for _ in range(val_steps)]": "loadBatches",
    "val_loss = 0": "resetLoss",
    "val_loss /= val_steps": "scale",
    "dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)": "reduce",
}

TRAINING_BATCH_STEPS = {
    "warmup_batches = ScheduledBatches(train_loader(), training_schedule, steps=warmup_steps)": "openSource true",
    "batches = ScheduledBatches(train_loader(), training_schedule, steps=range(training_schedule.total_steps))": "openSource false",
    "batch = batches.peek(step)": "access false",
    "batches.take(step)": "access true",
}


def quoted(value):
    # Lean string escapes agree with these JSON escapes on this ASCII source.
    return json.dumps(value, ensure_ascii=False)


def call_name(node):
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return None


def check_slot_boundary(source, *, candidate):
    train_step = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "train_step")
    assignments = [index for index, stmt in enumerate(train_step.body) if
                   isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ngram_slots"
                                                        for t in stmt.targets)]
    forwards = [index for index, stmt in enumerate(train_step.body)
                if isinstance(stmt, (ast.Assign, ast.Expr)) and call_name(stmt.value) == "step_graphs.forward"]
    [forward] = forwards
    [routing] = list(index for index in assignments if index != forward)
    if routing >= forward or (forward in assignments) != candidate:
        raise ValueError("Unexpected n-gram slot binding at the forward graph")
    if candidate:
        batch_bindings = [stmt for stmt in train_step.body[:routing]
                          if any(isinstance(node, ast.Name) and node.id == "batch" and
                                 isinstance(node.ctx, ast.Store) for node in ast.walk(stmt))]
        if [ast.unparse(stmt) for stmt in batch_bindings] != ["batch = batches.peek(step)"]:
            raise ValueError("The hashed slot request must use the scheduled training batch")
        expected = "row_prefetch.forward_slots(step, batch.ngram_ids, defer_slots=True, hashed_rows=True)"
        if ast.unparse(train_step.body[routing].value) != expected:
            raise ValueError("The fused slot request must use the uploaded hash rows")
    for stmt in train_step.body[routing + 1:forward]:
        if any(isinstance(node, ast.Name) and node.id == "ngram_slots" for node in ast.walk(stmt)):
            raise ValueError("A slot reader precedes the forward graph")


def diagnostic(node):
    if isinstance(node, ast.Expr) and call_name(node.value) in DIAGNOSTIC_CALLS:
        return True
    if isinstance(node, ast.Assign):
        return all(ast.unparse(t) in DIAGNOSTIC_TARGETS for t in node.targets)
    if isinstance(node, ast.AugAssign):
        return ast.unparse(node.target) in DIAGNOSTIC_TARGETS
    return False


def block(nodes, level=1):
    if not nodes:
        return "[]"
    indent = "  " * level
    return "[\n" + ",\n".join(indent + statement(node, level + 1) for node in nodes) + "\n" + indent[:-2] + "]"


def statement(node, level):
    line = node.lineno
    code = ast.unparse(node)
    text = quoted(code)
    if isinstance(node, ast.If):
        return f".branch {line} {quoted(ast.unparse(node.test))} {block(node.body, level)} {block(node.orelse, level)}"
    if isinstance(node, ast.For):
        if node.orelse:
            raise ValueError(f"Unexpected for/else at line {line}")
        return f".loop {line} {quoted(ast.unparse(node.target))} {quoted(ast.unparse(node.iter))} {block(node.body, level)}"
    if isinstance(node, ast.With):
        items = ", ".join(quoted(ast.unparse(item)) for item in node.items)
        return f".within {line} [{items}] {block(node.body, level)}"
    if isinstance(node, ast.Break):
        return f".breakLoop {line}"
    if isinstance(node, ast.Expr) and call_name(node.value) == "train_step":
        return f".trainStep {line} {text}"
    if isinstance(node, ast.Assign) and call_name(node.value) == "training_manager.step_optimizers":
        return f".optimizerStep {line} {text}"
    if isinstance(node, ast.Assign) and call_name(node.value) == "TrainingSchedule":
        return f".setupSchedule {line}"
    if isinstance(node, ast.Expr) and call_name(node.value) == "training_manager.advance_schedule":
        return f".advanceSchedule {line}"
    if isinstance(node, ast.Assign) and ast.unparse(node) == "ngram_slots = row_prefetch.forward_slots(step, batch.ngram_ids)":
        return f".rowForward {line}"
    if code == "sampled_loss = sampled_softmax.gather(step, lm_head_f8_col)":
        return f".sampledGather {line}"
    if code == "step_graphs.forward(step, batch, ngram_slots, training_manager.get_forward_args(sampled_loss))":
        return f".graphForward {line}"
    if code == "ngram_grad = step_graphs.backward()":
        return f".graphBackward {line}"
    if code == ("ngram_table.accumulate_grad(ngram_slots, step_graphs.hold_ngram_grad(ngram_grad, "
                "pending=len(ngram_table.pending), event_this_step=row_prefetch.is_update(step)))"):
        return f".graphHoldNgram {line}"
    if isinstance(node, ast.Expr) and ast.unparse(node) == "ngram_table.land(pull)":
        return f".rowLand {line}"
    if isinstance(node, ast.Expr) and ast.unparse(node) == "tail_averages.tick(step)":
        return f".tailTick {line}"
    if isinstance(node, ast.Expr) and ast.unparse(node) == "tail_averages.ship()":
        return f".tailShip {line}"
    if code in VALIDATION_STEPS:
        return f".validationCalc {line} .{VALIDATION_STEPS[code]}"
    if code in TRAINING_BATCH_STEPS:
        return f".trainingBatch {line} (.{TRAINING_BATCH_STEPS[code]})"
    if isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "val_loss":
        value = node.value
        if (not isinstance(node.op, ast.Add) or not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Attribute) or value.func.attr != "mean"
                or value.args or value.keywords or call_name(value.func.value) != "model"):
            raise ValueError(f"Unrecognized validation loss accumulation at line {line}")
        return f".validationCalc {line} (.accumulate {quoted(ast.unparse(value.func.value))})"
    if isinstance(node, ast.Expr) and call_name(node.value) == "print0":
        values = [child for child in ast.walk(node.value) if isinstance(child, ast.FormattedValue)]
        if any(ast.unparse(child.value) == "val_loss" for child in values):
            return f".validation {line} {text}"
    if diagnostic(node):
        return f".diagnostics {line} {text}"
    if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Assert,
                             ast.Delete, ast.Import, ast.ImportFrom, ast.FunctionDef)):
        raise ValueError(f"Unsupported driver statement {type(node).__name__} at line {line}")
    return f".shared {line} {text}"


def render(source):
    tree = ast.parse(source)
    symbols = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    entry = tree.body[-1]
    if not isinstance(entry, ast.If) or ast.unparse(entry.test) != "__name__ == '__main__'":
        raise ValueError("Unexpected Python driver entry point")
    prefix = block(tree.body[:-1])
    return f"""import models.NanoGPT.Track1.DriverSyntax

/-! Generated by export_driver.py from the pinned train_gpt.py. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

def upstreamRevision : String := {quoted(BASELINE)}
def upstreamDriverSHA256 : String := {quoted(hashlib.sha256(source.encode()).hexdigest())}

def moduleInitialization : List Stmt := {prefix}

def trainingStep : List Stmt := {block(symbols["train_step"].body)}

def main : List Stmt := {block(symbols["main"].body)}

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
"""


def check_checkout(checkout, *, candidate=False):
    if git(checkout, "rev-parse", "HEAD").strip() != BASELINE:
        raise ValueError(f"Expected {BASELINE}")
    source = git(checkout, "show", f"HEAD:{DRIVER}")
    expected = replace_driver(source) if candidate else source
    actual = (checkout / DRIVER).read_text()
    if expected != actual:
        raise ValueError("The actual trainer differs from the pinned driver")
    check_slot_boundary(actual, candidate=candidate)
    if ARTIFACT.read_text() != render(source):
        raise ValueError("The Lean driver does not match the pinned Python source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        if git(args.checkout, "rev-parse", "HEAD").strip() != BASELINE:
            raise ValueError(f"Expected {BASELINE}")
        ARTIFACT.write_text(render(git(args.checkout, "show", f"HEAD:{DRIVER}")))
    check_checkout(args.checkout)
