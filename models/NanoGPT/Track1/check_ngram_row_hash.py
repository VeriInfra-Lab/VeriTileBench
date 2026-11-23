"""Check the installed CPU row hash against the pinned PyTorch implementation."""

import argparse
import ast
from pathlib import Path
import timeit
from types import SimpleNamespace

import numpy as np
import torch

from install_candidate import NGRAM_TABLE
from verify_candidate import check_checkout


SIZES = (16384, 32768, 40960, 49152, 262144)
MODULUS = 42301439


def load_hash(root):
    path = root / NGRAM_TABLE
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = {"NGRAM_VOCAB_SIZE", "BIGRAM_ROW_MULS", "TRIGRAM_ROW_MULS"}
    functions = {"ngram_row_ids", "_ngram_cpu_rows", "_ngram_remainder_"}
    definitions = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        or isinstance(node, ast.FunctionDef) and node.name in functions
    )]
    namespace = dict(torch=torch, Tensor=torch.Tensor, np=np)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["NGRAM_VOCAB_SIZE"] // 2 - 1 == MODULUS
    return SimpleNamespace(**namespace)


def check(baseline, candidate, benchmark):
    check_checkout(baseline, candidate)
    before = load_hash(baseline).ngram_row_ids
    installed = load_hash(candidate)
    after = installed.ngram_row_ids

    # Check both sides of every quotient boundary, plus the signed endpoints.
    lower, upper = -(1 << 31), (1 << 31) - 1
    boundaries = [k * MODULUS + delta for k in range(-51, 52) for delta in range(-3, 4)
                  if lower <= k * MODULUS + delta <= upper]
    words = np.array([lower, lower + 1, upper - 1, upper, *boundaries], dtype=np.int32)
    expected = words % MODULUS
    installed._ngram_remainder_(words, np.empty(words.size, dtype=np.int64), MODULUS)
    np.testing.assert_array_equal(words, expected)

    rng = np.random.default_rng(7460)
    cases = [(2, 0, 65536), (16383, 0, 65536)]
    cases += [(size, lo, hi) for size in SIZES for lo, hi in ((0, 65536), (lower, upper + 1))]
    for size, lo, hi in cases:
        tokens = torch.from_numpy(rng.integers(lo, hi, size, dtype=np.int32))
        snapshot = tokens.clone()
        observed = after(tokens)
        assert observed.dtype == torch.int32 and observed.is_contiguous()
        assert torch.equal(observed, before(tokens)) and torch.equal(tokens, snapshot), (size, lo, hi)

    strided = torch.from_numpy(rng.integers(0, 65536, 2 * SIZES[0], dtype=np.int32))[::2]
    assert torch.equal(before(strided), after(strided))
    print(f"Matched {words.size} remainder boundaries and all default batch sizes, including signed overflow.", flush=True)

    if benchmark:
        print(f"CPU threads {torch.get_num_threads()}", flush=True)
        for size in SIZES:
            tokens = torch.from_numpy(rng.integers(0, 50257, size, dtype=np.int32))
            number = max(20, 1_000_000 // size)
            a, b = (min(timeit.repeat(lambda: f(tokens), number=number, repeat=5)) * 1e6 / number
                    for f in (before, after))
            print(f"{size:6d} tokens   PyTorch {a:8.1f} µs   NumPy {b:8.1f} µs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.benchmark)
