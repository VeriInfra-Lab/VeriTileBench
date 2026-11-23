"""Replay the installed FineWeb batch loaders on synthetic shards."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import glob
from itertools import pairwise
import math
import os
from pathlib import Path
import threading
from tempfile import TemporaryDirectory
from textwrap import dedent, indent
import timeit
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import torch

from check_ngram_row_hash import load_hash
from install_candidate import FINEWEB_DATA, FINEWEB_TORCH_CASTS, FINEWEB_TORCH_PACK
from verify_candidate import check_checkout


def load_data(root):
    path = root / FINEWEB_DATA
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = {"HEADER_BYTES", "SHARD_READ_THREADS", "BOS_ID", "TRAIN_MAX_NUM_DOCS"}
    functions = {"pread_parallel", "_load_data_shard", "cu_seqlens_rows", "split_attention_segments",
                 "_pack_bos_batch", "distributed_data_generator", "Batch", "Shard", "ScheduledBatches"}
    selected = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        or isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions
    )]
    namespace = dict(torch=torch, Tensor=torch.Tensor, np=np, os=os, glob=glob, threading=threading,
                     Path=Path, NamedTuple=NamedTuple, Sequence=list, PinnedBatchStaging=object,
                     ThreadPoolExecutor=ThreadPoolExecutor, TrainingSchedule=object, pairwise=pairwise,
                     ngram_row_ids=load_hash(root).ngram_row_ids, VIRTUAL_SEQ_CAP=2560,
                     next_multiple_of_n=lambda v, n: math.ceil(v / n) * n)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def write_shard(path, documents, rng):
    tokens = rng.integers(0, 40000, sum(documents), dtype=np.uint16)
    tokens[np.cumsum([0, *documents[:-1]])] = 50256
    header = np.zeros(256, dtype="<i4")
    header[:3] = [20240520, 1, tokens.size]
    with path.open("wb") as stream:
        header.tofile(stream)
        tokens.astype("<u2", copy=False).tofile(stream)


def snapshot(namespace, pattern, rank, world_size, geometries, coalesced=None):
    namespace["dist"] = SimpleNamespace(get_rank=lambda: rank, get_world_size=lambda: world_size)
    def upload(*arrays, coalesce=False):
        if coalesced is not None:
            coalesced.append(coalesce)
        return arrays

    staging = SimpleNamespace(upload=upload)
    tokens, length = geometries[0]
    loader = namespace["distributed_data_generator"](pattern, world_size * tokens, length, staging, align_to_bos=True)
    steps = [3 * i for i in range(len(geometries))]

    def lookup(step):
        count, cap = geometries[step // 3]
        return SimpleNamespace(batch_size=world_size * count, train_max_seq_len=cap), 0.0

    schedule = SimpleNamespace(lookup=lookup)
    queue = namespace["ScheduledBatches"](loader, schedule, steps)
    empty = torch.empty

    def unpinned(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    with patch.object(torch, "empty", unpinned):
        last = queue.peek(steps[-1])
        batches = [queue.take(step) for step in steps]
    assert batches[-1] is last and not queue.ahead
    return batches


def benchmark_pack(before, after):
    source = """def pack_reference(tokens, starts, ends, num_tokens_local, max_seq_len, max_num_docs):
    start_idxs, end_idxs = torch.tensor(starts), torch.tensor(ends)
""" + indent(dedent(FINEWEB_TORCH_PACK), "    ") + indent(dedent(FINEWEB_TORCH_CASTS), "    ") + """
    return _inputs, _targets, _cum_lengths
"""
    exec(compile(source, "<pinned pack>", "exec"), before)
    rng = np.random.default_rng(7460)
    source_tokens = torch.from_numpy(rng.integers(0, 65536, 70000, dtype=np.uint16))
    for count, cap, rows in ((16384, 896, 64), (32768, 2048, 96), (40960, 3072, 128), (49152, 3072, 128)):
        starts, ends = [], []
        remaining, pos = count + 1, 0
        while remaining:
            length = min(remaining, cap, int(rng.integers(256, 900)))
            starts.append(pos)
            ends.append(pos + length)
            remaining -= length
            pos += length + 11
        args = (source_tokens, starts, ends, count, cap, rows)
        reference, optimized = before["pack_reference"], after["_pack_bos_batch"]
        assert all(torch.equal(a, b) for a, b in zip(reference(*args), optimized(*args), strict=True))
        a, b = (min(timeit.repeat(lambda: f(*args), number=80, repeat=5)) * 1e6 / 80
                for f in (reference, optimized))
        print(f"{count:6d} tokens   PyTorch {a:7.1f} µs   NumPy {b:7.1f} µs", flush=True)


def check(baseline, candidate, benchmark):
    check_checkout(baseline, candidate)
    before, after = (load_data(path) for path in (baseline, candidate))
    geometries = [(16384, 896), (32768, 2048), (40960, 3072), (49152, 3072), (300, 896)]
    world_size = 8
    with TemporaryDirectory(prefix="veritile-fineweb-") as tmp:
        directory = Path(tmp)
        rng = np.random.default_rng(7460)
        write_shard(directory / "00.bin", [2048] * 5, rng)
        write_shard(directory / "01.bin", [3000 if i % 2 else 1530 for i in range(1200)], rng)
        expected_pack = after["_pack_bos_batch"]
        packed = []
        coalesced = []

        def capture(tokens, starts, ends, count, length, rows):
            packed.append(count)
            return expected_pack(tokens, starts, ends, count, length, rows)

        for rank in range(world_size):
            reference = snapshot(before, str(directory / "*.bin"), rank, world_size, geometries)
            with patch.dict(after, {"_pack_bos_batch": capture}):
                optimized = snapshot(after, str(directory / "*.bin"), rank, world_size, geometries, coalesced)
            for a, b in zip(reference, optimized, strict=True):
                for x, y in zip(a[:4], b[:4], strict=True):
                    assert x.dtype == y.dtype and torch.equal(x, y)
        assert packed == [count for count, _ in geometries[:-1]] * world_size
        assert coalesced == [True, True, True, True, False] * world_size
    print("Matched all eight ranks through rollover, four optimized geometries, fallback, and queued lookahead.", flush=True)
    if benchmark:
        benchmark_pack(before, after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.benchmark)
