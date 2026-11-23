"""Check the installed n-gram slot lookup, pinned packet, and emitted SM90 PTX."""

import argparse
import ast
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from check_graph_stage import instructions
from install_candidate import NGRAM_SLOTS_KERNEL, NGRAM_TABLE, ROW_PREFETCH
from qkv_ptx_replay import INT32_OPS, PREDICATES, WORD32, address, operands_in, value
from verify_candidate import check_checkout, load_candidate_scale


INPUTS = ("Want", "Directory", "Rows", "Slots")
CONSTANTS = ("COUNT", "VOCAB", "TILE")
BASE_STRIDE = 1 << 32


def signed32(words):
    words = np.asarray(words, dtype=np.int64) & WORD32
    return (words ^ (1 << 31)) - (1 << 31)


def operand_index(pointer, bank, width):
    offset = pointer - (bank + 1) * BASE_STRIDE
    assert np.all(offset >= 0) and np.all(offset % width == 0)
    return offset // width


def transfer(op, operand):
    *vector, word = op.split(".")[2:]
    regs = operands_in(operand) if operand.startswith("{") else [operand]
    assert vector == ([] if len(regs) == 1 else [f"v{len(regs)}"])
    assert word in ("b32", "b64")
    return int(word[1:]) // 8, regs


def replay_buffers(ptx, kernel, ctas, inputs, outputs, want_bank):
    """Check every PTX load and store with explicit per-bank byte widths."""
    code, labels, threads = instructions(ptx, kernel)
    lanes = np.arange(ctas * threads, dtype=np.int64)
    regs = {"%tid.x": lanes % threads, "%ctaid.x": lanes // threads}
    active = np.ones(lanes.size, dtype=bool)
    returned = np.zeros(lanes.size, dtype=bool)
    branches = {}
    written = [np.zeros(expected.size, dtype=bool) for _, expected in outputs]
    observed = [np.empty_like(expected, dtype=np.int64) for _, expected in outputs]
    probes = np.zeros(lanes.size, dtype=np.uint8)

    for pc, (predicate, op, args) in enumerate(code):
        if pc in branches:
            active |= branches.pop(pc)
        if not active.any():
            continue
        mask = active.copy()
        if predicate:
            pred = np.asarray(regs[predicate.removeprefix("!")], dtype=bool)
            mask &= ~pred if predicate.startswith("!") else pred
        if not mask.any():
            continue
        if op == "ret":
            returned |= mask
            active &= ~mask
            continue
        dst, *src = args
        if op in ("bra", "bra.uni"):
            target = labels[dst]
            assert target > pc
            if target in branches:
                branches[target] |= mask
            else:
                branches[target] = mask
            active &= ~mask
        elif op == "ld.param.b64":
            param, = src
            bank = int(param[:-1].rsplit("_", 1)[1])
            regs[dst] = np.where(mask, (bank + 1) * BASE_STRIDE, regs.get(dst, 0))
        elif op.startswith("ld.global."):
            width, names = transfer(op, dst)
            ptr = np.broadcast_to(address(regs, src[0]), lanes.shape)[mask]
            indices = np.flatnonzero(mask)
            for j, name in enumerate(names):
                one = ptr + width * j
                banks = one // BASE_STRIDE - 1
                assert np.all((banks >= 0) & (banks < len(inputs)))
                words = np.broadcast_to(regs.get(name, 0), lanes.shape).copy()
                for bank in np.unique(banks):
                    word_bytes, buffer = inputs[bank]
                    assert width == word_bytes
                    from_bank = banks == bank
                    ix = operand_index(one[from_bank], bank, width)
                    assert np.all(ix < buffer.size)
                    loaded = buffer[ix].astype(np.int64)
                    words[indices[from_bank]] = loaded & WORD32 if width == 4 else loaded
                    if bank == want_bank:
                        probes[indices[from_bank]] += 1
                regs[name] = words
        elif op.startswith("st.global."):
            width, names = transfer(op, src[0])
            ptr = np.broadcast_to(address(regs, dst), lanes.shape)[mask]
            for j, name in enumerate(names):
                one = ptr + width * j
                banks = one // BASE_STRIDE - 1 - len(inputs)
                assert np.all((banks >= 0) & (banks < len(outputs)))
                words = np.broadcast_to(value(regs, name), lanes.shape)[mask]
                for bank in np.unique(banks):
                    word_bytes, expected = outputs[bank]
                    assert width == word_bytes
                    from_bank = banks == bank
                    ix = operand_index(one[from_bank], bank + len(inputs), width)
                    assert np.all(ix < expected.size) and np.unique(ix).size == ix.size and not written[bank][ix].any()
                    observed[bank][ix] = words[from_bank]
                    written[bank][ix] = True
        else:
            val = arithmetic(op, src, regs)
            regs[dst] = np.where(mask, val, regs.get(dst, False if dst.startswith("%p") else 0))

    assert returned.all() and not branches
    for mask, values, (_, expected) in zip(written, observed, outputs, strict=True):
        assert mask.all() and np.array_equal(values, expected)
    assert probes.max() <= 8
    return probes


def arithmetic(op, src, regs):
    operands = [value(regs, arg) for arg in src]
    if op == "mov.pred":
        return bool(operands[0])
    if op in ("mov.u32", "mov.b32", "mov.u64"):
        return operands[0]
    if op in INT32_OPS:
        return INT32_OPS[op](*operands) & WORD32
    if op == "min.u32":
        left, right = operands
        return np.minimum(left & WORD32, right & WORD32)
    if op == "shr.s32":
        word, distance = operands
        return (signed32(word) >> distance) & WORD32
    if op == "add.s64":
        return operands[0] + operands[1]
    if op in ("mad.wide.s32", "mad.wide.u32", "mul.wide.s32", "mul.wide.u32"):
        left, right, *base = operands
        if op.endswith(".s32"):
            left, right = signed32(left), signed32(right)
        else:
            left, right = left & WORD32, right & WORD32
        return left * right + (base[0] if base else 0)
    if op == "bfe.u32":
        word, start, width = operands
        return ((word & WORD32) >> start) & ((1 << width) - 1)
    if op.startswith("setp."):
        _, compare, dtype = op.split(".")
        assert dtype in ("s32", "u32", "b32")
        left, right = operands
        if dtype == "s32":
            left, right = signed32(left), signed32(right)
        else:
            left, right = left & WORD32, right & WORD32
        return PREDICATES[compare](left, right)
    if op == "selp.b32":
        left, right, predicate = operands
        return np.where(predicate, left, right)
    if op == "and.pred":
        return operands[0] & operands[1]
    raise AssertionError(f"Unexpected opcode {op}")


def replay(ptx, want, directory, queries, expected):
    probes = replay_buffers(ptx, "bucket_slots", triton.cdiv(queries.size, 128),
                            ((4, want), (4, directory), (4, queries)), ((4, expected),), 0)
    return probes.mean(), int(probes.max())


def compile_and_check(module, ir_dir):
    signature = dict.fromkeys(INPUTS, "*i32") | dict.fromkeys(CONSTANTS, "constexpr")
    alignment = {(i,): [["tt.divisibility", 16]] for i in range(len(INPUTS))}
    rng = np.random.default_rng(7460)
    want = np.unique(rng.integers(0, module.VOCAB_SIZE, 393216, dtype=np.int32))
    directory = module.make_directory(want)
    assert directory is not None
    special = np.array([-1 << 31, -1, 0, 32767, 32768, module.VOCAB_SIZE - 1,
                        module.VOCAB_SIZE, (1 << 31) - 1], dtype=np.int32)
    assert module.make_directory(np.arange(256, dtype=np.int32)) is None
    widest = np.arange(255, dtype=np.int32)
    widest_directory = module.make_directory(widest)
    assert widest_directory is not None
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)

    for count in module.QUERY_COUNTS:
        source = ASTSource(module.bucket_slots, signature,
                           {"COUNT": count, "VOCAB": module.VOCAB_SIZE, "TILE": module.BLOCK}, attrs=alignment)
        compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32), options={"num_warps": 4, "num_stages": 1})
        queries = np.concatenate((special, want[:count // 2 - len(special)],
                                  rng.integers(0, module.VOCAB_SIZE, count // 2, dtype=np.int32)))
        expected = np.searchsorted(want, queries, side="left")
        mean, peak = replay(compiled.asm["ptx"], want, directory, queries, expected)
        limits = np.resize(np.arange(-1, 257, dtype=np.int32), count)
        assert replay(compiled.asm["ptx"], widest, widest_directory, limits,
                      np.searchsorted(widest, limits, side="left"))[1] == 8
        if ir_dir is not None:
            for stage in ("ttir", "ttgir", "ptx"):
                (ir_dir / f"ngram-slots-{count}.{stage}").write_text(compiled.asm[stage])
        print(f"{count:6d} queries   {mean:.2f} mean, {peak} max executed want probes", flush=True)


def load(path, functions=(), classes=(), methods=(), **env):
    root = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in root.body if (isinstance(node, ast.FunctionDef) and node.name in functions)
                or (isinstance(node, ast.ClassDef) and node.name in classes)]
    for class_name, method_name in methods:
        owner, = (node for node in root.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        method, = (node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        selected.append(method)
    assert len(selected) == len(functions) + len(classes) + len(methods)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return env


def check_dispatch(baseline, candidate, module):
    original_sort = load(baseline / NGRAM_TABLE, functions=("sorted_unique_rows",), np=np)["sorted_unique_rows"]
    sort_rows = load(candidate / NGRAM_TABLE, functions=("sorted_unique_rows",), np=np)["sorted_unique_rows"]
    rng = np.random.default_rng(9283)
    for parts in ([np.empty(0, dtype=np.int32)],
                  [np.array([32768, 0, 0, -1], dtype=np.int32), np.array([0, -1, 5], dtype=np.int32)],
                  list(np.array_split(rng.integers(0, 32768, 393216, dtype=np.int32), 4))):
        saved = [part.copy() for part in parts]
        rows = sort_rows(parts)
        assert rows.dtype == np.int32 and np.array_equal(rows, original_sort(parts))
        assert all(np.array_equal(old, new) for old, new in zip(saved, parts, strict=True))

    prefetch = load(candidate / ROW_PREFETCH, functions=("cycle_wants",),
                    classes=("PinnedUploadRing", "NextPull"),
                    methods=(("RowPrefetch", "_request"), ("RowPrefetch", "land_next_cycle"),
                             ("RowPrefetch", "forward_slots")),
                    np=np, torch=torch, Tensor=torch.Tensor, PINNED_SLOTS=4, Batch=object, Callable=Callable,
                    RowPull=object, dataclass=dataclass, sorted_unique_rows=sort_rows,
                    directory_start=module.directory_start, make_directory=module.make_directory, ref=ref)
    # The dispatcher selects the Triton result when supplied and otherwise calls ATen.
    result = object()
    plan = SimpleNamespace(materialize=lambda: result)
    modes = []

    def prepare(want, directory, ids, *, hashed_rows, member_rows):
        modes.append((hashed_rows, member_rows))
        return plan if directory is not None else None

    entries = load(candidate / NGRAM_TABLE, methods=(("NgramTable", "slots"),),
                   torch=torch, Tensor=torch.Tensor, RowPull=object, SlotQuery=module.SlotQuery,
                   prepare_bucket_slots=prepare)
    want_host = np.array([0, 3, 32768, module.VOCAB_SIZE - 1], dtype=np.int32)
    host_dir = module.make_directory(want_host)
    batch = SimpleNamespace(ngram_ids_cpu=want_host[::-1], inputs_cpu=np.array([37], dtype=np.int32),
                            ngram_ids=torch.from_numpy(want_host[::-1].copy()))
    prepared = prefetch["cycle_wants"]([batch], np.concatenate)
    assert all(np.array_equal(a, b) for a, b in zip(prepared, (want_host, batch.inputs_cpu, host_dir)))
    assert len(prepared[3]) == 1 and prepared[3][0]() is batch.ngram_ids
    ids = torch.tensor([0, 1, 32768], dtype=torch.int32)
    ready = Future()
    ready.set_result(prepared)
    # The request consumes the directory from the prep future and uploads it
    # in the same copy and event scope as the wanted rows.
    log = []
    orig_empty = torch.empty
    orig_copy = torch.Tensor.copy_

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return orig_empty(*args, **kwargs)

    def copy(dst, src, *, non_blocking=False):
        if non_blocking:
            log.append(("transfer", dst.numel()))
        return orig_copy(dst, src, non_blocking=non_blocking)

    event = lambda: SimpleNamespace(synchronize=lambda: log.append("ready"), record=lambda: log.append("record"))
    with (patch.object(torch, "empty", empty), patch.object(torch.Tensor, "copy_", copy),
          patch.object(torch.cuda, "Event", event)):
        ring = prefetch["PinnedUploadRing"](16 + module.EXTRA_WORDS, "cpu")
        stub = SimpleNamespace(
            _cycle_from=lambda _: [1], prep_pending=(0, ready),
            staging=SimpleNamespace(ngram_ring=ring, value_embed_ring=SimpleNamespace(upload=torch.from_numpy)),
            ngram=SimpleNamespace(request=lambda want, host: SimpleNamespace(want=want, slot_directory=None)),
            value_embeds=SimpleNamespace(request=lambda *_: None),
        )
        prefetch["_request"](stub, 0)
        pull = stub.next_pull.ngram
        assert stub.prep_pending is None and stub.next_pull.cycle == [1]
        assert np.array_equal(pull.want.numpy(), want_host)
        assert np.array_equal(pull.slot_directory.numpy(), host_dir)
        assert pull.slot_directory.data_ptr() % 16 == 0
        assert log == ["ready", ("transfer", host_dir.size + 4), "record"]
        for i in range(5):
            ring.upload(np.array([10 + i], dtype=np.int32))
        assert np.array_equal(pull.want.numpy(), want_host) and np.array_equal(pull.slot_directory.numpy(), host_dir)

    assert entries["slots"](None, pull, ids) is result and module.try_bucket_slots(pull.want, pull.slot_directory, ids) is None
    assert entries["slots"](None, pull, ids, defer=True) is plan
    stub.position = {17: 1}
    stub.is_update = lambda _: False
    stub.ngram.complete_land = lambda: log.append("land")
    stub.ngram.start_cycle = lambda active: vars(stub.ngram).update(live=active, cached=active)
    stub.ngram.slots = lambda *args, **kwargs: entries["slots"](None, *args, **kwargs)
    prefetch["land_next_cycle"](stub)
    assert log[-1] == "land" and stub.next_pull is None and stub.live_cycle == [1] and stub.ngram.live is pull
    assert prefetch["forward_slots"](stub, 17, batch.ngram_ids, defer_slots=True, hashed_rows=True) is plan
    assert modes[-1] == (True, True)
    assert prefetch["forward_slots"](stub, 17, batch.ngram_ids.clone(), defer_slots=True, hashed_rows=True) is plan
    assert modes[-1] == (True, False)
    assert entries["slots"](None, pull, batch.ngram_ids, defer=True, hashed_rows=True, member_index=1) is plan
    assert modes[-1] == (True, False)
    pull.slot_directory = None
    assert torch.equal(entries["slots"](None, pull, ids), torch.searchsorted(pull.want, ids, out_int32=True))
    print("Installed request publishes the directory with a single transfer. No-directory dispatch uses ATen.", flush=True)


def check_launcher(module):
    def tensor(count):
        return SimpleNamespace(shape=(count,), ndim=1, numel=lambda: count, is_contiguous=lambda: True,
                               is_cuda=True, device=torch.device("cuda", 0), dtype=torch.int32)

    launched = []
    want, directory = tensor(5000), tensor(module.DIRECTORY_LENGTH)

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **opts: launched.append((grid, args, opts))

    with (patch.object(module, "bucket_slots", Kernel()),
          patch.object(torch, "empty_like", lambda source: tensor(source.numel()))):
        for count in module.QUERY_COUNTS:
            rows = tensor(count)
            slots = module.try_bucket_slots(want, directory, rows)
            assert launched[-1] == ((count // module.BLOCK,),
                                    (want, directory, rows, slots, count, module.VOCAB_SIZE, module.BLOCK),
                                    {"num_warps": 4, "num_stages": 1})
        assert module.try_bucket_slots(want, directory, tensor(7)) is None
        rows.dtype = torch.int64
        assert module.try_bucket_slots(want, directory, rows) is None
        assert len(launched) == len(module.QUERY_COUNTS)


def main(baseline, candidate, ir_dir):
    check_checkout(baseline, candidate)
    module = load_candidate_scale(candidate / NGRAM_SLOTS_KERNEL)
    check_dispatch(baseline, candidate, module)
    check_launcher(module)
    compile_and_check(module, ir_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ir-dir", type=Path)
    args = parser.parse_args()
    main(args.baseline, args.candidate, args.ir_dir)
