"""Check the installed launcher and replay every staging store in SM90 PTX."""

import argparse
import ast
from dataclasses import dataclass, replace
import importlib
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton

from install_candidate import STEP_GRAPHS
from qkv_ptx_replay import INT32_OPS, PREDICATES, WORD32, address, operands_in, parse, signed32, value
from verify_candidate import check_checkout


LAYOUTS = ((16384, 64, 3), (32768, 96, 2), (49152, 128, 1), (40960, 128, 1), (16384, 64, 1))
WIDTHS = (4, 8, 4, 4, 4, 4)
BASES = tuple((i + 1) << 32 for i in range(12))
TYPES = ("*i32", "*i64", "*i32", "*i32", "*fp32", "*fp32") * 2 + ("constexpr",) * 3


def field_sizes(tokens, ends, mtp):
    return (tokens, tokens, ends, 2 * tokens, mtp, 1)


def load_graph_stage(candidate):
    with patch.object(sys, "path", [str(candidate), *sys.path]):
        return importlib.import_module("track_1_short.perf.kernels.graph_stage")


def load_step_operations(path, names, **env):
    tree = ast.parse(path.read_text(), filename=str(path))
    graph, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StepGraphs")
    code = [node for node in (*tree.body, *graph.body) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(code) == len(names)
    exec(compile(ast.Module(body=code, type_ignores=[]), str(path), "exec"), env)
    return tuple(env[name] for name in names)


def instructions(ptx, name="stage_step"):
    entry = ptx.split(f".entry {name}(", 1)[1]
    body = entry.split("\n{\n", 1)[1].split("\n}", 1)[0]
    threads, = map(int, re.findall(r"\.reqntid (\d+)", entry))
    assert ".target sm_90" in ptx and threads == 128
    code, labels = [], {}
    for line in body.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith("."):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(code)
        else:
            code.extend(parse([line]))
    return code, labels, threads


def transfer_width(op, registers):
    *vector, word = op.split(".")[2:]
    assert vector == ([] if len(registers) == 1 else [f"v{len(registers)}"])
    assert word in ("b32", "b64")
    return int(word[1:]) // 8


def register_names(operand):
    return operands_in(operand) if operand.startswith("{") else [operand]


def replay_thread(code, labels, cta, thread, source, stores):
    regs = {"%ctaid.x": cta, "%tid.x": thread}
    pc = 0
    while pc < len(code):
        pred, op, operands = code[pc]
        pc += 1
        if pred and regs[pred.removeprefix("!")] == pred.startswith("!"):
            continue
        if op == "ret":
            return
        dst, *args = operands
        if op == "bra":
            target = labels[dst]
            assert target >= pc, "This staging kernel has no loops"
            pc = target
        elif op == "ld.param.b64":
            parameter, = args
            index = int(parameter.removesuffix("]").rsplit("_", 1)[1])
            regs[dst] = BASES[index]
        elif op.startswith("ld.global."):
            names = register_names(dst)
            width = transfer_width(op, names)
            start = address(regs, args[0])
            for index, name in enumerate(names):
                location = start + index * width
                assert source[location] == width
                regs[name] = (location, width)
        elif op.startswith("st.global."):
            names = register_names(args[0])
            width = transfer_width(op, names)
            start = address(regs, dst)
            for index, name in enumerate(names):
                location = start + index * width
                word = regs[name]
                assert location not in stores and word[1] == width
                stores[location] = word
        elif op in ("mov.b32", "mov.u32", "mov.u64", "mov.pred"):
            regs[dst] = {"0": False, "1": True, "-1": True}[args[0]] if op == "mov.pred" else value(regs, args[0])
        elif op in INT32_OPS:
            regs[dst] = INT32_OPS[op](*(value(regs, arg) for arg in args)) & WORD32
        elif op == "add.s64":
            regs[dst] = value(regs, args[0]) + value(regs, args[1])
        elif op in ("mul.wide.s32", "mul.wide.u32"):
            words = (value(regs, arg) for arg in args)
            left, right = (signed32(word) if op.endswith(".s32") else word & WORD32 for word in words)
            regs[dst] = left * right
        elif op == "bfe.u32":
            bits, start, length = (value(regs, arg) for arg in args)
            regs[dst] = (bits >> start) & ((1 << length) - 1)
        elif op in ("setp.eq.b32", "setp.ne.b32", "setp.lt.u32"):
            compare = PREDICATES[op.split(".")[1]]
            regs[dst] = compare(*(value(regs, arg) & WORD32 for arg in args))
        elif op == "and.pred":
            regs[dst] = regs[args[0]] and regs[args[1]]
        else:
            raise ValueError(f"Unexpected staging PTX instruction {op}")
    raise ValueError("Missing return in staging PTX")


def check_ptx(ptx, shape, grid):
    code, labels, threads = instructions(ptx)
    source, expected, stores = {}, {}, {}
    for index, (count, width) in enumerate(zip(field_sizes(*shape), WIDTHS, strict=True)):
        for offset in range(0, count * width, width):
            addr = BASES[index] + offset
            source[addr] = width
            expected[BASES[index + 6] + offset] = (addr, width)
    for cta in range(grid[0]):
        for thread in range(threads):
            replay_thread(code, labels, cta, thread, source, stores)
    assert stores == expected, f"Store coverage or source mismatch for {shape}"
    return len(stores)


@dataclass(frozen=True)
class Tensor:
    count: int
    dtype: torch.dtype
    width: int
    pointer: int
    operations: list
    device: torch.device = torch.device("cuda", 0)
    contiguous: bool = True
    negative: bool = False
    requires_grad: bool = False
    inference: bool = False

    @property
    def shape(self):
        return (self.count,)

    def numel(self):
        return self.count

    def element_size(self):
        return self.width

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_inference(self):
        return self.inference

    def copy_(self, source):
        self.operations.append(("copy", self.pointer, source.pointer))


def check_launcher(candidate, module, shape):
    operations, launches = [], []
    tensors = tuple(Tensor(size, dtype, width, address, operations)
                    for size, dtype, width, address in zip(field_sizes(*shape) * 2, module.DTYPES * 2,
                                                           WIDTHS * 2, BASES, strict=True))

    class Launch:
        def __getitem__(self, grid):
            return lambda *args, **options: (launches.append((grid, args, options)), operations.append("stage"))

    forward, _ = load_step_operations(
        candidate / STEP_GRAPHS, ("forward", "step_inputs"), Batch=object, Tensor=object,
        ForwardScheduleConfig=object, live_key=lambda *_: 0, try_stage_step_inputs=module.try_stage_step_inputs,
        SlotQuery=module.SlotQuery, try_stage_step_query=module.try_stage_step_query,
    )
    cfg = SimpleNamespace(mtp_weights=tensors[4], prefix_weight=tensors[5], sampled_loss=None)
    batch = SimpleNamespace(inputs=tensors[0], targets=tensors[1], cum_seqlens=tensors[2])

    def run(source_inputs):
        batch.inputs = source_inputs
        capture = SimpleNamespace(attn_scales=(), cfg=cfg, static=SimpleNamespace(buffers=lambda: tensors[6:]),
                                  forward=SimpleNamespace(replay=lambda: operations.append("replay")))
        graph = SimpleNamespace(step_keys=[0], captured={0: capture}, _attn_scales=lambda: (), unchecked=set())
        assert forward(graph, 0, batch, tensors[3], cfg) is tensors[3]
        assert graph.current is capture

    kernel = module.stage_step
    module.stage_step = Launch()
    try:
        with (patch.object(torch.cuda, "current_device", return_value=0),
              patch.object(torch.autograd.graph, "increment_version", side_effect=operations.append)):
            run(tensors[0])
            (grid, args, options), = launches
            assert operations == ["stage", tensors[6:], "replay"]
            assert args == (*tensors, shape[1], shape[2], 256)
            assert grid == (shape[0] // 256,) and options == dict(num_warps=4, num_stages=1)
            for invalid in (replace(tensors[0], pointer=tensors[6].pointer),
                            replace(tensors[0], pointer=tensors[0].pointer + 4),
                            replace(tensors[0], device=torch.device("cpu")),
                            replace(tensors[0], device=torch.device("cuda", 1)),
                            replace(tensors[0], dtype=torch.float32),
                            replace(tensors[0], negative=True),
                            replace(tensors[0], requires_grad=True),
                            replace(tensors[0], inference=True),
                            replace(tensors[0], contiguous=False), replace(tensors[0], count=7)):
                operations.clear()
                run(invalid)
                expected = [("copy", dst.pointer, src.pointer)
                            for dst, src in zip(tensors[6:], (invalid, *tensors[1:6]), strict=True)]
                assert operations == [*expected, "replay"] and len(launches) == 1
    finally:
        module.stage_step = kernel
    return grid, options


def check_backward(candidate):
    backward, = load_step_operations(
        candidate / STEP_GRAPHS, ("backward",), torch=torch, Tensor=torch.Tensor,
    )
    for dtype in (torch.float32, torch.bfloat16):
        incoming = torch.tensor([0.25, 3], dtype=dtype)
        persistent = torch.empty_like(incoming)
        previous = torch.tensor([2, -1], dtype=dtype)
        anvil = SimpleNamespace(grad=torch.zeros_like(incoming))
        first = SimpleNamespace(grad=None)
        later = SimpleNamespace(grad=previous)
        replays = []
        sink = torch.tensor([0x8000, 0x3f80, 0x4000], dtype=torch.uint16).view(torch.bfloat16)
        current = SimpleNamespace(
            backward=SimpleNamespace(replay=lambda: replays.append(1)), anvil_grads=[(anvil, incoming)],
            adam_grads=[(first, incoming, persistent), (later, incoming, previous)], ngram_grad=sink,
        )
        state = SimpleNamespace(current=current)
        assert backward(state) is sink and replays == [1]
        assert anvil.grad is incoming and first.grad is persistent and later.grad is previous
        assert torch.equal(persistent, incoming)
        assert torch.equal(previous, torch.tensor([2.25, 2], dtype=dtype))
        assert backward(state) is sink and replays == [1, 1]
        assert torch.equal(persistent, torch.tensor([0.5, 6], dtype=dtype))
        assert torch.equal(previous, torch.tensor([2.5, 5], dtype=dtype))
    print("Backward wrapper: ANVIL links, Adam FP32/BF16 first copy and accumulation, n-gram sink identity", flush=True)


def check(baseline, candidate, ir_dir):
    check_checkout(baseline, candidate)
    module = load_graph_stage(candidate)
    assert module.LAYOUTS == set(LAYOUTS)
    signature = dict(zip(module.stage_step.arg_names, TYPES, strict=True))
    align = {(index,): [["tt.divisibility", 16]] for index in range(12)}
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)
    for shape in LAYOUTS:
        tokens, ends, mtp = shape
        grid, options = check_launcher(candidate, module, shape)
        source = ASTSource(module.stage_step, signature, dict(END_ROWS=ends, MTP_ROWS=mtp, BLOCK=256), attrs=align)
        code = triton.compile(source, target=GPUTarget("cuda", 90, 32), options=options)
        stores = check_ptx(code.asm["ptx"], shape, grid)
        if ir_dir is not None:
            for kind in ("ttir", "ttgir", "ptx"):
                (ir_dir / f"stage-{tokens}-{ends}-{mtp}.{kind}").write_text(code.asm[kind])
        print(f"{shape}: {stores} unique bit-preserving stores, staging precedes replay", flush=True)
    check_backward(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ir-dir", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.ir_dir)
