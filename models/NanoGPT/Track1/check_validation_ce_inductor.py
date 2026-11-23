"""Extract the pinned evaluation loss and check its offline Hopper lowering."""
import argparse
import ast
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import ReductionHint
import triton
from triton.compiler import ASTSource

from check_value_adam_inductor import HOPPER, TORCH_REVISION, capture
from install_candidate import BASELINE, MODEL, git
from validation_ce_replay import ROWS, SCHEDULES, SLABS, WIDTH, check_ptx

CHANNELS = 768


def evaluation_loss(root):
    if git(root, "rev-parse", "HEAD").strip() != BASELINE:
        raise ValueError(f"Expected the pinned Track 1 commit at {root}")
    tree = ast.parse(git(root, "show", f"HEAD:{MODEL}"))
    extent, = (n for n in tree.body if isinstance(n, ast.Assign)
               and [ast.unparse(t) for t in n.targets] == ["EVAL_CE_SLAB_ROWS"])
    assert ast.literal_eval(extent.value) == ROWS
    loss, = (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_loss")
    train, = (n for n in loss.body if isinstance(n, ast.If))
    dense, = train.orelse
    _, _, init, loop, joined = dense.orelse
    assert ast.unparse(loop.iter) == "range(0, x.size(0), EVAL_CE_SLAB_ROWS)"
    assert ast.unparse(joined) == "loss_per_token = torch.cat(slab_losses)"
    assert ast.unparse(init) == "slab_losses = []"
    hi, *_, save = loop.body
    assert ast.unparse(hi) == "hi = min(lo + EVAL_CE_SLAB_ROWS, x.size(0))"
    assert ast.unparse(save.value.func) == "slab_losses.append"
    layer_path = MODEL.with_name("layers.py")
    layers = ast.parse(git(root, "show", f"HEAD:{layer_path}"))
    head, = (node for node in layers.body if isinstance(node, ast.ClassDef) and node.name == "CastedLinearT")
    forward, = (node for node in head.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    projection, = forward.body

    class Substitute(ast.NodeTransformer):
        def __init__(self, replacements):
            self.replacements = replacements

        def visit(self, node):
            binding = self.replacements.get(ast.unparse(node)) if isinstance(node, ast.expr) else None
            return ast.parse(binding, mode="eval").body if binding else super().visit(node)

    projection = Substitute({"x": "x[lo:hi]", "self.weight": "weight"}).visit(projection.value)
    wrapper = ast.parse("def loss(x, weight, canon, input_seq, target_seq): pass")
    wrapper.body[0].body = [*dense.orelse, ast.Return(ast.Name("loss_per_token", ast.Load()))]
    wrapper = ast.fix_missing_locations(Substitute({
        "self.lm_head(x[lo:hi])": ast.unparse(projection),
        "self.canon_mask": "canon",
    }).visit(wrapper))
    env = {"torch": torch, "F": torch.nn.functional, "EVAL_CE_SLAB_ROWS": ROWS}
    exec(compile(wrapper, str(root / MODEL), "exec"), env)
    return env["loss"]


def check_wrapper(wrapper, name):
    """Match all eight mm slices and the output aliases of the fused reducer."""
    runner, = (n for n in wrapper.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    call, = (n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    projection_offsets, output_offsets = {}, {}
    for node in ast.walk(call):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "extern_kernels.mm":
            matrix, weight = node.args
            assert ast.unparse(weight) == "arg1_1"
            assert ast.unparse(matrix.func) == "reinterpret_tensor"
            source, shape, stride, offset = matrix.args
            assert source.id == "arg0_1"
            assert ast.literal_eval(shape) == (ROWS, CHANNELS)
            assert ast.literal_eval(stride) == (CHANNELS, 1)
            output, = (kw.value.id for kw in node.keywords if kw.arg == "out")
            projection_offsets[output] = ast.literal_eval(offset)
        if isinstance(node, ast.Assign) and isinstance(value := node.value, ast.Call):
            if ast.unparse(value.func) == "reinterpret_tensor" and ast.literal_eval(value.args[1]) == (ROWS,):
                name_out, = (target.id for target in node.targets)
                assert ast.literal_eval(value.args[2]) == (1,)
                output_offsets[name_out] = (value.args[0].id, ast.literal_eval(value.args[3]))

    launch, = (n for n in ast.walk(call) if isinstance(n, ast.Call) and ast.unparse(n.func) == name + ".run")
    inputs, mask, *operands = launch.args
    assert (inputs.id, mask.id, operands[SLABS].id) == ("arg2_1", "arg3_1", "arg4_1")
    assert tuple(map(ast.literal_eval, operands[-2:])) == (ROWS, WIDTH)
    projections, outputs = operands[:SLABS], operands[SLABS + 1:-2]
    assert len(projection_offsets) == len(output_offsets) == len(outputs) == SLABS
    result, = (n.value.elts[0].id for n in ast.walk(call) if isinstance(n, ast.Return))
    for slab, (projection, output) in enumerate(zip(projections, outputs, strict=True)):
        assert projection_offsets[projection.id] == slab * ROWS * CHANNELS
        assert output_offsets[output.id] == (result, slab * ROWS)


def check_scan(ir, tile):
    """Match the vocabulary loop's bounds, column range, and masked yields."""
    extent = tile[1]
    lines = dict(re.findall(r"^\s*(%\w+) = ([^\n]+)", ir, re.MULTILINE))
    definitions = {name: rhs.split(" : ", 1)[0] for name, rhs in lines.items()}
    loop = re.search(
        r"scf\.for (%\w+) = (%\w+) to (%\w+) step (%\w+) "
        r"iter_args\(([^)]+)\)", ir
    )
    index, start, stop, stride, loop_args = loop.groups()
    assert definitions[start].strip() == "arith.constant 0"
    assert definitions[stop].strip() == f"arith.constant {WIDTH}"
    assert definitions[stride].strip() == f"arith.constant {extent}"
    outputs = re.search(r"scf\.yield ([^:]+) :", ir)[1].split(", ")
    args = [binding.split(" = ") for binding in loop_args.split(", ")]
    assert len(args) == len(outputs) == 2 * SLABS
    guards = set()
    for i, (output, (arg, start)) in enumerate(zip(outputs, args, strict=True)):
        initial = "0xFF800000" if i % 2 == 0 else "0.000000e+00"
        assert definitions[start].strip() == f"arith.constant dense<{initial}>"
        select, guard, _, previous = definitions[output].replace(",", "").split()
        assert select == "arith.select" and previous == arg
        guards.add(guard)

    max_guard, = guards
    guard = definitions[max_guard].split()
    while guard[0] == "tt.broadcast":
        guard = definitions[guard[1]].split()
    compare, predicate, column, width = [part.removesuffix(",") for part in guard]
    assert (compare, predicate) == ("arith.cmpi", "slt")
    assert definitions[width].strip() == f"arith.constant dense<{WIDTH}>"
    op, base, slot = definitions[column].replace(",", "").split()
    assert op == "arith.addi"
    assert definitions[base].strip() == f"tt.splat {index}"
    expand, vector, *_ = definitions[slot].split()
    assert expand == "tt.expand_dims"
    assert lines[vector].startswith(f"tt.make_range {{end = {extent} : i32, start = 0 : i32}}")


def compile_loss(artifact, out):
    wrapper = ast.parse(artifact.read_text())
    call, = (n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "async_compile.triton")
    name, source = map(ast.literal_eval, call.args[:2])
    check_wrapper(wrapper, name)

    fn, = (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    heuristic, jit = fn.decorator_list
    assert ast.unparse(heuristic.func) == "triton_heuristics.reduction"
    assert ast.unparse(jit) == "triton.jit"
    kwargs = {k.arg: k.value for k in heuristic.keywords}
    assert ast.unparse(kwargs.pop("reduction_hint")) == "ReductionHint.DEFAULT"
    table = kwargs.pop("triton_meta")
    metadata = {ast.literal_eval(k): ast.literal_eval(v)
                for k, v in zip(table.keys, table.values, strict=True)
                if ast.literal_eval(k) != "device"}
    assert tuple(metadata["signature"].values()) == (
        "*i32", "*u8", *("*bf16",) * SLABS, "*i64", *("*fp32",) * SLABS,
        "i32", "i32", "constexpr", "constexpr"
    )
    props = SimpleNamespace(type="cuda", index=0, multi_processor_count=132, cc=90, major=9,
                            regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048,
                            max_threads_per_block=1024, warp_size=32)
    metadata["device"] = props
    kwargs = {k: ast.literal_eval(v) for k, v in kwargs.items() if k != "filename"}
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kw: configs):
        configs = triton_heuristics.reduction(triton_meta=metadata, reduction_hint=ReductionHint.DEFAULT, **kwargs)
    assert kwargs["inductor_meta"]["dynamic_scale_rblock"]
    assert not kwargs["inductor_meta"].get("coordinate_descent_tuning")

    fn.decorator_list = [jit]
    kernel_file = out / "kernel.py"
    kernel_file.write_text(
        "import triton\nimport triton.language as tl\n"
        "from torch._inductor.runtime import triton_helpers\n"
        "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n"
        + ast.unparse(fn) + "\n"
    )
    spec = spec_from_file_location(kernel_file.stem, kernel_file)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cuobjdump = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"

    def compile_tile(config):
        tile = (config.kwargs["XBLOCK"], config.kwargs["R0_BLOCK"], config.num_warps)
        path = out / "-".join(map(str, tile))
        src = ASTSource(getattr(module, name), metadata["signature"], config.kwargs, metadata["configs"][0])
        code = triton.compile(src, target=HOPPER, options={
            "num_warps": config.num_warps, "num_stages": config.num_stages,
            "debug": kwargs["inductor_meta"]["assert_indirect_indexing"], "sanitize_overflow": False,
            "enable_fp_fusion": metadata["enable_fp_fusion"],
        })
        cubin = path.with_suffix(".cubin")
        cubin.write_bytes(code.asm["cubin"])
        for fmt in ("ttgir", "ptx"):
            path.with_suffix(f".{fmt}").write_text(code.asm[fmt])
        resources = subprocess.check_output([cuobjdump, "-res-usage", cubin], text=True)
        regs = int(re.search(r"\bREG:(\d+)\b", resources)[1])
        check_scan(code.asm["ttgir"], tile)
        check_ptx(code.asm["ptx"], tile)
        print(f"Evaluation loss tile {tile}, {regs} registers: checked", flush=True)
        return tile, regs

    checked = set()
    for config in configs:
        tile, regs = compile_tile(config)
        checked.add(tile)
        x, r, warps = tile
        occupancy = max(props.regs_per_multiprocessor // (regs * props.warp_size * warps), 1)
        if r > 64 and regs > 32 and ROWS // x > occupancy * props.multi_processor_count:
            # Inductor's newly compiled variants are bound after the scale-down loop.
            smaller = deepcopy(config)
            smaller.kwargs["R0_BLOCK"] //= 2
            scaled, _ = compile_tile(smaller)
            checked.add(scaled)
    assert checked == SCHEDULES.keys()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None:
        raise RuntimeError("Use the pinned CPU build of Torch 2.10.0")
    if triton.__version__ != "3.6.0":
        raise RuntimeError("Use Triton 3.6.0")
    args.out.mkdir(parents=True, exist_ok=True)
    wrapper = args.out / "validation-ce.py"
    capture(evaluation_loss(args.baseline), wrapper, (
        torch.empty((SLABS * ROWS, CHANNELS), dtype=torch.bfloat16),
        torch.empty((CHANNELS, WIDTH), dtype=torch.bfloat16),
        torch.empty((WIDTH, WIDTH // 8), dtype=torch.uint8),
        torch.empty((SLABS * ROWS,), dtype=torch.int32),
        torch.empty((SLABS * ROWS,), dtype=torch.int64),
    ))
    compile_loss(wrapper, args.out)


if __name__ == "__main__":
    main()
