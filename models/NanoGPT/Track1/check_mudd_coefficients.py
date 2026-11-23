"""Capture the post-MUDD graph and compare the reference and fused Hopper stores."""
import argparse
import ast
import importlib.util
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime import triton_heuristics
from torch._inductor.utils import ValueWithLineMap
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from check_value_adam_inductor import capture
from export_source_link import lm_head_scales
from install_candidate import MUDD_BACKWARD_KERNEL, MUDD_COEFFICIENTS_KERNEL
from mudd_backward_replay import check_backward
from mudd_norm_replay import check_rows
from qkv_ptx_replay import INT32_OPS, PREDICATES, WORD32, address, operands_in, parse, signed32
from verify_candidate import check_checkout


class CompileOnlyCE:
    def set_shared_memory_config(self, extent):
        pass

    def __call__(self, *args):
        raise RuntimeError("offline wrapper capture must not execute CE")


def load_heads(checkout):
    """Import the head wrappers and custom-op schemas without invoking NVRTC."""
    with patch.object(sys, "path", [str(checkout), *sys.path]), patch.object(
            torch.cuda, "_compile_kernel", return_value=CompileOnlyCE()):
        from track_1_short.perf.kernels.cross_entropy import FusedSoftcappedCrossEntropy
        from track_1_short.perf.kernels.sampled_cross_entropy import SampledSoftcappedCrossEntropy
    return FusedSoftcappedCrossEntropy, SampledSoftcappedCrossEntropy


class ExposeWeights(ast.NodeTransformer):
    def __init__(self, scale):
        self.scale = scale

    def visit_Attribute(self, node):
        if not isinstance(node.value, ast.Name) or node.value.id != "self":
            return self.generic_visit(node)
        if node.attr == "_mudd_scale":
            return ast.Constant(self.scale)
        return ast.Name(dict(mudd_w1="w1", mudd_w2="w2", mudd_b2="b2", mudd_w2g="w2g")[node.attr], ast.Load())


def post_mudd_probe(baseline, head):
    model = baseline / "track_1_short/model"
    path = model / "gpt.py"
    module = ast.parse(path.read_text())
    gpt, = (node for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "GPT")
    groups, = (ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
               and [ast.unparse(target) for target in node.targets] == ["MUDD_GROUPS"])
    scale, = (ast.literal_eval(node.value) for node in ast.walk(gpt) if isinstance(node, ast.Assign)
              and [ast.unparse(target) for target in node.targets] == ["self._mudd_scale"])
    forward, = (node for node in gpt.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    coefficient, = (node for node in gpt.body
                    if isinstance(node, ast.FunctionDef) and node.name == "forward_mudd_grouped")
    start, = (i for i, stmt in enumerate(forward.body) if isinstance(stmt, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "grouped" for target in stmt.targets))
    end, = (i for i, stmt in enumerate(forward.body) if isinstance(stmt, ast.Return))
    stmts = ["mu, deltas = forward_mudd_grouped(x, 1, len(sources), w1, w2, b2, w2g)",
             *(ast.unparse(node) for node in forward.body[start:end]),
             "return Head.apply(norm(x).view(-1, 768), *head_args).sum()"]
    probe = ast.parse("def forward(x, sources, w1, w2, b2, w2g, *head_args):\n" +
                      "\n".join("    " + line for stmt in stmts for line in stmt.splitlines())).body
    norm, = (node for node in ast.parse((model / "layers.py").read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == "norm")
    coefficient.args.args = coefficient.args.args[1:] + [ast.arg(name) for name in ("w1", "w2", "b2", "w2g")]
    linked = ast.fix_missing_locations(ast.Module([norm, ExposeWeights(scale).visit(coefficient), *probe], []))
    scope = dict(torch=torch, Tensor=torch.Tensor, F=torch.nn.functional, Head=head, MUDD_GROUPS=groups)
    exec(compile(linked, str(path), "exec"), scope)
    return scope["forward"]


def default_graphs(baseline):
    package = baseline / "track_1_short"
    stages = runpy.run_path(str(package / "config.py"))["TRAINING_STAGES"]
    module = ast.parse((package / "sampled_softmax.py").read_text())
    counts, = (ast.literal_eval(node.value) for node in module.body if isinstance(node, ast.Assign)
               and [ast.unparse(target) for target in node.targets] == ["CANDIDATES_BY_STAGE"])
    return [(stage.batch_size // 8, width, len(stage.mtp_weights_start))
            for index, stage in enumerate(stages) for width in dict.fromkeys(counts.get(index, (0,)))]


def training_head_args(rows, classes, predictions, scales):
    targets = torch.zeros(rows, dtype=torch.int32)
    prefix = torch.zeros_like(targets)
    mtp = torch.ones(predictions, dtype=torch.float32)
    weight = torch.ones(1, dtype=torch.float32)
    lm = torch.empty((768, 50304), dtype=torch.bfloat16, requires_grad=True)
    if classes:
        row = torch.empty((classes, 768), dtype=torch.float8_e4m3fn)
        transposed = torch.empty((768, classes), dtype=torch.float8_e4m3fn)
        vocab_positions = torch.zeros(50304, dtype=torch.int32)
        return (mtp, targets, prefix, weight, lm, row, transposed, vocab_positions, *scales)
    column = torch.empty((50304, 768), dtype=torch.float8_e4m3fn).T
    row = torch.empty((768, 50304), dtype=torch.float8_e4m3fn)
    return (targets, mtp, prefix, weight, lm, column, row, *scales)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_fused(candidate, output):
    names = ("MU", "BIAS", "DELTA", *(f"OUT{i}" for i in range(10)))
    kernel = ASTSource(candidate.mudd_coefficients, dict.fromkeys(names, "*bf16") | {"TILE": "constexpr"},
                       {"TILE": 512}, {(i,): [["tt.divisibility", 16]] for i in range(13)})
    compiled = triton.compile(kernel, target=GPUTarget("cuda", 90, 32),
                              options=dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
    for fmt in ("ttgir", "ptx", "cubin"):
        value = compiled.asm[fmt]
        (output / f"mudd-coefficients.{fmt}").write_bytes(value if isinstance(value, bytes) else value.encode())
    return compiled.asm["ptx"]


def compile_norm(candidate, output, *, cached=False):
    names = ("X", *(name for i in range(10) for name in (f"C{i}", f"S{i}")))
    signature = {"MU": "*bf16", "BIAS": "*bf16", "DELTA": "*bf16", "MIXED": "*fp32", "SUM": "*fp32",
                 **dict.fromkeys(names, "*bf16"), "FP8": "*fp8e4nv", "CACHE_INVERSE": "constexpr"}
    kernel = GluonASTSource(candidate.mudd_final_norm, signature,
                           {"CACHE_INVERSE": cached},
                           attrs={(i,): [["tt.divisibility", 16]] for i in range(27)})
    return compile_gluon(kernel, output / ("mudd-norm-cached" if cached else "mudd-norm"))


def compile_backward(candidate, output, *, encoded, retain=False):
    signature = {"GRAD": "*bf16", "MIXED": "*fp32", "INVERSE": "*fp32",
                 "ENCODED": "*fp8e4nv" if encoded else "constexpr", "WIDTH": "constexpr", "ACTIVATION": "constexpr"}
    const = {"WIDTH": 16 if encoded or retain else 8, "ACTIVATION": encoded}
    if not encoded:
        const["ENCODED"] = None
    kernel = GluonASTSource(candidate.mudd_backward, signature, const,
                           attrs={(i,): [["tt.divisibility", 16]] for i in range(4 if encoded else 3)})
    label = "retained" if retain else ("encode" if encoded else "gradient")
    return compile_gluon(kernel, output / f"mudd-backward-{label}")


def compile_gluon(kernel, prefix):
    compiled = triton.compile(kernel, target=GPUTarget("cuda", 90, 32),
                              options=dict(num_warps=1, num_stages=1, enable_fp_fusion=True))
    for fmt in ("ttgir", "ptx", "cubin"):
        value = compiled.asm[fmt]
        prefix.with_suffix(f".{fmt}").write_bytes(value if isinstance(value, bytes) else value.encode())
    return compiled.asm["ptx"]


def compile_reference(source, output, name, *, norm=False):
    tree = ast.parse(source)
    fn, = (node for node in tree.body if isinstance(node, ast.FunctionDef))
    heuristics, jit = fn.decorator_list
    fn.decorator_list = [jit]
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.Expr) and ast.unparse(node) == "triton_helpers.set_driver_to_gpu()")]
    output.write_text(ast.unparse(tree))
    module = load(output)
    args = {kw.arg: eval(compile(ast.Expression(kw.value), str(output), "eval"), vars(module))
            for kw in heuristics.keywords}
    meta = args["triton_meta"]
    heuristic = triton_heuristics.persistent_reduction if norm else triton_heuristics.pointwise
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kwargs: configs), \
            patch.object(torch.cuda, "is_available", return_value=norm):
        configs = heuristic(**args)
    for cfg in configs:
        kernel = ASTSource(getattr(module, name), meta["signature"], cfg.kwargs, meta["configs"][0])
        compiled = triton.compile(kernel, target=GPUTarget("cuda", 90, 32),
                                  options=dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                                               enable_fp_fusion=True))
        yield cfg.kwargs["XBLOCK"], cfg.num_warps * 32, compiled.asm["ptx"]


STRIDE = 1 << 40
INPUTS = (STRIDE, 2 * STRIDE, 3 * STRIDE)
OUTPUTS = tuple((plane + 4) * STRIDE for plane in range(10))


def stores(ptx, cta, plane, params, threads):
    """Replay one CTA with each BF16 word labeled by its input address."""
    instructions = ptx.split("{\n", 1)[1].split("ret;", 1)[0].splitlines()
    instructions = list(parse(line.strip() for line in instructions
                              if line.strip() and not line.strip().startswith((".", "//", "$"))))
    written = {}

    def registers(operand):
        return operands_in(operand) if operand.startswith("{") else [operand]

    for tid in range(threads):
        regs = {"%ctaid.x": cta, "%ctaid.y": plane, "%tid.x": tid}

        def read(arg):
            if arg.startswith("%"):
                return regs[arg]
            if arg.startswith("0f"):
                return ("f32", int(arg[2:], 16))
            return int(arg, 0)

        for pred, op, (dst, *src) in instructions:
            if pred and regs[pred.lstrip("!")] == pred.startswith("!"):
                continue
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*map(read, src)) & WORD32
            elif op == "mul.hi.s32":
                a, b = (signed32(read(arg)) for arg in src)
                regs[dst] = (a * b >> 32) & WORD32
            elif op == "mul.hi.u32":
                a, b = (read(arg) & WORD32 for arg in src)
                regs[dst] = (a * b) >> 32
            elif op == "shr.s32":
                regs[dst] = (signed32(read(src[0])) >> read(src[1])) & WORD32
            elif op in ("mad.lo.s32", "mad.wide.s32", "mad.wide.u32"):
                a, b, c = map(read, src)
                regs[dst] = a * b + c
            elif op in ("mul.wide.s32", "mul.wide.u32"):
                a, b = map(read, src)
                regs[dst] = a * b
            elif op == "add.s64":
                regs[dst] = sum(map(read, src))
            elif op.startswith("setp."):
                regs[dst] = PREDICATES[op.split(".")[1]](*map(read, src))
            elif op == "selp.b64":
                regs[dst] = read(src[0] if read(src[2]) else src[1])
            elif op == "ld.param.b64":
                regs[dst] = params[int(src[0].rstrip("]").rsplit("_", 1)[1])]
            elif op.startswith("createpolicy."):
                regs[dst] = 0
            elif op == "mov.b32" and dst.startswith("{"):
                regs.update(zip(registers(dst), read(src[0]), strict=True))
            elif op == "mov.b32" and src[0].startswith("{"):
                regs[dst] = tuple(map(read, registers(src[0])))
            elif op in ("mov.u16", "mov.u32", "mov.u64", "mov.b32", "cvt.s64.s32"):
                regs[dst] = read(src[0])
            elif op.startswith("ld.global."):
                start = address(regs, src[0])
                if op.endswith(".b16"):
                    regs[registers(dst)[0]] = ("bf16", start)
                else:
                    assert op.endswith(".b32")
                    for lane, reg in enumerate(registers(dst)):
                        regs[reg] = (("bf16", start + 4 * lane), ("bf16", start + 4 * lane + 2))
            elif op == "cvt.f32.bf16":
                regs[dst] = ("widen", read(src[0]))
            elif op in ("add.f32", "add.rn.f32", "mul.f32", "mul.rn.f32", "fma.rn.f32"):
                regs[dst] = (op.replace(".rn", ""), *map(read, src))
            elif op == "cvt.rn.bf16x2.f32":
                regs[dst] = tuple(("narrow", read(arg)) for arg in reversed(src))
            elif op.startswith("st.global."):
                start = address(regs, dst)
                for lane, word in enumerate(word for reg in registers(src[0]) for word in read(reg)):
                    target = start + 2 * lane
                    assert target not in written
                    written[target] = word
            else:
                raise ValueError(f"Unexpected coefficient instruction {op}")
    return written


def check_backend(candidate, source, rewritten):
    import importlib

    compiler = importlib.import_module("torch._inductor.compile_fx")
    wrapper = ValueWithLineMap(source, [])
    def compile_probe(graph, inputs, config_patches):
        assert config_patches["post_grad_custom_post_pass"].uuid()
        return GraphLowering._compile_to_module_lines(None, wrapper)

    def accept(_lowering, resolved):
        assert resolved.value == rewritten and resolved.line_map == wrapper.line_map
        return resolved

    with patch.object(compiler, "compile_fx", compile_probe), patch.object(
            GraphLowering, "_compile_to_module_lines", accept):
        assert candidate._compile_with_coefficients(None, []).value == rewritten
        assert GraphLowering._compile_to_module_lines is accept


def check_wrapper(candidate, source, rows, rewritten):
    original = ast.parse(source)
    kernels, _ = candidate._candidate_assignments(original)
    guard, start, (_, inputs, outputs, _, _) = next(
        (node, i, region) for node in ast.walk(original) if isinstance(node, ast.With)
        for i in range(len(node.body) - 29) if (region := candidate._identify_region(kernels, node.body, i, 0)))
    match_line = guard.body[start].lineno
    body = next(node.body for node in ast.walk(ast.parse(rewritten))
                if isinstance(node, ast.With) and any(stmt.lineno == match_line for stmt in node.body))
    region = [stmt for stmt in body if match_line <= stmt.lineno <= guard.body[start + 29].end_lineno]
    allocations = []
    launches = []

    def allocate(shape, stride, dtype):
        assert (shape, stride, dtype) == ((1, rows, 12), (rows * 12, 12, 1), torch.bfloat16)
        value = object()
        allocations.append(value)
        return value

    scope = dict(zip(inputs, INPUTS, strict=True)) | dict(
        torch=SimpleNamespace(bfloat16=torch.bfloat16), empty_strided_cuda=allocate,
        get_raw_stream=lambda device: 100 + device,
        launch_mudd_coefficients=lambda *args: launches.append(args),
    )
    # Resolve the replacement's import above, using the loaded candidate in tests.
    exec(compile(ast.Module(region[1:], []), "mudd-wrapper.py", "exec"), scope)
    assert len(allocations) == 10 and len(set(allocations)) == 10
    assert launches == [(rows, *INPUTS, *(scope[outputs[plane]] for plane in range(10)))]
    assert scope["stream0"] == 100
    other_rank = (source.replace("_DeviceGuard(0)", "_DeviceGuard(7)")
                  .replace("set_device(0)", "set_device(7)")
                  .replace("get_raw_stream(0)", "get_raw_stream(7)").replace("stream0", "stream7"))
    rank_wrapper, count = candidate.patch_wrapper(other_rank, fuse_norm=False)
    assert count == 10 and "stream7 = get_raw_stream(7)" in rank_wrapper
    wrong_rank = other_rank.replace("get_raw_stream(7)", "get_raw_stream(0)")
    assert candidate.patch_wrapper(wrong_rank) == (wrong_rank, 0)

    for old, new in [("'out_ptr0': '*bf16'", "'out_ptr0': '*fp16'"),
                     ("'enable_fp_fusion': True", "'enable_fp_fusion': False"),
                     ("tmp0 + tmp2", "tmp0 - tmp2"), ("stream=stream0", "stream=unknown")]:
        mutation = source.replace(old, new)
        assert mutation != source and candidate.patch_wrapper(mutation) == (mutation, 0)


def check_norm_wrapper(candidate, source, rows, rewritten):
    tree = ast.parse(source)
    coefficients, norms = candidate._candidate_assignments(tree)
    guard, start, (allocations, arguments, deletions, launch) = next(
        (node, i, norm) for node in ast.walk(tree) if isinstance(node, ast.With)
        for i in range(len(node.body) - 37)
        if (region := candidate._identify_region(coefficients, node.body, i, 0))
        if (norm := candidate._identify_norm(norms, node.body, i + 30, region)))
    first, last = guard.body[start].lineno, launch.end_lineno
    assert source.splitlines()[last:] == rewritten.splitlines()[last:]
    block = next(node.body for node in ast.walk(ast.parse(rewritten))
                 if isinstance(node, ast.With) and any(stmt.lineno == first for stmt in node.body))
    statements = [stmt for stmt in block if first <= stmt.lineno <= last]
    originals = {name: object() for name in arguments[:3] + (arguments[5],) + arguments[7:-1:2]}
    launches = []
    allocated = []

    def allocate(shape, stride, dtype):
        obj = object()
        allocated.append((shape, stride, dtype, obj))
        return obj

    def launch_norm(tokens, *args):
        assert tokens == rows
        assert args == tuple(scope[name] for name in arguments)
        candidate.launch_mudd_norm(tokens, *args)

    class LaunchProbe:
        def __getitem__(self, grid):
            assert grid == (rows,)
            return self.launch

        def launch(self, *args, **options):
            assert options == dict(num_warps=1, num_stages=1, enable_fp_fusion=True)
            launches.append(args)

    scope = originals | dict(torch=torch, empty_strided_cuda=allocate,
                             reinterpret_tensor=lambda buf, *view: buf,
                             get_raw_stream=lambda device: 100 + device, launch_mudd_norm=launch_norm)
    with patch.object(candidate, "mudd_final_norm", LaunchProbe()):
        exec(compile(ast.Module(statements[1:], []), "mudd-norm-wrapper.py", "exec"), scope)
    assert len(launches) == 1 and len(allocated) == 10 + len(allocations)
    assert all(allocated[i][:3] == ((1, rows, 12), (rows*12, 12, 1), torch.bfloat16) for i in range(10))
    expected_transpose = ((768, rows), (rows, 1), torch.float8_e4m3fn)
    assert all(a[:3] == expected_transpose for a in allocated[10:-3])
    assert allocated[-3][:3] == ((1, rows, 768), (rows * 768, 768, 1), torch.float32)
    assert allocated[-2][:3] == ((1, rows, 1), (rows, 1, rows), torch.float32)
    assert allocated[-1][:3] == ((rows, 768), (768, 1), torch.float8_e4m3fn)
    assert launches[0][-1] is False
    assert set(launches[0][6:-2:2]) == {item[-1] for item in allocated[:10]}
    assert all(ast.unparse(d.targets[0]) not in scope for d in deletions)
    rank = (source.replace("_DeviceGuard(0)", "_DeviceGuard(7)")
            .replace("set_device(0)", "set_device(7)").replace("get_raw_stream(0)", "get_raw_stream(7)")
            .replace("stream0", "stream7"))
    assert candidate.patch_wrapper(rank, cache_backward=False)[1] == 11
    for old, new in [("tmp45 = tl.where(r0_mask", "tmp45 = tl.where(xmask"),
                     ("'max_autotune': False", "'max_autotune': True")]:
        invalid = source.replace(old, new)
        assert invalid != source and candidate.patch_wrapper(invalid)[1] == 10
    altered = ast.unparse(launch)
    wrong_plane = altered.replace(arguments[6], arguments[8], 1)
    launch_line = source.splitlines(keepends=True)
    launch_line[launch.lineno - 1] = " " * launch.col_offset + wrong_plane + "\n"
    assert candidate.patch_wrapper("".join(launch_line))[1] == 10


def check_cached_wrapper(candidate, source, ordinary, rewritten, name, rows, encoded):
    """Check that the launch pair shares precisely one private statistic allocation."""
    launch, = (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == f"{name}.run")
    gradient, mixed, statistic, *fp8 = [arg.id for arg in launch.args[:-2]]
    assert len(fp8) == encoded
    old, new = ordinary.splitlines(), rewritten.splitlines()
    assert len(old) == len(new)
    assert new[launch.lineno - 1].strip() == (
        f"launch_mudd_backward_cached({rows}, {', '.join((gradient, mixed, statistic, *fp8))})")
    assert sum("launch_mudd_norm_cached(" in line for line in new) == 1
    assert [line.strip() for line in old if "empty_strided_cuda(" in line] == [
        line.strip() for line in new if "empty_strided_cuda(" in line]
    old_stream = [line.strip() for line in old[:launch.lineno - 1] if line.strip()]
    new_stream = [line.strip().replace("launch_mudd_norm_cached", "launch_mudd_norm")
                  for line in new[:launch.lineno - 1] if line.strip()
                  and "import launch_mudd_backward_cached" not in line]
    assert old_stream == new_stream
    assert old[launch.end_lineno:] == new[launch.end_lineno:]
    changes = [
        source.replace("tmp8 / tmp9", "tmp8 * tmp9"),
        source.replace(ast.unparse(launch), ast.unparse(launch).replace("stream=stream0", "stream=stream7")),
    ]
    lines = source.splitlines(keepends=True)
    original_launch = lines[launch.lineno - 1]
    lines[launch.lineno - 1] = " " * launch.col_offset + f"observe({statistic})\n" + original_launch
    changes.append("".join(lines))
    for mutant in changes:
        assert mutant != source and candidate.patch_wrapper(mutant)[1] == 11
    rank = (source.replace("_DeviceGuard(0)", "_DeviceGuard(7)")
            .replace("set_device(0)", "set_device(7)").replace("get_raw_stream(0)", "get_raw_stream(7)")
            .replace("stream0", "stream7"))
    assert candidate.patch_wrapper(rank)[1] == 12


def check_retained_wrapper(candidate, source, cached, retained, rows):
    refs = ast.parse(cached)
    forward, = (n for n in ast.walk(refs) if candidate._call(n, "launch_mudd_norm_cached", 28))
    backward, = (n for n in ast.walk(refs) if candidate._call(n, "launch_mudd_backward_cached", 5))
    assert ast.literal_eval(forward.args[0]) == ast.literal_eval(backward.args[0]) == rows
    saved, remat = forward.args[-1].id, backward.args[-1].id
    old, new = [text.splitlines() for text in (cached, retained)]
    assert len(old) == len(new)
    expected = {
        "from track_1_short.perf.kernels.mudd_backward import launch_mudd_backward_cached":
            "from track_1_short.perf.kernels.mudd_backward import launch_mudd_backward_retained",
        f"del {saved}": "",
        f"{remat} = empty_strided_cuda(({rows}, 768), (768, 1), torch.float8_e4m3fn)": f"{remat} = {saved}",
        ast.unparse(backward): f"launch_mudd_backward_retained({', '.join(ast.unparse(x) for x in backward.args[:-1])})",
        f"del {remat}": f"del {remat}, {saved}",
    }
    assert {a.strip(): b.strip() for a, b in zip(old, new, strict=True) if a != b} == expected
    consumer, = (n for n in ast.walk(ast.parse(source))
                 if candidate._call(n, "torch.ops.nanogpt.track1_ce_transpose_operands.default", 4))
    altered = source.replace(ast.unparse(consumer), ast.unparse(consumer).replace("default", "other"))
    assert altered != source and candidate.patch_wrapper(altered) == candidate.patch_wrapper(altered, retain_activation=False)


def check(baseline, checkout, output, *, baseline_heads=False):
    check_checkout(baseline, checkout)
    candidate = load(checkout / MUDD_COEFFICIENTS_KERNEL)
    backward = load(checkout / MUDD_BACKWARD_KERNEL)
    if (torch.version.git_version != candidate.TORCH_REVISION or torch.version.cuda is not None
            or triton.__version__ != "3.6.0"):
        raise RuntimeError("Use the pinned CPU Torch wheel and Triton 3.6.0 for offline capture")
    output.mkdir(parents=True, exist_ok=True)
    fused = compile_fused(candidate, output)
    fused_norm = compile_norm(candidate, output)
    cached_norm = compile_norm(candidate, output, cached=True)
    cached_backward = {encoded: compile_backward(backward, output, encoded=encoded) for encoded in (False, True)}
    retained_backward = compile_backward(backward, output, encoded=False, retain=True)
    full, sampled = load_heads(baseline if baseline_heads else checkout)
    sampled_head = post_mudd_probe(baseline, sampled)
    full_head = post_mudd_probe(baseline, full)
    _, source_scales = lm_head_scales((baseline / "track_1_short/model/gpt.py").read_text())
    scales = tuple(source_scales[name] for name in ("x_s", "w_s", "grad_s"))
    plans = default_graphs(baseline)
    assert {rows for rows, _, _ in plans} == set(candidate.ROWS)
    for rows, classes, predictions in plans:
        inputs = [torch.empty((1, rows, 768), dtype=torch.bfloat16, requires_grad=True) for _ in range(11)]
        weights = [torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
                   for shape in ((2, 64, 768), (2, 14, 64), (2, 14), (14, 12, 64))]
        label = f"{rows}-{classes or 'full'}-{predictions}"
        path = output / f"mudd-wrapper-{label}.py"
        probe = sampled_head if classes else full_head
        capture(probe, path,
                [inputs[0], tuple(inputs[1:]), *weights, *training_head_args(rows, classes, predictions, scales)],
                user_triton=True)
        source = path.read_text()
        ordinary, ordinary_count = candidate.patch_wrapper(source, cache_backward=False)
        assert ordinary_count == 11 and len(ordinary.splitlines()) == len(source.splitlines())
        check_norm_wrapper(candidate, source, rows, ordinary)
        cached, count = candidate.patch_wrapper(source, retain_activation=False)
        assert count == 12
        rewritten, count = candidate.patch_wrapper(source)
        assert count == 12
        check_backend(candidate, source, rewritten)
        coefficients_only, count = candidate.patch_wrapper(source, fuse_norm=False)
        assert count == 10 and len(coefficients_only.splitlines()) == len(source.splitlines())
        check_wrapper(candidate, source, rows, coefficients_only)

        module = ast.parse(source)
        recognized, norms = candidate._candidate_assignments(module)
        defs = {node.targets[0].id: node for node in module.body if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
        (norm_name, size), = norms.items()
        assert rows == size
        norm_reference = ast.literal_eval(defs[norm_name].value.args[1])
        (block, threads, ptx), = compile_reference(
            norm_reference, output / f"mudd-reference-norm-{label}.py", norm_name, norm=True)
        assert (block, threads) == (1, 32)
        (output / f"mudd-reference-norm-{label}.ptx").write_text(ptx)
        check_rows(rows, ptx, fused_norm, cached_norm)
        (backward_name, (backward_rows, encoded)), = candidate._backward_assignments(module).items()
        assert backward_rows == rows
        check_cached_wrapper(candidate, source, ordinary, cached, backward_name, rows, encoded)
        if not baseline_heads and encoded:
            check_retained_wrapper(candidate, source, cached, rewritten, rows)
        else:
            assert rewritten == cached
        backward_reference = ast.literal_eval(defs[backward_name].value.args[1])
        (block, threads, backward_ptx), = compile_reference(
            backward_reference, output / f"mudd-reference-backward-{label}.py", backward_name, norm=True)
        assert (block, threads) == (1, 32)
        (output / f"mudd-reference-backward-{label}.ptx").write_text(backward_ptx)
        check_backward(rows, encoded, backward_ptx, cached_backward[encoded])
        if not baseline_heads and encoded:
            check_backward(rows, encoded, backward_ptx, retained_backward, retain=True)
        assert sorted(plane for count, plane in recognized.values() if count == rows) == list(range(10))
        for name, (count, plane) in recognized.items():
            old = ast.literal_eval(defs[name].value.args[1])
            compared = set()
            for block, threads, ptx in compile_reference(old, output / f"mudd-reference-{label}-{plane}.py", name):
                assert block in (512, 1024) and count * 12 % block == 0
                for cta in (0, 19, count * 12 // block - 1):
                    before = stores(ptx, cta, 0, (*INPUTS, OUTPUTS[plane]), threads)
                    after = {addr: value for tile in range(block // 512)
                             for addr, value in stores(fused, cta * (block // 512) + tile, plane,
                                                       (*INPUTS, *OUTPUTS), 128).items()}
                    assert len(before) == block and before == after, (count, plane, block, cta)
                compared.add(block)
            assert compared == {512, 1024}
        print(f"{label}: all forward and backward stores checked, including the paired inverse RMS buffer.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-heads", action="store_true",
                        help="capture with the unmodified head wrappers")
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.output, baseline_heads=args.baseline_heads)
