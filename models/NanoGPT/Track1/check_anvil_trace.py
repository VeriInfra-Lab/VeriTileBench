"""Verify the fused ANVIL trace against the real Hopper Inductor reductions."""
import argparse
import ast
from pathlib import Path
from struct import pack, unpack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from torch._inductor import config
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import ReductionHint
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from anvil_cascade_replay import SPLIT_TRACE, canonical_kernel, cname, first_gram, match_cascades, parse_wrapper
from tail_norm_replay import instructions
from anvil_trace_replay import check_trees
from check_anvil_epilogue import artifact as compile_pointwise, run as replay_pointwise
from check_anvil_gram import CASES, build_full, check_full_guards, check_full_pair, load, load_cascade, load_polar, lowered_schedule
from check_replicated_adam import term
from check_value_adam_inductor import HOPPER, TORCH_REVISION, capture
from export_source_link import anvil_frobenius
from install_candidate import ANVIL, POLAR
from verify_candidate import check_checkout


def compile_polar(part, path):
    name, source = part
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    fn, helper = functions
    assert fn.name == name and helper.name == "_pid_to_block"
    heuristic, jit = fn.decorator_list
    assert ast.unparse(heuristic.func) == "triton_heuristics.user_autotune"
    kw = {node.arg: node.value for node in heuristic.keywords}
    metadata = kw["triton_meta"]
    meta = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(metadata.keys, metadata.values, strict=True)
            if ast.literal_eval(k) != "device"}
    fn.decorator_list = [jit]
    path.write_text("import triton\nimport triton.language as tl\n" +
                    ast.unparse(ast.Module(body=functions, type_ignores=[])) + "\n")
    kernel = getattr(load(path, path.stem), name)
    return triton.compile(ASTSource(kernel, meta["signature"], meta["constants"], attrs=meta["configs"][0]),
                          target=HOPPER, options={**ast.literal_eval(kw["configs"])[0], "sanitize_overflow": False})


def check_polar_wrappers(split, full, gram, shape, coefficients, polar):
    labels = (gram, "ba_plus_cAA_kernel")
    lowered_schedule(split, gram, shape, coefficients, True)
    lowered_schedule(full, gram, shape, coefficients, True, full=True)
    kernels, _ = parse_wrapper(split)
    canonical = {}
    for name, source in kernels.values():
        if name in labels:
            code = canonical_kernel(name, source)
            assert name not in canonical or canonical[name][0] == code
            canonical[name] = (code, source)
    assert canonical.keys() == set(labels)
    aliases = {}
    transfers = 0
    batch, m, k = shape
    width = min(m, k)
    for name, dimensions in zip(labels, (shape, (batch, width, width)), strict=True):
        code, source = canonical[name]
        aliases[code] = name
        path = full.with_name(f"{full.stem}-{name}-0.py")
        original = compile_polar((name, source), path)
        # The two explicitly mutating operators launch this checked Gluon body.
        candidate = build_full(polar, name, dimensions[1], dimensions[2])
        for index, artifact in enumerate((original, candidate)):
            stem = full.with_name(f"{full.stem}-{name}-{index}")
            for kind in ("ttgir", "ptx"):
                stem.with_suffix(f".{kind}").write_text(artifact.asm[kind])
        _, avoided = check_full_pair(original, candidate, name, dimensions)
        transfers += avoided
    return aliases, transfers


def replace_split_trace(kernels, program, batch, dim):
    """Verify both original launches before interpreting their shared trace."""
    start = next(i + 1 for i, stmt in enumerate(program)
                 if cname(stmt).startswith(("XXT_kernel_", "XTX_kernel_")))
    gram = program[start - 1].value.args[1].id
    scan, = [name for name in kernels if name.startswith("triton_red_") and "diagonal_sum" in name]
    finish, = [name for name in kernels if name.startswith("triton_per_") and "diagonal_sum" in name]
    assert cname(program[start + 2]) == scan + ".run"
    assert cname(program[start + 5]) == finish + ".run"
    partial = program[start + 2].value.args[1].id
    trace = program[start + 5].value.args[1].id
    chunks = dim // 128
    expected = ast.parse(f"""{partial} = empty_strided_cuda(({batch}, {chunks}), ({chunks}, 1), torch.float32)
stream0 = get_raw_stream(0)
{scan}.run({gram}, {partial}, {batch * chunks}, 128, stream=stream0)
{trace} = empty_strided_cuda(({batch},), (1,), torch.float32)
stream0 = get_raw_stream(0)
{finish}.run({partial}, {trace}, {batch}, {chunks}, stream=stream0)
del {partial}""").body
    end = start + len(expected)
    assert [ast.dump(s) for s in program[start:end]] == [ast.dump(s) for s in expected]
    program[start:end] = ast.parse(f"{trace} = {SPLIT_TRACE}({gram})").body
    return kernels.pop(scan), kernels.pop(finish)


def compile_reduction(part, directory, label):
    name, source = part
    tree = ast.parse(source)
    fn, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    heuristic, jit = fn.decorator_list
    attr = heuristic.func.attr
    assert attr in ("reduction", "persistent_reduction")
    kw = {node.arg: node.value for node in heuristic.keywords}
    metadata = kw["triton_meta"]
    meta = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(metadata.keys, metadata.values, strict=True)
            if ast.literal_eval(k) != "device"}
    hints = ast.literal_eval(kw["size_hints"])
    inductor_meta = ast.literal_eval(kw["inductor_meta"])
    reduction_hint = getattr(ReductionHint, kw["reduction_hint"].attr)
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kw: configs):
        choices = getattr(triton_heuristics, attr)(size_hints=hints, reduction_hint=reduction_hint,
                                                   triton_meta=meta, inductor_meta=inductor_meta)
    fn.decorator_list = [jit]
    path = directory / f"{label}-{attr}.py"
    path.write_text("import triton\nimport triton.language as tl\n" + ast.unparse(fn) + "\n")
    jit_kernel = getattr(load(path, f"trace_check_{label}_{attr}"), name)
    artifacts = {}
    for choice in choices:
        constants = {**meta["constants"], **choice.kwargs}
        tile = (choice.kwargs["XBLOCK"], choice.kwargs.get("R0_BLOCK"), choice.num_warps)
        compiled = triton.compile(ASTSource(jit_kernel, meta["signature"], constants, attrs=meta["configs"][0]),
                                  target=HOPPER, options=dict(num_warps=choice.num_warps, num_stages=choice.num_stages,
                                                             sanitize_overflow=False, enable_fp_fusion=meta["enable_fp_fusion"]))
        ptx = compiled.asm["ptx"]
        path.with_suffix(f".{tile[0]}.ptx").write_text(ptx)
        artifacts[tile] = ptx
    return artifacts


def compile_fused(kernel, dim, warps, out):
    signature = {"Input": "*bf16", "Divisors": "*fp32", "DIM": "constexpr", "CHUNKS": "constexpr"}
    constants = {"DIM": dim, "CHUNKS": warps}
    attrs = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]]}
    compiled = triton.compile(GluonASTSource(kernel, signature, constants, attrs=attrs), target=HOPPER,
                              options=dict(num_warps=warps, num_stages=1))
    out.with_suffix(".ttgir").write_text(compiled.asm["ttgir"])
    out.write_text(compiled.asm["ptx"])
    return compiled.asm["ptx"]


def check_launch(module, shape, dim):
    batch = shape[0]
    gram = torch.empty((batch, dim, dim), dtype=torch.bfloat16)
    chunks = 2 if dim == 256 else 8

    class CaptureKernel:
        def __init__(self):
            self.calls = 0

        def __getitem__(self, grid):
            assert grid == (batch,)

            def launch(source, dest, width, count, *, num_warps, num_stages):
                assert source is gram and dest.shape == (batch, 2, 1)
                assert dest.is_contiguous() and dest.dtype == torch.float32
                assert (width, count, num_warps, num_stages) == (dim, chunks, chunks, 1)
                self.calls += 1
            return launch

    spy = CaptureKernel()
    with patch.object(module, "_gram_divisors", spy):
        result = module.gram_divisors(gram)
        assert result.shape == (batch, 2, 1) and spy.calls == 1
    device = torch.device("cuda")
    props = SimpleNamespace(major=9, minor=0, multi_processor_count=132)
    with patch.object(torch.cuda, "get_device_properties", return_value=props):
        assert module.can_fuse_gram_trace(shape, torch.bfloat16, device)
        assert not module.can_fuse_gram_trace(shape, torch.float16, device)
        assert not module.can_fuse_gram_trace(shape, torch.bfloat16, gram.device)
        with config.patch(max_autotune=True):
            assert not module.can_fuse_gram_trace(shape, torch.bfloat16, device)
        props.multi_processor_count = 114
        assert not module.can_fuse_gram_trace(shape, torch.bfloat16, device)


def check_normalization(path, shape, dim, margin, epsilon):
    batch, rows, cols = shape
    kernels, runner = parse_wrapper(path)
    calls = sorted((n for n in ast.walk(runner) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "run"), key=lambda n: n.lineno)
    first_gram_call = next(n for n in calls if n.func.value.id.startswith(("XXT_kernel_", "XTX_kernel_")))
    gram_input = first_gram_call.args[1].id
    cases = {
        "prepared": (rows * cols, ("prepared", "trace"), ("prepared",)),
        "gram": (dim * dim, ("gram", "trace", "scaled", "retained"), ("scaled", "retained")),
    }
    for phase, (cells, buffers, outputs) in cases.items():
        call, = [n for n in calls if n.func.value.id.startswith("triton_poi_")
                 and "sqrt" in n.func.value.id and (n.args[0].id == gram_input) == (phase == "gram")]
        assert len(call.args) == len(buffers) + 1 and ast.literal_eval(call.args[-1]) == batch * cells
        name, source = kernels[call.func.value.id]
        fn, = [item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name]
        programs = compile_pointwise((fn, ()), path.with_suffix(""), phase)
        half = set(buffers) - {"trace"}
        for (tile, warps), code in programs.items():
            assert cells % tile == 0
            tiles = cells // tile
            boundaries = {b * tiles + edge for b in range(batch) for edge in (0, tiles - 1)}
            for cta in sorted(boundaries):
                observed = replay_pointwise(code, buffers, tile, warps, cta, half=half)
                expected = {}
                for index in range(cta * tile, (cta + 1) * tile):
                    matrix = index // cells
                    root = term("sqrt.rn.f32", ("trace", matrix))
                    d = term("fma.rn.f32", root, margin, epsilon)
                    divisor = term("mul.f32", d, d) if phase == "gram" else d
                    value = term("cvt.f32.bf16", (phase, index))
                    stored = term("cvt.rn.bf16.f32", term("div.full.f32", value, divisor))
                    expected.update({(output, index): stored for output in outputs})
                assert observed == expected, (shape, phase, tile, cta)


def check_fused_normalization(module, shape, dim, path):
    batch, rows, cols = shape
    x = torch.empty(shape, dtype=torch.bfloat16)
    a = torch.empty((batch, dim, dim), dtype=torch.bfloat16)
    divisors = torch.empty((batch, 2, 1), dtype=torch.float32)
    launches = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def run(prepared, gram, table, x_tiles, a_tiles, block, **options):
                assert prepared is x and gram is a and table is divisors
                assert block == 2048 and grid == (x_tiles + a_tiles, batch)
                assert (x_tiles * block, a_tiles * block) == (rows * cols, dim * dim)
                assert options == dict(num_warps=4, num_stages=1)
                launches.append((dict(X_TILES=x_tiles, A_TILES=a_tiles, BLOCK=block), options))
            return run

    kernel = module._normalize_gram_inputs
    with patch.object(module, "_normalize_gram_inputs", CaptureKernel()):
        assert module.normalize_gram_inputs(x, a, divisors) is None
    (constants, options), = launches
    signature = dict(X="*bf16", A="*bf16", Divisors="*fp32",
                     X_TILES="constexpr", A_TILES="constexpr", BLOCK="constexpr")
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(3)}
    binary = triton.compile(GluonASTSource(kernel, signature, constants, attrs), target=HOPPER, options=options)
    assert binary.metadata.shared == 0
    code, _ = instructions(binary.asm["ptx"])
    roles = ("prepared", "gram", "divisors")
    block = constants["BLOCK"]
    start = 0
    for phase, cells in zip(roles[:2], (rows * cols, dim * dim), strict=True):
        tiles = cells // block
        for b in range(batch):
            divisor = ("divisors", 2 * b + int(phase == "gram"))
            for tile in (0, tiles - 1):
                cta = start + tile
                observed = replay_pointwise(code, roles, block, 4, cta, half=set(roles[:2]), cta_y=b)
                begin = b * cells + tile * block
                expected = {(phase, i): term("cvt.rn.bf16.f32", term(
                    "div.full.f32", term("cvt.f32.bf16", (phase, i)), divisor))
                    for i in range(begin, begin + block)}
                assert observed == expected, (shape, phase, b, tile)
        start += tiles
    for kind in ("ttgir", "ptx", "cubin"):
        data = binary.asm[kind]
        path.with_suffix(f".{kind}").write_bytes(data.encode() if isinstance(data, str) else data)


def check_projections(path, shape, coefficients):
    kernels, runner = parse_wrapper(path)
    calls = sorted((n for n in ast.walk(runner) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id.startswith("triton_poi_fused_add_mul_")), key=lambda n: n.lineno)
    if shape[1] <= shape[2]:
        assert not calls
        return {}
    matrix = shape[1] * shape[2]
    result = {}
    for round, (call, (alpha, _, _)) in enumerate(zip(calls, coefficients, strict=True)):
        kernel = call.func.value.id
        name, source = kernels[kernel]
        fn, = [item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name]
        pointers = len(call.args) - 1
        assert pointers in (2, 3)
        roles = ("first", "second", "out")[:pointers]
        programs = compile_pointwise((fn, ()), path.with_suffix(""), f"project-{round}")
        factor, = unpack("<I", pack("<f", alpha))
        possible = set()
        for (block, warps), code in programs.items():
            for tile in (0, matrix // block, shape[0] * matrix // block - 1):
                observed = replay_pointwise(code, roles, block, warps, tile, half=set(roles))
                for product, x in ((0, 1), (1, 0)):
                    dest = 2 if pointers == 3 else 0
                    cells = range(tile * block, (tile + 1) * block)
                    expected = {(roles[dest], i): term("cvt.rn.bf16.f32", term(
                        "fma.rn.f32", factor, term("cvt.f32.bf16", (roles[x], i)),
                        term("cvt.f32.bf16", (roles[product], i)))) for i in cells}
                    if observed == expected:
                        possible.add((product, x, dest, factor))
                        break
                else:
                    raise AssertionError((path, kernel, tile, block, next(iter(observed.items()))))
        assert len(possible) == 1
        result[kernel], = possible
    return result


def capture_cascades(module, out, label, split_baddbmm, stage, grad, velocity):
    args = [grad, velocity, stage[0], split_baddbmm, stage[2], stage[3]]
    split, compact = [out / f"{label}-{tag}.py" for tag in ("split", "fused")]
    capture(module.anvil_cascade, split, [*args, False], user_triton=True)
    capture(module.anvil_cascade, compact, [*args, True], user_triton=True)
    return split, compact


def check(baseline, candidate, out):
    check_checkout(baseline, candidate)
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None or triton.__version__ != "3.6.0":
        raise RuntimeError("Use the pinned CPU PyTorch build and Triton 3.6.0")
    out.mkdir(parents=True, exist_ok=True)
    # Compare the two trace paths on the same compact-grid cascade.
    # check_anvil_gram.py separately links the split trace on that grid to upstream.
    polar = load_polar(candidate / POLAR, "polar_trace")
    check_full_guards(polar)
    module = load_cascade(candidate / ANVIL, polar)
    _, margin, epsilon = anvil_frobenius((baseline / ANVIL).read_text())
    stage = torch.empty(4, dtype=torch.float32)
    copies = 0
    for label, gram, batch, m, k in CASES[:3]:
        shape = (batch, m, k)
        dim = k if gram == "XTX_kernel" else m
        grad = torch.empty(shape, dtype=torch.bfloat16)
        velocity = torch.empty((2, *shape), dtype=torch.float32)
        split, compact = capture_cascades(module, out, label, label == "mlp-gram", stage, grad, velocity)
        before = first_gram(split, shape)[:4]
        after = first_gram(compact, shape, fused=True)[:4]
        first, second = replace_split_trace(*before[:2], batch, dim)
        aliases, avoided = check_polar_wrappers(split, compact, gram, shape, module.ANVIL_MAPS, polar)
        copies += len(module.ANVIL_MAPS) * avoided
        projections = [check_projections(path, shape, module.ANVIL_MAPS) for path in (split, compact)]
        match_cascades((*before, projections[0]), (*after, projections[1]), shape, aliases)
        partials = compile_reduction(first, out, label)
        final = compile_reduction(second, out, label)
        assert list(partials) == [(2, 64, 2)]
        assert set(final) == ({(1, None, 2), (8, None, 2)} if dim == 256 else {(1, None, 2)})
        check_launch(module.trace_module, shape, dim)
        fused = compile_fused(module.trace_module._gram_divisors, dim, 2 if dim == 256 else 8,
                              out / f"{label}-divisors.ptx")
        check_trees(dim, batch, partials[(2, 64, 2)], [(tile[0], ptx) for tile, ptx in final.items()],
                    fused, margin, epsilon)
        check_normalization(split, shape, dim, margin, epsilon)
        check_fused_normalization(module.trace_module, shape, dim, out / f"{label}-normalize")
        print(f"{label}: matched FP32 divisors for all {batch} matrices, both BF16 outputs and the cascade", flush=True)
    payload = copies * 128 * 128 * 2
    assert payload == 146_276_352
    print(f"Verified {copies:,} fewer logical TMA box reads per rank and ANVIL update ({payload:,} bytes of requested payload).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
