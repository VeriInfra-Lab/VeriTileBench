"""Compare the installed triangular launch against the upstream SM90 tile bodies."""

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch

from anvil_tile_replay import expected_stores, instructions, replay
from anvil_cascade_replay import TMA_GRAM, TMA_POLYNOMIAL
from install_candidate import ANVIL, ANVIL_RAILS_KERNEL, ANVIL_TRACE_KERNEL, POLAR
from verify_candidate import check_checkout


# QK, VO, MLP Gram, followed by both distinct polynomial shapes.
CASES = (
    ("qk-gram", "XXT_kernel", 6, 256, 768),
    ("vo-gram", "XXT_kernel", 2, 768, 768),
    ("mlp-gram", "XTX_kernel", 3, 2816, 768),
    ("qk-poly", "ba_plus_cAA_kernel", 6, 256, 256),
    ("vo-mlp-poly", "ba_plus_cAA_kernel", 3, 768, 768),
)

SSA = r"%[\w-]+"


def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_polar(path, name):
    trace_file = path.with_name(ANVIL_TRACE_KERNEL.name)
    if not trace_file.is_file():
        return load(path, name)
    trace = load(trace_file, f"{name}_trace")
    with patch.dict(sys.modules, {"track_1_short.perf.kernels.anvil_trace": trace}):
        polar = load(path, name)
    sys.modules[name] = polar
    polar.trace_module = trace
    return polar


def build(module, name, m, k, compact):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    kernel = getattr(module, name)
    side = (k if name == "XTX_kernel" else m) // 128
    width = side * 128
    actual = dict(M=m, K=k, a_stride_b=m * k, a_stride_r=k,
                  c_stride_b=width * width, c_stride_r=width)
    constants = dict(a_stride_c=1, c_stride_c=1, BLOCK_SIZE_M=128,
                     BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, LOWER_UPPER=1)
    if "COMPACT_SIDE" in kernel.arg_names:
        constants["COMPACT_SIDE"] = side if compact else 0
    signature = {
        arg: ("constexpr" if arg in constants
              else "*bf16" if arg.endswith("_ptr")
              else "fp32" if arg in ("alpha", "beta") else "i32")
        for arg in kernel.arg_names
    }
    target = GPUTarget("cuda", 90, 32)
    attrs = {
        (i,): make_backend(target).parse_attr("D")
        for i, (arg, ty) in enumerate(signature.items())
        if ty == "*bf16" or (ty == "i32" and actual[arg] % 16 == 0)
    }
    artifact = triton.compile(ASTSource(kernel, signature, constants, attrs=attrs), target=target,
                              options=dict(num_stages=4, num_warps=8))
    return actual, artifact


def build_full(module, name, m, k):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.experimental.gluon._runtime import GluonASTSource
    from triton.experimental.gluon.language import NVMMASharedLayout

    tall = name == "XTX_kernel"
    polynomial = name == "ba_plus_cAA_kernel"
    layout = NVMMASharedLayout(128, 16)
    constants = dict(K=k, M=m, Tall=tall, Polynomial=polynomial)
    signature = dict(A=f"tensordesc<bf16[128,128],{layout}>", C=f"tensordesc<bf16[128,128],{layout}>")
    if polynomial:
        signature.update(A_ptr="*bf16", alpha="fp32", beta="fp32")
    else:
        constants.update(A_ptr=None, alpha=None, beta=None)
    kernel = module.polar
    if "L2Bias" in kernel.arg_names:
        constants["L2Bias"] = False
    if "ReturnOffdiag" in kernel.arg_names:
        constants["ReturnOffdiag"] = False
    signature.update(dict.fromkeys(constants, "constexpr"))
    signature = {name: signature[name] for name in kernel.arg_names}
    target = GPUTarget("cuda", 90, 32)
    attrs = {(i,): make_backend(target).parse_attr("D")
             for i, ty in enumerate(signature.values()) if ty == "*bf16"}
    return triton.compile(GluonASTSource(kernel, signature, constants, attrs=attrs),
                          target=target, options=dict(num_stages=1, num_warps=8))


@dataclass
class IR:
    layouts: list
    signature: str
    constants: dict
    prefix: list
    body: list
    coordinates: tuple


def split_ir(text):
    text = re.sub(r'loc\("(?:[^"\\]|\\.)*"\(#\w+\)\)|loc\(#\w+\)', "", text)
    lines = [line.split("//", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith("#loc")]
    head, = [i for i, line in enumerate(lines) if line.startswith("tt.func public ")]
    cut, = [i for i, line in enumerate(lines) if re.match(
        rf"{SSA} = arith.muli {SSA}, %a_stride_b : i32$", line)]
    batch = re.findall(SSA, lines[cut])[1]
    prefix = lines[head + 1:cut]
    coords = [line.split(" =", 1)[0] for line in prefix
              if re.fullmatch(rf"{SSA} = arith.muli {SSA}, %c128_i32 : i32", line)]
    # The two coordinates are the offsets in rows and columns, before buffer arithmetic.
    assert len(coords) == 2, coords
    row, col = coords
    signature = re.sub(r"\s+", " ", lines[head])
    constants = dict(re.findall(rf"^({SSA}) = arith\.constant (.*)$",
                                "\n".join(prefix), re.MULTILINE))
    return IR(lines[:head], signature, constants, prefix, lines[cut:], (batch, row, col))


def scalar_prefix(ir, actual, pid):
    # All program IDs and tile sizes are nonnegative. The reference's group is
    # at most six rows, so its signed divide/remainder agree with natural ones.
    env = {"%" + key: value for key, value in actual.items()}
    for line in ir.prefix:
        if line.startswith("^bb") or line == "tt.return":
            continue
        if line.startswith("cf.cond_br "):
            assert re.fullmatch(rf"cf.cond_br {SSA}, \^bb1, \^bb2", line), line
            assert ir.prefix[-3:] == ["^bb1:", "tt.return", "^bb2:"]
            if env[re.findall(SSA, line)[0]]:
                return None
            continue
        result, statement = line.split(" = ", 1)
        op, _, args = statement.partition(" ")
        if op == "arith.constant":
            if re.fullmatch(r"-?\d+ : i\d+", args):
                env[result] = int(args.split()[0])
            continue
        if op == "tt.get_program_id":
            assert args == "x : i32"
            env[result] = pid
            continue
        operands = [env[x] for x in re.findall(SSA, args)]
        if op == "arith.cmpi":
            assert args.startswith("sle, ")
            env[result] = operands[0] <= operands[1]
            continue
        if op in ("arith.extsi", "arith.extui"):
            env[result] = operands[0]
            continue
        if op == "arith.trunci":
            env[result] = operands[0] & 0xffffffff
            continue
        a, b = operands
        digits = int(args.rsplit("i", 1)[1])
        if op == "arith.shrui":
            value = (a % (1 << digits)) >> b
        elif op == "arith.andi":
            value = a & b
        elif op == "arith.addi":
            value = a + b
        elif op == "arith.subi":
            value = a - b
        elif op == "arith.muli":
            value = a * b
        elif op in ("arith.divsi", "arith.remsi"):
            assert 0 <= a and 0 < b
            value = a // b if op == "arith.divsi" else a % b
        elif op == "arith.minsi":
            value = min(a, b)
        else:
            raise AssertionError(line)
        env[result] = value
    return tuple(env[x] for x in ir.coordinates)


def body(ir):
    """Rename SSA values and expand constants in the common body."""
    names = dict(zip(ir.coordinates, ("BATCH", "ROW", "COL"), strict=True))
    arguments = set(re.findall(SSA, ir.signature))

    def rename(match):
        full, value, result = match.group(), match[1], match[2] or ""
        assert not result or result.startswith("#")
        if value in names:
            return names[value] + result
        if value in ir.constants:
            return "CONST(" + ir.constants[value] + ")" + result
        if value in arguments:
            return full
        if value not in names:
            names[value] = f"v{len(names)}"
        return names[value] + result

    return [re.sub(rf"({SSA})(#\d+)?", rename, re.sub(r"\s+", " ", line))
            for line in ir.body]


def mma_trace(ptx):
    result = []
    for line in ptx.splitlines():
        line = line.strip()
        if not line.startswith("wgmma."):
            continue
        if "mma_async" in line:
            op, rest = line.split(maxsplit=1)
            outputs, rest = rest.split("}", 1)
            fields = [x.strip() for x in rest.removesuffix(";").strip().lstrip(",").split(",")]
            assert all(operand.startswith("%") for operand in fields[:3]), line
            result.append((op.strip(), outputs.count(",") + 1, fields[3:]))
        else:
            result.append(line)
    return result


def polynomial_epilogue(ptx):
    params = re.findall(r"\.param \.f32 (\w+)", ptx)
    alpha, beta = [re.search(r"ld.param.b32\s+(%\w+), \[" + name + r"\];", ptx)[1]
                   for name in params]
    completed = ptx.rsplit("// wait for regs:", 1)[1].splitlines()[0]
    accumulator = set(re.findall(r"%\w+", completed))
    epilogue = ptx.rsplit("wgmma.wait_group.sync.aligned 0;", 1)[1]
    pattern = r"(?m)^\s+(cvt\.f32\.bf16|mul\.f32|fma\.rn\.f32|cvt\.rn\.bf16x2\.f32)\s+([^;]+);"
    extended, products, fused = set(), set(), set()
    counts = Counter()
    for op, operands in re.findall(pattern, epilogue):
        dst, *src = [term.strip() for term in operands.split(",")]
        if op == "cvt.f32.bf16":
            extended.add(dst)
        elif op == "mul.f32":
            assert len(src) == 2 and beta in src and set(src).intersection(extended), (op, src)
            products.add(dst)
        elif op == "fma.rn.f32":
            scale, acc, bias = src
            assert scale == alpha and acc in accumulator and bias in products, src
            fused.add(dst)
        else:
            assert set(src) <= fused, src
        counts[op] += 1
    assert len(counts) == 4
    return counts


def check_full_store_wait(ptx, count):
    """Check every output path reaches one full wait after its commits."""
    wait = r"(?m)^[ \t]*cp\.async\.bulk\.wait_group 0;"
    store = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
    fence = "fence.proxy.async.shared::cta;"
    commit = "cp.async.bulk.commit_group;"
    assert "cp.async.bulk.wait_group.read" not in ptx
    phases = re.split(wait, ptx)
    assert len(phases) == count + 1
    for phase in phases[:-1]:
        _, separator, suffix = phase.rpartition(fence)
        assert separator and phase.count(store) == 2
        body = [" ".join(line.split()) for row in suffix.splitlines()
                if (line := row.split("//", 1)[0].strip())
                and not line.startswith((".loc", "$L__tmp"))]
        first, second = [i for i, line in enumerate(body) if store in line]
        assert body[first + 1] == body[second + 1] == commit
        assert body.count(commit) == 2
        branches = [(i, line) for i, line in enumerate(body)
                    if re.match(r"^(?:@!?%p[0-9]+ )?bra(?:\.uni)? ", line)]
        assert len(branches) in (1, 2)
        branch_at, condition = branches[0]
        conditional = re.fullmatch(r"@%p[0-9]+ bra (\$L__BB[0-9_]+);", condition)
        assert conditional
        if len(branches) == 1:
            serve_at = branch_at
            skip = conditional[1]
        else:
            jump_at, jump = branches[1]
            unconditional = re.fullmatch(r"bra\.uni (\$L__BB[0-9_]+);", jump)
            assert unconditional and jump_at == branch_at + 1
            serve_at = body.index(conditional[1] + ":")
            assert jump_at + 1 == serve_at
            skip = unconditional[1]
        skip_to = body.index(skip + ":")
        assert serve_at < first < first + 1 < skip_to < second
        assert "bar.sync 0;" in body[serve_at:first] and "bar.sync 0;" in body[skip_to:second]
        assert not any(re.search(r"\b(?:ret|exit)\b", line) or "stmatrix." in line
                       or re.search(r"\bst\.shared", line) for line in body)
    assert store not in phases[-1] and commit not in phases[-1]


def check_full_pair(before, after, name, shape):
    batches, m, k = shape
    tall = name == "XTX_kernel"
    width = k if tall else m
    polynomial = name == "ba_plus_cAA_kernel"
    launch = dict(M=m, K=k, a_stride_b=m * k, a_stride_r=k,
                  c_stride_b=width * width, c_stride_r=width)
    bodies = [instructions(a.asm["ttgir"]) for a in (before, after)]
    direct, mirror = [re.search(r"tensor<128x128x!tt.ptr<bf16>, (#blocked\w*)>$", line)[1]
                      for line in bodies[0] if line.startswith("tt.store ")]
    # On the diagonal the two pointer tensors agree pointwise. Equal
    # layouts order the upstream overlapping writes in the same CUDA thread.
    assert direct == mirror
    pid = 0
    transfers_avoided = 0
    chunks = (m if tall else k) // 128
    for batch in range(batches):
        for col in range(0, width, 128):
            for row in range(col, width, 128):
                expected = expected_stores(batch, row, col, m, k, tall=tall,
                                           polynomial=polynomial)
                original, _ = replay(bodies[0], launch, pid, batches * m * k, batches * width * width)
                candidate, transfers = replay(bodies[1], launch, pid, batches * m * k,
                                              batches * width * width, tma=True)
                assert original == expected, (name, shape, pid)
                assert transfers == chunks * (1 if row == col else 2), (name, shape, pid, transfers)
                if row == col:
                    assert expected[0][0] == expected[1][0]
                    expected = expected[1:]
                assert candidate == expected, (name, shape, pid)
                transfers_avoided += 2 * chunks - transfers
                pid += 1

    trace = mma_trace(before.asm["ptx"])
    assert trace[-1] == "wgmma.wait_group.sync.aligned 0;"
    first, *steps, commit, wait, drain = trace
    assert len(steps) == 4 and all(isinstance(step, tuple) for step in steps)
    wide = [first, *steps, *steps, commit]
    ptx = after.asm["ptx"]
    assert mma_trace(ptx) == [*wide, wait, drain, *wide, drain]
    old_ptx = before.asm["ptx"]
    # The second path drains the current group inside the loop. All
    # warpgroups must synchronize before the fixed right box is refilled.
    right_refill = ptx.rsplit(drain, 1)[1]
    refill = "cp.async.bulk.tensor.2d.shared::cluster.global"
    assert right_refill.count(refill) == 1
    cta_barrier = re.search(r"bar\.sync\s+0;", right_refill)
    assert cta_barrier and cta_barrier.end() < right_refill.index(refill)
    if polynomial:
        assert polynomial_epilogue(old_ptx) == polynomial_epilogue(ptx)
    else:
        pattern = r"(?m)^\s*(cvt\.rn\.bf16(?:x2)?\.f32)\s"
        left, right = [re.findall(pattern, text) for text in (old_ptx, ptx)]
        assert left and left == right
    check_full_store_wait(ptx, 1)
    assert ptx.index("cp.async.bulk.wait_group 0;") < ptx.rfind("ret;")
    assert before.metadata.shared == 131072 and after.metadata.shared == 98320
    return pid, transfers_avoided


def load_cascade(path, polar):
    """Load the pinned cascade with its real launchers and no training imports."""
    import torch

    parsed = ast.parse(path.read_text(), filename=str(path))
    names = {"ANVIL_MAPS", "RAIL_SLOW_BETA", "FROBENIUS_MARGIN", "FROBENIUS_EPS"}
    declarations = [node for node in parsed.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in names
                            for lhs in node.targets for target in ast.walk(lhs))]
    fn, = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
           and node.name == "anvil_cascade"]
    decorator, = fn.decorator_list
    assert ast.unparse(decorator) == "torch.compile(dynamic=False, fullgraph=True)"
    fn.decorator_list = []
    module = ModuleType(polar.__name__ + "_cascade")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, XXT=polar.XXT, XTX=polar.XTX, ba_plus_cAA=polar.ba_plus_cAA,
                           anvil_cooperative_round=getattr(polar, "anvil_cooperative_round", None),
                           anvil_initial_gram=getattr(polar, "anvil_initial_gram", None))
    trace_file = path.parents[2] / ANVIL_TRACE_KERNEL
    if trace_file.is_file():
        module.trace_module = getattr(polar, "trace_module", None)
        if module.trace_module is None:
            module.trace_module = load(trace_file, module.__name__ + "_trace")
        module.gram_divisors = module.trace_module.gram_divisors
        module.normalize_gram_inputs = module.trace_module.normalize_gram_inputs
        module.rails_module = load(path.parents[2] / ANVIL_RAILS_KERNEL, module.__name__ + "_rails")
        module.prepare_rails = module.rails_module.prepare_rails
        module.prepare_rails_and_seed = getattr(module.rails_module, "prepare_rails_and_seed", None)
    code = compile(ast.Module(body=declarations + [fn], type_ignores=[]), str(path), "exec")
    exec(code, module.__dict__)
    return module


def lowered_schedule(path, gram, shape, coefficients, compact, *, full=False):
    tree = ast.parse(path.read_text(), filename=str(path))
    batches, m, k = shape
    side = (k if gram == "XTX_kernel" else m) // 128
    width = side * 128
    tiles = side * (side + 1) // 2 if compact else side * side
    kernels = {}
    user_kernels = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if ast.unparse(node.value.func) != "async_compile.triton":
            continue
        name, source = map(ast.literal_eval, node.value.args[:2])
        kernel = ast.parse(source)
        if name not in (gram, "ba_plus_cAA_kernel"):
            kernels[name] = ast.dump(kernel)
            continue
        assert not full
        target, = node.targets
        fn, = [item for item in kernel.body if isinstance(item, ast.FunctionDef) and item.name == name]
        config, = [dec for dec in fn.decorator_list if isinstance(dec, ast.Call)
                   and ast.unparse(dec.func) == "triton_heuristics.user_autotune"]
        kw = {arg.arg: arg.value for arg in config.keywords}
        assert ast.literal_eval(kw["custom_kernel"])
        assert ast.literal_eval(kw["configs"]) == [{"num_warps": 8, "num_stages": 4}]
        meta = {ast.literal_eval(key): value for key, value in zip(
            kw["triton_meta"].keys, kw["triton_meta"].values, strict=True)}
        constants = dict(a_stride_c=1, c_stride_c=1, BLOCK_SIZE_M=128,
                         BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, LOWER_UPPER=1)
        if compact:
            constants.update(COMPACT_SIDE=side)
        assert ast.literal_eval(meta["constants"]) == constants
        signature = ast.literal_eval(meta["signature"])
        expected = {
            arg.arg: ("constexpr" if arg.arg in constants
                      else "*bf16" if arg.arg.endswith("_ptr")
                      else "fp32" if arg.arg in ("alpha", "beta") else "i32")
            for arg in fn.args.args
        }
        assert signature == expected
        aligned = [{(i,): [["tt.divisibility", 16]]
                    for i, ty in enumerate(expected.values()) if ty in ("*bf16", "i32")}]
        assert ast.literal_eval(meta["configs"]) == aligned
        user_kernels[target.id] = name

    runner, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Runner"]
    calls = []
    for node in ast.walk(runner):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func)
        if func in (TMA_GRAM, TMA_POLYNOMIAL):
            assert full and not node.keywords
            name = gram if func == TMA_GRAM else "ba_plus_cAA_kernel"
            assert len(node.args) == (2 if func == TMA_GRAM else 4)
            assert all(isinstance(arg, ast.Name) for arg in node.args[:2])
            calls.append((node, name))
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == "run"
              and isinstance(node.func.value, ast.Name) and node.func.value.id in user_kernels):
            assert not full
            calls.append((node, user_kernels[node.func.value.id]))
    calls.sort(key=lambda pair: pair[0].lineno)
    assert [name for _, name in calls] == [gram, "ba_plus_cAA_kernel"] * 6
    for i, (node, _) in enumerate(calls):
        if i % 2:
            _, beta, alpha = coefficients[i // 2]
            dims, inner, scalars = [width], width, [alpha, beta]
        else:
            dims, inner, scalars = [m, k], k, []
        if full:
            assert [ast.literal_eval(arg) for arg in node.args[2:]] == scalars
            continue
        strides = [dims[0] * inner, inner, 1, width * width, width, 1]
        assert [ast.literal_eval(arg) for arg in node.args[2:]] == dims + strides + scalars + [batches * tiles, 1, 1]
        node.args[-3] = ast.Constant(0)
    return kernels, ast.dump(runner)


def check_dispatch(baseline, candidate, orig, opt, out):
    import torch
    from check_value_adam_inductor import TORCH_REVISION, capture

    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None:
        raise RuntimeError("Use the pinned CPU PyTorch build for offline Inductor capture")
    original = load_cascade(baseline / ANVIL, orig)
    optimized = load_cascade(candidate / ANVIL, opt)
    coefficients = original.ANVIL_MAPS
    assert coefficients == optimized.ANVIL_MAPS and len(coefficients) == 6
    stage = torch.empty(4, dtype=torch.float32)
    for bank, gram, batches, m, k in CASES[:3]:
        shape = (batches, m, k)
        gradient = torch.empty(shape, dtype=torch.bfloat16)
        velocity = torch.empty((2, *shape), dtype=torch.float32)
        args = [gradient, velocity, stage[0], bank == "mlp-gram", stage[2], stage[3]]
        checked = []
        for tag, module in (("reference", original), ("compact", optimized)):
            path = out / f"{bank}-{tag}-inductor.py"
            capture(module.anvil_cascade, path, args, user_triton=True)
            checked.append(lowered_schedule(path, gram, shape, coefficients, tag == "compact"))
        assert checked[0] == checked[1], f"Inductor changed a bank operation outside its grid: {bank}"
        print(f"{bank}: Inductor emitted all 12 launches, with matching surrounding kernels and wrapper", flush=True)


def check_full_guards(module):
    import torch
    from triton.experimental.gluon.language import NVMMASharedLayout

    class CaptureLaunch:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))
            return launch

    cases = (*CASES, ("vo-poly", "ba_plus_cAA_kernel", 2, 768, 768))
    layout = NVMMASharedLayout(128, 16)
    for _, name, batches, m, k in cases:
        tall = name == "XTX_kernel"
        polynomial = name == "ba_plus_cAA_kernel"
        width = k if tall else m
        tiled, fallback = CaptureLaunch(), CaptureLaunch()
        wrapper = getattr(module, name.removesuffix("_kernel"))
        args = (1.0, 2.0) if polynomial else ()
        source = torch.empty((batches, m, k), dtype=torch.bfloat16)
        dest = torch.empty((batches, width, width), dtype=torch.bfloat16)
        with patch.object(module, "polar", tiled), patch.object(module, name, fallback):
            assert wrapper(source, *args, out=dest, full_tiles=True) is dest
            (grid, launch_args, options), = tiled.calls
            src, dst, a, alpha, beta, launch_k, launch_m, transposed, poly, *bias_cache = launch_args
            assert bias_cache == ([False] if hasattr(module, "cooperative_round") else [])
            side = width // 128
            assert grid == (batches * side * (side + 1) // 2,)
            assert options == {"num_warps": 8, "num_stages": 1}
            assert (launch_k, launch_m, transposed, poly) == (k, m, tall, polynomial)
            assert src.layout == layout and src.padding == "zero"
            assert src.base.data_ptr() == source.data_ptr() and src.block_shape == [128, 128]
            assert tuple(src.shape) == (batches * m, k) and tuple(src.strides) == (k, 1)
            assert dst.layout == layout and dst.padding == "zero"
            assert dst.base.data_ptr() == dest.data_ptr() and dst.block_shape == [128, 128]
            assert tuple(dst.shape) == (batches * width, width) and tuple(dst.strides) == (width, 1)
            if polynomial:
                assert a is source and (alpha, beta) == args
            else:
                assert (a, alpha, beta) == (None, None, None)
            assert not fallback.calls
            strided = torch.empty((batches, k, m), dtype=source.dtype).mT
            for a, out, full_tiles in ((source, dest, False), (strided, dest, True),
                                       (source, dest.mT, True), (source, torch.empty_like(dest, dtype=torch.float16), True),
                                       (source[:1], dest[:1], True)):
                assert wrapper(a, *args, out=out, full_tiles=full_tiles) is out
                (_, positional, forwarded), = fallback.calls
                assert not positional and forwarded["A_ptr"] is a and forwarded["C_ptr"] is out
                fallback.calls.clear()
            assert len(tiled.calls) == 1


def check(baseline, candidate, out):
    import triton

    check_checkout(baseline, candidate)
    if triton.__version__ != "3.6.0":
        raise RuntimeError("Use Triton 3.6.0 to reproduce the pinned lowering")
    orig = load(baseline / POLAR, "polar_upstream")
    opt = load_polar(candidate / POLAR, "polar_compact")
    check_full_guards(opt)
    out.mkdir(parents=True, exist_ok=True)
    for label, name, batches, m, k in CASES:
        actual, before = build(orig, name, m, k, False)
        _, after = build(opt, name, m, k, True)
        _, fallback = build(opt, name, m, k, False)
        left, right, unchanged = [split_ir(a.asm["ttgir"]) for a in (before, after, fallback)]
        assert left.layouts == right.layouts == unchanged.layouts
        assert left.signature == right.signature == unchanged.signature
        a, b = body(left), body(right)
        assert a == b, next(((n, x, y) for n, (x, y) in enumerate(zip(a, b)) if x != y), (len(a), len(b)))
        assert left == unchanged
        side = (k if name == "XTX_kernel" else m) // 128
        expected = [(batch, row * 128, col * 128)
                    for batch in range(batches) for col in range(side) for row in range(col, side)]
        dense_count = batches * side * side
        old_grid = [tile for pid in range(dense_count)
                    if (tile := scalar_prefix(left, actual, pid)) is not None]
        new_grid = [scalar_prefix(right, actual, pid) for pid in range(len(expected))]
        assert old_grid == new_grid == expected
        trace = mma_trace(before.asm["ptx"])
        assert trace == mma_trace(after.asm["ptx"])
        assert sum("mma_async" in op[0] for op in trace if isinstance(op, tuple)) == 4
        full = build_full(opt, name, m, k)
        tiles, transfers = check_full_pair(after, full, name, (batches, m, k))
        assert tiles == len(expected)
        for tag, artifact in (("reference", before), ("compact", after), ("full", full)):
            for kind in ("ttir", "ttgir", "ptx", "cubin"):
                if kind in artifact.asm:
                    path = out / f"{label}-{tag}.{kind}"
                    if kind == "cubin":
                        path.write_bytes(artifact.asm[kind])
                    else:
                        path.write_text(artifact.asm[kind])
        print(f"{label}: {dense_count} -> {tiles} CTAs, same FP operands and stores, "
              f"{transfers} TMA reads avoided", flush=True)
    check_dispatch(baseline, candidate, orig, opt, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
