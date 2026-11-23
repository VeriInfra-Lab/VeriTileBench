#!/usr/bin/env python3
"""Capture the full pinned GPT and check the MUDD wrapper rewrites."""
import argparse
import ast
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import triton
from torch._inductor.runtime import triton_heuristics
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.experimental.gluon._runtime import GluonASTSource

from check_mudd_coefficients import CompileOnlyCE, capture, check_backend, load
from install_candidate import MUDD_COEFFICIENTS_KERNEL
from verify_candidate import check_checkout

SSA = re.compile(r"%[\w.]+")
LOCATION = re.compile(r" loc\((?:[^()]|\([^()]*\))*\)")

@torch.library.custom_op("veritile_full_capture::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_q: torch.Tensor,
              cu_k: torch.Tensor, max_q: int, max_k: int, scale: float, window: int) -> torch.Tensor:
    raise RuntimeError("offline FlashAttention must not execute")

@attention.register_fake
def attention_fake(q, k, v, cu_q, cu_k, max_q, max_k, scale, window):
    return v.new_empty((q.shape[0], q.shape[1], v.shape[2]))

def setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3])

def backward(ctx, grad):
    return (*torch.ops.veritile_full_capture.attention_grads(*ctx.saved_tensors, grad), *(None,) * 6)

@torch.library.custom_op("veritile_full_capture::attention_grads", mutates_args=())
def attention_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise RuntimeError("offline FlashAttention backward must not execute")

@attention_grads.register_fake
def attention_grads_fake(q, k, v, grad):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)

attention.register_autograd(backward, setup_context=setup)

def flash(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
          causal, softmax_scale, window_size):
    assert causal
    return attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, window_size[0])

def load_attention(candidate):
    key = "track_1_short.model.attention"
    module = ModuleType(key)
    path = candidate / "track_1_short/model/attention.py"
    module.__file__ = str(path)
    sys.modules[key] = module
    tree = ast.parse(path.read_text(), filename=str(path))
    module.flash_attn_interface = SimpleNamespace(flash_attn_varlen_func=flash)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "kernels")
                 and not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "flash_attn_interface")]
    exec(compile(tree, str(path), "exec"), vars(module))


def compile_reduction(text, output, symbol):
    tree = ast.parse(text)
    fn, = (s for s in tree.body if isinstance(s, ast.FunctionDef))
    heuristic, jit = fn.decorator_list
    fn.decorator_list = [jit]
    tree.body = [s for s in tree.body
                 if not (isinstance(s, ast.Expr) and ast.unparse(s) == "triton_helpers.set_driver_to_gpu()")]
    output.write_text(ast.unparse(tree))
    module = load(output)
    args = {kw.arg: eval(compile(ast.Expression(kw.value), str(output), "eval"), vars(module))
            for kw in heuristic.keywords}
    meta = args["triton_meta"]
    heuristic_name = heuristic.func.attr
    assert heuristic_name in ("reduction", "persistent_reduction", "pointwise")
    select_configs = getattr(triton_heuristics, heuristic_name)
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kwargs: configs), \
            patch.object(torch.cuda, "is_available", return_value=True):
        configs = select_configs(**args)
    assert configs
    for cfg in configs:
        source = ASTSource(getattr(module, symbol), meta["signature"], cfg.kwargs, meta["configs"][0])
        compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                  options=dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                                               enable_fp_fusion=meta["enable_fp_fusion"]))
        xblock = cfg.kwargs["XBLOCK"]
        rblock = cfg.kwargs.get("R0_BLOCK", 0 if heuristic_name == "pointwise" else 16)
        if heuristic_name == "persistent_reduction":
            assert cfg.kwargs.keys() == {"XBLOCK"} and cfg.num_stages == 1 and cfg.num_ctas == 1
            assert meta["enable_fp_fusion"]
        name = f"{output.stem}-x{xblock}-r{rblock}-w{cfg.num_warps}"
        for fmt in ("ttir", "ttgir", "ptx", "cubin"):
            value = compiled.asm[fmt]
            output.with_name(f"{name}.{fmt}").write_bytes(value if isinstance(value, bytes) else value.encode())
        yield (xblock, rblock, cfg.num_warps), compiled.asm


def compare_ir(text, redundant=None):
    """Erase one output pointer's address calculation and store, then rename SSA values."""
    names = {}
    dead = {redundant} if redundant else set()
    stores = 0
    result = []
    for line in text.splitlines():
        if line.startswith("#loc"):
            continue
        line = LOCATION.sub("", line).strip()
        if not line:
            continue
        destination, assign, expression = line.partition(" = ")
        expression = expression if assign else line
        if "tt.func public @" not in line and dead.intersection(SSA.findall(expression)):
            opcode = expression.split()[0]
            if opcode == "tt.store":
                stores += 1
            else:
                assert assign and opcode in ("tt.splat", "tt.addptr", "ttg.convert_layout"), line
                dead.add(destination)
            continue
        result.append(SSA.sub(lambda m: f"%{names.setdefault(m[0], len(names))}", line))
    assert stores == int(redundant is not None)
    return result


def verify_rewrite(optimizer, source, rewritten, output, label):
    tree = ast.parse(source)
    (symbol, _, rows, kernel, store), = optimizer._duplicated_stores(tree)
    old = kernel.value.splitlines()
    old[store.lineno - 1] = ""
    new_kernel, = (n.value.args[1].value for n in ast.parse(rewritten).body
                   if isinstance(n, ast.Assign) and optimizer._call(n.value, "async_compile.triton", 2)
                   and n.value.args[0].value == symbol)
    assert new_kernel.splitlines() == old
    discarded = f"%{store.value.args[0].left.id}"
    for (schedule, before), (new_schedule, after) in zip(
            compile_reduction(kernel.value, output / f"{label}-original.py", symbol),
            compile_reduction(new_kernel, output / f"{label}-single.py", symbol), strict=True):
        assert schedule == new_schedule and rows % schedule[0] == 0
        for ir in ("ttir", "ttgir"):
            assert compare_ir(before[ir], discarded) == compare_ir(after[ir]), (schedule, ir)
    line = kernel.lineno + store.lineno - 2
    lines = source.splitlines(keepends=True)
    for old, replacement in (("r0_mask)", "xmask)"), (" + 768", " + 767")):
        mutant = lines.copy()
        mutant[line] = mutant[line].replace(old, replacement)
        assert mutant != lines and optimizer.patch_wrapper("".join(mutant), retain_residual=False, group_gradients=False)[1] == 0
    transpose, = (n for n in ast.walk(tree) if optimizer._call(
        n, "torch.ops.nanogpt.track1_ce_transpose_operands.default", 4))
    extra_reader = " " * transpose.col_offset + f"observe({ast.unparse(transpose.args[0])})\n"
    mutant = "".join(lines[:transpose.lineno - 1] + [extra_reader] + lines[transpose.lineno - 1:])
    assert optimizer.patch_wrapper(mutant, retain_residual=False, group_gradients=False) == (mutant, 0)
    rank = (source.replace("_DeviceGuard(0)", "_DeviceGuard(7)")
            .replace("set_device(0)", "set_device(7)").replace("get_raw_stream(0)", "get_raw_stream(7)")
            .replace("stream0", "stream7"))
    assert optimizer.patch_wrapper(rank, retain_residual=False, group_gradients=False)[1] == 1
    assert rows == 16384
    print(f"{label}: all reduction schedules removed the duplicate E4M3 store.", flush=True)


BF16_POINTER_TILE = r"tensor<(\w+x!tt.ptr<bf16>(?:, #[\w.]+)?)>"
RESIDUAL_STORE = re.compile(r"^tt.store (%[\w.]+), (%[\w.]+), (%[\w.]+) : " + BF16_POINTER_TILE + r"$")
RESIDUAL_LOAD = re.compile(
    r"^(%[\w.]+) = tt.load (%[\w.]+), (%[\w.]+), (%[\w.]+) evictionPolicy = evict_first : "
    + BF16_POINTER_TILE + r"$"
)


def ir_lines(text):
    return [LOCATION.sub("", line).strip() for line in text.splitlines() if not line.startswith("#loc") and line.strip()]


def erase_residual_reload(lines):
    stores = {}
    matches = []
    for line in lines:
        if match := RESIDUAL_STORE.fullmatch(line):
            pointer, value, mask, tile = match.groups()
            stores[pointer, mask, tile] = value
        if match := RESIDUAL_LOAD.fullmatch(line):
            load, pointer, mask, _, tile = match.groups()
            if (pointer, mask, tile) in stores:
                matches.append((load, stores[pointer, mask, tile], mask))
    (load, stored, mask), = matches
    users = {load}
    published = 0
    result = []
    for line in lines:
        name, assign, rhs = line.partition(" = ")
        if name == load:
            continue
        expression = rhs if assign else line
        if users.intersection(SSA.findall(expression)):
            if assign:
                assert expression.split()[0] in ("arith.extf", "arith.mulf", "tt.fp_to_fp"), line
                users.add(name)
            else:
                assert line.startswith("tt.store "), line
                pointer, value, guard = map(str.strip, line[9:].split(" : ", 1)[0].split(","))
                assert pointer not in users and value in users and guard == mask
                published += 1
        result.append(SSA.sub(lambda m: stored if m[0] == load else m[0], line))
    assert published in (1, 2)
    return result


def canonical_residual_ir(lines):
    # A removed load can also remove its zero literal and renumber the
    # remaining constants. Compare every surviving operand by value.
    constants = {}
    names = {}
    result = []
    for line in lines:
        name, assign, rhs = line.partition(" = ")
        if assign and rhs.startswith("arith.constant "):
            constants[name] = f"({rhs})"
        else:
            operands = SSA.sub(lambda m: constants.get(m[0], m[0]), line)
            result.append(SSA.sub(lambda m: f"%{names.setdefault(m[0], len(names))}", operands))
    return result


def verify_residual(optimizer, reference, rewritten, output, label):
    tree = ast.parse(reference)
    (symbol, rows, kernel, edits), = optimizer._residual_rewrites(tree)
    compiled, = (node.value.args[1].value for node in ast.parse(rewritten).body
                 if isinstance(node, ast.Assign) and optimizer._call(node.value, "async_compile.triton", 2)
                 and node.value.args[0].value == symbol)
    output.mkdir(parents=True, exist_ok=True)
    schedules = []
    for (schedule, before), (new_schedule, after) in zip(
            compile_reduction(kernel.value, output / f"{label}-original.py", symbol),
            compile_reduction(compiled, output / f"{label}-register.py", symbol), strict=True):
        assert schedule == new_schedule and rows % schedule[0] == 0
        schedules.append(schedule)
        for ir in ("ttir", "ttgir"):
            original = ir_lines(before[ir])
            if schedule[1] >= 768:
                original = erase_residual_reload(original)
            assert canonical_residual_ir(original) == canonical_residual_ir(ir_lines(after[ir])), (schedule, ir)
    retained = [schedule for schedule in schedules if schedule[1] >= 768]
    assert len(set(schedules)) == 6 and 0 < len(retained) < 6
    lines = reference.splitlines(keepends=True)
    for stmt, _ in edits[1:]:
        i = kernel.lineno + stmt.lineno - 2
        mutant = lines.copy()
        mutant[i] = mutant[i].replace("r0_mask", "xmask")
        changed = "".join(mutant)
        assert changed != reference and optimizer.patch_wrapper(changed, group_gradients=False) == (changed, 0)
    print(f"{label}: six schedules checked, register reuse at {retained}.", flush=True)


def normalize_gradient_plane(text, lane=None, first=0):
    result = []
    integers = {}
    names = {}
    input_was_selected = False
    output_was_rebased = False
    for line in ir_lines(text):
        if line.startswith('tt.func public @'):
            result.append('tt.func public @reduce')
            continue
        lhs, _, rhs = line.partition(' = ')
        if lane is not None:
            if match := re.fullmatch(r'arith.constant (-?\d+) : i32', rhs):
                integers[lhs] = int(match[1])
            if rhs == 'tt.get_program_id y : i32':
                integers[lhs] = lane
                continue
            if match := re.fullmatch(r'arith.cmpi eq, (%[\w.]+), (%[\w.]+) : i32', rhs):
                if all(v in integers for v in match.groups()):
                    integers[lhs] = int(integers[match[1]] == integers[match[2]])
                    continue
            if match := re.fullmatch(r'arith.select (%[\w.]+), (%[\w.]+), (%[\w.]+) : !tt.ptr<bf16>', rhs):
                if match[1] in integers:
                    v = match[2 if integers[match[1]] else 3]
                    names[lhs] = names.get(v, v)
                    continue
            if match := re.fullmatch(r'arith.addi (%[\w.]+), (%[\w.]+) : i32', rhs):
                if all(v in integers for v in match.groups()):
                    integers[lhs] = integers[match[1]] + integers[match[2]]
                    continue
            if match := re.fullmatch(r'tt.addptr %OUT, (%[\w.]+) : !tt.ptr<bf16>, i32', rhs):
                assert not output_was_rebased and integers[match[1]] == first + lane
                names[lhs] = '%verified_output'
                output_was_rebased = True
                continue
        if lane is not None:
            line = SSA.sub(lambda m: names.get(m[0], m[0]), line)
            aliases = re.findall(r'%IN\d+', line)
            if aliases:
                assert aliases == [f'%IN{lane}'] and not input_was_selected
                input_was_selected = True
                line = line.replace(aliases[0], '%verified_input')
            assert '%OUT' not in line
        else:
            line = SSA.sub(lambda m: {'%in_ptr0': '%verified_input', '%out_ptr0': '%verified_output'}.get(m[0], m[0]), line)
        result.append(line)
    if lane is not None:
        assert input_was_selected and output_was_rebased
    return canonical_residual_ir(result)


def combine_integer(op, left, right):
    if isinstance(left, int) and isinstance(right, int):
        return {'arith.addi': lambda: left + right, 'arith.muli': lambda: left * right,
                'arith.divsi': lambda: left // right, 'arith.remsi': lambda: left % right}[op]()
    if op in ('arith.addi', 'arith.muli'):
        left, right = sorted((left, right), key=repr)
    return (op, left, right)


def replay_packing(text, plane):
    names = {f'%in_ptr{i}': ('input', 9-i) for i in range(10)}
    names['%in_out_ptr0'] = 'output'
    stores = []
    loads = 0
    for line in ir_lines(text):
        if line.startswith(('module', 'tt.func public @', '#')) or line in ('tt.return', '}'):
            continue
        name, assigned, expr = line.partition(' = ')
        code, _, ty = (expr if assigned else line).rpartition(' : ')
        op, _, operands = code.partition(' ')
        refs = [names[x] for x in SSA.findall(operands)]
        if op == 'arith.constant':
            if ty in ('i32', 'i64') or re.search(r'xi(?:32|64)(?:, #[\w.]+)?>$', ty):
                value = int(operands.removeprefix('dense<').removesuffix('>'))
            else:
                assert re.search(r'xf32(?:, #[\w.]+)?>$', ty), line
                value = ('constant-f32', operands)
        elif op == 'tt.get_program_id':
            assert operands == 'x'
            value = ('cta',)
        elif op == 'tt.make_range':
            match = re.fullmatch(r'\{end = (\d+) : i32, start = 0 : i32\}', operands)
            assert match, line
            value = ('range', int(match[1]))
        elif op in ('tt.splat', 'ttg.convert_layout'):
            value, = refs
        elif op in ('arith.addi', 'arith.muli', 'arith.divsi', 'arith.remsi'):
            value = combine_integer(op, *refs)
            if op == 'arith.remsi' and value[2] == 10 and value[1][0] == 'arith.divsi' and value[1][2] == 12:
                value = plane
        elif op == 'arith.cmpi':
            assert operands.startswith('eq, ') and all(isinstance(r, int) for r in refs), line
            value = refs[0] == refs[1]
        elif op == 'arith.select':
            assert isinstance(refs[0], bool), line
            value = refs[1 if refs[0] else 2]
        elif op == 'tt.addptr':
            assert len(refs) == 2 and '!tt.ptr<bf16>' in ty, line
            value = ('address', *refs)
        elif op == 'tt.load':
            assert len(refs) == 1 and operands.endswith('evictionPolicy = evict_last') and '!tt.ptr<bf16>' in ty, line
            value = ('load-bf16', refs[0])
            loads += 1
        elif op == 'tt.store':
            assert not assigned and len(refs) == 2 and '!tt.ptr<bf16>' in ty, line
            stores.append(tuple(refs))
            continue
        elif op in ('arith.extf', 'arith.truncf', 'arith.addf', 'arith.mulf'):
            value = (op, *refs)
            if op == 'arith.extf':
                assert 'xbf16' in ty and 'xf32' in ty
            elif op == 'arith.truncf':
                assert 'xf32' in ty and 'xbf16' in ty
            else:
                assert 'xf32' in ty
        else:
            raise ValueError(line)
        assert assigned and name not in names, line
        names[name] = value
    assert len(stores) == 1
    return stores[0], loads


def verify_packing_expression(store, plane, block):
    index = combine_integer('arith.addi', combine_integer('arith.muli', ('cta',), block), ('range', block))
    row = ('arith.divsi', index, 120)
    column = ('arith.remsi', index, 12)
    offset = combine_integer('arith.addi', column, combine_integer('arith.muli', row, 12))
    addr = ('address', 'output', index)
    data = ('arith.extf', ('load-bf16', ('address', ('input', plane), offset)))
    def selected(i):
        return data if i == plane else ('constant-f32', 'dense<0.000000e+00>')
    initial = selected(9)
    for i in range(8, -1, -1):
        initial = ('arith.addf', initial, selected(i))
    value = ('arith.truncf', ('arith.mulf', initial, ('constant-f32', 'dense<1.000000e-01>')))
    assert store == (addr, value), (store, (addr, value))


def verify_packing_loads(optimizer, before, launches, after, output, label, symbol):
    def kernel(source):
        result, = (node.value.args[1].value for node in ast.parse(source).body
                   if isinstance(node, ast.Assign) and optimizer._simple_target(node) == symbol
                   and optimizer._call(node.value, "async_compile.triton", 2))
        return result

    raw = kernel(before)
    rewritten = kernel(after)
    assert raw == kernel(launches) and raw != rewritten
    first = []
    for (cfg, original), (chosen, routed) in zip(
            compile_reduction(raw, output / f"{label}-packing-original.py", symbol),
            compile_reduction(rewritten, output / f"{label}-packing-routed.py", symbol), strict=True):
        assert cfg == chosen and cfg[1] == 0
        first.append(cfg)
        for ir in ("ttir", "ttgir"):
            for column in range(10):
                old, loads = replay_packing(original[ir], column)
                new, next_loads = replay_packing(routed[ir], column)
                assert (loads, next_loads) == (10, 1)
                verify_packing_expression(old, column, cfg[0])
                assert old == new, (label, cfg, column, ir)
    assert set(first) == {(1024, 0, 4), (512, 0, 8)} and len(first) == 2
    # Alter a plane predicate while retaining all memory accesses. The
    # launch reorder remains legal, but the single-load edit must decline.
    bad = before.replace("tmp46 = tl.full([1], 0, tl.int32)", "tmp46 = tl.full([1], 1, tl.int32)")
    assert bad != before
    held, count = optimizer._patch_column_reductions(bad, ast.parse(bad))
    assert count == 1 and kernel(held) == kernel(bad)


def save_gluon_packing(kernel, output, stem, constants, warps, *, scaled):
    pointer_count = 13 if scaled else 12
    signature = {arg: "*bf16" if i < pointer_count else "constexpr" for i, arg in enumerate(kernel.arg_names)}
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(pointer_count)}
    if not scaled:
        constants = {**constants, "SCALED": None}
    source = GluonASTSource(kernel, signature, constants, attrs)
    compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                              options=dict(num_warps=warps, num_stages=1, enable_fp_fusion=True))
    for fmt in ("ttgir", "ptx", "cubin"):
        code = compiled.asm[fmt]
        (output / f"{stem}.{fmt}").write_bytes(code if isinstance(code, bytes) else code.encode())


def compile_direct_packing(optimizer, output, name, tile, warps, planes, first, *, scaled):
    physical_tile = optimizer._direct_gradient_tile(tile)
    constants = dict(XBLOCK=physical_tile, PLANES=planes, FIRST=first)
    variant = "scaled" if scaled else "direct"
    stem = f"{name}-{variant}-x{physical_tile}-r16-w{warps}"
    if not scaled and tile == 128:
        assert (planes, first) == (10, 0)
        save_gluon_packing(optimizer.mudd_gradient_pack_warp, output, stem, {}, warps, scaled=False)
    else:
        save_gluon_packing(optimizer.mudd_gradient_pack_direct, output, stem, constants, warps, scaled=scaled)


def compile_mixed_packing(optimizer, output, label, *, scaled):
    tiles = (1, 8, 32, 128)
    for group in tiles:
        for zero in tiles:
            if group == zero:
                continue
            if (group, zero) == (128, 1):
                save_gluon_packing(optimizer.mudd_banded_gradients, output, f"{label}-mixed-g128-z1-x128-w8",
                                   {}, 8, scaled=scaled)
            else:
                tile, warps = ((64, 8) if (group, zero) == (1, 128)
                               else {8: (8, 2), 32: (32, 4), 128: (128, 8)}[max(group, zero)])
                save_gluon_packing(optimizer.mudd_mixed_gradients, output, f"{label}-mixed-g{group}-z{zero}-x{tile}-w{warps}",
                                   dict(GROUP_TILE=group, ZERO_TILE=zero), warps, scaled=scaled)


def capture_fused_packing(optimizer, tree, node, group, packer, kernels, output, label):
    device = node.items[0].context_expr.args[0].value
    aligned = dict(optimizer._column_reduction_kernels(tree, aligned_output=True))
    zero = optimizer._packing_column_zero(node.body, group, packer, aligned, device)
    assert zero is not None
    fused = optimizer.mudd_gradient_and_pack
    pointer_count = 12
    signature = {name: ("*bf16" if i < pointer_count else "constexpr") for i, name in enumerate(fused.arg_names)}
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(pointer_count)}

    def compile_fused(name, tile, warps, planes, first):
        source = ASTSource(fused, signature, dict(XBLOCK=tile, PLANES=planes, FIRST=first), attrs)
        candidate = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                   options=dict(num_warps=warps, num_stages=1, enable_fp_fusion=True))
        for fmt in ("ttir", "ttgir", "ptx", "cubin"):
            code = candidate.asm[fmt]
            path = output / f"{name}-fused-x{tile}-r16-w{warps}.{fmt}"
            path.write_bytes(code if isinstance(code, bytes) else code.encode())
        if tile != 128 or planes == 10:
            compile_direct_packing(optimizer, output, name, tile, warps, planes, first, scaled=False)

    for symbol, planes, first in ((group[0].kernel, 9, 1), (zero.kernel, 1, 0)):
        name = f"{label}-p{planes}-o{first}"
        schedules = []
        for (tile, rblock, warps), _ in compile_reduction(
                kernels[symbol], output / f"{name}-original.py", symbol):
            schedules.append((tile, warps))
            assert rblock == 16 and zero.rows % tile == 0
            compile_fused(name, tile, warps, planes, first)
        assert set(schedules) == optimizer._LAUNCH_CONFIG and len(schedules) == 4
    for tile, warps in sorted(optimizer._LAUNCH_CONFIG):
        compile_fused(f"{label}-p10-o0", tile, warps, 10, 0)
    compile_mixed_packing(optimizer, output, label, scaled=False)
    return zero


def capture_scaled_packing(optimizer, tree, node, group, kernels, output, label):
    aligned = dict(optimizer._column_reduction_kernels(tree, aligned_output=True))
    plan = optimizer._full_packer(tree, node, group, aligned)
    assert plan is not None
    zero, packing, _, scale_lines = plan
    scale = ast.unparse(scale_lines[1].value.func).removesuffix(".run")
    for symbol, variant in ((zero.kernel, "p1-o0"), (scale, "projection-scaling"),
                            (packing.kernel, "plane-packing")):
        schedules = [cfg for cfg, _ in compile_reduction(kernels[symbol], output / f"{label}-{variant}-original.py", symbol)]
        assert len(schedules) == (4 if variant == "p1-o0" else 2)
    for planes, first in ((1, 0), (9, 1), (10, 0)):
        name = f"{label}-p{planes}-o{first}"
        for tile, warps in sorted(optimizer._LAUNCH_CONFIG):
            compile_direct_packing(optimizer, output, name, tile, warps, planes, first, scaled=True)
    compile_mixed_packing(optimizer, output, label, scaled=True)
    return zero, packing, scale_lines


def verify_gradient_groups(optimizer, reference, rewritten, output, label, sampled):
    tree = ast.parse(reference)
    groups = list(optimizer._column_reduction_groups(tree))
    assert [(len(group), group[-1].column) for _, group, _ in groups] == [(9, 1)]
    fallback = list(optimizer._column_reduction_groups(tree, hoist=False)) if sampled else []
    assert [(len(group), group[-1].column) for _, group, _ in fallback] == ([(6, 4), (3, 1)] if sampled else [])
    assert rewritten == optimizer._patch_column_reductions(reference, tree)[0]
    fallback_rewrite = optimizer._patch_column_reductions(reference, tree, fuse_packing=False)[0]
    assert fallback_rewrite.count(" = try_grouped_gradients(") == 1
    assert rewritten.count(" = try_fused_gradients(") == int(sampled)
    compile(rewritten, label + "-rewritten.py", "exec")
    kernels = {node.value.args[0].value: node.value.args[1].value for node in tree.body
               if isinstance(node, ast.Assign) and optimizer._call(node.value, "async_compile.triton", 2)}
    grouped = optimizer.mudd_gradient_groups
    sig = {name: ("*bf16" if index < 10 else "constexpr") for index, name in enumerate(grouped.arg_names)}
    attrs = {(index,): [["tt.divisibility", 16]] for index in range(10)}
    output.mkdir(parents=True, exist_ok=True)
    for _, group, _ in groups + fallback:
        planes, first = len(group), group[-1].column
        symbol = group[0].kernel
        tag = f"{label}-p{planes}-o{first}"
        schedules = []
        for (tile, rblock, warps), original in compile_reduction(kernels[symbol],
                                                                 output / f"{tag}-original.py", symbol):
            assert rblock == 16 and group[0].rows % tile == 0
            schedules.append((tile, warps))
            source = ASTSource(grouped, sig, dict(XBLOCK=tile, PLANES=planes, FIRST=first), attrs)
            candidate = triton.compile(source, target=GPUTarget("cuda", 90, 32),
                                       options=dict(num_warps=warps, num_stages=1, enable_fp_fusion=True))
            for fmt in ("ttir", "ttgir", "ptx", "cubin"):
                code = candidate.asm[fmt]
                path = output / f"{tag}-grouped-x{tile}-r{rblock}-w{warps}.{fmt}"
                path.write_bytes(code if isinstance(code, bytes) else code.encode())
            for ir in ("ttir", "ttgir"):
                one = normalize_gradient_plane(original[ir])
                for lane in range(planes):
                    assert normalize_gradient_plane(candidate.asm[ir], lane, first) == one, (tag, tile, lane, ir)
        assert set(schedules) == optimizer._LAUNCH_CONFIG and len(schedules) == 4
    wrong_stride = reference.replace("out_ptr0 + (10*x0), tmp4, None", "out_ptr0 + (9*x0), tmp4, None")
    assert wrong_stride != reference and optimizer._patch_column_reductions(wrong_stride, ast.parse(wrong_stride))[1] == 0
    node, combined, packer = groups[0]
    first = combined[0]
    original = reference.splitlines(keepends=True)
    i = node.body[first.index].lineno - 1
    for a, b in ((first.src, combined[1].src), (f"stream={first.stream}", "stream=other_stream")):
        changed = original.copy()
        changed[i] = changed[i].replace(a, b, 1)
        mutant = "".join(changed)
        assert mutant != reference and optimizer._patch_column_reductions(mutant, ast.parse(mutant))[1] == int(sampled)
    assert bool(packer) == sampled
    if sampled:
        original = reference.splitlines(keepends=True)
        packed_line = next(i for i, line in enumerate(original)
                           if line.strip().startswith("triton_poi_fused_add_mul_select_backward_squeeze_")
                           and ".run(" in line)
        packed_call = ast.parse(original[packed_line].strip()).body[0].value
        for before, after in ((f"stream={first.stream}", "stream=foreign_stream"),
                              (f".run({packed_call.args[0].id},", f".run({first.base},")):
            mutant = original.copy()
            mutant[packed_line] = mutant[packed_line].replace(before, after, 1)
            assert mutant != original
            assert optimizer._patch_column_reductions("".join(mutant), ast.parse("".join(mutant)))[1] == 2
        invalid_load = reference.replace("tl.load(in_ptr9 + (x0 + 12*x2), None, eviction_policy='evict_last')",
                                         "tl.load(in_out_ptr0 + (x0 + 12*x2), None, eviction_policy='evict_last')")
        assert invalid_load != reference and optimizer._patch_column_reductions(invalid_load, ast.parse(invalid_load))[1] == 2
        rewritten_lines = rewritten.splitlines(keepends=True)
        assert rewritten_lines[packed_line].strip() == "if not _mudd_fused_packing_0: " + original[packed_line].strip()
        for launch in combined:
            i = node.body[launch.index].lineno - 1
            assert rewritten_lines[i].strip() == "if not _mudd_column_reduction_0: " + original[i].strip()
        launches = optimizer._patch_column_reductions(reference, tree, route_packing=False, fuse_packing=False)[0]
        fused_launches = optimizer._patch_column_reductions(reference, tree, route_packing=False)[0]
        verify_packing_loads(optimizer, reference, launches, fallback_rewrite, output, label, packer.kernel)
        zero = capture_fused_packing(optimizer, tree, node, combined, packer, kernels, output, label)
        zero_line = node.body[zero.index].lineno - 1
        assert rewritten_lines[zero_line].strip() == "if not _mudd_fused_packing_0: " + original[zero_line].strip()
        mutated = original.copy()
        mutated[zero_line] = mutated[zero_line].replace(f"stream={zero.stream}", "stream=other_stream")
        wrong_stream = "".join(mutated)
        wrong_scale = reference.replace("tmp51 = 0.1\n    tmp52 = tmp50", "tmp51 = 0.2\n    tmp52 = tmp50")
        for mutant in (wrong_stream, wrong_scale):
            assert mutant != reference
            assert " = try_fused_gradients(" not in optimizer._patch_column_reductions(mutant, ast.parse(mutant))[0]
        (output / f"{label}-before.py").write_text(reference)
        (output / f"{label}-launches.py").write_text(launches)
        (output / f"{label}-after.py").write_text(fallback_rewrite)
        (output / f"{label}-fused-launches.py").write_text(fused_launches)
        (output / f"{label}-fused-after.py").write_text(rewritten)
    else:
        zero, packing, scale_lines = capture_scaled_packing(optimizer, tree, node, combined, kernels, output, label)
        assert rewritten.count(" = try_fused_full_gradients(") == 1
        transformed = rewritten.splitlines()
        old = reference.splitlines()
        for stmt in (node.body[zero.index], *scale_lines, node.body[packing.index]):
            line = stmt.lineno - 1
            assert transformed[line].strip() == "if not _mudd_fused_packing_0: " + old[line].strip()
        (output / f"{label}-before.py").write_text(reference)
        (output / f"{label}-scaled-after.py").write_text(rewritten)
        print(f"{label}: captured the scaled packing, original reductions and original projection operand", flush=True)
    print(f"{label}: one grouped backward region, with unchanged TTIR and TTGIR after plane selection.", flush=True)


def check(baseline, candidate, output):
    check_checkout(baseline, candidate)
    optimizer = load(candidate / MUDD_COEFFICIENTS_KERNEL)
    assert torch.version.git_version == optimizer.TORCH_REVISION and torch.version.cuda is None
    assert triton.__version__ == "3.6.0"
    verify_fused_dispatch(optimizer)
    output.mkdir(parents=True, exist_ok=True)
    with patch.object(sys, "path", [str(candidate), *sys.path]):
        capture_graphs(candidate, optimizer, output)


def verify_fused_dispatch(optimizer):
    launches = []

    class LaunchRecorder:
        def __getitem__(self, grid):
            def launch(*args, **options):
                launches.append((grid, args, options))
            return launch

    def selected(tile, warps):
        tuner = object.__new__(triton_heuristics.CachingAutotuner)
        tuner.triton_interpret = tuner.dump_launch_params = False
        tuner.inductor_meta = {}
        config = triton.Config({"XBLOCK": tile}, num_warps=warps, num_stages=1)
        tuner.launchers = [SimpleNamespace(config=config, store_cubin=False)]
        return tuner

    output, packed = SimpleNamespace(device="cuda:0"), object()
    inputs = tuple(object() for _ in range(10))
    group, small, medium, large = selected(32, 4), selected(1, 2), selected(8, 2), selected(128, 8)
    with (
        patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=1)),
        patch.object(optimizer, "mudd_gradient_and_pack", LaunchRecorder()),
        patch.object(optimizer, "mudd_gradient_pack_direct", LaunchRecorder()),
        patch.object(optimizer, "mudd_mixed_gradients", LaunchRecorder()),
        patch.object(optimizer, "mudd_banded_gradients", LaunchRecorder()),
        patch.object(optimizer, "mudd_gradient_pack_warp", LaunchRecorder()),
        patch.object(torch, "empty") as allocate,
    ):
        for first, second, stream in ((group, None, 1), (None, small, 1), (group, small, 2),
                                      (group, selected(16, 4), 1)):
            assert not optimizer.try_fused_gradients(first, second, 16384, output, packed, inputs, stream)
        assert not launches
        assert optimizer.try_fused_gradients(group, small, 16384, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(small, group, 16384, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(group, group, 16384, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(large, small, 32768, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(medium, large, 49152, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(large, large, 49152, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(small, medium, 16384, output, packed, inputs, 1)
        assert optimizer.try_fused_gradients(medium, small, 16384, output, packed, inputs, 1)
        allocate.assert_not_called()
    assert launches == [
        ((512, 10), (*inputs, output, packed, None, 32, 1), {"num_warps": 4, "num_stages": 1}),
        ((512, 10), (*inputs, output, packed, None, 1, 32), {"num_warps": 4, "num_stages": 1}),
        ((512, 10), (*inputs, output, packed, None, 32, 10, 0), {"num_warps": 4, "num_stages": 1}),
        ((256, 17), (*inputs, output, packed, None), {"num_warps": 8, "num_stages": 1}),
        ((384, 10), (*inputs, output, packed, None, 8, 128), {"num_warps": 8, "num_stages": 1}),
        ((384, 10), (*inputs, output, packed, None), {"num_warps": 8, "num_stages": 1}),
        ((2048, 10), (*inputs, output, packed, None, 1, 8), {"num_warps": 2, "num_stages": 1}),
        ((2048, 10), (*inputs, output, packed, None, 8, 1), {"num_warps": 2, "num_stages": 1}),
    ]

    launches.clear()
    scaled = object()
    with (
        patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=1)),
        patch.object(optimizer, "mudd_gradient_pack_direct", LaunchRecorder()),
        patch.object(optimizer, "mudd_mixed_gradients", LaunchRecorder()),
        patch.object(optimizer, "mudd_banded_gradients", LaunchRecorder()),
        patch.object(torch, "empty", return_value=scaled) as allocate,
    ):
        for first, second, rows, stream in ((group, None, 16384, 1), (None, small, 16384, 1),
                                            (group, small, 16384, 2), (group, small, 32768, 1)):
            assert optimizer.try_fused_full_gradients(first, second, rows, output, packed, inputs, stream) is None
        assert not launches
        allocate.assert_not_called()
        assert optimizer.try_fused_full_gradients(group, small, 16384, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(small, small, 40960, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(small, medium, 16384, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(group, large, 16384, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(large, medium, 40960, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(small, large, 16384, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(large, small, 16384, output, packed, inputs, 1) is scaled
        assert optimizer.try_fused_full_gradients(large, small, 40960, output, packed, inputs, 1) is scaled
    assert launches == [
        ((512, 10), (*inputs, output, packed, scaled, 32, 1), {"num_warps": 4, "num_stages": 1}),
        ((10240, 10), (*inputs, output, packed, scaled, 4, 10, 0), {"num_warps": 2, "num_stages": 1}),
        ((2048, 10), (*inputs, output, packed, scaled, 1, 8), {"num_warps": 2, "num_stages": 1}),
        ((128, 10), (*inputs, output, packed, scaled, 32, 128), {"num_warps": 8, "num_stages": 1}),
        ((320, 10), (*inputs, output, packed, scaled, 128, 8), {"num_warps": 8, "num_stages": 1}),
        ((256, 10), (*inputs, output, packed, scaled, 1, 128), {"num_warps": 8, "num_stages": 1}),
        ((128, 17), (*inputs, output, packed, scaled), {"num_warps": 8, "num_stages": 1}),
        ((320, 17), (*inputs, output, packed, scaled), {"num_warps": 8, "num_stages": 1}),
    ]


def capture_graphs(candidate, optimizer, output):
    load_attention(candidate)
    with patch.object(torch.cuda, "_compile_kernel", return_value=CompileOnlyCE()):
        from track_1_short.model.gpt import GPT, ForwardScheduleConfig
    from track_1_short.ngram_table import NGRAM_DIM
    from track_1_short.sampled_softmax import CANDIDATES_BY_STAGE, SampledLoss
    from track_1_short.data import cu_seqlens_rows
    from track_1_short.config import BLOCK_SIZE, Hyperparameters, TRAINING_STAGES, VIRTUAL_SEQ_CAP

    device = torch.device("cpu")
    with torch.device(device):
        model = GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
                    max_seq_len=Hyperparameters().val_batch_size // 8, ngram_dim=NGRAM_DIM, world_size=8, device=device)
        model.cast_matrix_weights_bf16()

    def step(*args):
        return model(*args).sum()

    for index, stage in enumerate(TRAINING_STAGES):
        rows = stage.batch_size // 8
        for classes in dict.fromkeys(CANDIDATES_BY_STAGE.get(index, (0,))):
            predictions = len(stage.mtp_weights_start)
            with torch.device(device):
                sample = (SampledLoss(torch.empty(classes, 768, dtype=torch.float8_e4m3fn),
                                      torch.empty(768, classes, dtype=torch.float8_e4m3fn),
                                      torch.empty(rows, dtype=torch.int64), torch.empty(rows, dtype=torch.int64),
                                      torch.empty(model.vocab_size, dtype=torch.int32)) if classes else None)
                short, long = (w * BLOCK_SIZE for w in stage.window_sizes)
                schedule = ForwardScheduleConfig(torch.empty(predictions), torch.empty(1), short, long,
                                                min(stage.train_max_seq_len, VIRTUAL_SEQ_CAP), sample)
                args = (torch.empty(rows, dtype=torch.int32), torch.empty(rows, dtype=torch.int64),
                        torch.empty(cu_seqlens_rows(rows), dtype=torch.int32), torch.empty(2 * rows, dtype=torch.int32),
                        schedule, torch.empty(2 * rows, NGRAM_DIM, dtype=torch.bfloat16, requires_grad=True),
                        torch.empty_like(model.value_embeds, dtype=torch.float16))
            label = f"whole-mudd-{rows}-{classes}-{predictions}"
            target = output / f"{label}.py"
            capture(step, target, args, user_triton=True)
            raw = target.read_text()
            separate, duplicates = optimizer.patch_wrapper(raw, retain_residual=False, group_gradients=False)
            retained, retained_count = optimizer.patch_wrapper(raw, group_gradients=False)
            rewritten, count = optimizer.patch_wrapper(raw)
            assert retained_count == 1 and count == 2
            assert len(raw.splitlines()) == len(rewritten.splitlines())
            (output / f"{label}-rewritten.py").write_text(rewritten)
            check_backend(optimizer, raw, rewritten)
            if (rows, classes, predictions) == (16384, 10240, 3):
                assert duplicates == 1
                verify_rewrite(optimizer, raw, separate, output, label)
            else:
                assert duplicates == 0 and raw == separate
            verify_residual(optimizer, separate, retained, output / "residual", label)
            verify_gradient_groups(optimizer, retained, rewritten, output / "gradients", label, bool(classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.baseline.resolve(), args.candidate.resolve(), args.output.resolve())
