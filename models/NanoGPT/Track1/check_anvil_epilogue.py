"""Check the shared ANVIL reductions and pointwise fused Hopper epilogue."""
import argparse
import ast
from pathlib import Path

import torch
from torch._inductor.runtime import triton_heuristics
import triton
from triton.compiler import ASTSource
from unittest.mock import patch

from check_replicated_adam import load_module, term
from check_value_adam_inductor import HOPPER, TORCH_REVISION, capture
from install_candidate import ANVIL, ANVIL_EPILOGUE_KERNEL
from qkv_ptx_replay import address, operands_in, PREDICATES
from rope_ptx_replay import signed
from tail_norm_replay import instructions
from validation_ce_replay import Pair, bits, integer_op, selected
from verify_candidate import check_checkout, load_sparse_method


BANK = 1 << 40
HALF = {'value', 'scaled', 'upper', 'lower'}
ENTRIES = {
    'eq': ('value', 'power', 'energy', 'pre', 'post', 'scaled'),
    'decay': ('upper', 'lower', 'gate', 'wd', 'lr', 'scaled', 'upper', 'lower'),
    'fused': ('value', 'power', 'energy', 'pre', 'post', 'lr', 'upper', 'lower', 'gate', 'wd', 'upper', 'lower'),
}
OPS = {
    'add.f32', 'sub.f32', 'mul.f32', 'neg.f32', 'abs.f32', 'fma.rn.f32', 'div.full.f32', 'cvt.f32.bf16',
    'cvt.rn.bf16.f32', 'cvt.u32.u16', 'sqrt.rn.f32', 'rsqrt.approx.ftz.f32',
    'setp.gt.f32', 'setp.nan.f32', 'setp.ge.f32', 'selp.f32',
}


def split(x):
    if isinstance(x, Pair):
        return x.lo, x.hi
    if isinstance(x, int):
        return x & 0xffff, x >> 16 & 0xffff
    return term('low', x), term('high', x)


def run(code, phase, block, warps, cta, intermediate=None, *, half=HALF, intermediate_role="scaled", cta_y=0):
    refs = ENTRIES[phase] if isinstance(phase, str) else phase
    out = {}
    shared, pending = {}, {}
    lanes = [{'%ctaid.x': cta, '%ctaid.y': cta_y, '%tid.x': lane, 'global_smem': 0} for lane in range(warps*32)]

    def read(ptr, width):
        bank, loc = divmod(ptr, BANK)
        role = refs[bank-1]
        if role in half and width == 4:
            return Pair(read(ptr, 2), read(ptr+2, 2))
        assert width == (2 if role in half else 4)
        ix = loc//width
        return intermediate[(role, ix)] if intermediate is not None and role == intermediate_role else (role, ix)
    for pc, (pred, op, args) in enumerate(code):
        if op in ('add.rn.f32', 'sub.rn.f32', 'mul.rn.f32'):
            op = op.replace('.rn', '')
        if op == 'ret':
            assert not pending
            return out
        if op.startswith('createpolicy'):
            continue
        if op == 'bar.sync':
            assert pred is None and args == ["0"]
            shared.update(pending)
            pending.clear()
            continue
        dst, *src = args
        for regs in lanes:
            if not selected(regs, pred):
                continue

            def vals(): return [bits(regs, a) for a in src]
            if op == 'ld.param.b64':
                regs[dst] = (int(src[0][:-1].rsplit('_', 1)[1])+1)*BANK
            elif op.startswith(('ld.shared.', 'st.shared.')):
                assert op.endswith('.b32')
                load = op.startswith('ld')
                group, where = (dst, src[0]) if load else (src[0], dst)
                ptr = address(regs, where)
                names = operands_in(group) if group.startswith('{') else [group]
                for i, name in enumerate(names):
                    key = ptr + i * 4
                    if load:
                        assert key not in pending
                        regs[name] = shared[key]
                    else:
                        assert pending.setdefault(key, regs[name]) == regs[name]
            elif op.startswith(('ld.global.', 'st.global.')):
                assert op.endswith('.b32')
                load = op.startswith('ld')
                group, where = (dst, src[0]) if load else (src[0], dst)
                ptr = address(regs, where)
                entries = operands_in(group) if group.startswith("{") else [group]
                for i, name in enumerate(entries):
                    addr = ptr+i*4
                    if load:
                        regs[name] = read(addr, 4)
                    else:
                        bank, off = divmod(addr, BANK)
                        role = refs[bank-1]
                        if role not in half:
                            key = (role, off//4)
                            assert key not in out
                            out[key] = bits(regs, name)
                            continue
                        for j, word in enumerate(split(regs[name])):
                            key = (role, off//2+j)
                            assert key not in out
                            out[key] = word
            elif op.startswith('mov.'):
                v, = vals()
                if dst.startswith('{'):
                    for reg, word in zip(operands_in(dst), split(v), strict=True):
                        if reg != '_':
                            regs[reg] = word
                else:
                    regs[dst] = v
            elif op == 'cvt.rn.bf16x2.f32':
                regs[dst] = Pair(*(term('cvt.rn.bf16.f32', v) for v in reversed(vals())))
            elif op == 'prmt.b32':
                a, b, k = vals()
                assert k == 0x7632
                regs[dst] = Pair(split(a)[1], split(b)[1])
            elif op == 'and.pred':
                a, b = vals()
                regs[dst] = a and b
            elif op.startswith('setp.') and not op.endswith('.f32'):
                _, test, kind = op.split('.')
                a, b = vals()
                if kind == 's32':
                    a, b = signed(a, 32), signed(b, 32)
                else:
                    assert kind in ('u32', 'b32')
                regs[dst] = PREDICATES[test](a, b)
            elif op.startswith('selp.') and isinstance(bits(regs, src[2]), bool):
                a, b, p = vals()
                regs[dst] = a if p else b
            elif op in OPS:
                regs[dst] = term(op, *vals())
            elif op == 'shr.u32' and isinstance(bits(regs, src[0]), Pair):
                assert bits(regs, src[1]) == 16
                regs[dst] = split(bits(regs, src[0]))[1]
            else:
                a = vals()
                try:
                    regs[dst] = integer_op(op, a) if all(isinstance(x, int) for x in a) else term(op, *a)
                except Exception as e:
                    raise ValueError((pc, op, args, a)) from e
    raise ValueError('missing return')


def expected_equalized(shape, index):
    _, rows, columns = shape
    wide = rows < columns
    matrix = index // (rows * columns)
    lane = matrix * columns + index % columns if wide else index // columns
    length = 0x43800000 if wide else 0x44400000
    floor = 0x2edbe6ff

    def clamp(x):
        return term("selp.f32", x, term("selp.f32", x, floor, term("setp.nan.f32", x, x)),
                    term("setp.gt.f32", x, floor))

    mean = term("div.full.f32", ("power", lane), length)
    energy = term("fma.rn.f32", term("sub.f32", mean, ("energy", lane)), 0x3dcccccd, ("energy", lane))
    gain = term("rsqrt.approx.ftz.f32", clamp(energy))
    pre = term("sqrt.rn.f32", term("mul.f32", ("pre", matrix), length))
    post = term("sqrt.rn.f32", ("post", matrix))
    ratio = term("div.full.f32", pre, clamp(post))
    gradient = term("mul.f32", term("cvt.f32.bf16", ("value", index)), term("mul.f32", gain, ratio))
    return term("cvt.rn.bf16.f32", gradient)


def expected_weight(shape, index):
    _, rows, columns = shape
    exact = term("or.b32", term("shl.b32", term("cvt.u32.u16", ("upper", index)), 16),
                 term("cvt.u32.u16", ("lower", index)))
    active = term("selp.f32", 0x3f800000, 0, term("setp.ge.f32", term("mul.f32", ("gate", index), exact), 0))
    lr = ("lr", index // (rows * columns))
    decay = term("mul.f32", term("mul.f32", exact, active), ("wd", 0))
    decayed = term("fma.rn.f32", term("neg.f32", decay), lr, exact)
    grad = term("cvt.f32.bf16", expected_equalized(shape, index))
    return term("fma.rn.f32", term("neg.f32", lr), grad, decayed)


def launch_plan(wrapper):
    kernels = {}
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "async_compile.triton":
            name, source = map(ast.literal_eval, node.args[:2])
            fn, = (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
            kernels[name] = fn

    runner, = (n for n in wrapper.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    main, = (n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    available = {}
    plan = []
    for node in ast.walk(main):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            available[node.targets[0].id] = node.value.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            name = node.func.value.id
            args = tuple(available.get(ast.unparse(a), ast.unparse(a)) for a in node.args)
            plan.append((kernels[name], args))
    return plan


def artifact(plan, out, phase):
    fn, args = plan
    heuristic, jit = fn.decorator_list
    assert ast.unparse(heuristic.func) == "triton_heuristics.pointwise"
    kwargs = {k.arg: k.value for k in heuristic.keywords}
    triton_meta = kwargs.pop("triton_meta")
    meta = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(triton_meta.keys, triton_meta.values, strict=True)
            if ast.literal_eval(k) != "device"}
    kwargs = {k: ast.literal_eval(v) for k, v in kwargs.items() if k not in ("filename", "tile_hint")}
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kw: configs):
        configs = triton_heuristics.pointwise(triton_meta={"device": None}, **kwargs)
    fn.decorator_list = [jit]
    file = out.with_name(out.name + f"-{phase}-kernel.py")
    file.write_text("import triton\nimport triton.language as tl\n"
                    "from torch._inductor.runtime import triton_helpers\n"
                    "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n\n"
                    + ast.unparse(fn) + "\n")
    kernel = getattr(load_module(file.stem, file), fn.name)
    results = {}
    for config in configs:
        tile = (config.kwargs["XBLOCK"], config.num_warps)
        code = triton.compile(ASTSource(kernel, meta["signature"], config.kwargs, meta["configs"][0]),
                              target=HOPPER, options={
                                  "num_warps": config.num_warps, "num_stages": config.num_stages,
                                  "debug": kwargs["inductor_meta"]["assert_indirect_indexing"],
                                  "sanitize_overflow": False, "enable_fp_fusion": meta["enable_fp_fusion"],
        })
        file = out.with_name(f"{out.name}-{phase}-{tile[0]}-{tile[1]}")
        file.with_suffix(".ttgir").write_text(code.asm["ttgir"])
        file.with_suffix(".ptx").write_text(code.asm["ptx"])
        file.with_suffix(".cubin").write_bytes(code.asm["cubin"])
        results[tile] = instructions(code.asm["ptx"])[0]
    assert set(results) == {(1024, 4), (512, 8)}
    return results


def check_epilogue(shape, wrappers, out):
    equalizer, decay, fused = [launch_plan(ast.parse(f.read_text())) for f in wrappers]
    assert len(equalizer) == len(fused) and len(decay) == 1
    for (old, old_args), (new, new_args) in zip(equalizer[:-2] + equalizer[-1:], fused[:-2] + fused[-1:], strict=True):
        assert ast.dump(old) == ast.dump(new)
        assert old_args == new_args
    old_eq, old_decay, new = equalizer[-2], decay[0], fused[-2]
    power, pre, post = old_eq[1][1], old_eq[1][3], old_eq[1][4]
    assert old_eq[1][:-1] == ("arg0_1", power, "arg1_1", pre, post, "arg0_1")
    assert old_decay[1][:-1] == ("arg0_1", "arg1_1", "arg5_1", "arg3_1", "arg4_1", "arg2_1", "arg0_1", "arg1_1")
    assert new[1][:-1] == ("arg0_1", power, "arg1_1", pre, post, "arg6_1", "arg2_1", "arg3_1",
                          "arg4_1", "arg5_1", "arg2_1", "arg3_1")
    matrices, rows, channels = shape
    size = matrices * rows * channels
    assert old_eq[1][-1] == old_decay[1][-1] == new[1][-1] == str(size)
    phases = dict(zip(ENTRIES, (old_eq, old_decay, new), strict=True))
    programs = {phase: artifact(program, out, phase) for phase, program in phases.items()}

    for tile in programs["eq"]:
        block, warps = tile
        boundary = rows * channels // block
        for cta in sorted({0, 1, boundary - 1, boundary, size // block - 1}):
            values = run(programs["eq"][tile], "eq", block, warps, cta)
            original = run(programs["decay"][tile], "decay", block, warps, cta, values)
            candidate = run(programs["fused"][tile], "fused", block, warps, cta)
            begin = cta * block
            reference = {}
            for index in range(begin, begin + block):
                assert values[("scaled", index)] == expected_equalized(shape, index)
                raw = expected_weight(shape, index)
                reference[("upper", index)] = term("high", raw)
                reference[("lower", index)] = term("low", raw)
            assert original == reference and candidate == reference
        print(f"ANVIL {shape}, tile {block}, {warps} warps: matched both weight halves", flush=True)


def check(baseline, candidate, out):
    check_checkout(baseline, candidate)
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None or triton.__version__ != "3.6.0":
        raise RuntimeError("Use the pinned CPU build of Torch 2.10.0 and Triton 3.6.0")
    eq = load_sparse_method(baseline, ANVIL, "AnvilAndAdam", "_rail_equalizer", strip_decorators=True, torch=torch)
    decay = load_sparse_method(baseline, ANVIL, "AnvilAndAdam", "_sign_aligned_decay_update",
                               strip_decorators=True, torch=torch)
    module = load_module("installed_anvil_epilogue", candidate / ANVIL_EPILOGUE_KERNEL)
    out.mkdir(parents=True, exist_ok=True)
    offsets = {(6, 256, 768): (1, 12), (2, 768, 768): (5, 18), (3, 2816, 768): (9, 20)}
    assert module.BANKS.keys() == offsets.keys()
    for shape, red_dim in module.BANKS.items():
        matrices, rows, columns = shape
        lane_shape = (matrices, rows, 1) if red_dim == -1 else (matrices, 1, columns)
        v = torch.empty(shape, dtype=torch.bfloat16)
        energy = torch.empty(lane_shape, dtype=torch.float32)
        p = torch.empty(shape, dtype=torch.uint16)
        mantissa = torch.empty_like(p)
        staging = torch.empty(23, dtype=torch.float32)
        wd_offset, lr_offset = offsets[shape]
        wd = staging[wd_offset]
        lr = staging[lr_offset:lr_offset + matrices].view(matrices, 1, 1)
        gate = torch.empty(shape, dtype=torch.float32)
        eq_args = (v, energy, 0.9, red_dim)
        prefix = out / ("anvil-" + "-".join(map(str, shape)))
        wrappers = [prefix.with_name(f"{prefix.name}-{phase}.py") for phase in ENTRIES]
        for path, fn, inputs in zip(wrappers, (eq, decay, module._epilogue),
                                    (eq_args, (p, mantissa, v, wd, lr, gate),
                                     (*eq_args, p, mantissa, wd, lr, gate)), strict=True):
            capture(fn, path, inputs)
        check_epilogue(shape, wrappers, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
