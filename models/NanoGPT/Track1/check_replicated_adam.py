"""Compile and replay the two replicated Adam update kernels for Hopper."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from install_candidate import REPLICATED_ADAM, REPLICATED_SPAN_KERNEL
from qkv_ptx_replay import PREDICATES, address, operands_in, value
from tail_norm_replay import instructions
from validation_ce_replay import bits, float32, halves, integer_op, selected
from verify_candidate import check_checkout

BANK = 1 << 40
PARAMS = tuple(i * BANK for i in range(1, 7))
FP_OPS = {
    "add.f32", "sub.f32", "mul.f32", "div.full.f32", "sqrt.approx.ftz.f32", "fma.rn.f32",
    "cvt.f32.bf16", "cvt.rn.bf16.f32",
}


def term(op, *operands):
    if op in {"mul.f32", "add.f32", "fma.rn.f32"}:
        a, b, *rest = operands
        operands = (*sorted((a, b), key=repr), *rest)
    return (op, *operands)


def write(memory, pointer, count, word):
    assert pointer % count == 0
    if isinstance(word, int):
        for i in range(count):
            memory[pointer + i] = (word >> (8 * i)) & 255
    else:
        memory[pointer] = word


def read(memory, pointer, count):
    first = memory[pointer]
    if not isinstance(first, int):
        return first
    return sum(memory[pointer + i] << (8 * i) for i in range(count))


def input_word(pointer, count, size, segments, spans, active, bf16, tiled):
    bank, offset = divmod(pointer, BANK)
    if bank <= 4:
        expected = 2 if bf16 and bank in (1, 2) else 4
        assert count == expected and offset % count == 0 and offset // count < size
        return ("load", bank, offset // count)
    assert count == 4 and offset % 4 == 0
    index = offset // count
    if bank == 5:
        return spans[index // 3][index % 3] if tiled else segments[index]
    segment, column = divmod(index, 8)
    if column == 7:
        return active[segment]
    return ("scalar", segment, column)


def replay(ptx, blocks, size, segments, spans, active, bf16, tiled):
    code, _ = instructions(ptx)
    outputs = {}
    for block in blocks:
        lanes = [{"%ctaid.x": block, "%tid.x": lane, "global_smem": 0} for lane in range(128)]
        shared = {}
        pending = {}
        for pred, op, args in code:
            if op == "ret":
                assert not pending
                break
            if op == "bar.sync":
                shared.update(pending)
                pending.clear()
                continue
            dst, *src = args
            for regs in lanes:
                if not selected(regs, pred):
                    continue
                if op.startswith("ld.param"):
                    i = int(src[0][:-1].rsplit("_", 1)[1])
                    regs[dst] = PARAMS[i] if op == "ld.param.b64" else size
                elif op.startswith(("ld.global.", "st.global.", "ld.shared.", "st.shared.")):
                    load = op.startswith("ld.")
                    global_ = ".global." in op
                    group, addr = (dst, src[0]) if load else (src[0], dst)
                    names = operands_in(group) if group.startswith("{") else [group]
                    count = int(op.split(".")[-1][1:]) // 8
                    start = address(regs, addr)
                    for i, name in enumerate(names):
                        ptr = start + i * count
                        if load and global_:
                            regs[name] = input_word(ptr, count, size, segments, spans, active, bf16, tiled)
                        elif load:
                            regs[name] = read(shared, ptr, count)
                        elif global_:
                            bank, offset = divmod(ptr, BANK)
                            assert (bank, count) in {(1, 2 if bf16 else 4), (3, 4), (4, 4)}
                            assert offset % count == 0 and offset // count < size
                            key = (bank, offset // count)
                            assert key not in outputs
                            outputs[key] = regs[name]
                        else:
                            write(pending, ptr, count, regs[name])
                elif op.startswith("mov."):
                    word = bits(regs, src[0])
                    if dst.startswith("{"):
                        lo, hi = operands_in(dst)
                        regs[lo], regs[hi] = halves(word)
                    else:
                        regs[dst] = word
                elif op in FP_OPS:
                    regs[dst] = term(op, *(bits(regs, a) for a in src))
                elif op.startswith("setp."):
                    _, cmp, ty = op.split(".")
                    a, b = (bits(regs, arg) for arg in src)
                    if ty == "f32":
                        reg = (PREDICATES[cmp](float32(a), float32(b))
                               if isinstance(a, int) and isinstance(b, int)
                               else term(op, a, b))
                    else:
                        width = int(ty[1:])
                        cast = (lambda x: x if x < (1 << (width - 1)) else x - (1 << width)) if ty[0] == "s" else int
                        mask = (1 << width) - 1
                        reg = PREDICATES[cmp](cast(a & mask), cast(b & mask))
                    regs[dst] = reg
                elif op.startswith("selp."):
                    a, b, p = (bits(regs, arg) for arg in src)
                    regs[dst] = (a if p else b) if isinstance(p, bool) else term(op, a, b, p)
                elif op == "and.pred":
                    a, b = (bits(regs, arg) for arg in src)
                    assert isinstance(a, bool) and isinstance(b, bool)
                    regs[dst] = a and b
                elif op == "prmt.b32":
                    a, b, k = (value(regs, arg) for arg in src)
                    indexes = [(k >> (4 * i)) & 15 for i in range(4)]
                    assert all(i < 8 for i in indexes)
                    both = a | b << 32
                    regs[dst] = sum(((both >> (8 * j)) & 255) << (8 * i)
                                    for i, j in enumerate(indexes))
                elif op == "mul.wide.s32":
                    a, b = (value(regs, arg) for arg in src)
                    regs[dst] = a * b
                else:
                    regs[dst] = integer_op(op, [value(regs, arg) for arg in src])
        else:
            raise ValueError("Expected a return after the Adam writes")
    return outputs


def expected_store(index, segment, bf16):
    b1, a1, b2, a2, eps, step, decay = (("scalar", segment, i) for i in range(7))
    p, g, m, v = (("load", bank, index) for bank in range(1, 5))
    if bf16:
        p, g = (term("cvt.f32.bf16", x) for x in (p, g))
    first = term("fma.rn.f32", a1, g, term("mul.f32", m, b1))
    second = term("fma.rn.f32", term("mul.f32", a2, g), g, term("mul.f32", v, b2))
    denom = term("add.f32", term("sqrt.approx.ftz.f32", second), eps)
    u = term("mul.f32", term("div.full.f32", first, denom), step)
    positive = term("setp.gt.f32", term("mul.f32", u, p), 0)
    keep = term("selp.f32", 0x3f800000, 0, positive)
    weight = term("sub.f32", p, term("fma.rn.f32", term("mul.f32", decay, p), keep, u))
    if bf16:
        weight = term("cvt.rn.bf16.f32", weight)
    return weight, first, second


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(baseline, candidate, out):
    check_checkout(baseline, candidate)
    if triton.__version__ != "3.6.0":
        raise RuntimeError("Use Triton 3.6.0 to reproduce the Hopper artifacts")
    original = load_module("replicated_adam_baseline", baseline /
                           REPLICATED_ADAM.parent / "kernels" / REPLICATED_ADAM.name)
    optimized = load_module("replicated_adam_spans", candidate / REPLICATED_SPAN_KERNEL)
    out.mkdir(parents=True, exist_ok=True)
    dump = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    for fmt in ("bf16", "fp32"):
        ptx = []
        for tag, entry, tile in (
            ("reference", original._fused_adam_kernel, original.BLOCK),
            ("candidate", optimized._span_adam, optimized.BLOCK),
        ):
            selector = "SEG" if tag == "reference" else "PLAN"
            fields = {"P": "*" + fmt, "G": "*" + fmt, "EA": "*fp32", "ES": "*fp32", selector: "*i32", "SC": "*fp32"}
            if tag == "reference":
                fields["n_elements"] = "i32"
            fields["BLOCK"] = "constexpr"
            attrs = {(i,): [["tt.divisibility", 16]] for i in range(6)}
            code = triton.compile(ASTSource(entry, fields, {"BLOCK": tile}, attrs),
                                  target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
            name = out / f"replicated-adam-{fmt}-{tag}"
            name.with_suffix(".ptx").write_text(code.asm["ptx"])
            name.with_suffix(".ttgir").write_text(code.asm["ttgir"])
            cubin = name.with_suffix(".cubin")
            cubin.write_bytes(code.asm["cubin"])
            resources = subprocess.check_output([dump, "-res-usage", cubin], text=True).splitlines()[-1].strip()
            loads = sum("ld.global" in line for line in code.asm["ptx"].splitlines())
            print(f"{fmt} {tag}, tile {tile}, {loads} static global load instructions, {resources}", flush=True)
            ptx.append(code.asm["ptx"])
        for layout in ([(3, 13), (1, 1300), (4, 256), (0, 0), (6, 933), (2, 7)], [(1, 1024), (3, 1536)]):
            segments = [segment for segment, length in layout for _ in range(length)]
            spans = optimized.span_tiles(layout)
            size = len(segments)
            for active in ([0x3f800000] * 7, [0, 0x80000000, 0, 0x3f800000, 0x3f800000, 0, 0]):
                expected = {(bank, i): word for i, segment in enumerate(segments)
                            if float32(active[segment]) != 0
                            for bank, word in zip((1, 3, 4), expected_store(i, segment, fmt == "bf16"), strict=True)}
                ref = replay(ptx[0], range(triton.cdiv(size, original.BLOCK)), size, segments, spans,
                             active, fmt == "bf16", False)
                opt = replay(ptx[1], range(len(spans)), size, segments, spans, active, fmt == "bf16", True)
                for stores in (ref, opt):
                    assert stores.keys() == expected.keys()
                    for key, word in expected.items():
                        assert stores[key] == word, (fmt, key, stores[key], word)
        print(f"{fmt}: matched all moment and parameter stores, including inactive and partial tiles", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
