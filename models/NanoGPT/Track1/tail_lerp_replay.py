"""Replay both tail lerp kernels, keeping the ordered FP terms and addresses."""
from collections import Counter

from qkv_ptx_replay import address, INT32_OPS, operands_in, parse, PREDICATES, signed32, value
from rope_ptx_replay import atom, pair


WEIGHT, FIRST, SECOND = (1 << 40, 2 << 40, 3 << 40)
RATE_FIRST, RATE_SECOND = ("rate-first", "rate-second")
WORDS = {"b16": 2, "b32": 4}


def replay(code, block, parameters, terms):
    threads = [{"%tid.x": i, "%ctaid.x": block, "global_smem": 0} for i in range(256)]
    stores, scratch = {}, {}
    reads = Counter()
    for predicate, op, operands in code:
        if op == "ret":
            return stores, reads
        if op == "bar.sync":
            assert predicate is None and operands == ["0"]
            continue
        dst, *args = operands
        for regs in threads:
            if predicate and not regs[predicate]:
                continue
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(value(regs, arg) for arg in args)) & 0xFFFFFFFF
            elif op == "neg.s32":
                regs[dst] = -value(regs, args[0]) & 0xFFFFFFFF
            elif op == "add.s64":
                regs[dst] = sum(value(regs, arg) for arg in args)
            elif op in ("mad.wide.s32", "mul.wide.s32"):
                index, scale, *base = (value(regs, arg) for arg in args)
                regs[dst] = signed32(index) * scale + sum(base)
            elif op.startswith("setp."):
                _, test, kind = op.split(".")
                assert kind == "s32"
                regs[dst] = PREDICATES[test](*(signed32(value(regs, arg)) for arg in args))
            elif op in ("ld.param.b64", "ld.param.b32"):
                regs[dst] = parameters[int(args[0].removesuffix("]").rsplit("_", 1)[1])]
            elif op in ("mov.b32", "mov.u32"):
                src = value(regs, args[0])
                if dst.startswith("{"):
                    left, right = operands_in(dst)
                    regs[left], regs[right] = pair(src)
                else:
                    regs[dst] = src
            elif op.startswith("ld.global.") or op == "ld.shared.v4.b32":
                start = address(regs, args[0])
                width = WORDS[op.rsplit(".", 1)[1]]
                for word, target in enumerate(operands_in(dst)):
                    offset = start + word * width
                    if op == "ld.shared.v4.b32":
                        datum = scratch[offset]
                    else:
                        reads[offset >> 40] += width
                        if offset < FIRST:
                            datum = tuple(atom(terms, "weight", offset + half) for half in range(0, width, 2))
                            if width == 2:
                                datum = datum[0]
                        else:
                            assert width == 4
                            datum = atom(terms, "average", offset)
                    regs[target] = datum
            elif op in ("cvt.f32.bf16", "sub.f32", "fma.rn.f32"):
                regs[dst] = atom(terms, op, *(value(regs, arg) for arg in args))
            elif op in ("st.global.v4.b32", "st.shared.v4.b32"):
                start = address(regs, dst)
                target = stores if op == "st.global.v4.b32" else scratch
                for word, reg in enumerate(operands_in(args[0])):
                    offset = start + word * 4
                    if target is stores:
                        assert offset not in target
                    target[offset] = regs[reg]
            else:
                raise ValueError(f"Unexpected tail lerp instruction {op}")
    raise ValueError("Missing tail lerp return")


def verify_pair(reference, merged):
    reference, merged = list(parse(reference)), list(parse(merged))
    # VO and MLP shards at eight ranks, including their last block.
    for elements in (2 * 768**2, 3 * 2816 * 768):
        for block in (0, elements // 4096 - 1):
            terms = {}
            first, a = replay(reference, block, (WEIGHT, FIRST, elements, RATE_FIRST), terms)
            second, b = replay(reference, block, (WEIGHT, SECOND, elements, RATE_SECOND), terms)
            actual, fused_reads = replay(merged, block, (WEIGHT, FIRST, SECOND, RATE_FIRST, RATE_SECOND), terms)
            expected = {}
            for i in range(block * 4096, (block + 1) * 4096):
                weight = atom(terms, "cvt.f32.bf16", atom(terms, "weight", WEIGHT + i * 2))
                for base, rate in ((FIRST, RATE_FIRST), (SECOND, RATE_SECOND)):
                    address = base + i * 4
                    average = atom(terms, "average", address)
                    delta = atom(terms, "sub.f32", weight, average)
                    expected[address] = atom(terms, "fma.rn.f32", rate, delta, average)
            assert actual == (first | second) == expected, (elements, block)
            assert a + b == {1: 16384, 2: 16384, 3: 16384}, (elements, block, a, b)
            assert fused_reads == {1: 8192, 2: 16384, 3: 16384}, (elements, block, fused_reads)
