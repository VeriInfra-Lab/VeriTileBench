"""Replay the grouped norm-restoration PTX at both ends of each bank."""
from qkv_ptx_replay import address, INT32_OPS, operands_in, parse, PREDICATES, signed32, value
from rope_ptx_replay import atom


def instructions(ptx):
    code, labels = [], {}
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        if line.startswith(".section"):
            break
        if line.startswith("$") and line.endswith(":"):
            labels[line[:-1]] = len(code)
        if line and not line.startswith(".") and line.endswith(";"):
            code.extend(parse([line]))
    return code, labels


def replay(code, labels, cta, pointers, terms):
    output = {}
    for tid in range(256):
        regs = {"%tid.x": tid, "%ctaid.x": cta}
        pc = 0
        for _ in code:
            pred, opcode, operands = code[pc]
            pc += 1
            if pred and not regs[pred]:
                continue
            if opcode == "ret":
                break
            dst, *src = operands
            if opcode.startswith("bra"):
                pc = labels[dst]
            elif opcode in INT32_OPS:
                regs[dst] = INT32_OPS[opcode](*(value(regs, arg) for arg in src)) & 0xFFFFFFFF
            elif opcode in ("mov.u32", "mov.b64"):
                regs[dst] = value(regs, src[0])
            elif opcode == "ld.param.b64":
                regs[dst] = pointers[int(src[0].removesuffix("]").rsplit("_", 1)[1])]
            elif opcode.startswith("setp."):
                _, test, kind = opcode.split(".")
                assert kind == "u32"
                regs[dst] = PREDICATES[test](*(value(regs, arg) & 0xFFFFFFFF for arg in src))
            elif opcode in ("selp.b32", "selp.b64"):
                regs[dst] = value(regs, src[0] if regs[src[2]] else src[1])
            elif opcode == "shl.b64":
                regs[dst] = value(regs, src[0]) << value(regs, src[1])
            elif opcode == "add.s64":
                regs[dst] = sum(value(regs, arg) for arg in src)
            elif opcode == "mad.wide.s32":
                index, scale, base = (value(regs, arg) for arg in src)
                regs[dst] = signed32(index) * scale + base
            elif opcode == "ld.global.b32":
                regs[operands_in(dst)[0]] = atom(terms, "ratio", address(regs, src[0]))
            elif opcode == "ld.global.v4.b32":
                start = address(regs, src[0])
                for j, reg in enumerate(operands_in(dst)):
                    regs[reg] = tuple(atom(terms, "weight", start + 4 * j + h) for h in (0, 2))
            elif opcode == "mov.b32":
                first, second = operands_in(dst)
                regs[first], regs[second] = regs[src[0]]
            elif opcode in ("cvt.f32.bf16", "mul.f32"):
                regs[dst] = atom(terms, opcode, *(value(regs, arg) for arg in src))
            elif opcode == "cvt.rn.bf16x2.f32":
                regs[dst] = tuple(atom(terms, "narrow", regs[arg]) for arg in reversed(src))
            elif opcode == "st.global.v4.b32":
                start = address(regs, dst)
                for j, reg in enumerate(operands_in(src[0])):
                    for half, datum in enumerate(regs[reg]):
                        offset = start + j * 4 + half * 2
                        assert offset not in output
                        output[offset] = datum
            else:
                raise ValueError(f"Unexpected norm restoration instruction {opcode}")
        else:
            raise ValueError("Missing norm restoration return")
    return output


def verify_tail_norm(ptx, ends, block):
    code, labels = instructions(ptx)
    pointers = tuple(i << 40 for i in range(1, 7))
    for slot, (start, end) in enumerate(zip((0, *ends), ends)):
        for cta in (start, end - 1):
            terms = {}
            actual = replay(code, labels, cta, pointers, terms)
            expected = {}
            ratio = atom(terms, "ratio", pointers[5] + slot * 4)
            for lane in range(block):
                offset = pointers[slot] + 2 * ((cta - start) * block + lane)
                weight = atom(terms, "cvt.f32.bf16", atom(terms, "weight", offset))
                expected[offset] = atom(terms, "narrow", atom(terms, "mul.f32", ratio, weight))
            assert actual == expected, (slot, cta)
