"""Compare the Hopper candidate on full-sized boundary tiles using source tags."""
from itertools import chain

from qkv_ptx_replay import address, parse, operands_in, value, INT32_OPS, signed32, WORD32


SOURCE, POSITION, DEST = (1 << 40, 2 << 40, 3 << 40)


def replay_tile(commands, column_tile, row_tile, positions, p):
    output = {}
    for thread in range(256):
        regs = {"%tid.x": thread, "%ctaid.x": column_tile, "%ctaid.y": row_tile}
        for pred, op, params in commands:
            if op == "ret":
                break
            dst, *args = params
            if pred is not None and not regs[pred]:
                continue
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(value(regs, x) for x in args)) & WORD32
            elif op == "mad.lo.s32":
                a, b, c = (value(regs, x) for x in args)
                regs[dst] = (a * b + c) & WORD32
            elif op in ("add.s64", "add.u64"):
                regs[dst] = sum(value(regs, x) for x in args)
            elif op == "bfe.u32":
                a, b, c = (value(regs, x) for x in args)
                regs[dst] = (a >> b) & ((1 << c) - 1)
            elif op == "ld.param.b64":
                regs[dst] = (SOURCE, POSITION, DEST)[int(args[0].removesuffix("]").rsplit("_", 1)[1])]
            elif op in ("mad.wide.s32", "mul.wide.s32"):
                a, b, *extra = (value(regs, x) for x in args)
                regs[dst] = signed32(a) * signed32(b) + (extra[0] if extra else 0)
            elif op in ("mov.u16", "mov.b16", "mov.u32", "mov.b32"):
                arg = args[0]
                regs[dst] = tuple(regs[x] for x in operands_in(arg)) if arg.startswith("{") else value(regs, arg)
            elif op == "setp.gt.s32":
                regs[dst] = signed32(value(regs, args[0])) > signed32(value(regs, args[1]))
            elif op == "ld.global.v4.b32":
                off = address(regs, args[0]) - POSITION
                assert off % 16 == 0 and 0 <= off <= 4 * (len(positions) - 4)
                for k, r in enumerate(operands_in(dst)):
                    regs[r] = positions[off // 4 + k] & WORD32
            elif op == "ld.global.b16":
                off = address(regs, args[0]) - SOURCE
                assert off % 2 == 0 and 0 <= off < 2 * 768 * p
                regs[operands_in(dst)[0]] = off // 2 + 1
            elif op == "st.global.v4.b32":
                off = address(regs, dst) - DEST
                assert off % 16 == 0
                data = chain.from_iterable(regs[x] for x in operands_in(args[0]))
                for k, tag in enumerate(data):
                    index = off // 2 + k
                    assert index not in output
                    output[index] = tag
            else:
                raise ValueError((op, params))
    return output


def verify_densify_tiles(instructions, p):
    code = list(parse(instructions))
    for sort_candidates in (False, True):
        ids = [(137 + j * 20011) % 50304 for j in range(p)]
        if sort_candidates:
            ids.sort()
        positions = [-1] * 50304
        for i, c in enumerate(ids):
            positions[c] = i
        for rm, cn in ((0, 0), (0, 392), (11, 0), (11, 392)):
            out = replay_tile(code, cn, rm, positions, p)
            assert len(out) == 64 * 128
            for r in range(rm * 64, (rm + 1) * 64):
                for c in range(cn * 128, (cn + 1) * 128):
                    offset = r * 50304 + c
                    pos = positions[c]
                    expected = r * p + pos + 1 if pos >= 0 else 0
                    assert out[offset] == expected, (p, r, c, out[offset], expected)
