"""Replay the grouped QKV scan's control flow and vector load addresses."""

import operator

from qkv_ptx_replay import PREDICATES, WORD32, address, operands_in, parse


# Synthetic, disjoint base addresses for the two BF16 banks and three
# FP32 scratch buffers. The ABI may contain two trailing unused parameters.
BASES = tuple((i + 1) << 36 for i in range(5))
TILE_BYTES = 4096
BANK_BYTES = 2 * 1179648

WORD_OPS = {
    "add": operator.add,
    "and": operator.and_,
    "mul": operator.mul,
    "shl": operator.lshift,
    "shr": operator.rshift,
    "sub": operator.sub,
}


def _signed(bits, width):
    bits %= 1 << width
    return bits - (1 << width) if bits >> (width - 1) else bits


def _reads(program, labels, slot, part, thread):
    values = {"%ctaid.x": slot, "%ctaid.y": part, "%tid.x": thread}
    reads = []
    pc = 0

    def get(name):
        if name.startswith("%"):
            return values[name]
        if name.startswith("0f"):
            return int(name[2:], 16)
        return int(name.removesuffix("U"), 0)

    for _ in range(2048):
        predicate, opcode, args = program[pc]
        pc += 1
        if predicate is not None and get(predicate.removeprefix("!")) == predicate.startswith("!"):
            continue
        if opcode == "shfl.sync.bfly.b32":
            return reads
        dst, *src = args
        operation, *types = opcode.split(".")
        if operation == "bra":
            pc = labels[dst]
        elif opcode == "ld.param.b64":
            values[dst] = BASES[int(src[0].removesuffix("]").rsplit("_", 1)[1])]
        elif opcode == "ld.global.v4.b32":
            reads.append(address(values, src[0]))
            # Weight words can influence only the FP accumulator. A dependence
            # on them in an address or branch fails the integer evaluator.
            for reg in operands_in(dst):
                values[reg] = None
        elif opcode == "max.f32":
            values[dst] = None
        elif operation == "setp":
            cmp, ty = types
            a, b = (get(x) for x in src)
            width = int(ty[1:])
            if ty.startswith("s"):
                a, b = _signed(a, width), _signed(b, width)
            else:
                mask = (1 << width) - 1
                a, b = a & mask, b & mask
            values[dst] = PREDICATES[cmp](a, b)
        elif operation == "selp":
            values[dst] = get(src[0] if get(src[2]) else src[1])
        elif operation == "mov" and dst.startswith("{"):
            assert get(src[0]) is None
            for reg in operands_in(dst):
                values[reg] = None
        elif operation == "mov":
            values[dst] = get(src[0])
        elif operation == "cvt":
            ty, source_ty = types
            a = get(src[0])
            if a is None:
                values[dst] = None
            else:
                source_width = int(source_ty[1:])
                if source_ty.startswith("s"):
                    a = _signed(a, source_width)
                values[dst] = a % (1 << int(ty[1:])) if ty.startswith("u") else a
        elif opcode == "and.pred":
            values[dst] = get(src[0]) and get(src[1])
        elif operation in WORD_OPS or operation == "mad":
            a, b, *rest = map(get, src)
            ty = types[-1]
            width = int(ty[1:])
            mask = (1 << width) - 1
            if a is None or b is None:
                assert opcode == "and.b32"
                values[dst] = None
                continue
            if len(types) == 2 and types[0] == "wide":
                a = _signed(a, width) if ty.startswith("s") else a & mask
                b = _signed(b, width) if ty.startswith("s") else b & mask
                values[dst] = a * b + (rest[0] if rest else 0)
            elif len(types) == 2 and types[0] == "hi":
                a = _signed(a, width) if ty.startswith("s") else a & mask
                b = _signed(b, width) if ty.startswith("s") else b & mask
                values[dst] = ((a * b) >> width) & mask
            else:
                if operation == "shr":
                    a = _signed(a, width) if ty.startswith("s") else a & mask
                result = a * b + rest[0] if operation == "mad" else WORD_OPS[operation](a, b)
                values[dst] = result & mask
        elif opcode == "max.s32":
            values[dst] = max(_signed(get(x), 32) for x in src) & WORD32
        else:
            raise AssertionError(f"Unexpected scan address instruction {opcode}")
    raise AssertionError("Scan address replay exceeded its instruction bound")


def verify_grouped_scan(ptx):
    # Imported lazily because the offline verifier calls this replay.
    from verify_candidate import executable_ptx

    labels = {}
    program = list(parse(executable_ptx(ptx, labels=labels)))
    assert [args[2] for _, op, args in program if op == "shfl.sync.bfly.b32"] == [
        "16", "8", "4", "2", "1", "4", "2", "1",
    ]
    assert sum(op == "bar.sync" for _, op, _ in program) == 2
    for slot in range(7):
        first_tiles = 576 if slot >= 5 else 288
        second_tiles = 144 if 3 <= slot < 5 else 288
        for part in range(32):
            tiles = (
                [(0, tile + (tile // 24 * 24 if slot < 5 else 0))
                 for tile in range(part, first_tiles, 32)] +
                [(1, tile + (tile // 24 * 24 if 3 <= slot < 5 else 0))
                 for tile in range(part, second_tiles, 32)]
            )
            for thread in (0, 255):
                expected = [
                    BASES[bank] + slot * BANK_BYTES + tile * TILE_BYTES + thread * 16
                    for bank, tile in tiles
                ]
                assert _reads(program, labels, slot, part, thread) == expected, (slot, part, thread)
