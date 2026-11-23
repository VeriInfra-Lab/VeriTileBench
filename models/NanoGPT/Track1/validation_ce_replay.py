"""Check the evaluation softmax prologue and reduction at the CTA edges.

The replay carries exact symbolic FP expression trees. It checks the first
iteration and then checks an iteration with arbitrary loop-carried state.
The epilogue uses the resulting maximum and sum from each reduction slot.
"""
from dataclasses import dataclass
import math
from struct import pack, unpack
from qkv_ptx_replay import address, INT32_OPS, operands_in, PREDICATES, value
from rope_ptx_replay import signed
from tail_norm_replay import instructions

FP = {"add.f32", "sub.f32", "mul.f32", "fma.rn.f32", "fma.rn.ftz.f32", "fma.rm.ftz.f32",
      "neg.f32", "ex2.approx.f32", "ex2.approx.ftz.f32", "div.full.f32", "cvt.ftz.sat.f32.f32"}
NegInf = 0xff800000
WIDTH = 50304
ROWS = 32768
SLABS = 8
BANK = 1 << 40
SCHEDULES = {
    (1, 1024, 8): (256, [16, 8, 4, 2, 1, 128, 64, 32]),
    (1, 512, 8): (256, [16, 8, 4, 2, 1, 128, 64, 32]),
    (16, 32, 4): (8, [1, 4, 2]),
    (64, 4, 8): (4, [2, 1]),
    (64, 64, 16): (8, [4, 2, 1]),
    (8, 512, 16): (64, [2, 1, 32, 16, 8, 4]),
    (8, 256, 16): (64, [2, 1, 32, 16, 8, 4]),
}


def input_id(row):
    return 1 + 7 * row % (WIDTH - 1)


@dataclass(frozen=True)
class Term:
    id: int


@dataclass(frozen=True)
class Pair:
    lo: object
    hi: object


def halves(word):
    return (word.lo, word.hi) if isinstance(word, Pair) else (word & 0xffff, word >> 16 & 0xffff)


def bits(regs, operand):
    if operand.startswith("0f"):
        return int(operand[2:], 16)
    if operand.startswith("{"):
        return Pair(*(bits(regs, a) for a in operands_in(operand)))
    return regs.get(operand) if operand.startswith("%") else value(regs, operand)


def selected(regs, pred):
    return pred is None or regs[pred.removeprefix("!")] != pred.startswith("!")


def float32(word):
    return unpack('<f', pack('<I', word))[0]


def integer_op(op, args):
    if op in INT32_OPS:
        return INT32_OPS[op](*args) & 0xffffffff
    pattern, kind = op.rsplit(".", 1)
    width = int(kind[1:])
    mask = (1 << width) - 1
    operands = [a & mask for a in args]
    if kind.startswith("s"):
        operands = [signed(n, width) for n in operands]
    if pattern == "neg":
        return -operands[0] & mask
    a, b, *tail = operands
    if pattern == "bfe":
        return (a >> b) & ((1 << tail[0])-1)
    if pattern == "add":
        result = a + b
    elif pattern == "sub":
        result = a - b
    elif pattern == "and":
        result = a & b
    elif pattern == "or":
        result = a | b
    elif pattern == "shl":
        result = a << min(b, width)
    elif pattern == "shr":
        result = a >> min(b, width)
    elif pattern == "mul.lo":
        result = a * b
    elif pattern == "mul.hi":
        result = (a * b) >> width
    elif pattern == "mul.wide":
        return a * b
    elif pattern == "mad.lo":
        result = a * b + tail[0]
    elif pattern == "mad.wide":
        return a * b + args[2]
    else:
        raise ValueError(op)
    return result & mask


def lanes_for_matrix(lanes, tid, matrix, addr, transpose):
    lane = tid % 32
    first = tid - lane
    if transpose:
        return [address(lanes[first + matrix*8 + (lane % 4)*2+i], addr) + (lane//4)*2 for i in range(2)]
    offset = address(lanes[first + matrix*8 + lane//4], addr) + 4*(lane % 4)
    return [offset, offset+2]


class Replay:
    def __init__(self, extent, ignore=False):
        self.extent = extent
        self.ignore = ignore
        self.target_bank = 4 + SLABS
        inputs = tuple(i * BANK for i in range(2, self.target_bank + 1))
        outputs = tuple(BANK + 4 * ROWS * i for i in range(SLABS))
        self.params = inputs + outputs
        self.terms = []
        self.intern = {}

    def target(self, row):
        return -100 if self.ignore else WIDTH - 1 - row % WIDTH

    def term(self, op, *args):
        node = (op, *args)
        if node not in self.intern:
            self.intern[node] = Term(len(self.terms))
            self.terms.append(node)
        return self.intern[node]

    def node(self, term):
        return self.terms[term.id]

    def read_global(self, ptr, byte_width):
        bank, offset = divmod(ptr, BANK)
        if bank == 2 and byte_width == 4:
            assert offset % 4 == 0
            return input_id(offset // 4)
        if bank == 3 and byte_width == 1:
            return self.term("mask_byte", *divmod(offset, WIDTH // 8))
        if 4 <= bank < self.target_bank and byte_width == 2:
            assert offset % 2 == 0
            row, col = divmod(offset//2, WIDTH)
            row += (bank - 4) * ROWS
            return self.term("projection", row, col)
        if bank == self.target_bank and byte_width == 8:
            assert offset % 8 == 0
            return self.target(offset // 8)
        raise ValueError((bank, offset, byte_width))

    def expf(self, x):
        term = self.term
        rough = term("fma.rn.ftz.f32", x, 0x3bbb989d, 0x3f000000)
        y = term("fma.rm.ftz.f32", term("cvt.ftz.sat.f32.f32", rough), 0x437c0000, 0x4b400001)
        v = term("fma.rn.ftz.f32", x, 0x3fb8aa3b, term("neg.f32", term("add.f32", y, 0xcb40007f)))
        z = term("fma.rn.ftz.f32", x, 0x32a57060, v)
        return term("mul.f32", term("ex2.approx.ftz.f32", z), term("shl.b32", y, 23))

    def integer(self, op, a, b):
        if op in ("and.b16", "and.b32") and isinstance(a, Term) and isinstance(b, int):
            kind, *args = self.node(a)
            if kind == "mask_byte":
                if b == 255:
                    return a
                if 0 < b < 256 and b.bit_count() == 1:
                    return self.term("mask_bit", *args, b.bit_length() - 1)
            if kind == "mask_shift" and b == 1:
                return self.term("mask_bit", *args)
        if op == "shr.u16" and isinstance(a, Term) and self.node(a)[0] == "mask_byte":
            return self.term("mask_shift", *self.node(a)[1:], b)
        return integer_op(op, [a, b]) if isinstance(a, int) and isinstance(b, int) else self.term(op, a, b)

    def fp_pred(self, kind, a, b):
        if isinstance(a, int) and isinstance(b, int):
            x, y = float32(a), float32(b)
            if kind == "nan":
                return math.isnan(x) or math.isnan(y)
            return PREDICATES[kind](x, y)
        if kind == "gt" and a == NegInf:
            return False
        return self.term(kind+".f32", a, b)

    def maximum(self, a, b):
        if self.node(a)[0] == "prior_max":
            _, row, slot = self.node(a)
            assert self.node(b) == ("logit", row, slot + self.extent)
            return self.term("next_max", row, slot)
        return self.term("max", a, b)

    def select(self, a, b, p):
        if isinstance(p, bool) or a == b:
            return a if p else b
        test, l, r = self.terms[p.id]
        if test in ("eq.b16", "ne.b16", "eq.b32", "ne.b32") and r == 0:
            keep, drop = (a, b) if test.startswith("eq") else (b, a)
            assert drop == 0xc2700000
            kind, row, column = self.node(keep)
            assert kind == "softcap"
            assert self.node(l) == ("mask_bit", input_id(row), column // 8, column % 8)
            return self.term("logit", row, column)
        if test == "nan.f32" and a == l == r and isinstance(b, Term):
            op, *args = self.terms[b.id]
            if op == "selp" and args[0] == a and self.terms[args[2].id] == ("gt.f32", a, args[1]):
                return self.maximum(a, args[1])
        if test == "gt.f32" and l == a and isinstance(b, Term):
            op, *args = self.terms[b.id]
            if op == "selp" and args[:2] == [a, r] and self.terms[args[2].id] == ("nan.f32", a, a):
                return self.maximum(a, r)
        return self.term("selp", a, b, p)

    def online_sum(self, old_sum, lhs_scale, rhs_scale):
        op, one, _, pred = self.node(rhs_scale)
        assert op == "selp" and one == 0x3f800000
        op, maximum, negative_inf = self.node(pred)
        assert op == "eq.f32" and negative_inf == NegInf
        kind, row, slot = self.node(maximum)
        if old_sum == 0:
            assert kind == "logit"
            old_max, incoming = NegInf, maximum
        else:
            assert kind == "next_max" and self.node(old_sum) == ("prior_sum", row, slot)
            old_max = self.term("prior_max", row, slot)
            incoming = self.term("logit", row, slot + self.extent)
        scales = tuple(self.select(one, self.expf(self.term("sub.f32", value, maximum)), pred)
                       for value in (old_max, incoming))
        assert (lhs_scale, rhs_scale) == scales
        return self.term("loop_sum", row, slot)

    def fp(self, op, args):
        term = self.term
        if op == "cvt.f32.bf16":
            a = args[0]
            return a << 16 if isinstance(a, int) else term(op, a)
        if op == "mul.f32" and args[1] == 0x41b80000:
            a = args[0]
            code, one, den = self.terms[a.id]
            assert code == "div.full.f32" and one == 0x3f800000
            code, power, one = self.terms[den.id]
            assert code == "add.f32" and one == 0x3f800000
            code, exponent = self.terms[power.id]
            assert code == "ex2.approx.f32"
            code, affine, invlog = self.terms[exponent.id]
            assert code == "mul.f32" and invlog == 0x3fb8aa3b
            code, plus, ratio, zero = self.terms[affine.id]
            assert code == "fma.rn.f32" and ratio == 0xbe088889 and zero == 0
            code, wide, five = self.terms[plus.id]
            assert code == "add.f32" and five == 0x40a00000
            code, projection = self.terms[wide.id]
            assert code == "cvt.f32.bf16"
            leaf, *indices = self.terms[projection.id]
            assert leaf == "projection"
            return term("softcap", *indices)
        if op == "fma.rn.f32":
            first = args[0]
            if first == 0 or isinstance(first, Term) and self.node(first)[0] == "prior_sum":
                return self.online_sum(*args)
        if op in ("add.f32", "mul.f32") and isinstance(args[0], int) and isinstance(args[1], int):
            a, b = map(float32, args)
            return unpack('<I', pack('<f', a+b if op == "add.f32" else a*b))[0]
        return term(op, *args)

    def run(self, code, labels, cta, threads):
        lanes = [{"%tid.x": t, "%ctaid.x": cta, "global_smem": 0} for t in range(threads)]
        shared = {}
        shared32 = {}
        pc = 0
        iterations = 0
        stores = {}
        while pc < len(code):
            pred, op, operands = code[pc]
            pc += 1
            active = [(t, r) for t, r in enumerate(lanes) if selected(r, pred)]
            if not active:
                continue
            if op in ("bar.sync", "createpolicy.fractional.L2::evict_last.b64"):
                continue
            if op in ("trap", "call.uni"):
                raise ValueError((pc, op))
            if op == "ret":
                return stores
            dst, *src = operands
            if op.startswith("bra"):
                target = labels[dst]
                if dst == "$L__BB0_1":
                    assert len(active) == threads and iterations < 2
                    iterations += 1
                    replacements = {"logit": "prior_max", "loop_sum": "prior_sum"} if iterations == 1 else {
                        "next_max": "logit",
                    }
                    for regs in lanes:
                        for reg, value in regs.items():
                            if isinstance(value, Term):
                                kind, *args = self.node(value)
                                if kind in replacements:
                                    regs[reg] = self.term(replacements[kind], *args)
                    if iterations == 1:
                        pc = target
                    continue
                assert len(active) == threads
                pc = target
                continue
            if op == "setp.lt.f32" and src[-1] == "0f00800000":
                assert iterations == 2
                branch = next(args[0] for _, op, args in code[pc:pc + 40] if op == "bra")
                jump = labels[branch]
                normal, out = check_logf(code, pc - 1, jump, branch)
                for _, regs in active:
                    summed = bits(regs, src[0])
                    regs[out] = self.term("logf", summed)
                    regs[normal] = summed
                pc = jump
                continue
            if op == "shfl.sync.bfly.b32":
                assert len(active) == threads and src[2:] == ["31", "-1"]
                before = [bits(regs, src[0]) for regs in lanes]
                for tid, regs in active:
                    regs[dst] = before[tid ^ int(src[1])]
                continue
            if op.startswith(("ldmatrix.", "stmatrix.")):
                assert len(active) == threads
                load = op.startswith("ld")
                group, addr = (dst, src[0]) if load else (src[0], dst)
                for i, reg in enumerate(operands_in(group)):
                    for tid, regs in active:
                        slots = lanes_for_matrix(lanes, tid, i, addr, ".trans." in op)
                        if load:
                            regs[reg] = Pair(*(shared[i] for i in slots))
                        else:
                            for slot, word in zip(slots, halves(regs[reg]), strict=True):
                                shared[slot] = word
                continue
            for tid, regs in active:
                def read(a): return bits(regs, a)
                if op == "ld.param.b64":
                    regs[dst] = self.params[int(src[0][:-1].rsplit('_', 1)[1])]
                elif op == "st.global.b32":
                    pointer = address(regs, dst)
                    assert pointer not in stores
                    stores[pointer] = read(operands_in(src[0])[0])
                elif op.startswith(("ld.global.", "ld.shared.", "st.shared.")):
                    load = op.startswith("ld")
                    group, addr = (dst, src[0]) if load else (src[0], dst)
                    size = int(op.split(".")[-1][1:])//8
                    ptr = address(regs, addr)
                    names = operands_in(group) if group.startswith("{") else [group]

                    def read_shared(ptr, count):
                        if count == 2:
                            return shared[ptr]
                        if ptr in shared32:
                            return shared32[ptr]
                        return Pair(shared[ptr], shared[ptr + 2])

                    memory = self.read_global if ".global." in op else read_shared
                    for i, n in enumerate(names):
                        start = ptr+i*size
                        if load:
                            regs[n] = (Pair(memory(start, 2), memory(start+2, 2))
                                       if size == 4 and 4 <= ptr // BANK < self.target_bank and ".global." in op
                                       else memory(start, size))
                        elif size == 4:
                            shared32[start] = regs[n]
                            if isinstance(regs[n], Pair):
                                shared[start], shared[start+2] = halves(regs[n])
                        else:
                            datum = regs[n]
                            shared32.pop(start - start % 4, None)
                            shared[start] = datum.lo if isinstance(datum, Pair) else datum
                elif op.startswith("mov."):
                    if dst.startswith("{"):
                        low, high = operands_in(dst)
                        try:
                            regs[low], regs[high] = halves(read(src[0]))
                        except TypeError as e:
                            raise ValueError((pc, tid, op, operands)) from e
                    else:
                        regs[dst] = read(src[0])
                elif op.startswith("selp."):
                    regs[dst] = self.select(*(read(a) for a in src))
                elif op.startswith("setp."):
                    _, test, kind = op.split(".")
                    a, b = map(read, src)
                    if kind == "f32":
                        regs[dst] = self.fp_pred(test, a, b)
                    elif kind.startswith("s"):
                        regs[dst] = PREDICATES[test](signed(a, int(kind[1:])), signed(b, int(kind[1:])))
                    elif isinstance(a, Term):
                        regs[dst] = self.term(f"{test}.{kind}", a, b)
                    else:
                        regs[dst] = PREDICATES[test](a & ((1 << int(kind[1:]))-1), b & ((1 << int(kind[1:]))-1))
                elif op == "and.pred":
                    regs[dst] = read(src[0]) and read(src[1])
                elif op == "not.pred":
                    regs[dst] = not read(src[0])
                elif op in FP or op == "cvt.f32.bf16":
                    try:
                        regs[dst] = self.fp(op, [read(a) for a in src])
                    except Exception as e:
                        raise ValueError((pc, tid, op, operands, [read(a) for a in src])) from e
                elif op.startswith("cvt."):
                    _, dest, orig = op.split(".")
                    a = read(src[0])
                    if orig.startswith("s"):
                        a = signed(a, int(orig[1:]))
                    regs[dst] = a & ((1 << int(dest[1:]))-1)
                elif op == "prmt.b32":
                    a, b, k = map(read, src)
                    assert k in (0x5410, 0x7632), hex(k)
                    alo, ahi = halves(a)
                    blo, bhi = halves(b)
                    regs[dst] = Pair(alo, blo) if k == 0x5410 else Pair(ahi, bhi)
                else:
                    args = [read(a) for a in src]
                    if op == "shr.u32" and isinstance(args[0], Pair):
                        assert args[1] == 16
                        regs[dst] = args[0].hi
                    elif op in ("and.b32", "or.b32", "xor.b32") and any(isinstance(a, Pair) for a in args):
                        regs[dst] = Pair(*(self.integer(op, a, b)
                                         for a, b in zip(*(halves(x) for x in args), strict=True)))
                    elif len(args) == 2:
                        regs[dst] = self.integer(op, *args)
                    else:
                        regs[dst] = integer_op(op, args) if all(isinstance(a, int)
                                                                for a in args) else self.term(op, *args)
        raise ValueError("No loss reduction found")


def check_logf(code, begin, end, target):
    """Match every operand and both paths of the inlined logf polynomial."""
    suffix = code[begin:end]
    (branch,) = (i for i, (_, op, _) in enumerate(suffix) if op == "bra")
    regs = {suffix[0][2][1]: "sum"}
    for pred, op, operands in suffix[:branch]:
        assert pred is None
        dst, *args = operands
        args = tuple(bits(regs, arg) for arg in args)
        assert all(arg is not None for arg in args)
        regs[dst] = args[0] if op == "mov.b32" else (op, *args)

    tiny = ("setp.lt.f32", "sum", 0x00800000)
    normalized = ("selp.f32", ("mul.f32", "sum", 0x4b000000), "sum", tiny)
    adjustment = ("selp.f32", 0xc1b80000, 0, tiny)
    exponent = ("and.b32", ("add.s32", normalized, -1059760811), -8388608)

    def fma(a, b, c):
        return ("fma.rn.ftz.f32", a, b, c)

    power = fma(("cvt.rn.f32.s32", exponent), 0x34000000, adjustment)
    mantissa = ("add.f32", ("sub.s32", normalized, exponent), 0xbf800000)
    poly = 0xbe055027
    for coefficient in (0x3e1039f6, 0xbdf8cdcc, 0x3e0f2955, 0xbe2ad8b9,
                        0x3e4ced0b, 0xbe7fff22, 0x3eaaaa78, 0xbf000000):
        poly = fma(poly, mantissa, coefficient)
    log = fma(power, 0x3f317218, fma(("mul.f32", mantissa, poly), mantissa, mantissa))
    pred, _, label = suffix[branch]
    assert label == [target] and regs[pred] == ("setp.lt.u32", normalized, 0x7f800000)
    (move_pred, move_op, (inf, inf_word)), (fma_pred, fma_op, (out, x, a, b)) = suffix[branch + 1:]
    assert (move_pred, move_op, inf_word) == (None, "mov.b32", "0f7F800000")
    assert (fma_pred, fma_op, a, b) == (None, "fma.rn.ftz.f32", inf, inf)
    assert regs[x] == normalized and regs[out] == log
    return x, out


def expected_trees(tile):
    _, extent, _ = tile
    lanes, shuffles = SCHEDULES[tile]
    groups = extent // lanes
    maxima, sums = [], []
    for lane in range(lanes):
        columns = [lane + g * lanes for g in range(groups)]
        maximum = columns[0]
        for col in columns[1:]:
            maximum = ("max", maximum, col)
        maxima.append(maximum)
        if groups == 1:
            total = ("mul", columns[0])
        else:
            total = ("mul", columns[1])
            for col in [columns[0], *columns[2:]]:
                total = ("fma", col, total)
        sums.append(total)

    for offset in shuffles:
        maxima = [("max", m, maxima[lane ^ offset]) for lane, m in enumerate(maxima)]
        sums = [("add", s, sums[lane ^ offset]) for lane, s in enumerate(sums)]
    return maxima[0], sums[0]


def row_trees(replay, row, word):
    node, term = replay.node, replay.term
    op, out, zero = node(word)
    assert op == "add.f32" and zero == 0
    op, logged, target = node(out)
    assert op == "sub.f32"
    op, logit, maximum = node(target)
    assert op == "sub.f32" and node(logit) == ("logit", row, replay.target(row))
    op, negative_inf, answer, pred = node(logged)
    assert op == "selp" and negative_inf == NegInf
    op, total = node(answer)
    assert op == "logf" and node(pred) == ("eq.f32", total, 0)

    def max_tree(word):
        op, a, b = node(word)
        if op == "max":
            return ("max", max_tree(a), max_tree(b))
        assert op == "logit" and a == row
        return b

    def scaled_column(a, b):
        if node(b)[0] == "loop_sum":
            a, b = b, a
        op, r, col = node(a)
        assert op == "loop_sum" and r == row
        delta = term("sub.f32", term("logit", row, col), maximum)
        scale = replay.expf(replay.select(0, delta, replay.fp_pred("eq", maximum, NegInf)))
        assert scale == b
        return col

    def sum_tree(word):
        op, *args = node(word)
        if op == "add.f32":
            return ("add", sum_tree(args[0]), sum_tree(args[1]))
        if op == "fma.rn.f32":
            return ("fma", scaled_column(*args[:2]), sum_tree(args[2]))
        assert op == "mul.f32"
        return ("mul", scaled_column(*args))

    return max_tree(maximum), sum_tree(total)


def check_ptx(ptx, tile):
    code, labels = instructions(ptx)
    rows, extent, warps = tile
    expected = expected_trees(tile)
    for cta in (0, ROWS // rows - 1):
        replay = Replay(extent)
        stores = replay.run(code, labels, cta, warps * 32)
        assert len(stores) == rows * SLABS
        for slab in range(SLABS):
            begin = slab * ROWS + cta * rows
            for row in range(begin, begin + rows):
                assert row_trees(replay, row, stores[BANK + 4 * row]) == expected, (tile, cta, slab, row)
    ignored = Replay(extent, ignore=True).run(code, labels, 0, warps * 32)
    assert ignored == {BANK + 4 * (slab * ROWS + row): 0 for slab in range(SLABS) for row in range(rows)}
