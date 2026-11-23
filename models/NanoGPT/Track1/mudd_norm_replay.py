"""Symbolic SSA replay for the actual training RMSNorm and FP8 output stores."""
import re
from functools import lru_cache
from itertools import count
from dataclasses import dataclass
from qkv_ptx_replay import INT32_OPS, PREDICATES, WORD32, address, operands_in, parse, signed32

BANK = 1 << 40
MIXED, SQUARE_SUM, X, FP8, GRADIENT = (i * BANK for i in (1, 2, 3, 16, 17))
MAX_ROW = 49151


@dataclass(frozen=True, slots=True)
class RowOffset:
    """An affine integer over the entire supported blockIdx.x range."""
    stride: int
    constant: int

    def bounds(self):
        other = self.constant + MAX_ROW * self.stride
        return min(other, self.constant), max(other, self.constant)

    def __add__(self, other):
        if isinstance(other, RowOffset):
            return offset(self.stride + other.stride, self.constant + other.constant)
        return offset(self.stride, self.constant + other)

    __radd__ = __add__

    def __mul__(self, factor):
        assert isinstance(factor, int)
        return offset(self.stride * factor, self.constant * factor)

    __rmul__ = __mul__

    def __and__(self, mask):
        if mask == WORD32:
            low, high = self.bounds()
            assert 0 <= low <= high < 2**31
            return self
        assert self.stride % (1 << mask.bit_length()) == 0
        return self.constant & mask

    __rand__ = __and__

    def __or__(self, mask):
        assert self.stride % (1 << mask.bit_length()) == 0
        return offset(self.stride, self.constant | mask)

    __ror__ = __or__

    def __divmod__(self, denominator):
        low, high = self.bounds()
        if low // denominator == high // denominator:
            quotient = low // denominator
            return quotient, offset(self.stride, self.constant - quotient * denominator)
        assert self.stride % denominator == 0
        quotient, remainder = divmod(self.constant, denominator)
        return offset(self.stride // denominator, quotient), remainder

    def __floordiv__(self, denominator):
        return divmod(self, denominator)[0]

    def __mod__(self, denominator):
        return divmod(self, denominator)[1]


def offset(stride, constant):
    return RowOffset(stride, constant) if stride else constant

@dataclass(frozen=True, slots=True)
class Term:
    id: int

@dataclass(frozen=True, slots=True)
class Half:
    word: Term
    index: int


def unpack32(word):
    if isinstance(word, int):
        return word & 0xffff, word >> 16
    if isinstance(word, Term):
        return Half(word, 0), Half(word, 1)
    return word


def pack32(lo, hi):
    if isinstance(lo, Half) and isinstance(hi, Half):
        if lo.word == hi.word and (lo.index, hi.index) == (0, 1):
            return lo.word
    return lo, hi


next_id = count()
nodes = {}

@lru_cache(None)
def op(code, *xs):
    node = Term(next(next_id))
    nodes[node] = code, *xs
    return node

def mismatch(a, b):
    route = []
    while isinstance(a, Term) and isinstance(b, Term):
        left, right = nodes[a], nodes[b]
        if left[0] != right[0] or len(left) != len(right):
            return route, left, right
        step = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
        route.append((left[0], step))
        a, b = left[step], right[step]
    return route, a, b

def flatten(words):
    for word in words:
        if isinstance(word, tuple) and not isinstance(word[0], str):
            yield from flatten(word)
        else:
            yield word

def coefficient(plane, index):
    token, group = divmod(index, 12)
    bias = op("widen", op("bf16", 121, 14 + plane))
    mu = op("widen", op("bf16", 120, token * 10 + plane))
    delta = op("widen", op("bf16", 122, token * 120 + plane * 12 + group))
    scale = op("f32", 0x3dcccccd)
    return op("narrow", op("fma.f32", op("add.f32", bias, mu), scale, op("mul.f32", delta, scale)))


def inverse(square_sum):
    mean = op("div.full.f32", square_sum, op("f32", 0x44400000))
    return op("rsqrt.approx.ftz.f32", op("add.f32", mean, op("f32", 0x34000000)))


def replay(ptx, parameters, *, cached=False):
    body = ptx.split("{\n", 1)[1].split("ret;", 1)[0]
    instructions = parse(re.sub(r";\s*//.*", ";", line).strip() for line in body.splitlines()
                         if line.strip() and not line.strip().startswith((".", "//", "$", "}")))
    threads = [{"%tid.x": i, "%ctaid.x": RowOffset(1, 0), "global_smem": 0} for i in range(32)]
    shared, pending, out = {}, {}, {}

    def distribute(target, start, values, width=1):
        for i, word in enumerate(values):
            assert start + width * i not in target
            target[start + width * i] = word

    for pred, code, args in instructions:
        if code in ("bar.sync", "bar.warp.sync"):
            assert pred is None
            shared.update(pending)
            pending.clear()
            continue
        if code in ("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16", "ldmatrix.sync.aligned.m8n8.x4.shared.b16"):
            assert pred is None
            dst, pointer = args
            row_base = [address(r, pointer) for r in threads]
            for lane, regs in enumerate(threads):
                column, fragment = divmod(lane, 4)
                for matrix, name in enumerate(operands_in(dst)):
                    if ".trans." in code:
                        row = 8 * matrix + 2 * fragment
                        regs[name] = pack32(*(shared[row_base[row + j] + 2 * column] for j in range(2)))
                    else:
                        row = row_base[8 * matrix + column]
                        regs[name] = pack32(*(shared[row + 4 * fragment + 2 * j] for j in range(2)))
            continue
        if code.startswith("shfl.sync."):
            assert pred is None and args[3:] == ["31", "-1"]
            dst, source, index, *_ = args
            kind = code.split(".")[2]
            before = [r.get(source) for r in threads]
            for tid, regs in enumerate(threads):
                i = regs[index] if index.startswith("%") else int(index)
                lane = tid % 32 ^ i if kind == "bfly" else i
                regs[dst] = before[tid // 32 * 32 + lane]
            continue
        for regs in threads:
            def get(name):
                if name.startswith("%") or name == "global_smem":
                    return regs.get(name)
                if name.startswith("0f"):
                    return op("f32", int(name[2:], 16))
                return int(name, 0)

            if pred is not None and regs[pred.lstrip("!")] == pred.startswith("!"):
                if code == "ld.shared.b32":
                    regs.setdefault(args[0], None)
                continue
            dst, *src = args
            if code in INT32_OPS:
                regs[dst] = INT32_OPS[code](*map(get, src)) & WORD32
            elif code in ("mov.b32", "mov.u32") and dst.startswith("{"):
                word = get(src[0])
                parts = unpack32(word)
                for name, value in zip(operands_in(dst), parts, strict=True):
                    regs[name] = value
            elif code == "mov.b32" and src[0].startswith("{"):
                regs[dst] = pack32(*(get(reg) for reg in operands_in(src[0])))
            elif code in ("mov.u32", "mov.b32", "mov.u64", "mov.u16", "mov.b16", "mov.pred", "cvt.s64.s32"):
                regs[dst] = get(src[0])
            elif code == "cvt.u64.u32":
                regs[dst] = get(src[0]) & WORD32
            elif code == "add.s64":
                regs[dst] = sum(map(get, src))
            elif code in ("bfe.u32", "bfe.s32"):
                x, i, n = map(get, src)
                extracted = (x >> i) & ((1 << n) - 1)
                if code == "bfe.s32" and extracted >> (n - 1):
                    extracted -= 1 << n
                regs[dst] = extracted
            elif code in ("mad.wide.s32", "mad.wide.u32", "mad.lo.s32"):
                x, y, z = map(get, src)
                regs[dst] = x * y + z
            elif code == "mul.hi.s32":
                a, b = map(lambda arg: signed32(get(arg)), src)
                regs[dst] = (a * b >> 32) & WORD32
            elif code == "shr.s32":
                a, b = map(get, src)
                regs[dst] = (signed32(a) >> b) & WORD32
            elif code in ("mul.wide.s32", "mul.wide.u32"):
                a, b = map(get, src)
                regs[dst] = a * b
            elif code.startswith("createpolicy."):
                regs[dst] = 0
            elif code.startswith("setp."):
                regs[dst] = PREDICATES[code.split(".")[1]](*map(get, src))
            elif code == "and.pred":
                regs[dst] = get(src[0]) and get(src[1])
            elif code == "not.pred":
                regs[dst] = not get(src[0])
            elif code in ("selp.f32", "selp.b32"):
                regs[dst] = get(src[0] if get(src[2]) else src[1])
            elif code == "ld.param.b64":
                regs[dst] = parameters[int(src[0].rstrip("]").rsplit("_", 1)[1])]
            elif code.startswith("ld.global."):
                start = address(regs, src[0])
                bank, byte = divmod(start, BANK)
                names = operands_in(dst) if dst.startswith("{") else [dst]
                if bank in (MIXED // BANK, SQUARE_SUM // BANK):
                    assert byte % 4 == 0 and code.endswith(".b32")
                    for i, name in enumerate(names):
                        word = op("f32-load", bank, byte // 4 + i)
                        regs[name] = inverse(word) if cached and bank == SQUARE_SUM // BANK else word
                    continue
                assert byte % 2 == 0
                def load(addr):
                    index = (addr % BANK) // 2
                    if 40 <= bank < 50:
                        return coefficient(bank - 40, index)
                    return op("bf16", bank, index)
                if code.endswith(".b32"):
                    for i, name in enumerate(names):
                        regs[name] = tuple(load(start + 4 * i + 2 * j) for j in range(2))
                elif code.endswith(".b16"):
                    assert len(names) == 1
                    regs[names[0]] = load(start)
                else:
                    raise ValueError(code)
            elif code == "cvt.f32.bf16":
                x = get(src[0])
                regs[dst] = op("f32", 0) if x == 0 else op("widen", x)
            elif code == "cvt.rn.bf16x2.f32":
                regs[dst] = tuple(op("narrow", get(arg)) for arg in reversed(src))
            elif code == "cvt.rn.bf16.f32":
                regs[dst] = op("narrow", get(src[0]))
            elif code in ("mul.f32", "mul.rn.f32", "add.f32", "add.rn.f32", "fma.rn.f32",
                          "div.full.f32", "rsqrt.approx.ftz.f32"):
                values = (get(arg) for arg in src)
                operands = tuple(op("f32", v & WORD32) if isinstance(v, int) else v for v in values)
                arithmetic = code.replace(".rn", "")
                if arithmetic == "mul.f32" and operands == (op("f32", 0), op("f32", 0)):
                    regs[dst] = operands[0]
                else:
                    regs[dst] = op(arithmetic, *operands)
            elif code == "cvt.rn.satfinite.e4m3x2.f32":
                regs[dst] = tuple(op("e4m3", get(arg)) for arg in reversed(src))
            elif code in ("ld.shared.b16", "ld.shared.b32"):
                regs[dst] = shared[address(regs, src[0])]
            elif code.startswith("st.shared."):
                start = address(regs, dst)
                values = [get(reg) for reg in operands_in(src[0])] if src[0].startswith("{") else [get(src[0])]
                if code == "st.shared.v4.b32":
                    assert len(values) == 4
                    values = [part for value in values for part in
                              unpack32(value)]
                width = 2
                for i, word in enumerate(values):
                    assert start + width * i not in pending or pending[start + width * i] == word
                    pending[start + width * i] = word
            elif code.startswith("st.global."):
                start = address(regs, dst)
                values = [get(reg) for reg in operands_in(src[0])]
                bank = start // BANK
                if 40 <= bank < 50:
                    assert code == "st.global.b16"
                    distribute(out, start, values)
                elif bank == MIXED // BANK:
                    distribute(out, start, values, width=4)
                elif bank == SQUARE_SUM // BANK:
                    assert len(values) == 1
                    distribute(out, start, values)
                elif bank == GRADIENT // BANK:
                    distribute(out, start, flatten(values), width=2)
                else:
                    assert bank == FP8 // BANK
                    distribute(out, start, flatten(values))
            else:
                raise ValueError((code, args))
    return out

def reference_row(source_ports, coefficient_ports):
    """Construct the stores from the row recurrence and warp reduction."""
    row = RowOffset(1, 0)
    output = {}
    coefficients = []
    for plane, port in enumerate(coefficient_ports):
        bank = []
        for group in range(12):
            index = 12 * row + group
            term = coefficient(plane, index)
            output[port + 2 * index] = term
            bank.append(op("widen", term))
        coefficients.append(bank)
    mixed = []
    for column in range(768):
        index = row * 768 + column
        residual = op("widen", op("bf16", X // BANK, index))
        for plane, source in enumerate(source_ports):
            value = op("widen", op("bf16", source // BANK, index))
            residual = op("fma.f32", coefficients[plane][column // 64], value, residual)
        mixed.append(residual)
        output[MIXED + 4 * index] = residual
    partial = []
    for lane in range(32):
        words = mixed[lane*16:lane*16 + 16]
        total = op("mul.f32", words[1], words[1])
        for word in (words[0], *words[2:]):
            total = op("fma.f32", word, word, total)
        for word in range(512 + 16 * lane, 528 + 16 * lane):
            square = op("mul.f32", mixed[word], mixed[word]) if word < 768 else op("f32", 0)
            total = op("add.f32", square, total)
        partial.append(total)
    for span in (16, 8, 4, 2, 1):
        partial = [op("add.f32", partial[lane], partial[lane ^ span]) for lane in range(32)]
    output[SQUARE_SUM + 4 * row] = partial[0]
    inverse = []
    for magnitude in partial:
        mean = op("div.full.f32", magnitude, op("f32", 0x44400000))
        biased = op("add.f32", mean, op("f32", 0x34000000))
        inverse.append(op("rsqrt.approx.ftz.f32", biased))
    for column, residual in enumerate(mixed):
        lane = (column % 512) // 16
        scaled = op("mul.f32", op("mul.f32", residual, inverse[lane]), op("f32", 0x408f5c29))
        output[FP8 + row * 768 + column] = op("e4m3", scaled)
    return output


def check_rows(rows, reference_ptx, fused_ptx, cached_ptx):
    source_ports = tuple(BANK * (20 + i) for i in range(10))
    coefficient_ports = tuple(BANK * (40 + i) for i in range(10))
    operands = tuple(a for pair in zip(coefficient_ports, source_ports) for a in pair)
    separate = (X, *operands, MIXED, SQUARE_SUM, FP8)
    fused = (BANK * 120, BANK * 121, BANK * 122, MIXED, SQUARE_SUM, X, *operands, FP8)
    assert 0 < rows <= MAX_ROW + 1
    before = replay(reference_ptx, separate)
    for plane, port in enumerate(coefficient_ports):
        for group in range(12):
            index = RowOffset(12, group)
            before[port + 2 * index] = coefficient(plane, index)
    after = replay(fused_ptx, fused)
    cached = replay(cached_ptx, fused)
    expected = reference_row(source_ports, coefficient_ports)
    assert len(before) == 1657 and before.keys() == after.keys() == expected.keys() == cached.keys()
    statistic = SQUARE_SUM + RowOffset(4, 0)
    for ptr, term in before.items():
        assert term == expected[ptr], (rows, ptr, mismatch(term, expected[ptr]))
        assert term == after[ptr], (rows, ptr, mismatch(term, after[ptr]))
        saved = inverse(term) if ptr == statistic else term
        assert cached[ptr] == saved, (rows, ptr, mismatch(cached[ptr], saved))
    op.cache_clear()
    nodes.clear()
