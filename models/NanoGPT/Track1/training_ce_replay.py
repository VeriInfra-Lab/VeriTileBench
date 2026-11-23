"""Check the NVRTC pair conversion and its connection to the loss/gradient suffix."""
from dataclasses import dataclass
import re

from qkv_ptx_replay import operands_in, parse, PREDICATES
from validation_ce_replay import Term, integer_op

REG = re.compile(r"%([a-z]+)\d+")
SYMBOL = re.compile(r"%([a-z]+)\d+|\$L__BB0_\d+")
BANK = 1 << 40
LOGITS = BANK
TARGETS = 2 * BANK
PREFIX_WEIGHT = 5 * BANK
SHARED = 8 * BANK
SIGMOID_TABLE = 10 * BANK
EXP_TABLE = 11 * BANK
GRADIENT_FACTOR = 12 * BANK
TARGET_CACHE = 13 * BANK
FACTOR_SYMBOL = "_ZZ17ce_fwd_bwd_kernelE15gradient_factor"
TARGET_CACHE_SYMBOL = "_ZZ17ce_fwd_bwd_kernelE14cached_targets"
TABLE_SYMBOLS = {
    "_ZZ17ce_fwd_bwd_kernelE13sigmoid_table": SIGMOID_TABLE,
    "_ZZ17ce_fwd_bwd_kernelE9exp_table": EXP_TABLE,
}


@dataclass(frozen=True)
class Word:
    parts: tuple
    width: int

    def halves(self):
        first = len(self.parts) // 2
        return (Word(self.parts[:first], self.width), Word(self.parts[first:], self.width))


@dataclass(frozen=True)
class RowAddress:
    """A constant plus the same unsigned row-offset instruction in both prefixes."""
    row_offset: Term
    constant: int = 0

    def __add__(self, amount):
        return RowAddress(self.row_offset, self.constant + amount)

    __radd__ = __add__


@dataclass(frozen=True)
class ByteAddress:
    """Address indexed by a symbolic unsigned logit byte."""
    byte: Term
    stride: int
    base: int = 0

    def __add__(self, amount):
        assert isinstance(amount, int)
        return ByteAddress(self.byte, self.stride, self.base + amount)

    __radd__ = __add__


@dataclass(frozen=True)
class TargetInputs:
    row: int
    height: int
    targets: tuple[int, ...]

    def load(self, slot):
        assert 0 <= slot < len(self.targets) and self.row + slot < self.height
        return self.targets[slot] & ((1 << 64) - 1)

    def expected_cache(self, vocabulary):
        words = {}
        for slot in range(len(self.targets)):
            word = self.load(slot) if self.row + slot < self.height else -1
            words[4 * slot] = word if 0 <= word < vocabulary else 0xffffffff
        return words


def widen_bytes(word):
    assert word.width == 8 and len(word.parts) == 2
    return Word(word.parts + (0, 0), 8)


def select_byte(word, zero, selector):
    assert word.width == 8 and len(word.parts) == 4 and zero == 0
    assert 0x4440 <= selector <= 0x4443
    return Word((word.parts[selector - 0x4440], 0, 0, 0), 8)


def byte_integer(op, operands):
    if op == "add.s64":
        value = sum(operands)
        return value & ((1 << 64) - 1) if isinstance(value, int) else value
    value, shift, *_ = operands
    if op == "mul.wide.u32" and isinstance(value, (Term, Word)):
        if isinstance(value, Word):
            assert value.width == 8 and value.parts[1:] == (0, 0, 0)
            value = value.parts[0]
        return ByteAddress(value, shift)
    if isinstance(value, Word):
        assert value.width == 8
        if op == "shr.u16":
            assert len(value.parts) == 2 and shift == 8
            return Word((value.parts[1], 0), 8)
        assert op == "and.b32" and len(value.parts) == 4 and shift == 255
        return value.parts[0]
    return integer_op(op, operands)


def lines(text):
    result = []
    for raw in text.splitlines():
        line = " ".join(raw.split("//", 1)[0].split())
        if line.startswith(".section"):
            break
        if line and not line.startswith((".file", ".loc")):
            result.append(line)
    return result


def instructions(body):
    result = []
    labels = {}
    for line in body:
        if line.startswith("$") and line.endswith(":"):
            labels[line[:-1]] = len(result)
        elif line.startswith("{") and line.endswith("}"):
            result.extend(parse([part.strip() + ";" for part in line[1:-1].split(";")
                                 if part.strip() and not part.strip().startswith(".reg")]))
        elif not line.startswith(".") and line.endswith(";"):
            result.extend(parse([line]))
    return result, labels


def split(text):
    body = lines(text)
    entry = body.index(next(line for line in body if line.startswith(".visible .entry ")))
    register_file = next(i for i, line in enumerate(body) if i > entry and line.startswith(".reg "))
    boundary = next(i for i, line in enumerate(body) if line.startswith("shfl.sync.down.b32 "))
    header = body[:register_file] + [line for line in body[register_file:boundary]
                                    if line.startswith(".shared ")]
    return header, body[register_file:boundary], body[boundary:]


def renamed(body):
    names = {}

    def name(match):
        return names.setdefault(match[0], f"{match[1] or 'label'}_{len(names)}")

    return [SYMBOL.sub(name, line) for line in body], names


def live_registers(body):
    """Backwards liveness at the first shuffle, including conditional definitions."""
    code, labels = instructions(body)
    uses = []
    for pred, op, args in code:
        writes = set() if op.startswith(("st.", "bra", "bar", "ret")) else {
            m[0] for m in REG.finditer(args[0])
        }
        sources = args if not writes else args[1:]
        reads = {m[0] for part in sources for m in REG.finditer(part)}
        if pred:
            reads.add(pred.lstrip("!"))
            writes = set()
        uses.append((reads, writes))
    before = [set() for _ in range(len(code) + 1)]
    while True:
        updated = False
        for i in reversed(range(len(code))):
            pred, op, args = code[i]
            reads, writes = uses[i]
            if op == "ret":
                after = set()
            elif op.startswith("bra"):
                after = before[labels[args[0]]] | (before[i + 1] if pred else set())
            else:
                after = before[i + 1]
            need = reads | (after - writes)
            if need != before[i]:
                before[i] = need
                updated = True
        if not updated:
            return before[0]


class Prefix:
    def __init__(self):
        self.terms = {}

    def term(self, op, *args):
        return self.terms.setdefault((op, *args), Term(len(self.terms)))

    def table_values(self, byte):
        term = self.term
        a, b, c = (term("cvt.rn.ftz.f32.f64", term("parameter", i, "ld.param.f64")) for i in (9, 10, 11))
        inv = term("rcp.approx.ftz.f32", c)
        b_scale = term("mul.ftz.f32", inv, b)
        logit = term("cvt.f32.f16", term("fp8_to_half", byte))
        z = term("fma.rn.ftz.f32", inv, logit, b_scale)
        t = term("tanh.approx.f32", term("mul.ftz.f32", z, 0x3f000000))
        sigma = term("cvt.rn.f16.f32", term("fma.rn.ftz.f32", t, 0x3f000000, 0x3f000000))
        widened = term("cvt.f32.f16", sigma)
        scaled = term("mul.ftz.f32", widened, a)
        delta = term("sub.ftz.f32", scaled, a)
        exp = term("ex2.approx.ftz.f32", term("mul.ftz.f32", delta, 0x3fb8aa3b))
        return sigma, widened, scaled, exp

    def read_exp_table(self, ptr):
        assert isinstance(ptr, ByteAddress)
        assert (ptr.base, ptr.stride) == (EXP_TABLE, 4)
        return self.table_values(ptr.byte)[3]

    def gradient_factor(self):
        term = self.term
        a, c, grad_scale = (term("cvt.rn.ftz.f32.f64", term("parameter", i, "ld.param.f64"))
                            for i in (9, 11, 12))
        return term("mul.ftz.f32", term("mul.ftz.f32", term("rcp.approx.ftz.f32", c), a),
                    term("rcp.approx.ftz.f32", grad_scale))

    def logit(self, vocabulary, col):
        row = RowAddress(self.term("mul.lo.s32", self.term("row"), vocabulary), LOGITS)
        return self.term("logit", row + col)

    def expected(self, vocabulary, columns):
        result = {}
        total = 0
        for i, col in enumerate(columns):
            sigma, _, _, expf = self.table_values(self.logit(vocabulary, col))
            result[2 * col] = sigma
            operands = (expf, total) if i == 0 else (total, expf)
            total = self.term("add.ftz.f32", *operands)
        return result, total

    def run(self, code, labels, lane, vocabulary, *, table=False, target_inputs=None, table_registers=None):
        # Chunk counters and masks are concrete. The row and every FP
        # payload stay symbolic, even across a rolled chunk loop.
        row = self.term("row")
        regs = {"%tid.x": lane, "%ctaid.x": row, "smem": SHARED, "code_smem": SHARED,
                FACTOR_SYMBOL: GRADIENT_FACTOR, TARGET_CACHE_SYMBOL: TARGET_CACHE,
                **TABLE_SYMBOLS}
        scratch = {}
        initialization = {}
        target_words = {}
        synchronized = False

        def get(name):
            if name in regs:
                return regs[name]
            if name.startswith("0f"):
                return int(name[2:], 16)
            return int(name, 0)

        def address(token):
            base, *offset = token[1:-1].replace(" ", "").split("+")
            return get(base) + (int(offset[0]) if offset else 0)

        def unpack(out, packed):
            items = packed.parts if len(packed.parts) == 2 else packed.halves()
            for d, item in zip(operands_in(out), items, strict=True):
                regs[d] = item

        pc = 0
        max_chunks = (vocabulary + 2047) // 2048
        for _ in range(len(code) * (max_chunks + 1)):
            if pc == len(code):
                assert synchronized == table
                return regs, scratch, target_words
            pred, op, (dst, *src) = code[pc]
            pc += 1
            if pred and regs[pred.lstrip("!")] == pred.startswith("!"):
                continue
            if op.startswith("bra"):
                pc = labels[dst]
            elif op == "bar.sync":
                assert table and not synchronized and not pred and dst == "0" and not src
                _, widened, scaled, exp = self.table_values(lane)
                entry = SIGMOID_TABLE + lane * 8
                expected = {entry: widened, entry + 4: scaled, EXP_TABLE + lane * 4: exp}
                if table_registers is not None:
                    assert tuple(regs.get(reg) for reg in table_registers) == (widened, scaled)
                    expected = {EXP_TABLE + lane * 4: exp}
                if lane == 0:
                    expected[GRADIENT_FACTOR] = self.gradient_factor()
                assert initialization == expected
                expected_targets = target_inputs.expected_cache(vocabulary) if lane == 0 and target_inputs else {}
                assert target_words == expected_targets
                synchronized = True
            elif op.startswith("ld.param."):
                i = int(re.search(r"param_(\d+)\]", src[0])[1])
                regs[dst] = (i + 1) * BANK if i < 7 else self.term("parameter", i, op)
            elif op == "mov.b32" and src[0].startswith("{"):
                low, high = (get(part) for part in operands_in(src[0]))
                if isinstance(low, Word) and isinstance(high, Word):
                    assert low.width == high.width == 8 and len(low.parts) == len(high.parts) == 2
                    regs[dst] = Word(low.parts + high.parts, 8)
                else:
                    regs[dst] = Word((low, high), 16)
            elif op == "cvt.u32.u64":
                regs[dst] = get(src[0]) & 0xffffffff
            elif op == "prmt.b32":
                assert table and synchronized
                regs[dst] = select_byte(*map(get, src))
            elif op.startswith(("mov.", "cvta.to.global", "cvt.u64", "cvt.s64", "cvt.u16", "cvt.u32.u16")):
                value = get(src[0])
                if op == "cvt.u16.u32":
                    if isinstance(value, int):
                        value = value & 0xffff
                    else:
                        assert value.width == 16
                        value = value.parts[0]
                if op == "cvt.u32.u16":
                    value = widen_bytes(value)
                if dst.startswith("{"):
                    unpack(dst, value)
                else:
                    regs[dst] = value
            elif op in ("ld.global.nc.v4.u16", "ld.global.nc.v2.u32", "ld.global.v2.u32"):
                start = address(src[0])
                assert isinstance(start, RowAddress) and LOGITS <= start.constant <= LOGITS + vocabulary - 8
                width = 2 if op.endswith("u16") else 4
                for i, d in enumerate(operands_in(dst)):
                    regs[d] = Word(tuple(self.term("logit", start + (width * i + b)) for b in range(width)), 8)
            elif op == "ld.global.nc.f32":
                assert address(src[0]) == PREFIX_WEIGHT
                regs[dst] = self.term("prefix_weight")
            elif op == "ld.global.nc.u64":
                ptr = address(src[0])
                assert target_inputs and not synchronized and lane == 0
                assert isinstance(ptr, ByteAddress) and (ptr.byte, ptr.stride) == (row, 8)
                delta = ptr.base - TARGETS
                assert delta >= 0 and delta % 8 == 0
                regs[dst] = target_inputs.load(delta // 8)
            elif op == "st.shared.v2.f32":
                ptr = address(dst)
                assert table and not synchronized and ptr == SIGMOID_TABLE + lane * 8
                assert ptr not in initialization and ptr + 4 not in initialization
                for offset, reg in zip((0, 4), operands_in(src[0]), strict=True):
                    initialization[ptr + offset] = get(reg)
            elif op == "st.shared.f32":
                ptr = address(dst)
                assert table and not synchronized and ptr not in initialization
                assert ptr == EXP_TABLE + lane * 4 or (lane == 0 and ptr == GRADIENT_FACTOR)
                initialization[ptr] = get(src[0])
            elif op == "st.shared.u32":
                offset = address(dst) - TARGET_CACHE
                assert target_inputs and not synchronized and lane == 0
                assert offset % 4 == 0 and 0 <= offset < 4 * len(target_inputs.targets)
                assert offset not in target_words
                target_words[offset] = get(src[0]) & 0xffffffff
            elif op == "ld.shared.f32":
                assert table and synchronized
                regs[dst] = self.read_exp_table(address(src[0]))
            elif op in ("st.shared.v4.u32", "st.shared.v2.u32"):
                start = address(dst) - SHARED
                stride = 1 if table else 2
                assert synchronized == table and start >= 0 and start % 8 == 0
                for i, item in enumerate(operands_in(src[0])):
                    word = get(item)
                    assert word.width == 8 * stride and len(word.parts) == 4 // stride
                    for j, value in enumerate(word.parts):
                        loc = start + i * 4 + j * stride
                        assert loc not in scratch and loc < stride * vocabulary
                        scratch[loc] = value
            elif op == "cvt.rn.f16x2.e4m3x2":
                word = get(src[0])
                if isinstance(word, int):
                    word = Word((word & 255, word >> 8), 8)
                assert word.width == 8 and len(word.parts) == 2
                regs[dst] = Word(tuple(self.term("fp8_to_half", a) for a in word.parts), 16)
            elif op == "cvt.rn.f16x2.f32":
                regs[dst] = Word(tuple(self.term("cvt.rn.f16.f32", get(a)) for a in reversed(src)), 16)
            elif op in ("and.b16", "shr.u16"):
                value, mask = map(get, src)
                assert value.width == 8 and len(value.parts) == 2
                assert mask == (255 if op == "and.b16" else 8)
                regs[dst] = Word((value.parts[0 if op == "and.b16" else 1], 0), 8)
            elif op.startswith("setp."):
                _, comparison, kind = op.split(".")
                left, right = map(get, src)
                if kind == "s64":
                    assert target_inputs and not synchronized and comparison == "ge"
                    assert right == self.term("parameter", 7, "ld.param.u32")
                    if isinstance(left, ByteAddress):
                        assert (left.byte, left.stride) == (row, 1)
                        left = target_inputs.row + left.base
                    else:
                        assert left == row
                        left = target_inputs.row
                    right = target_inputs.height
                elif kind == "u64":
                    assert target_inputs and not synchronized and comparison == "lt"
                    left &= (1 << 64) - 1
                else:
                    assert kind in ("s32", "u32")
                regs[dst] = PREDICATES[comparison](left, right)
            elif op == "selp.b32":
                assert target_inputs and not synchronized
                regs[dst] = get(src[0] if get(src[2]) else src[1])
            elif op == "mul.lo.s32" and isinstance(get(src[0]), Term):
                regs[dst] = RowAddress(self.term(op, *map(get, src)))
            elif op == "and.pred":
                regs[dst] = all(map(get, src))
            elif op == "add.s64" and get(src[0]) == row:
                assert target_inputs and not synchronized
                regs[dst] = ByteAddress(row, 1, get(src[1]))
            elif op in ("mul.wide.s32", "mul.wide.u32", "mul.lo.s32", "shl.b32", "add.s32", "add.s64", "and.b32"):
                regs[dst] = byte_integer(op, list(map(get, src)))
            elif op in ("cvt.rn.ftz.f32.f64", "rcp.approx.ftz.f32", "mul.ftz.f32", "mul.rn.ftz.f32", "fma.rn.ftz.f32",
                        "tanh.approx.f32", "ex2.approx.ftz.f32", "cvt.rn.f16.f32",
                        "cvt.f32.f16", "add.ftz.f32", "sub.ftz.f32"):
                operands = tuple(map(get, src))
                if op == "add.ftz.f32" and operands[0] == 0:
                    operands = (operands[1], 0)
                # Floating instructions with no explicit rounding use .rn.
                regs[dst] = self.term("mul.ftz.f32" if op == "mul.rn.ftz.f32" else op, *operands)
            else:
                raise ValueError(f"Unhandled CE prefix instruction {op}")
        if pc == len(code):
            assert synchronized == table
            return regs, scratch, target_words
        raise ValueError("CE prefix exceeded the vocabulary scan bound")


def verify_training_ce(reference, candidate, vocabulary):
    header, original, suffix = split(reference)
    other_header, changed, other_suffix = split(candidate)
    assert header == other_header
    suffix_body, names = renamed(suffix)
    other_body, other_names = renamed(other_suffix)
    assert suffix_body == other_body, "warp reduction, loss or gradient PTX changed"
    live = live_registers(suffix)
    inverse = {renamed: name for name, renamed in other_names.items()}
    assert {inverse[names[name]] for name in live} == live_registers(other_suffix)
    code, labels = instructions(original)
    other_code, other_labels = instructions(changed)
    suffix_code, _ = instructions(suffix)
    _, _, (_, shuffle_input, *_) = suffix_code[0]
    prefix = Prefix()
    written = set()
    for lane in range(256):
        regs, scratch, _ = prefix.run(code, labels, lane, vocabulary)
        candidate_regs, candidate_scratch, _ = prefix.run(other_code, other_labels, lane, vocabulary)
        columns = [part + 8 * lane + i for part in range(0, vocabulary, 2048) for i in range(8)
                   if part + 8 * lane + i < vocabulary]
        expected, total = prefix.expected(vocabulary, columns)
        assert scratch == candidate_scratch == expected, (vocabulary, lane, "FP16 sigmoid stores")
        assert regs[shuffle_input] == total, (vocabulary, lane, "ordered lane sum")
        assert written.isdisjoint(expected)
        written.update(expected)
        for name in live:
            assert regs[name] == candidate_regs[inverse[names[name]]], (vocabulary, lane, name)
    assert len(written) == vocabulary
    return len(live), len(code), len(other_code)
