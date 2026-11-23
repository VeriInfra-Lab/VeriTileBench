"""Replay the dynamic and fixed-count CE control paths on symbolic FP payloads."""
from qkv_ptx_replay import operands_in, PREDICATES
from rope_ptx_replay import signed
from training_ce_replay import (BANK, EXP_TABLE, FACTOR_SYMBOL, GRADIENT_FACTOR, SHARED, SIGMOID_TABLE, TABLE_SYMBOLS,
                                TARGET_CACHE, TARGET_CACHE_SYMBOL, ByteAddress, Prefix, RowAddress, TargetInputs, Term, Word,
                                byte_integer, instructions, select_byte, split, widen_bytes)

WARP_SUMS = 9 * BANK
WARP_SUM_SYMBOL = "_ZZ17ce_fwd_bwd_kernelE10block_sums"
LOSSES = 6 * BANK
GRADIENT = 7 * BANK
FLOAT = {"add.ftz.f32", "sub.ftz.f32", "mul.ftz.f32", "mul.rn.ftz.f32", "fma.rn.ftz.f32", "ex2.approx.ftz.f32",
         "lg2.approx.ftz.f32", "cvt.f32.f16", "rcp.approx.ftz.f32", "cvt.rn.ftz.f32.f64"}
INTEGER = {"add.s32", "sub.s32", "add.s64", "mul.wide.s32", "mul.wide.u32", "mad.lo.s32",
           "shl.b64", "shl.b32", "and.b32", "shr.u16", "shr.u32", "mul.lo.s32"}


class Program:
    def __init__(self, ptx, vocabulary, prefix):
        header, head, tail = split(ptx)
        self.table = any("code_smem[]" in line for line in header)
        if self.table:
            for symbol, align, extent in zip(TABLE_SYMBOLS, (8, 4), (2048, 1024), strict=True):
                assert f".shared .align {align} .b8 {symbol}[{extent}];" in header
            assert f".shared .align 4 .f32 {FACTOR_SYMBOL};" in header
        self.vocabulary = vocabulary
        self.prefix = prefix
        declarations = [line for line in header if TARGET_CACHE_SYMBOL in line]
        self.cache_slots = 0
        if declarations:
            assert self.table and len(declarations) == 1
            layouts = {f".shared .align 4 .b8 {TARGET_CACHE_SYMBOL}[{4 * count}];": count
                       for count in range(1, 4)}
            layouts[f".shared .align 4 .u32 {TARGET_CACHE_SYMBOL};"] = 1
            assert declarations[0] in layouts
            self.cache_slots = layouts[declarations[0]]
        self.code, self.labels = instructions(tail)
        assert self.table == any(op == "st.shared.u8" for _, op, _ in self.code)
        self.suffix_barriers = 2
        self.output_layer = 2 if self.table else 1
        assert sum(op == "bar.sync" for _, op, _ in self.code) == self.suffix_barriers
        self.table_registers = None
        if self.table:
            sum_ready, caches_ready = [pc for pc, (_, op, _) in enumerate(self.code) if op == "bar.sync"]
            publication = self.code[sum_ready + 1:caches_ready]
            pair_stores = [(pred, args) for pred, op, args in publication if op == "st.shared.v2.f32"]
            if pair_stores:
                (pred, (_, pair)), = pair_stores
                assert not pred
                sigma, _ = operands_in(pair)
                _, _, (_, scaled, _) = next(insn for insn in publication if insn[1] == "sub.ftz.f32")
                assert sigma != scaled
                self.table_registers = (sigma, scaled)
        _, _, (_, shuffle_input, *_) = self.code[0]
        code, labels = instructions(head)
        self.head = (code, labels)
        assert sum(op == "bar.sync" for _, op, _ in code) == int(self.table)
        self.initial = []
        self.scratch = {}
        targets = TargetInputs(0, self.cache_slots, tuple(range(self.cache_slots))) if self.cache_slots else None
        for tid in range(256):
            registers, scratch, _ = prefix.run(code, labels, tid, vocabulary,
                                               table=self.table, target_inputs=targets,
                                               table_registers=self.table_registers)
            columns = [part + tid * 8 + i for part in range(0, vocabulary, 2048)
                       for i in range(8) if part + tid * 8 + i < vocabulary]
            expected, total = prefix.expected(vocabulary, columns)
            if self.table:
                expected = {col: prefix.logit(vocabulary, col) for col in columns}
            assert scratch == expected and registers[shuffle_input] == total
            assert self.scratch.keys().isdisjoint(scratch)
            self.scratch.update(scratch)
            self.initial.append(registers)
        assert len(self.scratch) == vocabulary

    def execute(self, predictions, row, height, targets, prefix_class, prefix_positive, *, reuse_logits=False):
        assert not reuse_logits or self.table
        gradient_base = BANK if reuse_logits else GRADIENT

        def parameter_address(port):
            return gradient_base if port == 6 else (port + 1) * BANK

        term = self.prefix.term
        parameters = {7: height, 8: predictions}
        parameter_words = {term("parameter", i, "ld.param.u32"): value for i, value in parameters.items()}
        row_term = term("row")
        row_offset = term("mul.lo.s32", row_term, self.vocabulary)
        prefix_scale = term("prefix_weight")
        warp_sums = {}
        probabilities = {}
        base_bytes = {}
        row_lse = None
        loss = {}
        base = {}
        corrections = {}

        def materialize(arg):
            if isinstance(arg, RowAddress):
                assert arg.row_offset == row_offset
                arg = (row * self.vocabulary & 0xffffffff) + arg.constant
            if isinstance(arg, ByteAddress) and arg.byte == row_term:
                return arg.base + row * arg.stride
            if isinstance(arg, Term):
                return row if arg == row_term else parameter_words.get(arg, arg)
            if isinstance(arg, int) and GRADIENT <= arg < GRADIENT + BANK:
                return gradient_base + arg - GRADIENT
            return arg

        lanes = [{name: materialize(arg) for name, arg in initial.items()} for initial in self.initial]
        cached_targets = {}
        if self.cache_slots:
            assert predictions == self.cache_slots
            leader, staged, cached_targets = self.prefix.run(*self.head, 0, self.vocabulary, table=True,
                                                             target_inputs=TargetInputs(row, height, targets),
                                                             table_registers=self.table_registers)
            assert all(self.scratch[col] == code for col, code in staged.items())
            lanes[0] = {name: materialize(arg) for name, arg in leader.items()}
        pc = [0] * 256
        barriers = set()
        layer = 0

        def get(regs, name):
            if name in regs:
                return regs[name]
            if name.startswith("0f"):
                return int(name[2:], 16)
            if name == FACTOR_SYMBOL:
                assert self.table
                return GRADIENT_FACTOR
            if name == WARP_SUM_SYMBOL:
                return WARP_SUMS
            if name == TARGET_CACHE_SYMBOL:
                assert self.cache_slots
                return TARGET_CACHE
            return int(name.removesuffix("U"), 0)

        def address(regs, location):
            pointer, *offset = location[1:-1].replace(" ", "").split("+")
            return get(regs, pointer) + (int(offset[0]) if offset else 0)

        def read_global(ptr, op):
            bank, offset = divmod(ptr, BANK)
            if bank == 2 and op == "ld.global.nc.u64":
                index = offset // 8
                assert offset % 8 == 0 and row <= index < height
                return targets[index - row]
            if bank == 3 and op == "ld.global.nc.f32":
                assert offset % 4 == 0 and offset // 4 < predictions
                return term("weight", offset // 4)
            assert bank == 4 and op == "ld.global.nc.u64" and offset == row * 8
            return prefix_class & ((1 << 64) - 1)

        def expected_lse():
            total = term("add.ftz.f32", warp_sums[0], 0)
            for warp in range(1, 8):
                total = term("add.ftz.f32", total, warp_sums[warp * 4])
            a = term("cvt.rn.ftz.f32.f64", term("parameter", 9, "ld.param.f64"))
            return term("fma.rn.ftz.f32", term("lg2.approx.ftz.f32", total), 0x3f317218, a)

        def expected_probability(byte):
            assert row_lse is not None
            scaled = self.prefix.table_values(byte)[2]
            diff = term("sub.ftz.f32", scaled, row_lse)
            return term("ex2.approx.ftz.f32", term("mul.ftz.f32", diff, 0x3fb8aa3b))

        def expected_base(byte):
            assert row_lse is not None
            total = term("add.ftz.f32", term("weight", 0), 0)
            for i in range(1, predictions):
                total = term("add.ftz.f32", total, term("weight", i))
            if prefix_class >= 0:
                total = term("add.ftz.f32", prefix_scale, total)
            sigma = self.prefix.table_values(byte)[1]
            normalizer = term("mul.ftz.f32", total, expected_probability(byte))
            scaled = term("mul.ftz.f32", self.prefix.gradient_factor(), normalizer)
            differentiated = term("mul.ftz.f32", sigma, scaled)
            one_minus_sigma = term("sub.ftz.f32", 0x3f800000, sigma)
            return term("to_e5m2", term("mul.ftz.f32", one_minus_sigma, differentiated))

        def shared_read(ptr, op, tid):
            if op in ("ld.shared.s32", "ld.shared.u32"):
                assert self.cache_slots and layer == self.output_layer
                word = cached_targets[ptr - TARGET_CACHE]
                return signed(word, 32) if op == "ld.shared.s32" else word
            if isinstance(ptr, ByteAddress):
                if self.table and (ptr.base, ptr.stride) == (EXP_TABLE, 1):
                    assert layer == 2 and op == "ld.shared.u8"
                    return expected_base(ptr.byte)
                assert self.table and layer == 2 and (ptr.base, ptr.stride) == (SIGMOID_TABLE, 8)
                sigma = self.prefix.table_values(ptr.byte)[1]
                if op == "ld.shared.v2.f32":
                    return sigma, expected_probability(ptr.byte)
                assert op == "ld.shared.f32"
                return sigma
            if self.table and ptr == SIGMOID_TABLE + 8 * tid and layer == 1:
                assert self.table_registers is None and op == "ld.shared.v2.f32" and tid not in probabilities
                return self.prefix.table_values(tid)[1:3]
            if self.table and ptr == SIGMOID_TABLE + 8 * tid + 4:
                assert self.table_registers is None and layer == 1 and op == "ld.shared.f32"
                assert tid not in probabilities
                return self.prefix.table_values(tid)[2]
            if ptr == GRADIENT_FACTOR:
                assert self.table and op == "ld.shared.f32"
                return self.prefix.gradient_factor()
            if op == "ld.shared.f32":
                return warp_sums[ptr - WARP_SUMS]
            assert layer >= 1
            return self.scratch[ptr - SHARED]

        # Both paths first publish the warp sums. The fixed path publishes
        # probabilities next. The generic path instead waits for base stores.
        # Divergent loss and correction paths keep their own PCs.
        while True:
            active = [tid for tid, current in enumerate(pc) if 0 <= current < len(self.code) and tid not in barriers]
            if not active:
                if barriers:
                    assert len(barriers) == 256
                    assert len({pc[tid] for tid in barriers}) == 1
                    if self.table and layer == 0:
                        row_lse = expected_lse()
                    if self.table and layer == 1:
                        assert len(probabilities) == 256
                        assert len(base_bytes) == 256
                    barriers.clear()
                    layer += 1
                    continue
                break
            current = min(pc[tid] for tid in active)
            participants = [tid for tid in active if pc[tid] == current]
            pred, op, args = self.code[current]
            for tid in participants:
                pc[tid] += 1
            if op == "ret":
                assert pred is None
                for tid in participants:
                    pc[tid] = -1
                continue
            if op == "bar.sync":
                assert pred is None and args == ["0"]
                barriers.update(participants)
                continue
            dst, *src = args
            if op == "shfl.sync.down.b32":
                assert len(participants) == 256 and pred is None
                read = [get(regs, src[0]) for regs in lanes]
                value, valid = dst.split("|")
                for tid, regs in enumerate(lanes):
                    distance, clamp, mask = (get(regs, a) for a in src[1:])
                    assert clamp == 31 and mask & 0xffffffff == 0xffffffff
                    live = tid % 32 + distance < 32
                    regs[value], regs[valid] = read[tid + distance if live else tid], live
                continue
            for tid in participants:
                regs = lanes[tid]
                if pred and regs[pred.lstrip("!")] == pred.startswith("!"):
                    continue
                if op.startswith("bra"):
                    pc[tid] = self.labels[dst]
                elif op in FLOAT:
                    values = tuple(get(regs, a) for a in src)
                    # The fixed-count code may reverse the initial add
                    # of positive zero. The Lean scalar model proves it.
                    if op == "add.ftz.f32" and values[0] == 0:
                        values = (values[1], 0)
                    regs[dst] = term("mul.ftz.f32" if op == "mul.rn.ftz.f32" else op, *values)
                elif op in INTEGER:
                    values = [get(regs, a) for a in src]
                    if op == "add.s64" and values[0] == EXP_TABLE and isinstance(values[1], Word):
                        code = values[1]
                        assert code.width == 8 and code.parts[1:] == (0, 0, 0)
                        regs[dst] = ByteAddress(code.parts[0], 1, EXP_TABLE)
                    else:
                        regs[dst] = byte_integer(op, values)
                elif op.startswith("setp."):
                    _, comparison, *kind = op.split(".")
                    left, right = (get(regs, a) for a in src)
                    if kind == ["ftz", "f32"]:
                        assert comparison in ("gt", "leu") and left == prefix_scale and right == 0
                        regs[dst] = prefix_positive if comparison == "gt" else not prefix_positive
                    else:
                        bits = int(kind[0][1:])
                        mask = (1 << bits) - 1
                        left, right = left & mask, right & mask
                        if kind[0].startswith("s"):
                            left, right = signed(left, bits), signed(right, bits)
                        regs[dst] = PREDICATES[comparison](left, right)
                elif op in ("and.pred", "or.pred", "not.pred"):
                    values = [regs[a] for a in src]
                    regs[dst] = all(values) if op == "and.pred" else any(values) if op == "or.pred" else not values[0]
                elif op.startswith("selp."):
                    regs[dst] = get(regs, src[0] if regs[src[2]] else src[1])
                elif op == "mov.b32" and dst.startswith("{"):
                    value = get(regs, src[0])
                    assert value.width == 16 and len(value.parts) == 2
                    lo, hi = operands_in(dst)
                    regs[lo], regs[hi] = value.parts
                elif op.startswith(("mov.", "cvta.to.global.")):
                    regs[dst] = get(regs, src[0])
                elif op in ("cvt.s64.s32", "cvt.u64.u32", "cvt.u32.u64", "cvt.u32.u16"):
                    arg = get(regs, src[0])
                    if op == "cvt.s64.s32":
                        regs[dst] = signed(arg, 32)
                    elif op == "cvt.u32.u16" and isinstance(arg, Word):
                        regs[dst] = widen_bytes(arg)
                    elif op == "cvt.u64.u32" and isinstance(arg, Word):
                        assert arg.width == 8 and len(arg.parts) == 4
                        regs[dst] = arg
                    else:
                        regs[dst] = arg & 0xffffffff
                elif op == "cvt.u16.u32":
                    word = get(regs, src[0])
                    assert self.table and word.width == 8 and len(word.parts) == 4
                    regs[dst] = Word(word.parts[:2], 8)
                elif op.startswith("ld.param."):
                    index = int(src[0].removesuffix("]").rsplit("_", 1)[1])
                    regs[dst] = parameter_address(index) if index < 7 else parameters.get(index, term("parameter", index, op))
                elif op.startswith("ld.global."):
                    regs[dst] = read_global(address(regs, src[0]), op)
                elif op == "ld.shared.v4.f32":
                    ptr = address(regs, src[0]) - WARP_SUMS
                    assert self.table and layer == 1 and ptr in (0, 16)
                    values = (warp_sums[ptr + 4 * i] for i in range(4))
                    regs.update(zip(operands_in(dst), values, strict=True))
                elif op == "ld.shared.v2.f32":
                    values = shared_read(address(regs, src[0]), op, tid)
                    regs.update(zip(operands_in(dst), values, strict=True))
                elif op in ("ld.shared.f32", "ld.shared.u16", "ld.shared.u8", "ld.shared.s32", "ld.shared.u32"):
                    regs[dst] = shared_read(address(regs, src[0]), op, tid)
                elif op in ("ld.shared.v4.u32", "ld.shared.v4.u16", "ld.shared.v2.u32"):
                    ptr = address(regs, src[0])
                    stride = 1 if self.table else 2
                    width = 2 if op.endswith("u16") else 4
                    for i, name in enumerate(operands_in(dst)):
                        regs[name] = Word(tuple(shared_read(ptr + i * width + b, op, tid)
                                                for b in range(0, width, stride)), 8 * stride)
                elif op == "st.shared.v2.f32":
                    ptr = address(regs, dst)
                    assert self.table_registers is not None and layer == 1
                    assert ptr == SIGMOID_TABLE + 8 * tid and tid not in probabilities
                    sigma, probability = (get(regs, reg) for reg in operands_in(src[0]))
                    assert sigma == self.prefix.table_values(tid)[1]
                    assert probability == expected_probability(tid)
                    probabilities[tid] = probability
                elif op == "st.shared.f32":
                    ptr = address(regs, dst)
                    value = get(regs, src[0])
                    if layer == 1 and self.table:
                        assert self.table_registers is None
                        assert ptr == SIGMOID_TABLE + 8 * tid + 4 and tid not in probabilities
                        assert value == expected_probability(tid)
                        probabilities[tid] = value
                    else:
                        ptr -= WARP_SUMS
                        assert layer == 0 and ptr not in warp_sums and ptr == tid // 32 * 4 and tid % 32 == 0
                        warp_sums[ptr] = value
                elif op == "st.shared.u8":
                    ptr = address(regs, dst)
                    packed = get(regs, src[0])
                    assert self.table and layer == 1 and ptr == EXP_TABLE + tid and tid not in base_bytes
                    assert packed.width == 8 and packed.parts[0] == expected_base(tid)
                    base_bytes[tid] = packed.parts[0]
                elif op == "cvt.rn.satfinite.e5m2x2.f32":
                    regs[dst] = Word(tuple(term("to_e5m2", get(regs, a)) for a in reversed(src)), 8)
                elif op == "prmt.b32":
                    first, second, selector = (get(regs, a) for a in src)
                    if self.table and 0x4440 <= selector <= 0x4443:
                        regs[dst] = select_byte(first, second, selector)
                        continue
                    if self.table and selector == 0x7604:
                        assert isinstance(first, Term) and isinstance(second, Term)
                        regs[dst] = Word((second, first, 0, 0), 8)
                        continue
                    assert first.width == second.width == 8 and selector == 0x5410
                    packed = first.parts + second.parts
                    regs[dst] = Word(tuple(packed[(selector >> (4 * i)) & 15] for i in range(4)), 8)
                elif op == "bfi.b64":
                    high, low, offset, width = (get(regs, a) for a in src)
                    assert offset == width == 32 and high.width == low.width == 8
                    regs[dst] = Word(low.parts + high.parts, 8)
                elif op == "st.global.f32":
                    ptr = address(regs, dst) - LOSSES
                    assert tid == 0 and ptr == row * 4 and ptr not in loss and layer == self.output_layer
                    loss[ptr] = get(regs, src[0])
                elif op in ("st.global.u64", "st.global.v4.u16"):
                    ptr = address(regs, dst) - gradient_base - row * self.vocabulary
                    if op == "st.global.v4.u16":
                        assert self.table
                        parts = [get(regs, n) for n in operands_in(src[0])]
                        assert all(word.width == 8 and len(word.parts) == 2 for word in parts)
                        packed = Word(tuple(byte for word in parts for byte in word.parts), 8)
                    else:
                        packed = get(regs, src[0])
                    assert packed.width == 8 and len(packed.parts) == 8 and layer == self.output_layer
                    assert 0 <= ptr <= self.vocabulary - 8 and ptr % 8 == 0 and ptr // 8 % 256 == tid
                    for i, byte in enumerate(packed.parts):
                        assert ptr + i not in base and ptr + i not in corrections
                        base[ptr + i] = byte
                elif op == "st.global.u8":
                    ptr = address(regs, dst) - gradient_base - row * self.vocabulary
                    packed = get(regs, src[0])
                    assert packed.width == 8 and ptr in base and layer == self.suffix_barriers
                    if self.table:
                        assert ptr // 8 % 256 == tid and ptr not in corrections
                    assert corrections.setdefault(ptr, packed.parts[0]) == packed.parts[0]
                else:
                    raise ValueError(f"Unhandled CE suffix instruction {op}")
        assert all(index < 0 for index in pc) and layer == self.suffix_barriers
        assert len(warp_sums) == 8 and len(loss) == 1 and base.keys() == set(range(self.vocabulary))
        target_columns = {target for target in targets[:min(predictions, height - row)]
                          if 0 <= target < self.vocabulary}
        if prefix_class >= 0 and prefix_positive:
            target_columns.add(prefix_class)
        assert corrections.keys() == target_columns
        return loss, base, corrections


class DefaultSoftcapPrefix(Prefix):
    """Interpret the launch constants that the fixed-width variant embeds."""

    def __init__(self, vocabulary):
        super().__init__()
        words = [(10, 0x40a00000), (11, 0x40f00000), (12, 0x395b6db7)]
        if vocabulary != 50304:
            words.append((9, 0x41b80000))
        term = super().term
        self.softcap = {term("parameter", port, "ld.param.f64"): bits for port, bits in words}

    def term(self, op, *args):
        if op == "cvt.rn.ftz.f32.f64" and args[0] in self.softcap:
            return self.softcap[args[0]]
        return super().term(op, *args)


def verify_specialized_ce(generic, specialized, vocabulary, predictions):
    prefix = DefaultSoftcapPrefix(vocabulary)
    reference, fixed = (Program(ptx, vocabulary, prefix) for ptx in (generic, specialized))
    unique = tuple([0, 2047, vocabulary - 1][:predictions])
    scenarios = (
        (0, 16384, unique, -1, False),
        (1, 16384, (vocabulary - 1,) * predictions, vocabulary - 1, True),
        (16384 - predictions, 16384, unique, 8, True),
        (16384 - 1, 16384, unique, 0, False),
        (8192, 16384, tuple(2048 * k for k in range(predictions)), 2048 * predictions, True),
        *((16384 - left, 16384, unique, unique[1], True) for left in range(2, predictions)),
    )
    for args in scenarios:
        lhs = reference.execute(predictions, *args)
        rhs = fixed.execute(predictions, *args, reuse_logits=True)
        for i, (original, changed) in enumerate(zip(lhs, rhs, strict=True)):
            if original != changed:
                changed_keys = [k for k in original.keys() | changed.keys() if original.get(k) != changed.get(k)]
                key = min(changed_keys)
                raise AssertionError((vocabulary, predictions, args, i, key, original.get(key), changed.get(key)))
    return len(scenarios)
