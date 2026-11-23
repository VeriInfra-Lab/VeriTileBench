"""Replay ordered RoPE FP instructions, loads, and packed stores on Hopper."""
import operator

from qkv_ptx_replay import address, operands_in, parse, INT32_OPS, signed32, value, PREDICATES

PARAMS = tuple(i << 40 for i in range(1, 10))
INT16_OPS = {"add.s16": operator.add, "sub.s16": operator.sub,
             "mul.lo.s16": operator.mul, "shr.u16": operator.rshift, "and.b16": operator.and_}
FP_OPS = {
    "add.f32", "mul.f32", "div.full.f32",
    "fma.rn.f32", "rsqrt.approx.ftz.f32", "neg.f32",
}


def atom(terms, op, *operands):
    term = (op, *operands)
    return ("value", terms.setdefault(term, len(terms)))


def read(terms, regs, name):
    return atom(terms, "f32-constant", name) if name.startswith("0f") else value(regs, name)


def signed(n, width):
    n %= 1 << width
    return n - (1 << width) if n >> (width - 1) else n


def pair(word):
    if isinstance(word, int):
        return word & 0xFFFF, (word >> 16) & 0xFFFF
    return word


def negative_bf16(word):
    if isinstance(word, int):
        return word ^ 0x8000
    return ("negative-bf16", word)


def replay(terms, code, cta_x, cta_y, num_threads, key_address=None, *, bf16_load=None):
    if bf16_load is None:
        bf16_load = lambda addr: atom(terms, "bf16-load", addr)
    threads = [
        {"%tid.x": lane, "%ctaid.x": cta_x, "%ctaid.y": cta_y, "global_smem": 0}
        for lane in range(num_threads)
    ]
    output = {}
    shared = {}
    key_inputs = 0
    for pred, op, operands in code:
        if op == "ret":
            return output, key_inputs
        if op == "bar.sync":
            assert pred is None and operands == ["0"]
            continue
        dst, *args = operands
        if op == "shfl.sync.bfly.b32":
            assert pred is None and args[2:] == ["31", "-1"]
            before = [regs[args[0]] for regs in threads]
            delta = int(args[1])
            for lane, regs in enumerate(threads):
                regs[dst] = before[lane ^ delta]
            continue
        for tid, regs in enumerate(threads):
            if pred and not regs[pred]:
                continue
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(read(terms, regs, x) for x in args)) & 0xFFFFFFFF
            elif op == "add.s64":
                regs[dst] = sum(read(terms, regs, x) for x in args)
            elif op in ("bfe.u32", "bfe.s32"):
                x, lo, width = (read(terms, regs, a) for a in args)
                bits = (x >> lo) & ((1 << width) - 1)
                regs[dst] = (signed(bits, width) if op == "bfe.s32" else bits) & 0xffffffff
            elif op in ("cvt.s64.s32", "cvt.u64.u32"):
                x = read(terms, regs, args[0])
                regs[dst] = signed32(x) if op == "cvt.s64.s32" else x & 0xffffffff
            elif op == "min.u32":
                regs[dst] = min(read(terms, regs, a) for a in args)
            elif op in ("shl.b64", "xor.b64"):
                a, b = (read(terms, regs, x) for x in args)
                regs[dst] = ((a << b) if op == "shl.b64" else a ^ b) & ((1 << 64) - 1)
            elif op in INT16_OPS:
                regs[dst] = INT16_OPS[op](*(read(terms, regs, x) for x in args)) & 0xFFFF
            elif op == "cvt.s32.s8":
                regs[dst] = signed(read(terms, regs, args[0]), 8) & 0xFFFFFFFF
            elif op == "cvt.u32.u16":
                regs[dst] = read(terms, regs, args[0]) & 0xFFFF
            elif op in ("cvt.s16.s8", "cvt.u16.u32"):
                a = read(terms, regs, args[0])
                regs[dst] = (signed(a, 8) if op == "cvt.s16.s8" else a) & 0xFFFF
            elif op in ("mad.lo.s32", "mad.wide.s16", "mad.wide.s32", "mad.wide.u32",
                        "mul.wide.s16", "mul.wide.s32", "mul.wide.u32"):
                a, b, *rest = (read(terms, regs, x) for x in args)
                w = 16 if op.endswith("s16") else 32
                if not op.endswith("u32"):
                    a, b = signed(a, w), signed(b, w)
                n = a * b + (rest[0] if rest else 0)
                regs[dst] = n & 0xFFFFFFFF if op == "mad.lo.s32" else n
            elif op in ("mul.hi.s32", "mul.hi.u32"):
                a, b = (read(terms, regs, x) & 0xffffffff for x in args)
                if op.endswith("s32"):
                    a, b = signed32(a), signed32(b)
                regs[dst] = ((a * b) >> 32) & 0xFFFFFFFF
            elif op == "shr.s32":
                a, b = (read(terms, regs, x) for x in args)
                regs[dst] = (signed32(a) >> b) & 0xFFFFFFFF
            elif op == "max.s32":
                regs[dst] = max(signed32(read(terms, regs, a)) for a in args) & 0xFFFFFFFF
            elif op.startswith("setp."):
                _, rel, kind = op.split(".")
                a, b = (read(terms, regs, x) & 0xFFFFFFFF for x in args)
                if kind == "s32":
                    a, b = signed32(a), signed32(b)
                regs[dst] = PREDICATES[rel](a, b)
            elif op in ("and.pred", "or.pred"):
                a, b = (regs[x] for x in args)
                regs[dst] = a and b if op == "and.pred" else a or b
            elif op == "selp.f32":
                regs[dst] = read(terms, regs, args[0] if regs[args[2]] else args[1])
            elif op in ("mov.b32", "mov.u32", "mov.u16", "mov.b16"):
                arg = args[0]
                x = tuple(regs[k] for k in operands_in(arg)) if arg.startswith("{") else read(terms, regs, arg)
                if dst.startswith("{"):
                    a, b = operands_in(dst)
                    regs[a], regs[b] = pair(x)
                else:
                    regs[dst] = x
            elif op in ("ld.global.b16", "ld.global.b32"):
                start = address(regs, args[0])
                if op.endswith("b16") and key_address is not None and key_address <= start < key_address + (1 << 40):
                    key_inputs += 2
                out = operands_in(dst)[0] if dst.startswith("{") else dst
                regs[out] = bf16_load(start) if op.endswith("b16") else atom(terms, "f32-load", start)
            elif op == "ld.param.b64":
                regs[dst] = PARAMS[int(args[0].removesuffix("]").rsplit("_", 1)[1])]
            elif op in ("ld.global.v2.b32", "ld.global.v4.b32"):
                start = address(regs, args[0])
                words = operands_in(dst)
                if key_address is not None and key_address <= start < key_address + (1 << 40):
                    key_inputs += 4 * len(words)
                for j, name in enumerate(words):
                    regs[name] = tuple(bf16_load(start + j * 4 + k * 2) for k in range(2))
            elif op in (
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
                ):
                lane = tid % 32
                first = tid - lane
                for matrix, name in enumerate(operands_in(dst)):
                    if ".trans." in op:
                        indexes = [address(threads[first + matrix * 8 + (lane % 4) * 2 + i], args[0]) +
                                   2 * (lane // 4) for i in range(2)]
                    else:
                        start = address(threads[first + matrix * 8 + lane // 4], args[0]) + 4 * (lane % 4)
                        indexes = [start, start + 2]
                    regs[name] = tuple(shared[i] for i in indexes)
            elif op in ("st.global.v4.b32", "st.shared.v4.b32"):
                start = address(regs, dst)
                for j, name in enumerate(operands_in(args[0])):
                    for k, x in enumerate(pair(regs[name])):
                        offset = start + j * 4 + k * 2
                        if op.startswith("st.shared"):
                            shared[offset] = x
                        else:
                            assert offset not in output
                            output[offset] = x
            elif op == "cvt.rn.bf16.f32":
                regs[dst] = atom(terms, "round-bf16", read(terms, regs, args[0]))
            elif op == "cvt.rn.satfinite.e4m3x2.f32":
                a, b = (read(terms, regs, x) for x in args)
                regs[dst] = (atom(terms, "round-e4m3", b), atom(terms, "round-e4m3", a))
            elif op == "xor.b16":
                a, b = (read(terms, regs, x) for x in args)
                assert b == 0x8000, (op, operands)
                regs[dst] = negative_bf16(a)
            elif op == "cvt.f32.bf16":
                word = read(terms, regs, args[0])
                if isinstance(word, tuple) and word[0] == "negative-bf16":
                    regs[dst] = atom(terms, "neg.f32", atom(terms, op, word[1]))
                else:
                    regs[dst] = atom(terms, op, word)
            elif op in FP_OPS:
                regs[dst] = atom(terms, op, *(read(terms, regs, x) for x in args))
            elif op == "cvt.rn.bf16x2.f32":
                a, b = (read(terms, regs, x) for x in args)
                regs[dst] = (atom(terms, "round-bf16", b), atom(terms, "round-bf16", a))
            else:
                raise ValueError((op, operands))
    raise ValueError("Missing ret")


def gather(terms, outputs, blocks, rows):
    result = {}
    valid = set(rows)
    for block in blocks:
        writes, _ = replay(terms, outputs, block, 0, 64)
        for addr, term in writes.items():
            plane_start = PARAMS[3] if addr < PARAMS[4] else PARAMS[4]
            tok = (addr - plane_start) // (2 * 768)
            if tok in valid:
                assert addr not in result
                result[addr] = term
    return result


def reference_word(terms, output_address):
    is_key = output_address >= PARAMS[4]
    plane_address = PARAMS[4] if is_key else PARAMS[3]
    t, idx = divmod((output_address - plane_address) // 2, 768)
    head, channel = divmod(idx, 128)
    use_previous = is_key and channel >= 64 and t > 0
    fetch_token = t - 1 if use_previous else t
    input_head = head + (6 if is_key else 0)
    packed = PARAMS[0] + 2 * (fetch_token * 2304 + input_head * 128)

    def field(base, offset):
        return atom(terms, "cvt.f32.bf16", atom(terms, "bf16-load", base + 2 * offset))

    def square_sum(lane):
        x = field(packed, 8 * lane + 1)
        subtotal = atom(terms, "mul.f32", x, x)
        for i in (0, 2, 3, 4, 5, 6, 7):
            x = field(packed, 8 * lane + i)
            subtotal = atom(terms, "fma.rn.f32", x, x, subtotal)
        return subtotal

    sums = [square_sum(lane) for lane in range(16)]
    for distance in (8, 4, 2, 1):
        sums = [atom(terms, "add.f32", sums[lane], sums[lane ^ distance]) for lane in range(16)]
    mean = atom(terms, "div.full.f32", sums[channel // 8], atom(terms, "f32-constant", "0f43000000"))
    biased = atom(terms, "add.f32", mean, atom(terms, "f32-constant", "0f34000000"))
    norm = atom(terms, "rsqrt.approx.ftz.f32", biased)
    y = atom(terms, "mul.f32", norm, field(packed, channel))
    if not use_previous:
        x_flip = atom(terms, "mul.f32", norm, field(packed, channel ^ 1))
        f1 = field(PARAMS[1], 128 * t + channel)
        f2 = field(PARAMS[2], 128 * t + channel)
        y = atom(terms, "fma.rn.f32", y, f1, atom(terms, "mul.f32", x_flip, f2))
    return atom(terms, "round-bf16", y)


def verify_rope_shift(reference, candidate, tokens):
    code = [list(parse(kernel)) for kernel in (reference, candidate)]
    terms = {}
    window = {0, 1, 2, tokens // 2, tokens - 2, tokens - 1}
    extra = {tok - 1 for tok in window if tok}
    base = gather(terms, code[0], [3 * tok + i for tok in sorted(window) for i in range(3)], window)
    got = gather(terms, code[1], [3 * tok + i for tok in sorted(window | extra) for i in range(3)], window)
    assert len(base) == len(got) == 2 * len(window) * 768
    for a in base:
        expected = reference_word(terms, a)
        assert got[a] == base[a] == expected, (tokens, a, got[a], base[a], expected)


def backward_words(terms, layout, head, block):
    """Build the ordered derivative independently of the emitted instruction stream."""
    tokens = layout["tokens"]
    qk_dim, v_dim, qkv_dim = (layout[key] for key in ("qk_dim", "v_dim", "qkv_dim"))
    paired, shifted = layout["PAIRED"], layout["KEY_OFFSET"]
    output_head = head % 3 if paired else head
    factor_offset = (head % 2) * qk_dim if paired else 0
    scale = atom(terms, "f32-load", PARAMS[8])
    width = atom(terms, "f32-constant", "0f43000000" if qk_dim == 128 else "0f42800000")
    eps = atom(terms, "f32-constant", "0f34000000")
    zero = atom(terms, "cvt.f32.bf16", 0)

    def load(base, offset):
        return atom(terms, "cvt.f32.bf16", atom(terms, "bf16-load", base + 2 * offset))

    def reduce_dot(left, right):
        chunks = []
        for start in range(0, qk_dim, 16):
            seed = start + 1
            total = atom(terms, "mul.f32", left[seed], right[seed])
            for i in (0, *range(2, 16)):
                j = start + i
                total = atom(terms, "fma.rn.f32", left[j], right[j], total)
            chunks.append(total)
        for distance in ((4, 2, 1) if qk_dim == 128 else (2, 1)):
            chunks = [atom(terms, "add.f32", x, chunks[i ^ distance]) for i, x in enumerate(chunks)]
        return [chunks[i // 16] for i in range(qk_dim)]

    words = {}
    for t in range(block * 32, (block + 1) * 32):
        output_token = 2 * t + head // 3 if paired else t
        factor_start = t * layout["factor_stride_t"] + factor_offset
        for band in range(3):
            if band == 2:
                feature = 12 * qk_dim + head * v_dim
                start = t * layout["stride_grad_v_t"] + head * layout["stride_grad_v_h"]
                y = [load(PARAMS[band], start + d) for d in range(v_dim)]
            else:
                feature = (band * 6 + head) * qk_dim
                which = "q" if band == 0 else "k"
                stride_t, stride_h = (layout[f"stride_grad_{which}_{axis}"] for axis in ("t", "h"))
                start = output_token * stride_t + output_head * stride_h
                y = [load(PARAMS[band], start + d) for d in range(qk_dim)]
                packed = t * layout["stride_qkt"] + (band * 6 + head) * layout["stride_qkh"]
                x = [load(PARAMS[3], packed + d) for d in range(qk_dim)]
                norm = [atom(terms, "rsqrt.approx.ftz.f32",
                             atom(terms, "add.f32", atom(terms, "div.full.f32", s, width), eps))
                        for s in reduce_dot(x, x)]
                unit = [atom(terms, "mul.f32", a, b) for a, b in zip(norm, x)]
                transformed = []
                for d in range(qk_dim):
                    if shifted and band == 1 and d >= 64:
                        successor = load(PARAMS[1], start + stride_t + d) if t + 1 < tokens else zero
                        transformed.append(atom(terms, "add.f32", y[d], successor) if t == 0 else successor)
                    else:
                        f1 = load(PARAMS[4], factor_start + d)
                        f2 = load(PARAMS[5], factor_start + (d ^ 1))
                        transformed.append(atom(terms, "fma.rn.f32", f1, y[d],
                                                atom(terms, "mul.f32", f2, y[d ^ 1])))
                pair = (unit, transformed) if shifted and band == 1 else (transformed, unit)
                correction = [atom(terms, "div.full.f32", x, width) for x in reduce_dot(*pair)]
                dx = [atom(terms, "mul.f32", n,
                           atom(terms, "fma.rn.f32", atom(terms, "neg.f32", u), c, g))
                      for n, u, c, g in zip(norm, unit, correction, transformed)]
                y = [atom(terms, "cvt.f32.bf16", atom(terms, "round-bf16", x)) for x in dx]
            assert feature + len(y) <= qkv_dim
            for d, x in enumerate(y):
                words[t, feature + d] = atom(terms, "round-e4m3", atom(terms, "div.full.f32", x, scale))
    return words


def verify_rope_backward(reference, candidate, layout):
    tokens = layout["tokens"]
    qkv_dim = layout["qkv_dim"]
    shifted = layout["KEY_OFFSET"]
    threads = 256 if layout["qk_dim"] == 128 else 128
    original, optimized = (list(parse(kernel)) for kernel in (reference, candidate))
    key_traffic = None
    sites = [(0, 0), (5, 0), (0, 1), (5, tokens // 32 - 1)]
    if layout["PAIRED"]:
        sites.extend((head, 0) for head in (1, 2, 3, 4))
    for head, block in sites:
        terms = {}
        key_address = PARAMS[1] if shifted else None
        old, original_reads = replay(terms, original, head, block, threads, key_address=key_address)
        new, trimmed_reads = replay(terms, optimized, head, block, threads, key_address=key_address)
        words = backward_words(terms, layout, head, block)
        row = {PARAMS[6] + t * qkv_dim + f for t, f in words if f % 2 == 0}
        transposed = {PARAMS[7] + f * tokens + t for t, f in words if t % 2 == 0}
        assert old.keys() == new.keys() == row | transposed
        assert old == new, (tokens, head, block)
        for (t, f), word in words.items():
            r = PARAMS[6] + t * qkv_dim + f
            c = PARAMS[7] + f * tokens + t
            assert old[r - f % 2][f % 2] == word, ("row", tokens, head, t, f)
            assert old[c - t % 2][t % 2] == word, ("transposed", tokens, head, t, f)
        if shifted:
            assert (original_reads, trimmed_reads) == (
                (16384, 8320) if block == 0 else
                (16128, 8064) if block == tokens // 32 - 1 else
                (16384, 8192)
            ), (tokens, head, block, original_reads, trimmed_reads)
            if block == 1:
                key_traffic = original_reads, trimmed_reads
    return key_traffic


def verify_compact_factors(reference, optimized, tokens, qk_dim, *, backward, stationary=False):
    """Compare stores on one common compact buffer and its original expansion."""
    code = [list(parse(kernel)) for kernel in (reference, optimized)]
    if backward:
        bases = PARAMS[4:6]
        threads = 256 if qk_dim == 128 else 128
        sites = [(h, 0) for h in range(6)] + [(0, 1), (5, tokens // 32 - 1)]
    else:
        bases = PARAMS[1:3]
        threads = 64
        blocks = tokens * 12 // (4 if qk_dim == 128 else 8)
        sites = [(b, 0) for b in (0, 1, 2, 3, blocks // 2, blocks - 2, blocks - 1)]
    for x, y in sites:
        terms = {}
        counts = [[0, 0], [0, 0]]
        outputs = []
        for variant, program in enumerate(code):
            def read_factor(addr):
                for i, base in enumerate(bases):
                    if base <= addr < base + (1 << 40):
                        counts[variant][i] += 2
                        if variant == 0:
                            d = (addr - base) // 2
                            if stationary:
                                row, pair = divmod(d, 64)
                                if pair >= 32:
                                    return 0 if i == 1 else 0x3f80
                                return atom(terms, "bf16-load", base + 2 * (row * 32 + pair))
                            word = atom(terms, "bf16-load", base + 2 * (d // 2))
                            return negative_bf16(word) if i == 1 and d % 2 else word
                        if stationary:
                            assert addr < base + 2 * tokens * 32, (backward, tokens, x, y, addr)
                        break
                return atom(terms, "bf16-load", addr)

            stores, _ = replay(terms, program, x, y, threads, bf16_load=read_factor)
            outputs.append(stores)
        assert outputs[0] and outputs[0] == outputs[1], (backward, qk_dim, tokens, x, y)
        assert [2 * count for count in counts[1]] == counts[0], (backward, qk_dim, tokens, x, y, counts)
    return tuple(sum(variant) for variant in counts)
