"""Replay the n-gram PTX with symbolic BF16 words and concrete addresses."""

from itertools import product

from check_graph_stage import instructions, register_names
from qkv_ptx_replay import INT32_OPS, PREDICATES, WORD32, address, signed32, value


WIDTH = 768
ROW_MASK = 8191
CACHE_ROWS = 5_000_003
PARAM_STRIDE = 1 << 44
HASHES = dict(B0=48271, B1=30011, T0=58699, T1=39779, T2=26801)


def token_word(token):
    # Vary the low bits and include both signed int32 halves.
    return (1_664_525 * token + 2_654_435_761) & WORD32


def slot_word(index):
    return (231_883 * index + 3_000_001) % CACHE_ROWS


def signs(token):
    cur = token_word(token)
    prev = token_word(token - 1)
    prior = token_word(token - 2)
    b = (HASHES["B0"] * cur ^ HASHES["B1"] * prev) & ROW_MASK if token else 0
    t = (HASHES["T0"] * cur ^ HASHES["T1"] * prev ^ HASHES["T2"] * prior) & ROW_MASK if token >= 2 else 0
    return b, t


def load_word(buffer, offset, tokens):
    if buffer == "input":
        assert offset % 4 == 0 and 0 <= offset < tokens * 4
        return token_word(offset // 4)
    if buffer == "slots":
        assert offset % 4 == 0 and 0 <= offset < 2 * tokens * 4
        return slot_word(offset // 4)
    elements = {"cache": CACHE_ROWS, "pool": ROW_MASK + 1, "grad": tokens}[buffer] * WIDTH
    assert offset % 4 == 0 and 0 <= offset and offset + 4 <= 2 * elements, (buffer, offset, elements)
    return (buffer, offset // 2), (buffer, offset // 2 + 1)


def run_thread(code, coordinates, parameters, banks, tokens, stores):
    regs = dict(zip(("%ctaid.x", "%ctaid.y", "%tid.x"), coordinates, strict=True))
    for predicate, op, args in code:
        if predicate and bool(regs[predicate.removeprefix("!")]) == predicate.startswith("!"):
            continue
        if op == "ret":
            return
        dst, *src = args
        if op in ("ld.param.b64", "ld.param.b32"):
            regs[dst] = parameters[int(src[0][:-1].rsplit("_", 1)[1])]
            continue
        if op.startswith("ld.global."):
            names = register_names(dst)
            assert op.endswith(".v4.b32" if len(names) == 4 else ".b32")
            ptr = address(regs, src[0])
            bank, offset = divmod(ptr, PARAM_STRIDE)
            for i, name in enumerate(names):
                regs[name] = load_word(banks[bank - 1], offset + 4 * i, tokens)
            continue
        if op == "st.global.v4.b32":
            bank, offset = divmod(address(regs, dst), PARAM_STRIDE)
            assert banks[bank - 1] == "output" and offset % 16 == 0
            values = [word for name in register_names(src[0]) for word in regs[name]]
            assert len(values) == 8
            for i, word in enumerate(values, offset // 2):
                assert i not in stores
                stores[i] = word
            continue
        if op == "createpolicy.fractional.L2::evict_last.b64":
            assert src == ["1.0"]
            regs[dst] = 0
            continue
        if op == "mov.b32":
            a, b = register_names(dst)
            regs[a], regs[b] = regs[src[0]]
            continue
        words = [value(regs, name) for name in src]
        if op in INT32_OPS:
            result = INT32_OPS[op](*words) & WORD32
        elif op in ("mov.u32", "mov.u64"):
            result, = words
        elif op == "add.s64":
            result = sum(words)
        elif op == "shr.s32":
            result = (signed32(words[0]) >> words[1]) & WORD32
        elif op == "mul.hi.s32":
            result = (signed32(words[0]) * signed32(words[1]) >> 32) & WORD32
        elif op.startswith(("mul.wide.", "mad.wide.", "mad.lo.")):
            lhs, rhs, *bias = words
            if op.endswith(".s32"):
                lhs, rhs = signed32(lhs), signed32(rhs)
            result = lhs * rhs + (bias[0] if bias else 0)
            if op.startswith("mad.lo."):
                result &= WORD32
        elif op == "bfe.u32":
            n, start, width = words
            result = (n >> start) & ((1 << width) - 1)
        elif op.startswith("setp."):
            _, compare, dtype = op.split(".")
            assert dtype in ("s32", "u32")
            result = PREDICATES[compare](*(map(signed32, words) if dtype == "s32" else words))
        elif op == "and.pred":
            result = all(words)
        elif op == "selp.b32":
            a, b, cond = words
            result = a if cond else b
        elif op == "cvt.f32.bf16":
            result = ("widen", words[0])
        elif op in ("fma.rn.f32", "mul.f32"):
            result = (op, *words)
        elif op == "cvt.rn.bf16x2.f32":
            result = tuple(("narrow", expr) for expr in reversed(words))
        elif op == "mul.bf16x2":
            result = tuple(("mul.bf16", a, b) for a, b in zip(*words, strict=True))
        else:
            raise AssertionError(f"Unexpected n-gram opcode {op}")
        regs[dst] = result
    raise AssertionError("Missing n-gram return")


def expected_word(forward, offset, tokens):
    plane, position = divmod(offset, tokens * WIDTH)
    token, channel = divmod(position, WIDTH)
    b, t = signs(token)
    sa, sb = (("pool", row * WIDTH + channel) for row in (b, t))
    if not forward:
        assert plane < 2
        return "mul.bf16", ("grad", position), (sa, sb)[plane]
    assert plane == 0
    a, b = (("cache", slot_word(i) * WIDTH + channel) for i in (token, tokens + token))
    a, b, sa, sb = (("widen", word) for word in (a, b, sa, sb))
    return "narrow", ("fma.rn.f32", a, sa, ("mul.f32", b, sb))


def replay(ptx, forward, layout, tokens, *, all_tiles):
    direction = "fwd" if forward else "bwd"
    code, _, threads = instructions(ptx, f"_ngram_embed_{direction}_kernel")
    banks = ("cache", "slots", "pool", "input", "output") if forward else ("grad", "pool", "input", "output")
    params = tuple(PARAM_STRIDE * i for i in range(1, len(banks) + 1))
    if forward or layout != "flat":
        params += (tokens,)
    if layout == "flat":
        params += (tokens * WIDTH,)
    nblocks = (tokens * WIDTH + 1023) // 1024 if layout == "flat" else (tokens + 3) // 4
    rows = range(nblocks) if all_tiles else (0, 1, nblocks - 1)
    tiles = range(3) if layout == "tiled" else (0,)
    stores, expected = {}, set()
    for row, tile in product(rows, tiles):
        for thread in range(threads):
            run_thread(code, (row, tile, thread), params, banks, tokens, stores)
        if layout == "flat":
            indices = range(row * 1024, min((row + 1) * 1024, tokens * WIDTH))
        else:
            channels = range(tile * 256, (tile + 1) * 256) if layout == "tiled" else range(WIDTH)
            indices = (t * WIDTH + c for t, c in product(range(row * 4, min((row + 1) * 4, tokens)), channels))
        for offset in indices:
            expected.add(offset)
            if not forward:
                expected.add(tokens * WIDTH + offset)
    assert stores.keys() == expected
    for offset, result in stores.items():
        assert result == expected_word(forward, offset, tokens), (layout, direction, offset, result)


def compile_and_replay(reference, candidate, build, ir_dir):
    types = dict(cache_ptr="*bf16", slots_ptr="*i32", pool_ptr="*bf16", inp_ptr="*i32",
                 out_ptr="*bf16", g_ptr="*bf16", T="i32", xnumel="i32")
    for label, kernels, tile in (
        ("flat", reference, dict(XBLOCK=1024)),
        ("tiled", candidate, dict(BT=4, BC=256, FULL_ROWS=False)),
        ("full", candidate, dict(BT=4, BC=256, FULL_ROWS=True)),
    ):
        for forward, kernel in zip((True, False), kernels[1:], strict=True):
            operands = {arg: types[arg] for arg in kernel.arg_names if arg in types}
            binary = build(kernel, dict(D=WIDTH, RMASK=ROW_MASK, **HASHES, **tile), operands,
                           aligned_args=True, num_warps=4, num_stages=1)
            ptx = binary.asm["ptx"]
            replay(ptx, forward, label, 48, all_tiles=True)
            replay(ptx, forward, label, 262144, all_tiles=False)
            if ir_dir is not None:
                direction = "fwd" if forward else "bwd"
                (ir_dir / f"ngram-embed-{label}-{direction}.ptx").write_text(ptx)
    print("N-gram embedding PTX: exact BF16 expressions, load addresses and both backward planes checked.")

