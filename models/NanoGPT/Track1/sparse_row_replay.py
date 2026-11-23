"""Replay the Hopper row kernels' integer routing and stored FP expressions."""
from qkv_ptx_replay import address, INT32_OPS, operands_in, PREDICATES, signed32, value
from rope_ptx_replay import atom, pair
from tail_norm_replay import instructions


WIDTH = 768
TILE = 4


class InvalidGradientRow(ValueError):
    """The replay reached the device assertion before any invalid load."""


def replay(program, labels, x, y, params, read_ids, terms, *, ids_param=1, word_kind="fp16-word"):
    writes = {}
    for thread in range(128):
        regs = {"%tid.x": thread, "%ctaid.x": x, "%ctaid.y": y}
        pc = 0
        for _ in program:
            pred, op, operands = program[pc]
            pc += 1
            if pred and regs[pred.removeprefix("!")] == pred.startswith("!"):
                continue
            if op == "ret":
                break
            if op == "bar.sync":
                assert operands == ["0"]
                continue
            dst, *src = operands
            if op in ("bra", "bra.uni"):
                pc = labels[dst]
            elif op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(value(regs, a) for a in src)) & 0xFFFFFFFF
            elif op == "min.s32":
                regs[dst] = min(signed32(value(regs, a)) for a in src) & 0xFFFFFFFF
            elif op == "min.u32":
                regs[dst] = min(value(regs, a) & 0xFFFFFFFF for a in src)
            elif op in ("mov.u32", "mov.b32"):
                item = (tuple(regs[reg] for reg in operands_in(src[0]))
                        if src[0].startswith("{") else value(regs, src[0]))
                if dst.startswith("{"):
                    a, b = operands_in(dst)
                    regs[a], regs[b] = pair(item)
                else:
                    regs[dst] = item
            elif op in ("ld.param.b32", "ld.param.b64"):
                regs[dst] = params[int(src[0].removesuffix("]").rsplit("_", 1)[1])]
            elif op == "bfe.u32":
                n, start, width = (value(regs, a) for a in src)
                regs[dst] = (n >> start) & ((1 << width) - 1)
            elif op.startswith("setp."):
                _, test, kind = op.split(".")
                convert = signed32 if kind == "s32" else lambda x: x & 0xFFFFFFFF
                regs[dst] = PREDICATES[test](*(convert(value(regs, a)) for a in src))
            elif op in ("and.pred", "or.pred"):
                a, b = (regs[name] for name in src)
                regs[dst] = a and b if op == "and.pred" else a or b
            elif op == "not.pred":
                regs[dst] = not regs[src[0]]
            elif op in ("selp.b32", "selp.b64"):
                regs[dst] = value(regs, src[0] if regs[src[2]] else src[1])
            elif op in ("mad.wide.s32", "mul.wide.u32"):
                index, scale, *base = (value(regs, a) for a in src)
                index = signed32(index) if op.startswith("mad") else index & 0xFFFFFFFF
                regs[dst] = index * scale + sum(base)
            elif op == "add.s64":
                regs[dst] = sum(value(regs, a) for a in src)
            elif op == "mov.b64":
                regs[dst] = src[0]
            elif op == "cvta.global.u64":
                regs[dst] = regs[src[0]]
            elif op in ("st.param.b32", "st.param.b64"):
                regs[dst] = value(regs, src[0])
            elif op == "call.uni" and dst == "__assertfail":
                raise InvalidGradientRow
            elif op == "ld.global.b32":
                ptr = address(regs, src[0])
                offset, remainder = divmod(ptr - params[ids_param], 4)
                assert remainder == 0 and 0 <= offset < len(read_ids)
                regs[operands_in(dst)[0]] = read_ids[offset]
            elif op == "ld.global.v4.b32":
                ptr = address(regs, src[0])
                for word, reg in enumerate(operands_in(dst)):
                    regs[reg] = tuple(writes.get(ptr + 4 * word + half, atom(terms, word_kind, ptr + 4 * word + half))
                                      for half in (0, 2))
            elif op == "cvt.f32.f16":
                regs[dst] = atom(terms, op, regs[src[0]])
            elif op == "cvt.rn.bf16.f32":
                regs[dst] = atom(terms, "bf16-rne", regs[src[0]])
            elif op == "st.global.v4.b32":
                ptr = address(regs, dst)
                for word, reg in enumerate(operands_in(src[0])):
                    for half, datum in enumerate(pair(regs[reg])):
                        offset = ptr + word * 4 + half * 2
                        assert offset not in writes
                        writes[offset] = datum
            else:
                raise ValueError(f"Unexpected sparse row instruction {op}")
        else:
            raise ValueError("Missing sparse row return")
    return writes


def verify_sent_clear(ptx):
    code, labels = instructions(ptx)
    source, ids = 1 << 40, 2 << 40
    for count in (0, 1, 2, 3, 4, 5, 6, 7, 37):
        # Include the last real row and the highest row allowed by the 32-bit element guard.
        want = [((1 << 31) - 1) // WIDTH - TILE - 1, 0, 201215, 50303, 150912]
        want.extend(202000 + i * 171 for i in range(max(0, count - len(want))))
        want = want[:count]
        for start in range(0, count + 1, TILE):
            actual = replay(code, labels, start // TILE, 0, (source, ids, count), want, {})
            expected = {source + 2 * (row * WIDTH + col): 0
                        for row in want[start:start + TILE] for col in range(WIDTH)}
            assert actual == expected, (count, start)


def verify_sparse_packets(value_ptx, ngram_ptx, cleared_ptx=None, value_wire_ptx=None):
    for ptx in (value_ptx, ngram_ptx, cleared_ptx, value_wire_ptx):
        if ptx is not None:
            assert "%ctaid.y" not in ptx
    value = instructions(value_ptx)
    ngram = instructions(ngram_ptx)
    value_wire = instructions(value_wire_ptx) if value_wire_ptx is not None else None
    cleared = instructions(cleared_ptx) if cleared_ptx is not None else None
    source, ids, local, wire = (bank << 40 for bank in range(1, 5))
    source_rows = 201216
    cuts = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (19, 19, 19), (23, 1, 2), (32, 7, 20), (37, 13, 37)]
    for rows, lo, hi in cuts:
        want = [(i * 1829 + 414) % source_rows for i in range(rows)]
        own_rows = hi - lo
        peer_rows = rows - own_rows
        own_tiles = (own_rows + TILE - 1) // TILE
        value_args = (source, ids, local, wire, own_rows, peer_rows, lo, source_rows)
        ngram_args = (source, wire, peer_rows, lo, own_rows)
        for bank, original in [(local, list(range(lo, hi))), (wire, [*range(lo), *range(hi, rows)])]:
            for first in range(0, len(original), TILE):
                cta = first // TILE
                value_cta = cta + (own_tiles if bank == wire else 0)
                limit = min(first + TILE, len(original))
                terms = {}
                expected = {bank + 2 * (row * WIDTH + col):
                            atom(terms, "fp16-word", source + 2 * (want[original[row]] * WIDTH + col))
                            for row in range(first, limit) for col in range(WIDTH)}
                actual = replay(*value, value_cta, 0, value_args, want, terms)
                assert actual == expected, (rows, lo, hi, bank, cta)
                if bank == wire and value_wire is not None:
                    assert replay(*value_wire, cta, 0, value_args, want, terms) == expected
                if bank == wire:
                    terms = {}
                    expected = {}
                    erased = {}
                    for row in range(first, limit):
                        for col in range(WIDTH):
                            src = source + 2 * (original[row] * WIDTH + col)
                            half = atom(terms, "fp16-word", src)
                            widened = atom(terms, "cvt.f32.f16", half)
                            expected[wire + 2 * (row * WIDTH + col)] = atom(terms, "bf16-rne", widened)
                            erased[src] = 0
                    assert replay(*ngram, cta, 0, ngram_args, None, terms) == expected
                    if cleared is not None:
                        assert replay(*cleared, cta, 0, ngram_args, None, terms) == expected | erased
        if cleared is not None and not peer_rows:
            assert not replay(*cleared, 0, 0, ngram_args, None, {})
    for invalid in (-1, source_rows):
        paths = [(value, 1, 0), (value, 0, 1)]
        if value_wire is not None:
            paths.append((value_wire, 0, 1))
        for program, own_rows, peer_rows in paths:
            params = (source, ids, local, wire, own_rows, peer_rows, 0, source_rows)
            try:
                replay(*program, 0, 0, params, [invalid], {})
            except InvalidGradientRow:
                continue
            raise AssertionError("The value-gradient kernel did not reject an out-of-range row")


def verify_sparse_fetch(serve_ptx, land_ptx):
    serve, serve_labels = instructions(serve_ptx)
    land, land_labels = instructions(land_ptx)
    shard, received, ids, destination = (bank << 40 for bank in range(1, 5))
    shard_rows = 84_602_880 // 8
    first = 7 * shard_rows
    word_kind = "bf16-word"

    def compare(code, labels, params, want, sources, ids_param):
        for start in range(0, len(want), TILE):
            terms = {}
            actual = replay(code, labels, start // TILE, 0, params, want, terms,
                            ids_param=ids_param, word_kind=word_kind)
            expected = {}
            for row in range(start, min(start + TILE, len(want))):
                bank, offset = sources[row]
                for col in range(WIDTH):
                    src = bank + 2 * (offset * WIDTH + col)
                    dst = destination + 2 * (row * WIDTH + col)
                    expected[dst] = atom(terms, word_kind, src)
            assert actual == expected, (len(want), start, params)

    for count in (1, 7, 32):
        want = [first + shard_rows - 1 - i * 127 for i in range(count)]
        params = (shard, ids, destination, count, first, shard_rows)
        compare(serve, serve_labels, params, want, [(shard, i - first) for i in want], 1)

    for count, lo, hi in [(1, 0, 0), (1, 0, 1), (19, 19, 19), (23, 1, 2), (32, 7, 20), (37, 13, 37)]:
        # Peer want IDs must never be used to read the local shard.
        want = [first + shard_rows - (hi - i) * 29 if lo <= i < hi else -1 for i in range(count)]
        params = (shard, received, ids, destination, count, lo, hi, first, shard_rows)
        sources = [(shard, want[i] - first) if lo <= i < hi
                   else (received, i if i < lo else lo + (i - hi)) for i in range(count)]
        compare(land, land_labels, params, want, sources, 2)

    for invalid in (first - 1, first + shard_rows):
        for code, labels, params, ids_param in [
            (serve, serve_labels, (shard, ids, destination, 1, first, shard_rows), 1),
            (land, land_labels, (shard, received, ids, destination, 1, 0, 1, first, shard_rows), 2),
        ]:
            try:
                replay(code, labels, 0, 0, params, [invalid], {}, ids_param=ids_param, word_kind=word_kind)
            except InvalidGradientRow:
                continue
            raise AssertionError("The sparse fetch kernel did not reject an out-of-range local row")
