"""Replay Hopper n-gram Adam paths and compare their stored expression trees."""
import re

from qkv_ptx_replay import address, INT32_OPS, operands_in, PREDICATES, signed32, value
from rope_ptx_replay import atom, pair, signed
from tail_norm_replay import instructions


TRAINER_BETA = (0x3F733333, 0x3F670A3D)


def multiply(terms, a, b):
    if a == 0 and b in TRAINER_BETA:
        return 0
    return atom(terms, "mul.rn.f32", a, b)


class DeviceAssertion(ValueError):
    pass


def run(code, labels, cta, lanes, params, memory, terms, event, *, reads=None):
    registers = [{"%ctaid.x": cta, "%tid.x": lane, "global_smem": 0} for lane in lanes]
    by_lane = {regs["%tid.x"]: regs for regs in registers}
    shared = {}
    writes = {}

    def fp(regs, name):
        if name.startswith("0f"):
            return int(name[2:], 16)
        if name.startswith("%") and name not in regs:
            return atom(terms, "undefined", regs["%tid.x"], name)
        return value(regs, name)

    def store(ptr, datum):
        assert ptr in memory and ptr not in writes
        writes[ptr] = datum

    pc = 0
    for _ in range(len(code) * (event + 2)):
        predicate, op, args = code[pc]
        pc += 1
        active = [regs for regs in registers
                  if not predicate or regs[predicate.removeprefix("!")] != predicate.startswith("!")]
        if not active:
            continue
        if op == "ret":
            assert len(active) == len(registers)
            return writes
        if op == "bar.sync":
            assert args == ["0"] and len(active) == len(registers)
            continue
        dst, *src = args
        if op in ("bra", "bra.uni"):
            assert len(active) == len(registers), "Expected a warp-uniform loop"
            pc = labels[dst]
            continue
        if op == "redux.sync.min.s32":
            assert len(active) == 32
            assert all(value(regs, src[1]) & 0xFFFFFFFF == 0xFFFFFFFF for regs in active)
            minimum = min(signed32(value(regs, src[0])) for regs in active)
            for regs in active:
                regs[dst] = minimum
            continue
        if op == "shfl.sync.bfly.b32":
            source, distance, limit, mask = src
            assert limit == "31" and mask == "-1"
            shuffled = [fp(by_lane[regs["%tid.x"] ^ int(distance)], source) for regs in active]
            for regs, datum in zip(active, shuffled, strict=True):
                regs[dst] = datum
            continue
        for regs in active:
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(value(regs, a) for a in src)) & 0xFFFFFFFF
            elif op == "neg.s32":
                regs[dst] = -value(regs, src[0]) & 0xFFFFFFFF
            elif op == "mad.lo.s32":
                a, b, c = (value(regs, word) for word in src)
                regs[dst] = (a * b + c) & 0xFFFFFFFF
            elif op in ("mov.b32", "mov.u32", "mov.u16"):
                item = (tuple(fp(regs, r) for r in operands_in(src[0]))
                        if src[0].startswith("{") else fp(regs, src[0]))
                if dst.startswith("{"):
                    a, b = operands_in(dst)
                    regs[a], regs[b] = pair(item)
                else:
                    regs[dst] = item
            elif op == "mov.pred":
                regs[dst] = bool(value(regs, src[0]))
            elif op in ("ld.param.b32", "ld.param.b64"):
                regs[dst] = params[int(src[0].removesuffix("]").rsplit("_", 1)[1])]
            elif op in ("ld.global.b32", "ld.global.L1::evict_last.L2::cache_hint.b32"):
                p = address(regs, src[0])
                if reads is not None:
                    reads.add(p)
                regs[operands_in(dst)[0]] = writes.get(p, memory[p])
            elif op == "st.global.b32":
                store(address(regs, dst), regs[operands_in(src[0])[0]])
            elif op == "ld.global.L1::evict_first.L2::cache_hint.b32":
                ptr = address(regs, src[0])
                regs[operands_in(dst)[0]] = (memory[ptr], memory[ptr + 2])
            elif op in ("ld.global.v4.b32", "ld.global.L1::evict_first.L2::cache_hint.v4.b32"):
                ptr = address(regs, src[0])
                for word, reg in enumerate(operands_in(dst)):
                    lo = ptr + word * 4
                    regs[reg] = (writes.get(lo, memory[lo]), writes.get(lo + 2, memory[lo + 2]))
            elif op == "st.global.v4.b32":
                ptr = address(regs, dst)
                for word, reg in enumerate(operands_in(src[0])):
                    for half, datum in enumerate(pair(regs[reg])):
                        store(ptr + word * 4 + half * 2, datum)
            elif op == "st.shared.b32":
                shared[address(regs, dst)] = fp(regs, src[0])
            elif op == "ld.shared.b32":
                regs[dst] = shared[address(regs, src[0])]
            elif op == "setp.gt.f32":
                regs[dst] = atom(terms, op, *(fp(regs, x) for x in src))
            elif op.startswith("setp."):
                _, test, ty = op.split(".")
                width = int(ty[1:])
                operands = [value(regs, a) for a in src]
                if ty.startswith("s"):
                    operands = [signed(n, width) for n in operands]
                regs[dst] = PREDICATES[test](*operands)
            elif op in ("and.pred", "or.pred"):
                a, b = (regs[x] for x in src)
                regs[dst] = a and b if op == "and.pred" else a or b
            elif op == "not.b32":
                regs[dst] = ~value(regs, src[0]) & 0xFFFFFFFF
            elif op == "bfe.u32":
                word, start, width = (value(regs, x) for x in src)
                regs[dst] = (word >> start) & ((1 << width) - 1)
            elif op in ("min.s32", "add.s64"):
                operands = [value(regs, a) for a in src]
                regs[dst] = min(map(signed32, operands)) if op == "min.s32" else sum(operands)
            elif op == "shl.b64":
                regs[dst] = (value(regs, src[0]) << value(regs, src[1])) & 0xFFFFFFFFFFFFFFFF
            elif op in ("mad.wide.s32", "mad.wide.u32", "mul.wide.s32", "mul.wide.u32", "mad.lo.s64"):
                index, scale, *base = (value(regs, a) for a in src)
                if op != "mad.lo.s64":
                    index = signed32(index) if op.endswith("s32") else index & 0xFFFFFFFF
                regs[dst] = index * scale + sum(base)
            elif op in ("cvt.s64.s32", "cvt.u64.u32", "cvt.u32.u64"):
                n = value(regs, src[0])
                regs[dst] = signed32(n) if op == "cvt.s64.s32" else n & 0xFFFFFFFF
            elif op in ("mul.f32", "mul.rn.f32"):
                regs[dst] = multiply(terms, *(fp(regs, a) for a in src))
            elif op == "selp.f32":
                pred = regs[src[2]]
                if isinstance(pred, bool):
                    regs[dst] = fp(regs, src[0] if pred else src[1])
                else:
                    regs[dst] = atom(terms, op, fp(regs, src[0]), fp(regs, src[1]), pred)
            elif op in ("selp.b32", "selp.b64"):
                regs[dst] = value(regs, src[0] if regs[src[2]] else src[1])
            elif op in ("add.f32", "sub.f32", "fma.rn.f32", "div.full.f32", "sqrt.approx.ftz.f32"):
                regs[dst] = atom(terms, op, *(fp(regs, x) for x in src))
            elif op == "cvt.f32.bf16":
                x = fp(regs, src[0])
                regs[dst] = x << 16 if isinstance(x, int) else atom(terms, op, x)
            elif op == "cvt.rn.bf16x2.f32":
                regs[dst] = tuple(atom(terms, "narrow", fp(regs, x)) for x in reversed(src))
            elif op in ("cvt.f32.f16", "cvt.rn.bf16.f32", "cvt.rn.f16.f32"):
                regs[dst] = atom(terms, op, fp(regs, src[0]))
            elif op in ("atom.global.gpu.relaxed.add.noftz.v2.bf16",
                        "atom.global.gpu.relaxed.add.noftz.v2.f16",
                        "atom.global.gpu.relaxed.add.noftz.v8.f16",
                        "atom.global.gpu.relaxed.add.noftz.v8.bf16"):
                ptr = address(regs, src[0])
                for half, (out, inp) in enumerate(zip(operands_in(dst), operands_in(src[1]), strict=True)):
                    addr = ptr + 2 * half
                    regs[out] = memory[addr]
                    store(addr, atom(terms, op.replace(".v8.", ".v2."), memory[addr], fp(regs, inp)))
            elif op == "atom.global.gpu.relaxed.or.b32":
                ptr = address(regs, src[0])
                regs[dst] = memory[ptr]
                store(ptr, memory[ptr] | value(regs, src[1]))
            elif op == "atom.global.gpu.relaxed.max.s32":
                ptr = address(regs, src[0])
                prev = writes.get(ptr, memory[ptr])
                regs[dst] = prev
                writes[ptr] = max(signed32(prev), signed32(value(regs, src[1]))) & 0xFFFFFFFF
            elif op == "atom.global.gpu.relaxed.exch.b32":
                ptr = address(regs, src[0])
                assert ptr in memory
                regs[dst] = writes.get(ptr, memory[ptr])
                writes[ptr] = value(regs, src[1])
            elif op in ("createpolicy.fractional.L2::evict_first.b64", "createpolicy.fractional.L2::evict_last.b64"):
                assert src == ["1.0"]
                regs[dst] = 0
            elif op == "mov.u64":
                regs[dst] = value(regs, src[0])
            elif op == "mov.b64":
                regs[dst] = src[0]
            elif op in ("cvta.global.u64", "st.param.b32", "st.param.b64"):
                regs[dst] = value(regs, src[0])
            elif op == "call.uni" and dst == "__assertfail":
                raise DeviceAssertion("n-gram or value-row device assertion")
            else:
                raise ValueError(f"Unexpected second-moment instruction {op}")
    raise ValueError("Second-moment replay exceeded the event bound")


def verify_moment_replay(reference, grouped, *, skip_zero=False):
    staged = instructions(reference)
    vector = instructions(grouped)
    moment, stamp, beta, local, want = (bank << 40 for bank in range(1, 6))
    shard_rows = 84_602_880 // 8
    first = 5 * shard_rows
    cases = [(0, 8, 1, True), (1, 0, 1, False), (15, 3, 16, True),
             (33, 29, 12, False), (72, 7, 89, False), (32, 0, 382, False)]
    if skip_zero:
        cases += [(31, 5, 382, False), (32, 0, 382, True)]
    for count, lo, event, fresh in cases:
        rows = [shard_rows - 1 - 307 * j for j in range(count)]
        terms = {}
        history = {beta + 4 * i: TRAINER_BETA[i % 2] if skip_zero else atom(terms, "history-f32", i)
                   for i in range(1, event + 1)}
        memory = dict(history)
        for k, row in enumerate(rows):
            all_zero = skip_zero and count == 31
            bits = 0 if all_zero or (skip_zero and k % 5 == 0) else (
                0x80000000 if skip_zero and k % 5 == 1 else atom(terms, "old-f32", row))
            memory[moment + row * 4] = bits
            memory[stamp + row * 4] = event if fresh or k % 7 == 1 else (k * 17) % (event + 1)
            memory[local + k * 4] = row
            memory[want + (k + lo) * 4] = row + first
        old_args = (None, moment, stamp, beta, None, local, None, event)
        new_args = (moment, stamp, beta, want, count, lo, first, event)
        for cta, start in enumerate(range(0, max(1, count), 32)):
            tile = {}
            for slot in range(start, min(start + 32, count)):
                writes = run(*staged, slot, (0,), old_args, memory, terms, event)
                assert len(writes) == 2 and not (set(tile) & set(writes))
                tile.update(writes)
                assert not run(*staged, slot, (31,), old_args, memory, terms, event)
            reads = set()
            fused = run(*vector, cta, range(32), new_args, memory, terms, event, reads=reads)
            stamps = [memory[stamp + row * 4] for row in rows[start:start + 32]
                      if not skip_zero or memory[moment + row * 4] != 0]
            first_stamp = min([event, *stamps])
            assert reads & history.keys() == {beta + 4 * i for i in range(first_stamp + 1, event + 1)}
            assert set(fused) <= set(tile), (count, lo, event, cta)
            for dst, expected in tile.items():
                assert fused.get(dst, memory[dst]) == expected, (count, lo, event, cta, dst)
                assert (dst in fused) == (memory[dst] != expected)


def verify_adam_entries(ptx):
    code, labels = instructions(ptx)
    want, asked, storage = (bank << 40 for bank in range(1, 4))
    for lo, own, remote, first in [(0, 0, 0, 0), (3, 0, 9, 50), (1, 3, 0, 0),
                                  (17, 127, 131, 57), (3, 128, 7, 0x7fffffff), (5, 257, 65, 107)]:
        count = own + remote
        mirror = storage + ((count + 3) // 4 * 4) * 4
        memory = {want + 4 * (lo + i): (i + 13) * 59 for i in range(own)}
        memory.update({asked + 4 * i: (11 + i % 4) * 43 for i in range(remote)})
        memory.update({storage + 4 * i: None for i in range(count)})
        memory.update({mirror + 4 * i: None for i in range(remote)})
        params = (want, asked, storage, mirror, own, count, lo, first)
        for cta, start in enumerate(range(0, max(count, 1), 128)):
            expected = {}
            for i in range(start, min(start + 128, count)):
                source = want + 4 * (lo + i) if i < own else asked + 4 * (i - own)
                row = (memory[source] - first) & 0xFFFFFFFF
                expected[storage + 4 * i] = row
                if i >= own:
                    expected[mirror + 4 * (i - own)] = row
            assert run(code, labels, cta, range(128), params, memory, {}, 0) == expected, params


def verify_adam_preparation(ptx):
    code, labels = instructions(ptx)
    want, asked, entries, claims, grad = (bank << 44 for bank in range(1, 6))
    shard_rows = 84_602_880 // 8
    first = 5 * shard_rows
    for lo, own, peer, ctas in [
        (3, 0, 0, (0,)), (5, 0, 9, (2, 0, 1)), (4, 3, 0, (0,)),
        (13, 3, 14, (4, 1, 2, 3, 0)), (7, 1 << 22, 7, ((1 << 20) + 1, 1 << 20, 0)),
    ]:
        count = own + peer
        mirror = entries + ((count + 3) // 4 * 4) * 4
        params = want, asked, entries, mirror, claims, grad, own, count, lo, first
        for cta in ctas:
            memory, expected = {}, {}
            # In the emitted scalar path, lanes 0..3 claim entries in
            # this replay order. Repeated IDs overwrite only the claim.
            for slot in range(4 * cta, min(4 * cta + 4, count)):
                row = shard_rows - 1 - (slot % 3)
                original = want + 4 * (lo + slot) if slot < own else asked + 4 * (slot - own)
                memory[original] = first + row
                expected[entries + 4 * slot] = row
                expected[claims + 4 * row] = slot
                if slot >= own:
                    expected[mirror + 4 * (slot - own)] = row
                for column in range(768):
                    expected[grad + 2 * (768 * slot + column)] = 0
            memory.update(dict.fromkeys(expected, 0x7FC1))
            assert run(code, labels, cta, range(128), params, memory, {}, 0) == expected, (lo, own, peer, cta)


def verify_adam_scatter(ptx, local, *, clear_source=False):
    code, labels = instructions(ptx)
    source, ids, accum, map_base = (bank << 40 for bank in range(1, 5))
    row = 84_602_880 // 8 - 9
    for cta, chosen in [(0, 132), (17, 3)]:
        terms = {}
        memory = {ids + 4 * cta: row, map_base + 4 * row: chosen}
        src, dst = source + cta * 1536, accum + chosen * 1536
        expected = {}
        for col in range(0, 768, 2):
            incoming = tuple(atom(terms, "fp16" if local else "bf16", col + h) for h in (0, 1))
            memory[src + 2 * col] = incoming
            if clear_source:
                expected[src + 2 * col] = 0
            for half, word in enumerate(incoming):
                target = dst + (col + half) * 2
                old = atom(terms, "old-bf16", col + half)
                memory[target] = old
                converted = atom(terms, "cvt.rn.bf16.f32", atom(terms, "cvt.f32.f16", word)) if local else word
                expected[target] = atom(terms, "atom.global.gpu.relaxed.add.noftz.v2.bf16", old, converted)
        assert run(code, labels, cta, range(128), (source, ids, accum, map_base), memory, terms, 0) == expected


def verify_tiled_adam_scatter(ptx, *, local):
    code, labels = instructions(ptx)
    source, ids, dest, row_map = (bank << 40 for bank in range(1, 5))
    for count in (0, 1, 3, 4, 5, 8, 37):
        for cta in range((count + 3) // 4 + 1):
            memory, expected, terms = {}, {}, {}
            for row in range(4 * cta, min(4 * cta + 4, count)):
                global_id = 10_575_359 - row
                # A destination beyond the signed 32-bit element offset.
                entry = 3_012_891 - row
                memory[ids + 4 * row] = global_id
                memory[row_map + 4 * global_id] = entry
                for col in range(768):
                    src = source + 2 * (row * 768 + col)
                    dst = dest + 2 * (entry * 768 + col)
                    incoming = atom(terms, "fp16" if local else "bf16", row, col)
                    prior = atom(terms, "bf16", entry, col)
                    memory[src], memory[dst] = incoming, prior
                    if local:
                        incoming = atom(terms, "cvt.rn.bf16.f32", atom(terms, "cvt.f32.f16", incoming))
                        expected[src] = 0
                    expected[dst] = atom(terms, "atom.global.gpu.relaxed.add.noftz.v2.bf16", prior, incoming)
            params = source, ids, dest, row_map, count
            assert run(code, labels, cta, range(128), params, memory, terms, 0) == expected, (count, cta)

    if not local:
        # A peer ID can occur in more than one warp or CTA. Run the warps
        # in different orders and check the resulting ordered BF16 adds.
        rows = (10_575_359, 10_575_358, 10_575_359, 10_575_359, 10_575_358, 10_575_359)
        entries = {10_575_359: 3_012_891, 10_575_358: 3_012_890}
        params = source, ids, dest, row_map, len(rows)
        initial = {row_map + 4 * row: entry for row, entry in entries.items()}
        terms = {}
        for i, row in enumerate(rows):
            initial[ids + 4 * i] = row
            for col in range(768):
                initial[source + 2 * (i * 768 + col)] = atom(terms, "peer-bf16", i, col)
                initial[dest + 2 * (entries[row] * 768 + col)] = atom(terms, "prior-bf16", row, col)
        for order in (range(6), range(5, -1, -1), (5, 2, 1, 4, 0, 3)):
            memory, expected = initial.copy(), initial.copy()
            for i in order:
                cta, warp = divmod(i, 4)
                memory.update(run(code, labels, cta, range(32 * warp, 32 * (warp + 1)), params, memory, terms, 0))
                for col in range(768):
                    incoming = initial[source + 2 * (i * 768 + col)]
                    dst = dest + 2 * (entries[rows[i]] * 768 + col)
                    expected[dst] = atom(terms, "atom.global.gpu.relaxed.add.noftz.v2.bf16", expected[dst], incoming)
            assert memory == expected, order


def verify_ngram_accumulation(ptx):
    code, labels = instructions(ptx)
    source, ids, accum = (bank << 40 for bank in range(1, 4))
    terms = {}
    rows = [393215, 0, 393215]
    destinations = {accum + 2 * (row * 768 + col) for row in rows for col in range(768)}
    memory = dict.fromkeys(destinations, 0)
    memory.update({ids + 4 * i: row for i, row in enumerate(rows)})
    gradient = {(i, col): atom(terms, "sink-bf16", i, col) for i in range(len(rows)) for col in range(768)}
    for i in range(len(rows)):
        for col in range(0, 768, 2):
            memory[source + 2 * (i * 768 + col)] = gradient[i, col], gradient[i, col + 1]

    # Exercise repeated rows in either order and the exact 768 channels.
    for order in ((0, 1, 2), (2, 1, 0)):
        actual = memory.copy()
        expected = dict.fromkeys(destinations, 0)
        for i in order:
            actual.update(run(code, labels, i, range(128), (source, ids, accum, ids), actual, terms, 0))
            for col in range(768):
                ptr = accum + 2 * (rows[i] * 768 + col)
                term = atom(terms, "cvt.f32.bf16", gradient[i, col])
                narrowed = atom(terms, "cvt.rn.f16.f32", term)
                expected[ptr] = atom(terms, "atom.global.gpu.relaxed.add.noftz.v2.f16", expected[ptr], narrowed)
        assert actual == memory | expected, order


def verify_value_scatter(reference, direct, marked, own_updates=()):
    old = instructions(reference)
    new = instructions(direct)
    mark = instructions(marked)
    owner_paths = [(track, instructions(code)) for track, code in own_updates]
    source, gathered_ids, accum, global_ids, found = (bank << 40 for bank in range(1, 6))
    for slot, lo, first, row in [(0, 0, 50304, 0), (3, 17, 50304, 25151),
                                 (11, 0, 176064, 25151), (0, 2**31-3, 222_222, 10_575_109)]:
        terms = {}
        memory = {gathered_ids + 4 * slot: row, global_ids + 4 * (slot + lo): first + row}
        src, dst = source + slot * 1536, accum + row * 1536
        expected = {}
        for col in range(0, 768, 2):
            incoming = tuple(atom(terms, "fp16", col + h) for h in (0, 1))
            memory[src + 2 * col] = incoming
            for half, word in enumerate(incoming):
                address = dst + (col + half) * 2
                prior = atom(terms, "prior-fp16", col + half)
                memory[address] = prior
                expected[address] = atom(terms, "atom.global.gpu.relaxed.add.noftz.v2.f16", prior, word)
        base = run(*old, slot, range(128), (source, gathered_ids, accum, None), memory, terms, 0)
        claim_start = 127
        args = (source, global_ids, accum, lo, first, found, claim_start)
        fused = run(*new, slot, range(128), args, memory, terms, 0)
        assert base == fused == expected, (slot, lo, first, row)
        flag = found + 4 * row
        ticket = claim_start + slot + 1
        for initial in (0, 1, ticket, ticket + 7):
            memory[flag] = initial
            writes = run(*mark, slot, range(128), args, memory, terms, 0)
            assert writes == expected | {flag: max(initial, ticket)}
        full_source = 6 << 40
        owned = full_source + row * 1536
        for col in range(0, 768, 2):
            memory[owned + col * 2] = memory[src + col * 2]
        source_zero = {owned + col * 2: 0 for col in range(0, 768, 2)}
        for track, program in owner_paths:
            own_args = (full_source, *args[1:-1], 0)
            memory[flag] = 0
            writes = run(*program, slot, range(128), own_args, memory, terms, 0)
            assert writes == expected | source_zero | ({flag: slot + 1} if track else {}), (slot, lo, first, row, track)


def verify_value_embed_accum(reference, fused, debug=False):
    old, new = instructions(reference), instructions(fused)
    banks = tuple(p << 40 for p in range(1, 5))
    ids, accumulator = 5 << 40, 6 << 40
    vocab, width = 50304, 768
    terms = {}
    # Replay both orders for duplicate tokens, including boundary IDs and
    # nonzero initial FP16 words in every target address.
    rows = {0: 0, 21: vocab - 1, 492: 86, 49151: vocab - 1}
    memory = {ids + 4 * cta: row for cta, row in rows.items()}
    for cta, row in rows.items():
        for plane, src in enumerate(banks):
            for col in range(width):
                memory[src + 2 * (cta * width + col)] = atom(terms, "value-embed-bf16", plane, cta, col)
                dst = accumulator + 2 * ((plane * vocab + row) * width + col)
                memory[dst] = atom(terms, "initial-fp16", plane, row, col)
    for order in (tuple(rows), tuple(reversed(rows))):
        staged, combined = dict(memory), dict(memory)
        for plane, src in enumerate(banks):
            dest = accumulator + plane * vocab * width * 2
            for cta in order:
                writes = run(*old, cta, range(128), (src, ids, dest), staged, terms, 0)
                assert len(writes) == width
                staged.update(writes)
        for cta in order:
            writes = run(*new, cta, range(128), (*banks, ids, accumulator), combined, terms, 0)
            assert len(writes) == 4 * width
            combined.update(writes)
        assert staged == combined, order
    if debug:
        for invalid in (vocab, 0xFFFFFFFF):
            for program, params in ((old, (banks[0], ids, accumulator)), (new, (*banks, ids, accumulator))):
                try:
                    run(*program, 0, range(128), params, {ids: invalid}, terms, 0)
                except DeviceAssertion:
                    continue
                raise AssertionError("Invalid value row did not assert before accessing the gradient")


def verify_value_normalization(ptx):
    # The inline assembly emits three scalar instructions in one PTX block.
    def expand(match):
        return ";\n".join(match[1].replace("wide", "%inline_wide").split("; ")) + ";"

    code = instructions(re.sub(r"(?m)^\s*\{ \.reg \.f32 wide; (.*); \}$", expand, ptx))
    grad, touched, want, asked = (bank << 40 for bank in range(1, 5))
    cases = ((0, 0, (), ()), (2, 10, (), (0,)),
             (1, 100, (4, 7, 0), ()), (7, 200, (0, 3, 99), (1, 0, 0, 3, 1, 8)),
             (3, 1000, (36, 0, 1, 7), (1, 8, 0, 7, 36, 8, 9)))
    for lo, first, local, peers in cases:
        rows = local + peers
        memory = {want + 4 * (lo + i): first + row for i, row in enumerate(local)}
        memory.update({asked + 4 * i: first + row for i, row in enumerate(peers)})
        winner = {row: slot for slot, row in enumerate(rows)}
        memory.update({touched + 4 * row: slot + 1 for row, slot in winner.items()})
        terms = {}
        for row in winner:
            for col in range(768):
                memory[grad + 2 * (row * 768 + col)] = atom(terms, "fp16", row, col)
        for cta in range((len(rows) + 3) // 4 + 1):
            expected = {}
            for row, slot in winner.items():
                if slot // 4 == cta:
                    for col in range(768):
                        ptr = grad + 2 * (row * 768 + col)
                        wide = atom(terms, "cvt.f32.f16", memory[ptr])
                        scaled = atom(terms, "mul.rn.f32", wide, 0x3e000000)
                        expected[ptr] = atom(terms, "cvt.rn.f16.f32", scaled)
            params = grad, touched, want, asked, len(local), len(rows), lo, first
            assert run(*code, cta, range(128), params, memory, terms, 0) == expected, (lo, len(rows), cta)


def verify_value_clear(ptx):
    code, labels = instructions(ptx)
    ops = [op for _, op, _ in code]
    barrier, = (i for i, op in enumerate(ops) if op == "bar.sync")
    assert ops.count("st.global.v4.b32") == 3
    assert all(i < barrier for i, op in enumerate(ops) if op in ("ld.global.b32", "st.global.v4.b32"))
    assert all(i > barrier for i, op in enumerate(ops) if op == "st.global.b32")
    grad, touched, want, asked = (bank << 40 for bank in range(1, 5))
    cases = ((0, 0, (), ()), (2, 10, (), (0,)),
             (1, 100, (4, 7, 0), ()), (7, 200, (0, 3, 99), (1, 0, 0, 3, 1, 8)),
             (3, 1000, (36, 0, 1, 7), (1, 8, 0, 7, 36, 8, 9)))
    for lo, first, local, peers in cases:
        rows = local + peers
        memory = {want + 4 * (lo + i): first + row for i, row in enumerate(local)}
        memory.update({asked + 4 * i: first + row for i, row in enumerate(peers)})
        winner = {row: slot for slot, row in enumerate(rows)}
        memory.update({touched + 4 * row: slot + 1 for row, slot in winner.items()})
        memory[touched + 4 * 100] = 0
        terms = {}
        for row in (*winner, 100):
            for col in range(768):
                ptr = grad + 2 * (row * 768 + col)
                memory[ptr] = atom(terms, "fp16", row, col)
        ctas = (len(rows) + 3) // 4 + 1
        for order in (range(ctas), range(ctas - 1, -1, -1)):
            observed = memory.copy()
            for cta in order:
                params = grad, touched, want, asked, len(local), len(rows), lo, first
                writes = run(code, labels, cta, range(128), params, observed, terms, 0)
                expected = {}
                for row, slot in winner.items():
                    if slot // 4 == cta:
                        expected[touched + 4 * row] = 0
                        expected.update((grad + 2 * (row * 768 + col), 0) for col in range(768))
                assert writes == expected, (lo, len(rows), cta, order)
                observed.update(writes)
            assert observed[grad + 2 * 100 * 768] == memory[grad + 2 * 100 * 768]


def verify_adam_row(ptx, *, guarded=False, skip_zero=False):
    code, labels = instructions(ptx)
    weights, moment, stamp, beta, gradient, entries, owner = (bank << 40 for bank in range(1, 8))
    row = 84_602_880 // 8 - 17
    cases = [(0, 1, 0, 0, None), (7, 93, 88, 7, None), (31, 382, 380, 30, None)]
    if skip_zero:
        cases += [(15, 382, 0, 15, 0), (19, 10, 2, 19, 0x80000000)]
    for slot, event, last, chosen, initial in cases:
        terms = {}
        old = atom(terms, "moment") if initial is None else initial
        history = {e: TRAINER_BETA[e % 2] if skip_zero else atom(terms, "beta", e) for e in range(1, event)}
        b, eps, size, decay, grad_mul, sq_mul = [atom(terms, n) for n in
                                                ("beta", "eps", "size", "decay", "grad-mul", "sq-mul")]
        params = (weights, moment, stamp, beta, gradient, entries, owner, event,
                  b, eps, size, decay, grad_mul, sq_mul)
        m_addr, s_addr = moment + row * 4, stamp + row * 4
        p_addr, g_addr = weights + row * 1536, gradient + slot * 1536
        memory = {m_addr: old, s_addr: last, entries + 4 * slot: row, owner + 4 * row: chosen}
        memory.update({beta + e * 4: datum for e, datum in history.items()})
        g, p = [], []
        for col in range(768):
            grad = atom(terms, "gradient-bf16", col)
            weight = atom(terms, "weight-bf16", col)
            memory[g_addr + col * 2], memory[p_addr + col * 2] = grad, weight
            g.append(atom(terms, "cvt.f32.bf16", grad))
            p.append(atom(terms, "cvt.f32.bf16", weight))
        reads = set()
        actual = run(code, labels, slot, range(128), params, memory, terms, event, reads=reads)
        if chosen != slot:
            assert not actual
            if guarded:
                assert reads == {entries + 4 * slot, owner + 4 * row}
            continue
        expected_reads = [] if skip_zero and old == 0 else range(last + 1, event)
        assert reads & {beta + e * 4 for e in history} == {beta + 4 * e for e in expected_reads}
        for e in range(last + 1, event):
            old = multiply(terms, old, history[e])

        def butterfly(values, stages):
            for distance in stages:
                values = [atom(terms, "add.f32", x, values[i ^ distance]) for i, x in enumerate(values)]
            return values[0]

        thread_sums = []
        padded = g + [0] * 256
        for t in range(128):
            x = padded[8*t:8*(t+1)]
            acc = atom(terms, "mul.rn.f32", x[1], x[1])
            for k in (0, 2, 3, 4, 5, 6, 7):
                acc = atom(terms, "fma.rn.f32", x[k], x[k], acc)
            thread_sums.append(acc)
        warp_sums = [butterfly(thread_sums[i:i+32], (16, 8, 4, 2, 1)) for i in range(0, 128, 32)]
        mean = atom(terms, "div.full.f32", butterfly(warp_sums, (2, 1)), 0x44400000)
        variance = atom(terms, "fma.rn.f32", b, old, atom(terms, "mul.rn.f32", sq_mul, mean))
        denom = atom(terms, "add.f32", eps, atom(terms, "sqrt.approx.ftz.f32", variance))
        expected = {m_addr: variance, s_addr: event}
        for col, (grad, weight) in enumerate(zip(g, p, strict=True)):
            direction = atom(terms, "div.full.f32", atom(terms, "mul.rn.f32", grad_mul, grad), denom)
            scaled = atom(terms, "mul.rn.f32", size, direction)
            gate = atom(terms, "setp.gt.f32", atom(terms, "mul.rn.f32", scaled, weight), 0)
            wd = atom(terms, "selp.f32", atom(terms, "mul.rn.f32", decay, weight), 0, gate)
            update = atom(terms, "fma.rn.f32", size, direction, wd)
            expected[p_addr + 2 * col] = atom(terms, "narrow", atom(terms, "sub.f32", weight, update))
        assert actual == expected, (slot, event, last)
