"""Replay every participating lane of the Hopper ANVIL trace reductions."""
import re

from qkv_ptx_replay import address, operands_in, PREDICATES
from rope_ptx_replay import signed
from tail_norm_replay import instructions
from validation_ce_replay import bits, halves, integer_op, selected


INPUT = 1 << 40
OUTPUT = 2 << 40
TEMP = 3 << 40


def add(left, right):
    if left == right == 0:
        return 0
    return ("add.f32", left, right)


def widen(word):
    return 0 if word == 0 else ("cvt.f32.bf16", word)


def butterfly(lanes, masks):
    for mask in masks:
        lanes = [add(x, lanes[i ^ mask]) for i, x in enumerate(lanes)]
    return lanes[0]


def expected_trace(input_words, dim, batch):
    """The original pair of additions, warp reductions, and final padded sum."""
    def load(row):
        return widen(input_words[INPUT + 2 * (batch * dim * dim + row * (dim + 1))])

    chunks = []
    for start in range(0, dim, 128):
        lanes = []
        for i in range(32):
            lo, hi = load(start + i), load(start + i + 32)
            lanes.append(add(add(add(lo, 0), load(start + i + 64)),
                             add(add(hi, 0), load(start + i + 96))))
        chunks.append(butterfly(lanes, (8, 4, 2, 1, 16)))
    if dim == 256:
        return add(*chunks)
    return butterfly(chunks + [0, 0], (4, 2, 1))


def replay(ptx, blocks, source, input_ptr, output_ptr, *, block_ids=None):
    """Model branch predicates, lane exchange and shared memory for these kernels."""
    threads = int(re.search(r"^\.reqntid (\d+)$", ptx, re.MULTILINE)[1])
    code, labels = instructions(ptx)
    outputs = {}
    for block in range(blocks) if block_ids is None else block_ids:
        lanes = [{"%tid.x": t, "%ctaid.x": block, "global_smem": 0} for t in range(threads)]
        shared, pending = {}, {}
        pc = 0
        for _ in code:
            pred, op, operands = code[pc]
            pc += 1
            active = [(t, r) for t, r in enumerate(lanes) if selected(r, pred)]
            if not active:
                continue
            if op == "ret":
                assert len(active) == threads and not pending
                break
            if op.startswith("bra"):
                assert len(active) == threads
                pc = labels[operands[0]]
                continue
            if op == "bar.sync":
                assert len(active) == threads and operands == ["0"]
                shared.update(pending)
                pending.clear()
                continue
            dst, *src = operands
            if op.startswith("shfl.sync."):
                assert len(active) == threads and src[2:] == ["31", "-1"]
                other = int(src[1])
                assert 0 <= other < 32
                before = [bits(regs, src[0]) for regs in lanes]
                if op == "shfl.sync.bfly.b32":
                    for tid, regs in active:
                        regs[dst] = before[tid ^ other]
                else:
                    assert op == "shfl.sync.idx.b32"
                    for tid, regs in active:
                        regs[dst] = before[(tid & ~31) | other]
                continue
            for _, regs in active:
                def read(a):
                    return bits(regs, a)

                if op == "ld.param.b64":
                    index = int(src[0][:-1].rsplit("_", 1)[1])
                    regs[dst] = (input_ptr, output_ptr)[index]
                elif op.startswith("ld.global."):
                    assert op.endswith(".b16" if input_ptr == INPUT else ".b32")
                    regs[operands_in(dst)[0]] = source[address(regs, src[0])]
                elif op == "st.global.b32":
                    addr = address(regs, dst)
                    assert addr not in outputs
                    outputs[addr] = read(operands_in(src[0])[0])
                elif op == "ld.shared.b32":
                    addr = address(regs, src[0])
                    assert addr not in pending
                    regs[dst] = shared[addr]
                elif op == "st.shared.b32":
                    addr = address(regs, dst)
                    word = read(operands_in(src[0])[0] if src[0].startswith("{") else src[0])
                    assert pending.setdefault(addr, word) == word
                elif op.startswith("mov."):
                    if dst.startswith("{"):
                        lo, hi = operands_in(dst)
                        regs[lo], regs[hi] = halves(read(src[0]))
                    else:
                        regs[dst] = read(src[0])
                elif op == "and.pred":
                    regs[dst] = read(src[0]) and read(src[1])
                elif op.startswith("selp."):
                    regs[dst] = read(src[0] if regs[src[2]] else src[1])
                elif op.startswith("setp."):
                    _, test, ty = op.split(".")
                    a, b = map(read, src)
                    if ty == "s32":
                        a, b = signed(a, 32), signed(b, 32)
                    else:
                        assert ty in ("u32", "b32")
                    regs[dst] = PREDICATES[test](a, b)
                elif op == "add.f32":
                    regs[dst] = add(*map(read, src))
                elif op in ("sqrt.rn.f32", "fma.rn.f32", "mul.f32"):
                    regs[dst] = (op, *map(read, src))
                elif op == "cvt.f32.bf16":
                    regs[dst] = widen(read(src[0]))
                elif op == "createpolicy.fractional.L2::evict_last.b64":
                    pass  # Cache hint only.
                else:
                    regs[dst] = integer_op(op, list(map(read, src)))
        else:
            raise ValueError("Missing trace kernel return")
    return outputs


def check_trees(dim, batch, split_ptx, finish_artifacts, compact_ptx, margin, epsilon):
    inputs = {INPUT + 2 * (b * dim * dim + row * (dim + 1)): ("bf16", b, row)
              for b in range(batch) for row in range(dim)}
    expected = {OUTPUT + 4 * b: expected_trace(inputs, dim, b) for b in range(batch)}
    chunks = dim // 128
    assert batch * chunks % 2 == 0
    partials = replay(split_ptx, batch * chunks // 2, inputs, INPUT, TEMP)
    assert set(partials) == {TEMP + i * 4 for i in range(batch * chunks)}
    for tile, ptx in finish_artifacts:
        assert replay(ptx, (batch + tile - 1) // tile, partials, TEMP, OUTPUT) == expected
    divisors = {}
    for b in range(batch):
        trace = expected[OUTPUT + 4 * b]
        d = ("fma.rn.f32", ("sqrt.rn.f32", trace), margin, epsilon)
        divisors[OUTPUT + 8 * b] = d
        divisors[OUTPUT + 8 * b + 4] = ("mul.f32", d, d)
    assert replay(compact_ptx, batch, inputs, INPUT, OUTPUT) == divisors
