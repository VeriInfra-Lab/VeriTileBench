"""Replay logical addresses and opaque FP operations in the pinned SM90 TTGIR."""

from dataclasses import dataclass, field
import operator
import re

import numpy as np


SSA = r"%[\w-]+(?:#\d+)?"
OUTPUT = 1 << 24
ZERO = ("zero",)
INTEGER_OPS = {
    "arith.addi": operator.add, "tt.addptr": operator.add, "arith.subi": operator.sub,
    "arith.muli": operator.mul, "arith.andi": operator.and_, "arith.remsi": operator.mod,
    "arith.divsi": operator.floordiv, "arith.shrui": operator.rshift, "arith.xori": operator.xor,
    "arith.shli": operator.lshift, "arith.ori": operator.or_,
}
COMPARE = {"slt": operator.lt, "sle": operator.le, "sgt": operator.gt, "sge": operator.ge, "ne": operator.ne, "eq": operator.eq}


def addresses(array):
    array = np.asarray(array, dtype=np.int32)
    return array.shape, array.tobytes()


def tensor_shape(types):
    dims = re.findall(r"\btensor<((?:\d+x)+)", types)[-1]
    return tuple(map(int, dims[:-1].split("x")))


def instructions(source):
    source = re.sub(r'loc\("(?:[^"\\]|\\.)*"\(#\w+\)\)|loc\(#\w+\)', "", source)
    body = source.split("tt.func public", 1)[1].split("\n", 1)[1]
    return [line.strip() for line in body.splitlines()
            if line.strip() and not line.lstrip().startswith(("#loc", "//"))]


@dataclass(frozen=True)
class SharedTile:
    buffer: str
    index: int
    transpose: bool = False


def box(row, col, tile, shape):
    tile_rows, tile_cols = tile
    rows, cols = shape
    assert 0 <= row and row + tile_rows <= rows
    assert 0 <= col and col + tile_cols <= cols
    assert col % 8 == 0
    return np.arange(row, row + tile_rows)[:, None] * cols + np.arange(col, col + tile_cols)[None, :]


@dataclass
class TransferBatch:
    parity: int = 0
    expected_bytes: int = 0
    loads: dict = field(default_factory=dict)


class TmaReplay:
    def __init__(self, input_shape, output_shape, published=None):
        self.input_shape = input_shape
        self.published = published
        self.output_shape = output_shape
        self.barriers = {}
        self.tiles = {}
        self.last_dot = {}
        self.issued_dots = 0
        self.input_transfers = 0
        self.retired_dots = 0
        self.fenced_outputs = set()
        self.outputs = {}
        self.pending_stores = {}
        self.pending_writes = []

    def expect(self, key, count):
        barrier = self.barriers[key]
        assert not barrier.expected_bytes and not barrier.loads
        assert count in (32768, 65536)
        barrier.expected_bytes = count

    def load(self, key, dest, row, col, dims):
        if self.published is not None:
            assert (row, col) in self.published, (row, col)
        barrier = self.barriers[key]
        assert barrier.expected_bytes and dest not in barrier.loads
        assert not dest.transpose and (dest.index == key.index or dest.index == 2)
        assert self.last_dot.get(dest, 0) <= self.retired_dots
        self.tiles.pop(dest, None)
        self.input_transfers += 1
        barrier.loads[dest] = box(row, col, dims, self.input_shape)

    def wait(self, key, parity):
        barrier = self.barriers[key]
        assert barrier.parity == parity
        assert barrier.expected_bytes > 0 and len(barrier.loads) == barrier.expected_bytes // 32768
        assert sum(2 * words.size for words in barrier.loads.values()) == barrier.expected_bytes
        self.tiles.update(barrier.loads)
        barrier.loads.clear()
        barrier.expected_bytes = 0
        barrier.parity ^= 1

    def operand(self, key):
        words = self.tiles[key]
        self.last_dot[key] = self.issued_dots
        return words

    def stage_output(self, key, value):
        assert self.issued_dots == self.retired_dots
        assert key not in self.pending_stores
        self.outputs[key] = value

    def store(self, key, row, col):
        assert key in self.fenced_outputs and key not in self.pending_stores
        region = box(row, col, (128, 128), self.output_shape)
        for previous, _ in self.pending_stores.values():
            assert not np.intersect1d(previous, region).size
        self.pending_stores[key] = region, self.outputs[key]

    def finish_store(self):
        assert self.pending_stores
        self.pending_writes.extend((addresses(region), value) for region, value in self.pending_stores.values())
        self.pending_stores.clear()

    def wait_for_writes(self):
        if self.pending_stores:
            self.finish_store()
        assert self.pending_writes
        writes, self.pending_writes = self.pending_writes, []
        return writes


def region(lines, first):
    depth = 0
    other = None
    for i, line in enumerate(lines[first:], first):
        if line == "}":
            if depth == 0:
                return i, other
            depth -= 1
        elif line == "} else {":
            if depth == 0:
                assert other is None
                other = i
        elif line.endswith("{") and ("scf.if " in line or "scf.for " in line):
            depth += 1
    raise AssertionError("Unterminated SCF region")


def replay(program, launch, pid, input_words, output_words, *, tma=False, cooperative=False):
    env = {f"%{n}": v for n, v in launch.items()}
    env.update({"%A_ptr": 0, "%C_ptr": OUTPUT, "%alpha": ("alpha",), "%beta": ("beta",)})
    stores = []
    gram = None
    slots = {}
    buffers = {}
    dma = None
    if tma:
        pitch = launch["K"]
        width = launch["c_stride_r"]
        dma = TmaReplay((input_words // pitch, pitch), (output_words // width, width))
        env.update({"%A": "input", "%C": "output"})
    if cooperative:
        assert tma
        env.update({"%X": "input", "%Gram": "output", "%Output": "unused",
                    "%Gram_ptr": 0, "%Counter": 0})
    committed, ready = 0, 0

    def run(lines):
        nonlocal committed, ready, dma, gram, input_words, stores
        cursor = 0
        while cursor < len(lines):
            line = lines[cursor]
            cursor += 1
            if line.startswith("scf.yield "):
                return [env[x] for x in re.findall(SSA, line.split(" : ", 1)[0])]
            if line == "tt.return":
                return None
            result, eq, expr = line.partition(" = ")
            if not eq or not result.startswith("%"):
                expr, result = line, None
            op, _, rest = expr.partition(" ")
            syntax, _, types = rest.partition(" : ")
            # Attributes contain their own colons in some instructions.
            operands = re.findall(SSA, rest.split(" {", 1)[0].split(" : ", 1)[0])
            if op == "scf.if":
                pred, = operands
                assert np.ndim(env[pred]) == 0
                end, other = region(lines, cursor)
                if env[pred]:
                    path = lines[cursor:other if other is not None else end]
                else:
                    path = lines[other + 1:end] if other is not None else []
                if result is None:
                    run([*path, "tt.return"])
                else:
                    env[result], = run(path)
                cursor = end + 1
                continue
            if op == "scf.for":
                iv, lo, hi, stride = operands[:4]
                args, initial = operands[4::2], operands[5::2]
                values = [env[x] for x in initial]
                end, other = region(lines, cursor)
                assert other is None
                for iteration in range(*(int(env[x]) for x in (lo, hi, stride))):
                    env[iv] = iteration
                    env.update(zip(args, values, strict=True))
                    values = run(lines[cursor:end])
                cursor = end + 1
                base, separator, count = result.partition(":")
                if separator:
                    assert int(count) == len(values)
                    env.update((f"{base}#{i}", value) for i, value in enumerate(values))
                else:
                    env[result], = values
                continue
            if op == "tt.elementwise_inline_asm" and "observe_gram:" in rest:
                assert cooperative and gram is None
                assert not dma.barriers and not dma.pending_stores and not dma.pending_writes
                assert dma.issued_dots == dma.retired_dots
                flag, bit, required = re.findall(
                    r'pure = false} (%[\w-]+), (%[\w-]+), (%[\w-]+) : !tt.ptr<i32>, i32, i32 -> i32$', rest)[0]
                side = launch["c_stride_r"] // 128
                coords = [(r, c) for c in range(side) for r in range(c, side)]
                batch, own = divmod(pid, len(coords))
                assert env[flag] == launch["round_offset"] + batch and env[bit] == 1 << own
                r, c = coords[own]
                published = set()
                mask = 0
                for slot, (i, j) in enumerate(coords):
                    if i in (r, c) or j in (r, c):
                        mask |= 1 << slot
                        published.add(((batch * side + i) * 128, j * 128))
                        published.add(((batch * side + j) * 128, i * 128))
                assert env[required] == mask
                gram = stores, dma.input_transfers
                stores = []
                input_words = output_words
                width = launch["c_stride_r"]
                dma = TmaReplay((input_words // width, width), (output_words // width, width), published)
                env.update({"%Gram": "input", "%Output": "output", "%X": "unused"})
                continue
            v = [env[x] for x in operands]
            if op == "arith.constant":
                literal = syntax.removeprefix("dense<").removesuffix(">")
                value = (ZERO if literal == "0.000000e+00" else
                         {"true": True, "false": False}[literal] if literal in ("true", "false") else int(literal))
            elif op == "tt.get_program_id":
                assert syntax == "x"
                value = pid
            elif op in INTEGER_OPS:
                if op in ("arith.divsi", "arith.remsi", "arith.shrui"):
                    assert np.all(v[0] >= 0) and np.all(v[1] >= (op != "arith.shrui"))
                value = INTEGER_OPS[op](*v)
            elif op == "tt.elementwise_inline_asm" and rest.startswith('"bmsk.clamp.b32 '):
                v = [env[x] for x in re.findall(SSA, rest.rsplit("}", 1)[1].split(" : ", 1)[0])]
                if rest.startswith('"bmsk.clamp.b32 $0, 0, $1;"'):
                    start, length = 0, v[0]
                else:
                    assert rest.startswith('"bmsk.clamp.b32 $0, $1, $2;"')
                    start, length = v
                assert 0 <= start <= start + length <= 32
                value = ((1 << length) - 1) << start
            elif op == "arith.cmpi":
                value = COMPARE[syntax.split(",", 1)[0]](*v)
            elif op == "arith.select":
                value = v[1] if v[0] else v[2]
            elif op in ("arith.extsi", "arith.extui", "arith.trunci", "ttg.convert_layout"):
                value, = v
                if op != "ttg.convert_layout":
                    assert 0 <= value < 2**31
            elif op == "tt.make_range":
                stop, start = map(int, re.findall(r"(?:end|start) = (\d+)", rest))
                value = np.arange(start, stop, dtype=np.int64)
            elif op in ("tt.splat", "tt.broadcast"):
                value, = v
                if not isinstance(value, tuple):
                    value = np.broadcast_to(value, tensor_shape(types))
            elif op == "tt.expand_dims":
                axis, = re.findall(r"axis = (\d+)", rest)
                value = np.expand_dims(v[0], int(axis))
            elif op in ("tt.trans", "ttg.memdesc_trans"):
                assert "order = array<i32: 1, 0>" in rest
                if op == "tt.trans":
                    value = ("transpose", v[0])
                else:
                    desc, = v
                    assert not desc.transpose
                    value = SharedTile(desc.buffer, desc.index, transpose=True)
            elif op == "ttg.local_alloc":
                if not v:
                    ring = re.fullmatch(
                        r": \(\) -> !ttg.memdesc<([1-4])x(?:64x128xbf16|128x64xbf16|128x128xbf16|1xi64), #shared\d*, #smem, mutable>",
                        rest,
                    )
                    if ring is not None:
                        buffers[result] = int(ring[1])
                    else:
                        assert dma is not None
                        assert re.search(r"-> !ttg.memdesc<128x128xbf16, #shared\d*, #smem, mutable>$", rest)
                else:
                    assert dma is not None and len(v) == 1
                    assert re.search(r"-> !ttg.memdesc<128x128xbf16, #shared\d*, #smem, mutable>$", rest)
                    dma.stage_output(result, v[0])
                value = result
            elif op == "ttg.local_store":
                assert dma is not None and len(v) == 2
                assert re.search(r"-> !ttg.memdesc<128x128xbf16, #shared\d*, #smem, mutable>$", rest)
                data, dest = v
                dma.stage_output(dest, data)
                continue
            elif op == "ttg.memdesc_index":
                buffer, index = v
                assert 0 <= index < buffers[buffer]
                value = SharedTile(buffer, index)
            elif op == "ttg.async_copy_global_to_local":
                ptr, desc, mask, *padding = v
                assert not desc.transpose and all(p == ZERO for p in padding)
                assert "contiguity = 8 : i32" in rest
                live = np.broadcast_to(mask, ptr.shape)
                assert np.all((0 <= ptr[live]) & (ptr[live] < input_words))
                value = desc, np.where(live, ptr, -1)
            elif op == "ttg.async_commit_group":
                (desc, words), = v
                committed += 1
                slots[desc] = committed, words
                value = committed
            elif op == "ttg.async_wait":
                pending, = re.findall(r"num = (\d+)", rest)
                assert all(0 < token <= committed for token in v)
                ready = max(ready, committed - int(pending))
                value = ready
            elif op == "ttng.init_barrier":
                assert dma is not None and syntax.endswith(", 1")
                key, = v
                assert key not in dma.barriers
                dma.barriers[key] = TransferBatch()
                continue
            elif op == "ttng.barrier_expect":
                assert dma is not None
                key, enabled = v
                count = int(syntax.split(", ")[1])
                if enabled:
                    dma.expect(key, count)
                continue
            elif op == "ttng.async_tma_copy_global_to_local":
                assert dma is not None
                descriptor, row, col, dst, barrier, enabled = v
                assert descriptor == "input"
                tile = re.search(r"tt.tensordesc<tensor<(\d+)x(\d+)xbf16,", types)
                assert tile is not None
                if enabled:
                    dma.load(barrier, dst, row, col, (int(tile[1]), int(tile[2])))
                continue
            elif op == "ttng.wait_barrier":
                assert dma is not None
                key, phase, enabled, *deps = v
                assert enabled
                assert set(deps) == set(dma.barriers[key].loads)
                dma.wait(key, phase)
                continue
            elif op == "ttng.inval_barrier":
                assert dma is not None and dma.issued_dots == dma.retired_dots
                barrier = dma.barriers.pop(v[0])
                assert not barrier.expected_bytes
                continue
            elif op == "ttng.fence_async_shared":
                assert dma is not None and syntax == "{bCluster = false}"
                dma.fenced_outputs.update(dma.outputs)
                continue
            elif op == "ttng.async_tma_copy_local_to_global":
                assert dma is not None
                descriptor, row, col, src = v
                assert descriptor == "output" and "!tt.tensordesc<tensor<128x128xbf16," in types
                dma.store(src, row, col)
                continue
            elif op == "ttng.async_tma_store_wait":
                assert dma is not None and rest == "{pendings = 0 : i32}"
                dma.finish_store()
                continue
            elif op == "tt.elementwise_inline_asm":
                assert dma is not None and not v
                assert rest.startswith('"cp.async.bulk.wait_group 0; mov.u32 $0, 0;"')
                assert 'constraints = "=r,~{memory}"' in rest and "pure = false" in rest
                stores.extend(dma.wait_for_writes())
                continue
            elif op == "ttng.warp_group_dot":
                assert "{inputPrecision = 0 : i32, isAsync = true}" in rest
                left, right, acc, *use_acc = v
                assert use_acc == ([True] if tma else [])
                terms = []
                if dma is not None:
                    dma.issued_dots += 1
                k_words = 128 if dma is not None else 64
                for desc, shape in zip((left, right), ((128, k_words), (k_words, 128)), strict=True):
                    key = SharedTile(desc.buffer, desc.index)
                    if dma is None:
                        token, array = slots[key]
                        assert token <= ready
                    else:
                        array = dma.operand(key)
                    assert np.all(array >= 0)
                    logical = array.T if desc.transpose else array
                    assert logical.shape == shape
                    terms.append(logical)
                a, b = terms
                value = acc
                for k in range(0, k_words, 64):
                    value = ("dot", value, addresses(a[:, k:k + 64]), addresses(b[k:k + 64, :]))
            elif op == "ttng.warp_group_dot_wait":
                if dma is not None:
                    dma.retired_dots = max(0, dma.issued_dots - int("pendings = 1 : i32" in rest))
                if ":" in result:
                    base, count = result.split(":")
                    assert int(count) == len(v) == 3 and "pendings = 1 : i32" in rest
                    env.update((f"{base}#{i}", val) for i, val in enumerate(v))
                    continue
                assert "pendings = 0 : i32" in rest or "pendings = 1 : i32" in rest
                value, = v
            elif op == "tt.load":
                ptr, *mask = v
                assert not mask or np.all(mask[0])
                assert np.all((0 <= ptr) & (ptr < input_words))
                if tma and dma.published is not None:
                    width = launch["c_stride_r"]
                    row_tiles = np.unique(ptr // width // 128)
                    col_tiles = np.unique(ptr % width // 128)
                    assert len(row_tiles) == len(col_tiles) == 1
                    assert (int(row_tiles[0]) * 128, int(col_tiles[0]) * 128) in dma.published
                value = ("load", addresses(ptr))
            elif op in ("arith.extf", "arith.mulf", "arith.addf", "arith.truncf"):
                value = (op, *v)
            elif op == "tt.store":
                ptr, data, *mask = v
                assert not mask or np.all(mask[0])
                assert np.all((OUTPUT <= ptr) & (ptr < OUTPUT + output_words))
                stores.append((addresses(ptr - OUTPUT), data))
                continue
            elif op == "ttg.local_dealloc":
                assert ready == committed
                continue
            else:
                raise AssertionError(line)
            env[result] = value
        raise AssertionError("Missing TTGIR return or yield")

    run(program)
    if dma is not None:
        assert not dma.barriers and not dma.pending_stores and not dma.pending_writes
        assert dma.issued_dots == dma.retired_dots
    result = stores, dma.input_transfers if tma else 0
    if cooperative:
        assert gram is not None
        return gram, result
    return result


def expected_stores(batch, row, col, m, k, *, tall, polynomial):
    width, reduction = (k, m) if tall else (m, k)
    rows = np.arange(row, row + 128)
    cols = np.arange(col, col + 128)
    base = batch * m * k
    acc = ZERO
    for start in range(0, reduction, 64):
        inner = np.arange(start, start + 64)
        if tall:
            left = rows[:, None] + inner[None, :] * k
            right = inner[:, None] * k + cols[None, :]
        else:
            left = rows[:, None] * k + inner[None, :]
            right = inner[:, None] + cols[None, :] * k
        acc = ("dot", acc, addresses(base + left), addresses(base + right))
    out_base = batch * width * width
    direct = addresses(out_base + rows[:, None] * width + cols[None, :])
    mirror = addresses(out_base + cols[:, None] * width + rows[None, :])
    if polynomial:
        bias = ("arith.extf", ("load", direct))
        acc = ("arith.addf", ("arith.mulf", acc, ("alpha",)), ("arith.mulf", bias, ("beta",)))
    result = ("arith.truncf", acc)
    return [(direct, result), (mirror, ("transpose", result))]
