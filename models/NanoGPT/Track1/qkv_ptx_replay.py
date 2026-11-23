"""Symbolic replay of the emitted Hopper packers and sampled row gather.

The replay tags BF16 loads by address, checks the scalar instruction chain,
and follows the resulting FP8 byte through the actual PTX. It also tracks
the BF16 inputs of each MLP amax without executing floating-point arithmetic.
"""

from dataclasses import dataclass
import operator
import re


PARAMETERS = tuple((i + 1) << 40 for i in range(5))
OPERAND = re.compile(r"\{[^}]*\}|\[[^]]*\]|[^,\s]+")
WORD32 = (1 << 32) - 1
INT32_OPS = {
    "add.s32": operator.add, "sub.s32": operator.sub, "mul.lo.s32": operator.mul,
    "and.b32": operator.and_, "or.b32": operator.or_,
    "shl.b32": operator.lshift, "shr.u32": operator.rshift, "xor.b32": operator.xor,
}
PREDICATES = {"lt": operator.lt, "le": operator.le, "ge": operator.ge, "gt": operator.gt,
              "eq": operator.eq, "ne": operator.ne}


@dataclass(frozen=True)
class Value:
    stage: str
    source: int
    scale: int | None = None


def signed32(bits):
    bits &= WORD32
    return bits - (1 << 32) if bits >> 31 else bits


def parse(instructions):
    for line in instructions:
        predicate = None
        if line.startswith("@"):
            predicate, line = line[1:].split(maxsplit=1)
        opcode, *operands = line.rstrip(";").split(maxsplit=1)
        yield predicate, opcode, OPERAND.findall(operands[0]) if operands else []


def operands_in(group):
    return group[1:-1].replace(" ", "").split(",")


def packed(registers, group):
    return tuple(byte for reg in operands_in(group) for byte in registers[reg])


def distribute(registers, group, data, word_bytes):
    names = operands_in(group)
    assert len(data) == len(names) * word_bytes
    for i, name in enumerate(names):
        registers[name] = data[i * word_bytes:(i + 1) * word_bytes]


def value(registers, name):
    if name.startswith("%") or name == "global_smem":
        return registers[name]
    return int(name.removesuffix("U"), 0)


def address(registers, memory):
    base, *offset = memory[1:-1].replace(" ", "").split("+")
    return registers[base] + (int(offset[0]) if offset else 0)


def write_bytes(target, start, data):
    for i, byte in enumerate(data):
        assert start + i not in target
        target[start + i] = byte


def replay_tile(instructions, batch, row_tile, col_tile, parameters=PARAMETERS, reductions=None,
                candidate_ids=None, raw_bytes=False, warp_local=True, labels=None, warp_scratch_banks=1, head_inverse_bits=None):
    packed_head = head_inverse_bits is not None
    packed_gain = head_inverse_bits == 0x435fffff
    threads = [
        {"%tid.x": t, "%ctaid.z": batch, "%ctaid.x": row_tile, "%ctaid.y": col_tile, "global_smem": 0}
        for t in range(256)
    ]
    output = {}
    scratch = {}
    warp_writes = {}
    scratch_reduction = {}
    pending_reduction = {}
    shared_exchanges = 0
    cta_barriers = 0
    indexed_writes = set()
    cursor = 0
    for _ in range(len(instructions) * (17 if labels is not None else 1)):
        predicate, opcode, args = instructions[cursor]
        cursor += 1
        if opcode == ".reg":
            quantize_registers = [".b32", "gain" if packed_gain else "rounded", "upper"]
            assert packed_head and predicate is None
            assert args in (quantize_registers, [".b16", "lo", "hi"], [".b32", "f16s"])
            scratch_registers = args[1:]
            for registers in threads:
                for name in scratch_registers:
                    registers.pop(name, None)
            continue
        if opcode in ("bra", "bra.uni"):
            active = [predicate is None or r[predicate.removeprefix("!")] != predicate.startswith("!")
                      for r in threads]
            assert labels is not None and all(v == active[0] for v in active)
            if active[0]:
                cursor = labels[args[0]]
            continue
        if opcode in ("bar.warp.sync", "bar.sync"):
            assert predicate is None and args == (["-1"] if opcode == "bar.warp.sync" else ["0"])
            if warp_local:
                scratch.update(warp_writes)
                warp_writes.clear()
            if opcode == "bar.sync":
                cta_barriers += 1
                scratch_reduction.update(pending_reduction)
                pending_reduction.clear()
            continue
        if opcode == "redux.sync.add.s32":
            assert predicate is None
            dst, src, mask = args
            for start in range(0, len(threads), 32):
                warp = threads[start:start + 32]
                assert all(value(r, mask) & WORD32 == WORD32 for r in warp)
                total = sum(signed32(value(r, src)) for r in warp) & WORD32
                for registers in warp:
                    registers[dst] = total
            continue
        if opcode == "shfl.sync.bfly.b32":
            assert predicate is None and args[3:] == ["31", "-1"]
            dst, src, distance = args[:3]
            before = [r[src] for r in threads]
            assert 0 < int(distance) < 32
            for tid, registers in enumerate(threads):
                registers[dst] = before[tid ^ int(distance)]
            continue
        if opcode == "ret":
            assert not args
            assert not pending_reduction
            return output
        if opcode == "ldmatrix.sync.aligned.m8n8.x4.shared.b16":
            shared_exchanges += 1
        for tid, registers in enumerate(threads):
            dst, *src = args
            if predicate is not None and registers[predicate.removeprefix("!")] == predicate.startswith("!"):
                if opcode == "ld.shared.b32":
                    registers.setdefault(dst, None)
                continue
            if opcode in INT32_OPS:
                registers[dst] = INT32_OPS[opcode](*(value(registers, a) for a in src)) & WORD32
            elif opcode == "mul.hi.u32":
                a, b = (value(registers, arg) & WORD32 for arg in src)
                registers[dst] = (a * b) >> 32
            elif opcode == "shr.s32":
                a, b = (value(registers, arg) for arg in src)
                registers[dst] = (signed32(a) >> b) & WORD32
            elif opcode == "min.u32":
                registers[dst] = min(value(registers, arg) & WORD32 for arg in src)
            elif opcode in ("bfe.u32", "bfe.s32"):
                word, offset, width = (value(registers, a) for a in src)
                bits = (word >> offset) & ((1 << width) - 1)
                if opcode == "bfe.s32" and bits >> (width - 1):
                    bits -= 1 << width
                registers[dst] = bits
            elif opcode == "add.s64":
                registers[dst] = sum(value(registers, a) for a in src)
            elif opcode == "mul.lo.s64":
                a, b = (value(registers, arg) for arg in src)
                registers[dst] = (a * b) & ((1 << 64) - 1)
            elif opcode == "shr.s64":
                a, b = (value(registers, arg) for arg in src)
                registers[dst] = (a - ((a >> 63) << 64)) >> b
            elif opcode in ("or.b64", "shl.b64"):
                a, b = (value(registers, arg) for arg in src)
                registers[dst] = (a | b) if opcode == "or.b64" else (a << b) & ((1 << 64) - 1)
            elif opcode == "mad.lo.s32":
                a, b, c = (value(registers, arg) for arg in src)
                registers[dst] = (a * b + c) & WORD32
            elif opcode in ("mul.wide.u32", "mul.wide.s32"):
                a, b = (value(registers, arg) & WORD32 for arg in src)
                if opcode == "mul.wide.s32":
                    a, b = signed32(a), signed32(b)
                registers[dst] = a * b
            elif opcode.startswith("mad.wide."):
                offset, factor, base = (value(registers, a) for a in src)
                offset = signed32(offset) if opcode.endswith(".s32") else offset & WORD32
                registers[dst] = offset * factor + base
            elif opcode in ("cvt.s64.s32", "cvt.u64.u32"):
                bits = value(registers, src[0])
                registers[dst] = signed32(bits) if opcode == "cvt.s64.s32" else bits & WORD32
            elif opcode == "cvt.u32.u64":
                registers[dst] = value(registers, src[0]) & WORD32
            elif opcode.startswith("setp."):
                _, comparison, kind = opcode.split(".")
                assert kind in ("s32", "u32", "b32")
                a, b = (value(registers, arg) for arg in src)
                if packed_head and isinstance(a, Value):
                    assert opcode == "setp.ne.b32" and a == Value("scale", parameters[3]) and b == 0x435fffff
                    a = head_inverse_bits
                if kind == "s32":
                    a, b = signed32(a), signed32(b)
                registers[dst] = PREDICATES[comparison](a, b)
            elif opcode in ("and.pred", "or.pred"):
                a, b = (registers[arg] for arg in src)
                registers[dst] = (a and b) if opcode == "and.pred" else (a or b)
            elif opcode in ("popc.b32", "bfind.u32"):
                bits = value(registers, src[0]) & WORD32
                registers[dst] = bits.bit_count() if opcode == "popc.b32" else (bits.bit_length() - 1) & WORD32
            elif opcode in ("selp.b32", "selp.b64"):
                registers[dst] = value(registers, src[0] if registers[src[2]] else src[1])
            elif opcode == "mov.u32":
                registers[dst] = value(registers, src[0]) & WORD32
            elif opcode == "mov.u64":
                registers[dst] = value(registers, src[0]) & ((1 << 64) - 1)
            elif opcode == "mov.b32":
                if dst.startswith("{"):
                    distribute(registers, dst, registers[src[0]], 2)
                elif src[0].startswith("{"):
                    registers[dst] = packed(registers, src[0])
                else:
                    registers[dst] = value(registers, src[0])
            elif opcode in ("ld.param.b32", "ld.param.b64", "ld.param.s32"):
                param = parameters[int(src[0].removesuffix("]").rsplit("_", 1)[1])]
                registers[dst] = signed32(param) if opcode == "ld.param.s32" else param
            elif opcode == "ld.global.v4.b32":
                start = address(registers, src[0])
                if raw_bytes or candidate_ids is not None:
                    data = tuple(Value("copied", start + byte) for byte in range(16))
                else:
                    data = tuple(byte for offset in range(0, 16, 2)
                                 for byte in (Value("bf16", start + offset), None))
                distribute(registers, dst, data, 4)
            elif opcode == "ld.global.v2.b64":
                assert candidate_ids is not None
                index, misalignment = divmod(address(registers, src[0]) - parameters[1], 8)
                assert not misalignment and 0 <= index < len(candidate_ids) - 1
                for offset, register in enumerate(operands_in(dst)):
                    registers[register] = candidate_ids[index + offset]
            elif opcode == "ld.global.b64":
                assert candidate_ids is not None
                index, misalignment = divmod(address(registers, src[0]) - parameters[1], 8)
                assert not misalignment and 0 <= index < len(candidate_ids)
                registers[operands_in(dst)[0]] = candidate_ids[index]
            elif opcode == "ld.global.v2.b32":
                start = address(registers, src[0])
                if raw_bytes or candidate_ids is not None:
                    data = tuple(Value("copied", start + lane) for lane in range(8))
                else:
                    data = tuple(byte for offset in range(0, 8, 2)
                                 for byte in (Value("bf16", start + offset), None))
                distribute(registers, dst, data, 4)
            elif opcode == "add.bf16x2":
                left, right = (registers[a] for a in src)
                assert left[1] is None and left[3] is None and right[1] is None and right[3] is None
                assert all(v.stage == "bf16" for v in (left[0], left[2], right[0], right[2]))
                registers[dst] = (Value("bf16-add", left[0].source, right[0].source), None,
                                  Value("bf16-add", left[2].source, right[2].source), None)
            elif opcode == "ld.global.b32":
                registers[operands_in(dst)[0]] = Value("scale", address(registers, src[0]))
            elif opcode == "cvt.f32.bf16":
                datum, unused = registers[src[0]]
                stage = {"bf16": "f32", "rounded-bf16": "rounded-f32"}[datum.stage]
                assert unused is None
                registers[dst] = Value(stage, datum.source, datum.scale)
            elif opcode == "mul.f32":
                left, right = (registers[a] for a in src)
                datum, scale = (right, left) if left.stage == "scale" else (left, right)
                assert datum.stage == "f32" and scale.stage == "scale"
                registers[dst] = Value("scaled", datum.source, scale.source)
            elif opcode == "cvt.rn.bf16.f32":
                datum = registers[src[0]]
                assert datum.stage == "scaled"
                registers[dst] = (Value("rounded-bf16", datum.source, datum.scale), None)
            elif opcode == "cvt.rn.bf16x2.f32":
                high, low = (registers[a] for a in src)
                assert packed_head and dst == "rounded" and high.stage == low.stage == "scaled"
                assert high.source == low.source + 2 and high.scale == low.scale
                registers[dst] = (Value("rounded-bf16", low.source, low.scale), None,
                                  Value("rounded-bf16", high.source, high.scale), None)
            elif opcode == "mul.rn.bf16x2":
                assert packed_gain and src[1] == "gain" and registers["gain"] == 0x43604360
                low, low_pad, high, high_pad = registers[src[0]]
                assert low_pad is high_pad is None and low.stage == high.stage == "bf16"
                assert high.source == low.source + 2
                registers[dst] = (Value("rounded-bf16", low.source, parameters[3]), None,
                                  Value("rounded-bf16", high.source, parameters[3]), None)
            elif opcode == "min.bf16x2":
                rounded = dst if packed_gain else "rounded"
                assert packed_head and src == [rounded, "upper"] and registers["upper"] == 0x43e043e0
                low, low_pad, high, high_pad = registers[rounded]
                assert low_pad is high_pad is None and low.stage == high.stage == "rounded-bf16"
                registers[dst] = (Value("upper-bounded-bf16", low.source, low.scale), None,
                                  Value("upper-bounded-bf16", high.source, high.scale), None)
            elif opcode == "cvt.rn.f16.bf16":
                datum, pad = registers[src[0]]
                assert packed_head and pad is None and datum.stage == "upper-bounded-bf16"
                registers[dst] = (Value("rounded-f16", datum.source, datum.scale), None)
            elif opcode == "cvt.rn.satfinite.e4m3x2.f16x2":
                low, low_pad, high, high_pad = registers[src[0]]
                assert packed_head and low_pad is high_pad is None
                assert low.stage == high.stage == "rounded-f16"
                assert high.source == low.source + 2 and high.scale == low.scale
                registers[dst] = tuple(
                    Value("e4m3", item.source, item.scale) for item in (low, high)
                )
            elif opcode == "div.full.f32":
                datum, scale = (registers[a] for a in src)
                assert datum.stage == "f32" and scale.stage == "scale"
                registers[dst] = Value("div", datum.source, scale.source)
            elif opcode == "abs.f32":
                datum = registers[src[0]]
                assert datum.stage == "f32"
                registers[dst] = frozenset((datum.source,))
            elif opcode == "min.f32":
                datum = registers[src[0]]
                assert datum.stage in ("div", "rounded-f32") and src[1] == "0f43E00000"
                registers[dst] = Value("min448", datum.source, datum.scale)
            elif opcode == "max.f32":
                datum = registers[src[0]]
                if src[1] == "0fC3E00000":
                    assert datum.stage == "min448"
                    registers[dst] = Value("clamped", datum.source, datum.scale)
                else:
                    other = registers[src[1]]
                    if datum is None or other is None:
                        registers[dst] = None
                    else:
                        assert isinstance(datum, frozenset) and isinstance(other, frozenset)
                        registers[dst] = datum | other
            elif opcode == "cvt.rn.satfinite.e4m3x2.f32":
                high, low = (registers[a] for a in src)
                assert not packed_head and high.stage == low.stage == "clamped"
                registers[dst] = tuple(Value("e4m3", item.source, item.scale) for item in (low, high))
            elif opcode == "st.shared.v4.b32":
                start = address(registers, dst)
                assert start % 16 == 0
                if warp_local:
                    assert 0 <= start < 4096 * warp_scratch_banks and start // 512 % 8 == tid // 32
                write_bytes(warp_writes if warp_local else scratch, start, packed(registers, src[0]))
            elif opcode == "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16":
                assert warp_local
                lane = tid % 32
                first = tid - lane
                for matrix, word in enumerate(operands_in(src[0])):
                    for half in range(2):
                        provider = threads[first + 8 * matrix + 2 * (lane % 4) + half]
                        row = address(provider, dst)
                        start = row + 2 * (lane // 4)
                        assert row % 16 == 0 and 0 <= start < 4096
                        assert start // 128 % 8 == tid // 32
                        write_bytes(warp_writes, start, registers[word][2 * half:2 * half + 2])
            elif opcode == "ld.shared.v4.b32":
                assert warp_local and not warp_writes
                start = address(registers, src[0])
                assert start % 16 == 0 and 0 <= start < 4096
                assert start // 128 % 8 == tid // 32
                distribute(registers, dst, tuple(scratch[start + i] for i in range(16)), 4)
            elif opcode == "st.shared.b32":
                assert reductions is not None and shared_exchanges == 2 and cta_barriers == 0
                word = registers[src[0]]
                offset = address(registers, dst)
                assert offset == 4096 + 8 * (tid // 32) + (tid & 4)
                assert isinstance(word, frozenset)
                # Lanes in the same warp and column half publish the same maximum.
                assert pending_reduction.setdefault(offset, word) == word
            elif opcode == "ld.shared.b32":
                registers[dst] = scratch_reduction[address(registers, src[0])]
            elif opcode == "ldmatrix.sync.aligned.m8n8.x4.shared.b16":
                warp_start = tid - tid % 32
                lane = tid % 32
                for matrix, reg in enumerate(operands_in(dst)):
                    provider = threads[warp_start + 8 * matrix + lane // 4]
                    start = address(provider, src[0]) + 4 * (lane % 4)
                    if warp_local:
                        assert not warp_writes and 0 <= start < 4096 * warp_scratch_banks
                        assert start // 512 % 8 == tid // 32
                    registers[reg] = tuple(scratch[start + i] for i in range(4))
            elif opcode == "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16":
                assert not warp_local
                first = tid - tid % 32
                lane = tid % 32
                for matrix, reg in enumerate(operands_in(dst)):
                    word = []
                    for half in range(2):
                        provider = threads[first + 8 * matrix + 2 * (lane % 4) + half]
                        start = address(provider, src[0]) + 2 * (lane // 4)
                        word.extend((scratch[start], scratch[start + 1]))
                    registers[reg] = tuple(word)
            elif opcode == "prmt.b32":
                left, right, selector = (value(registers, a) for a in src)
                select = [(selector >> (4 * i)) & 15 for i in range(4)]
                assert max(select) < 8
                registers[dst] = tuple((left + right)[i] for i in select)
            elif opcode in ("st.global.v2.b32", "st.global.v4.b32"):
                start = address(registers, dst)
                if predicate is None or not warp_local:
                    write_bytes(output, start, packed(registers, src[0]))
                else:
                    assert reductions is not None
                    for i, word in enumerate(operands_in(src[0])):
                        offset = start + 4 * i
                        assert offset not in reductions and isinstance(registers[word], frozenset)
                        reductions[offset] = registers[word]
            elif opcode == "st.global.b32" and reductions is not None:
                offset = address(registers, dst)
                word = registers[operands_in(src[0])[0]]
                assert offset not in reductions and isinstance(word, frozenset)
                reductions[offset] = word
            elif opcode == "st.global.b32" and candidate_ids is not None:
                word = signed32(value(registers, operands_in(src[0])[0]))
                target = address(registers, dst)
                if cta_barriers == 0:
                    assert target not in output and word == -1
                else:
                    assert cta_barriers == 1 and output[target] == -1 and target not in indexed_writes
                    assert 0 <= word < len(candidate_ids)
                    indexed_writes.add(target)
                output[target] = word
            else:
                raise ValueError(f"Unexpected QKV pack instruction {opcode}")
    raise ValueError("Missing PTX return")


def verify_writes(written, constants, pointers, batch, tile_m, tile_n, slot):
    rows, cols, split = (constants[name] for name in ("M", "N", "FIRST_ROWS"))
    first, second, row, col, scale = pointers
    assert len(written) == 2 * 64 * 128
    for m in range(tile_m * 64, (tile_m + 1) * 64):
        bank, stride, local_m = ((first, constants["FIRST_STRIDE_B"], m) if m < split else
                                 (second, constants["SECOND_STRIDE_B"], m - split))
        crop = constants["QK_CROP" if m < split else "V_CROP"]
        if crop:
            head, head_row = divmod(local_m, 64)
            local_m = head * 128 + head_row
        for n in range(tile_n * 128, (tile_n + 1) * 128):
            origin = bank + 2 * (slot * stride + local_m * cols + n)
            expected = Value("e4m3", origin, scale + 4 * batch)
            assert written[row + batch * rows * cols + m * cols + n] == expected
            assert written[col + batch * rows * cols + n * rows + m] == expected


def boundary_tiles(constants, batches):
    rows, cols, split = (constants[name] for name in ("M", "N", "FIRST_ROWS"))
    return ((0, 0, 0), (0, split // 64 - 1, cols // 128 - 1),
            (batches - 1, split // 64, 0), (batches - 1, rows // 64 - 1, cols // 128 - 1))


def verify_tiles(ptx_instructions, constants, batches):
    instructions = list(parse(ptx_instructions))
    for batch, tile_m, tile_n in boundary_tiles(constants, batches):
        written = replay_tile(instructions, batch, tile_m, tile_n)
        verify_writes(written, constants, PARAMETERS, batch, tile_m, tile_n, batch)


def verify_grouped_tiles(ptx_instructions, groups):
    instructions = list(parse(ptx_instructions))
    parameters = tuple((i + 1) << 40 for i in range(11))
    band_base, slot_base = 0, 0
    for group, (batches, first_rows, second_rows) in enumerate(groups):
        rows = first_rows + second_rows
        constants = dict(M=rows, N=768, FIRST_ROWS=first_rows, FIRST_STRIDE_B=1179648,
                         SECOND_STRIDE_B=1179648, QK_CROP=first_rows == 768, V_CROP=second_rows == 384)
        pointers = parameters[:2] + tuple(parameters[group + i] for i in (2, 5, 8))
        bands = rows // 384
        for batch, tile_m, tile_n in boundary_tiles(constants, batches):
            band = band_base + batch * bands + tile_m // 6
            written = replay_tile(instructions, band, tile_m % 6, tile_n, parameters)
            verify_writes(written, constants, pointers, batch, tile_m, tile_n, slot_base + batch)
        band_base += batches * bands
        slot_base += batches


def verify_mlp_tiles(ptx_instructions):
    instructions = list(parse(ptx_instructions))
    parameters = tuple((i + 1) << 40 for i in range(9))
    bank, *buffers = parameters
    for matrix, tile_h, pair_d in ((0, 0, 0), (1, 43, 5), (22, 43, 0), (23, 0, 5)):
        projection, slot = matrix % 2, matrix // 2
        row, col, scale, partial = (buffers[projection + i] for i in (0, 2, 4, 6))
        amax = {}
        written = replay_tile(instructions, matrix, pair_d, tile_h, parameters, reductions=amax)
        assert len(written) == 2 * 64 * 128
        expected = [set(), set()]
        for h in range(tile_h * 64, (tile_h + 1) * 64):
            for lane_d in range(128):
                d = pair_d * 128 + lane_d
                origin = bank + 2 * ((matrix * 2816 + h) * 768 + d)
                byte = Value("e4m3", origin, scale + 4 * slot)
                assert written[row + (slot * 2816 + h) * 768 + d] == byte
                assert written[col + (slot * 768 + d) * 2816 + h] == byte
                expected[lane_d // 64].add(origin)
        base = partial + 4 * (slot * 528 + tile_h * 12 + pair_d * 2)
        assert amax == {base: expected[0], base + 4: expected[1]}


def verify_lm_head_tiles(ptx_instructions, labels):
    instructions = list(parse(ptx_instructions))
    pointers = tuple((i + 1) << 40 for i in range(4))
    bank, row, transposed, inverse = pointers
    for gain in (0x435fffff, 0x40400000):
        for row_tile, col_tile in ((0, 0), (0, 392), (11, 0), (11, 392)):
            written = replay_tile(instructions, 0, col_tile, row_tile, pointers,
                                  labels=labels, head_inverse_bits=gain)
            assert len(written) == 2 * 64 * 128
            for m in range(row_tile * 64, (row_tile + 1) * 64):
                for n in range(col_tile * 128, (col_tile + 1) * 128):
                    offset = m * 50304 + n
                    byte = Value("e4m3", bank + 2 * offset, inverse)
                    assert written[row + offset] == byte
                    assert written[transposed + n * 768 + m] == byte


def verify_sampled_tiles(ptx_instructions, p):
    instructions = list(parse(ptx_instructions))
    pointers = tuple((i + 1) << 40 for i in range(4))
    bank, _, row, transposed = pointers
    # Use distinct, nonconsecutive class IDs so routing cannot depend on
    # candidate row order. The real candidate builder returns sorted IDs.
    ids = [(j * 20011 + 137) % 50304 for j in range(p)]
    for rm, cn in ((0, 0), (p // 128 - 1, 0), (0, 5), (p // 128 - 1, 5)):
        written = replay_tile(instructions, 0, cn, rm, pointers, candidate_ids=ids)
        assert len(written) == 2 * 128 * 128
        for m in range(rm * 128, (rm + 1) * 128):
            for n in range(cn * 128, (cn + 1) * 128):
                byte = Value("copied", bank + ids[m] * 768 + n)
                assert written[row + m * 768 + n] == byte
                assert written[transposed + n * p + m] == byte


def verify_sampled_inverse(ptx, p):
    from tail_norm_replay import instructions

    code, labels = instructions(ptx)
    pointers = tuple((i + 1) << 40 for i in range(5))
    bank, _, rows, transposed, inverse = pointers
    vocab = 50304
    ids = sorted((j * 20011 + 137) % vocab for j in range(p))
    for m, n in ((0, 0), (0, 5), (p // 128 - 1, 0), (p // 128 - 1, 5), (7, 3)):
        got = replay_tile(code, 0, 197 + 6 * m + n, 0, pointers, candidate_ids=ids, labels=labels)
        expected = {}
        for i in range(128 * m, 128 * (m + 1)):
            for j in range(128 * n, 128 * (n + 1)):
                byte = Value("copied", bank + ids[i] * 768 + j)
                expected[rows + i * 768 + j] = byte
                expected[transposed + j * p + i] = byte
        assert got == expected, (p, m, n)

    for candidates in (ids, list(range(p)), list(range(vocab - p, vocab))):
        by_class = {class_id: slot for slot, class_id in enumerate(candidates)}
        for tile in sorted({*range(0, 197, 31), 1, p // 256 - 1, p // 256, 195, 196}):
            got = replay_tile(code, 0, tile, 0, pointers, candidate_ids=candidates, labels=labels)
            expected = {inverse + 4 * j: by_class.get(j, -1)
                        for j in range(256 * tile, min(vocab, 256 * (tile + 1)))}
            assert got == expected, (p, tile)


def verify_byte_transpose_tiles(ptx_instructions):
    instructions = list(parse(ptx_instructions))
    src, dst = PARAMETERS[:2]
    # Eight-rank inputs to the MLP and to sampled and full CE. Also
    # exercise the minimum grid and both extremes of the launch bounds.
    shapes = [(m, 768) for m in (16384, 32768, 40960, 49152)]
    shapes.extend(((16384, 10240), (32768, 10240), (49152, 14336), (49152, 24576),
                   (16384, 50304), (40960, 50304), (128, 128),
                   (65535 * 128, 128), (128, 131071 * 128)))
    for m, n in shapes:
        mt, nt = m // 128, n // 128
        for rm, cn in {(0, 0), (mt - 1, 0), (0, nt - 1), (mt - 1, nt - 1)}:
            written = replay_tile(instructions, 0, cn, rm, (src, dst, mt, nt), raw_bytes=True)
            assert len(written) == 128 * 128
            for row in range(rm * 128, (rm + 1) * 128):
                for col in range(cn * 128, (cn + 1) * 128):
                    assert written[dst + col * m + row] == Value("copied", src + row * n + col)


def verify_ce_transpose_tiles(ptx_instructions):
    instructions = list(parse(ptx_instructions))
    x, xt, grad, gt = PARAMETERS[:4]
    for m, n in ((16384, 10240), (32768, 10240), (49152, 14336), (49152, 24576),
                 (16384, 50304), (40960, 50304), (128, 128), (21845 * 128, 128), (128, 131071 * 128)):
        mt, nt = m // 128, n // 128
        for tile_row, tile_col in ((0, 0), (mt - 1, 5), (0, 6), (mt - 1, 6 + nt - 1)):
            src, dst, width, local_col = (x, xt, 768, tile_col) if tile_col < 6 else (grad, gt, n, tile_col - 6)
            written = replay_tile(instructions, 0, tile_col, tile_row, (x, xt, grad, gt, mt, nt),
                                  raw_bytes=True, warp_scratch_banks=2)
            assert len(written) == 128 * 128
            for row in range(tile_row * 128, (tile_row + 1) * 128):
                for col in range(local_col * 128, (local_col + 1) * 128):
                    assert written[dst + col * m + row] == Value("copied", src + row * width + col)


def verify_bf16_transpose_tiles(original, hopper, add):
    reference = list(parse(original))
    candidate = list(parse(hopper))
    m, n = (50304, 768) if add else (768, 50304)
    src, dst = PARAMETERS[:2]
    for rm, cn in ((0, 0), (m // 64 - 1, 0), (0, n // 128 - 1), (m // 64 - 1, n // 128 - 1)):
        expected = replay_tile(reference, 0, rm, cn, (src, dst, m, n, n, m), warp_local=False)
        observed = replay_tile(candidate, 0, cn, rm, (src, dst))
        assert len(expected) == 2 * 64 * 128 and expected == observed
        for row in range(rm * 64, (rm + 1) * 64):
            for col in range(cn * 128, (cn + 1) * 128):
                out = dst + 2 * (col * m + row)
                origin = src + 2 * (row * n + col)
                assert observed[out] == (Value("bf16-add", out, origin) if add else Value("bf16", origin))
                assert observed[out + 1] is None
