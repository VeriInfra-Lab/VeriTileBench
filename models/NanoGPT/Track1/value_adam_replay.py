"""Check compiled value Adam operations and lane routing on each Hopper tile."""
from dataclasses import dataclass
from struct import pack, unpack

from qkv_ptx_replay import INT32_OPS, address, operands_in, signed32, value
from rope_ptx_replay import atom, pair
from tail_norm_replay import instructions


ELEMENTS = 25152 * 768
BANKS = {name: i << 40 for i, name in enumerate(("p", "m", "g", "v", "step", "wd"), 1)}
HALF_BANKS = {BANKS["p"], BANKS["g"]}
FP_OPS = {
    "add.f32", "sub.f32", "mul.f32", "div.full.f32", "sqrt.rn.f32", "fma.rn.f32",
    "setp.gt.f32", "selp.f32", "cvt.f32.bf16", "cvt.f32.f16", "mul.f16",
    "cvt.rn.bf16.f32", "cvt.rn.f16.f32",
}


@dataclass(frozen=True)
class Packed:
    low: object
    high: object


def f32(x):
    return unpack("<I", pack("<f", x))[0]


def read(regs, operand):
    if operand.startswith("0f"):
        return int(operand[2:], 16)
    return value(regs, operand)


def unpack_halves(word):
    if isinstance(word, Packed):
        return word.low, word.high
    assert isinstance(word, int), word
    return pair(word)


def memory_op(op, regs, dst, src, memory, stores, shared, pending):
    load, space, *ty = op.split(".")
    assert ty[-1] == "b32"
    group, addr = (dst, src) if load == "ld" else (src, dst)
    names = operands_in(group) if group.startswith("{") else [group]
    assert len(names) == (int(ty[0][1:]) if len(ty) == 2 else 1)
    start = address(regs, addr)
    assert start % (4 * len(names)) == 0
    for i, reg in enumerate(names):
        ptr = start + 4 * i
        if space == "shared":
            if load == "ld":
                assert ptr not in pending
                regs[reg] = shared[ptr]
            else:
                assert ptr not in pending
                pending[ptr] = regs[reg]
            continue
        half = ((ptr >> 40) << 40) in HALF_BANKS
        if load == "ld":
            regs[reg] = Packed(memory[ptr], memory[ptr + 2]) if half else memory[ptr]
        else:
            data = unpack_halves(regs[reg]) if half else (regs[reg],)
            for j, word in enumerate(data):
                target = ptr + 2 * j
                assert target in memory and target not in stores
                stores[target] = word


def replay(ptx, cta, threads, params, memory, terms):
    code, _ = instructions(ptx)
    lanes = [{"%ctaid.x": cta, "%tid.x": lane, "global_smem": 0} for lane in range(threads)]
    stores, shared, pending = {}, {}, {}
    for pred, op, args in code:
        assert pred is None, (pred, op)
        if op == "ret":
            assert not pending
            return stores
        if op == "bar.sync":
            assert args == ["0"]
            shared.update(pending)
            pending.clear()
            continue
        dst, *src = args
        for regs in lanes:
            if op in INT32_OPS:
                regs[dst] = INT32_OPS[op](*(value(regs, a) for a in src)) & 0xffffffff
            elif op == "add.s64":
                regs[dst] = sum(value(regs, a) for a in src)
            elif op == "mul.wide.s32":
                regs[dst] = signed32(value(regs, src[0])) * signed32(value(regs, src[1]))
            elif op == "neg.s32":
                regs[dst] = -value(regs, src[0]) & 0xffffffff
            elif op in ("mov.b16", "mov.b32", "mov.u32"):
                if dst.startswith("{"):
                    low, high = operands_in(dst)
                    regs[low], regs[high] = unpack_halves(read(regs, src[0]))
                elif src[0].startswith("{"):
                    regs[dst] = Packed(*(read(regs, a) for a in operands_in(src[0])))
                else:
                    regs[dst] = read(regs, src[0])
            elif op in ("ld.param.b64", "ld.param.b32"):
                regs[dst] = params[int(src[0][:-1].rsplit("_", 1)[1])]
            elif op.startswith(("ld.global.", "st.global.", "ld.shared.", "st.shared.")):
                memory_op(op, regs, dst, src[0], memory, stores, shared, pending)
            elif op in FP_OPS:
                regs[dst] = atom(terms, op, *(read(regs, a) for a in src))
            elif op == "cvt.rn.bf16x2.f32":
                regs[dst] = Packed(*(atom(terms, "cvt.rn.bf16.f32", read(regs, a)) for a in reversed(src)))
            elif op == "mul.f16x2":
                a, b = (unpack_halves(read(regs, x)) for x in src)
                regs[dst] = Packed(*(atom(terms, "mul.f16", x, y) for x, y in zip(a, b, strict=True)))
            else:
                raise ValueError(f"Unexpected value Adam instruction {op}")
    raise ValueError("Value Adam has no return")


def lane_outputs(terms, period, inputs, gradient, by_value):
    def expr(op, *operands):
        return atom(terms, op, *operands)

    def mul(a, b):
        return expr("mul.f32", a, b)

    g = expr("cvt.f32.f16", gradient)
    p = expr("cvt.f32.bf16", inputs["p"])
    beta1, beta2 = (0.75 ** (period // 2), 0.95 ** (period // 2))
    first = expr("fma.rn.f32", inputs["m"], f32(beta1), mul(g, f32(1.0 - beta1)))
    second = expr("fma.rn.f32", inputs["v"], f32(beta2), mul(mul(g, g), f32(1.0 - beta2)))
    den = expr("add.f32", expr("sqrt.rn.f32", second), f32(1e-10))
    direction = expr("div.full.f32", first, den)
    step = mul(inputs["step"], direction) if by_value else mul(direction, inputs["step"])
    gate = expr("setp.gt.f32", mul(step, p), 0)
    masked = mul(p, expr("selp.f32", f32(1.0), 0, gate))
    decay_inputs = (inputs["wd"], masked) if by_value else (masked, inputs["wd"])
    decay_fma = expr("fma.rn.f32", *decay_inputs, step)
    step_fma = expr("fma.rn.f32", direction, inputs["step"], mul(masked, inputs["wd"]))
    weights = {expr("cvt.rn.bf16.f32", expr("sub.f32", p, delta)): label
               for label, delta in (("decay", decay_fma), ("step", step_fma))}
    return first, second, weights


def check_tile(ptx, order, period, block, warps, cta, scaled, *, by_value=False):
    terms = {}
    start = cta * block
    memory, lanes = {}, []
    for i in range(start, start + block):
        addresses = {name: base + (2 if base in HALF_BANKS else 4) * i
                     for name, base in BANKS.items() if name not in {"step", "wd"}}
        addresses.update({name: BANKS[name] for name in ("step", "wd")})
        inputs = {name: atom(terms, "load", ptr) for name, ptr in addresses.items()}
        memory.update({ptr: inputs[name] for name, ptr in addresses.items()})
        lanes.append((i, addresses, inputs))
    params = [memory[BANKS[name]] if by_value and name in {"step", "wd"} else BANKS[name] for name in order]
    stores = replay(ptx, cta, 32 * warps, params, memory, terms)
    profile = []
    for i, addresses, inputs in lanes:
        gradient = atom(terms, "mul.f16", inputs["g"], 0x3000) if scaled else inputs["g"]
        first, second, weights = lane_outputs(terms, period, inputs, gradient, by_value)
        assert stores.pop(addresses["m"]) == first, (i, "first")
        assert stores.pop(addresses["v"]) == second, (i, "second")
        weight = stores.pop(addresses["p"])
        assert weight in weights, (i, "weight rounding or contraction")
        profile.append(weights[weight])
        if scaled:
            assert stores.pop(addresses["g"]) == gradient, (i, "gradient materialization")
    assert not stores
    return tuple(profile)
