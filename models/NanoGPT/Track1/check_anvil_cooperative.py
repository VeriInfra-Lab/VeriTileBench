"""Verify the matrix-scoped ANVIL initialization and cooperative rounds."""
from pathlib import Path
import argparse
from math import prod
import re
from unittest.mock import patch

import torch
from triton.compiler import make_backend
from triton.experimental.gluon._runtime import GluonASTSource
import triton

from check_anvil_gram import CASES, build_full, check_full_store_wait, load_cascade, load_polar, mma_trace, polynomial_epilogue
from check_anvil_rails import expected_cells
from check_value_adam_inductor import HOPPER, TORCH_REVISION, capture
from install_candidate import ANVIL, POLAR
from anvil_tile_replay import expected_stores, instructions, replay
from tail_norm_replay import instructions as pointwise_instructions
from anvil_cascade_replay import first_gram, match_cascades
from anvil_trace_replay import INPUT, OUTPUT, expected_trace, replay as replay_trace
from check_anvil_epilogue import run as pointwise_run
from check_anvil_trace import check_projections
from export_source_link import anvil_frobenius, anvil_rail_weight
from qkv_ptx_replay import parse
from validation_ce_replay import integer_op
from verify_candidate import check_checkout

class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **options):
            self.calls.append((grid, args, options))
        return launch


def compile_launch(jit, signature, constants, options, stem):
    signature.update(dict.fromkeys(constants, "constexpr"))
    attrs = {(i,): make_backend(HOPPER).parse_attr("D") for i, ty in enumerate(signature.values())
             if ty.startswith("*")}
    artifact = triton.compile(GluonASTSource(jit, signature, constants, attrs), target=HOPPER, options=options)
    for ext in ("ttgir", "ptx", "cubin"):
        data = artifact.asm[ext]
        stem.with_suffix(f".{ext}").write_bytes(data if isinstance(data, bytes) else data.encode())
    return artifact


register = re.compile(r"%[\w.]+")
builtins = {"%tid.x", "%ctaid.x", "global_smem"}


def operands(instruction):
    pred, op, args = instruction
    defined = set()
    sources = args
    if args and not op.startswith(("st.", "bra", "bar.", "ret")):
        defined = set(register.findall(args[0]))
        sources = args[1:]
    used = set(register.findall(" ".join(sources)))
    if pred is not None:
        used.add(pred.removeprefix("!"))
    return defined, used


def divisor_suffix(program, ticket):
    atomic = re.search(r"atom\.global\.gpu\.acq_rel\.add\.u32\s+(%r\d+),", program)
    assert atomic and program.count("atom.global.gpu.acq_rel.add.u32") == 1
    before = program[:atomic.start()]
    store = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
    assert before.count(store) == 2
    assert before.rfind(store) < before.index("cp.async.bulk.wait_group 0;")
    assert before.index("cp.async.bulk.wait_group 0;") < before.index("bar.sync 0; fence.proxy.async.global;")
    assert "ld.acquire" not in program and "ld.global.cg." in program
    after = program[atomic.end():]
    broadcast = re.search(r"ld.shared.b32\s+(%r\d+), \[global_smem\];", after)
    assert broadcast
    assert ("bar.sync" in after[:broadcast.start()]
            and re.search(rf"st.shared.b32 [^;]+, {atomic[1]};", after[:broadcast.start()]))
    pos = atomic.end() + broadcast.end()
    tail = f"\nmov.u32 {broadcast[1]}, {ticket};\n" + program[pos:]
    ending = re.search(r"(?m)^\s*ret;\s*$", tail)
    assert ending
    tail = tail[:ending.end()]
    tail, n = re.subn(r"(?m)^\s*fence\.proxy\.async\.global; mov\.u32 %r\d+, 0;\s*$", "", tail)
    assert n == 1
    # The typed launcher passes the raw Gram and divisor views immediately
    # after the two five-word descriptors.
    tail = tail.replace("[cooperative_initial_param_10]", "[trace_param_0]")
    tail = tail.replace("[cooperative_initial_param_11]", "[trace_param_1]")
    code, _ = pointwise_instructions(tail)
    external = set()
    internal = set()
    for instruction in code:
        written, read = operands(instruction)
        external.update(read - internal - builtins)
        internal.update(written)
    full = program[:pos] + "\n$trace_entry:\n" + program[pos:]
    waits = iter(range(20))

    def distinguish_wait(match):
        return match[0].replace("waitLoop", f"$wait_{next(waits)}")

    full = re.sub(r"waitLoop:.*?bra\.uni\s+waitLoop;", distinguish_wait, full, flags=re.DOTALL)
    code, labels = pointwise_instructions(full)
    entry = labels["$trace_entry"]

    def reaches_without(excluded):
        pending = [0]
        visited = {excluded}
        while pending:
            loc = pending.pop()
            if loc in visited:
                continue
            if loc == entry:
                return True
            visited.add(loc)
            pred, op, args = code[loc]
            if op == "ret":
                continue
            if op.startswith("bra"):
                pending.append(labels[args[0]])
                if pred is None:
                    continue
            pending.append(loc + 1)
        return False

    selected = []
    for pc in range(entry - 1, -1, -1):
        instruction = code[pc]
        written, read = operands(instruction)
        if written & external:
            pred, op, args = instruction
            assert pred is None and not op.startswith(("ld.", "cvta.", "ldmatrix", "st."))
            assert not reaches_without(pc), (pc, instruction)
            external.difference_update(written)
            external.update(read - builtins)
            selected.append(op + " " + ", ".join(args) + ";")
    assert not external, external
    return ".reqntid 256\n" + "\n".join(reversed(selected)) + "\n" + tail


def check_counter_address(asm, atomic_address, counter_parameter, batches, tiles):
    cta, = re.findall(r"mov.u32\s+(%r\d+), %ctaid.x;", asm)
    mad = re.search(rf"mad.wide.u32\s+{atomic_address}, (%r\d+), 4, (%rd\d+);", asm)
    assert mad and re.search(rf"ld.param.b64\s+{mad[2]}, \[{counter_parameter}\];", asm)
    available = {cta}
    steps = []
    for line in asm.splitlines():
        line = line.strip()
        if re.match(r"(mul\.hi\.u32|shr\.u32)\s+%r\d+", line):
            _, op, args = next(parse([line]))
            if args[1] in available and args[0] not in available:
                available.add(args[0])
                steps.append((op, args))
                if args[0] == mad[1]:
                    break
    assert mad[1] in available
    for cta_id in range(batches * tiles):
        registers = {cta: cta_id}
        for op, (dst, src, imm) in steps:
            registers[dst] = integer_op(op, [registers[src], int(imm)])
        assert registers[mad[1]] == cta_id // tiles


def check_round_publication(asm, batches, coordinates):
    check_full_store_wait(asm, 2)
    barrier = re.search(
        r"bar.sync 0;\s+mov.u32 tid, %tid.x;\s+setp.eq.u32 observer, tid, 0;\s+"
        r"@!observer bra gram_ready;\s+fence.proxy.async.global;\s+"
        r"atom.acq_rel.gpu.global.or.b32 bits, \[(%rd\d+)\], (%r\d+);\s+"
        r"or.b32 bits, bits, \2;\s+and.b32 covered, bits, (%r\d+);\s+"
        r"setp.eq.u32 ready, covered, \3;\s+@ready bra gram_ready;\s+"
        r"observe_gram:\s+ld.acquire.gpu.global.u32 bits, \[\1\];\s+"
        r"and.b32 covered, bits, \3;\s+setp.eq.u32 ready, covered, \3;\s+"
        r"@!ready bra observe_gram;\s+gram_ready:\s+bar.sync 0;\s+fence.proxy.async.global;", asm,
    )
    assert barrier
    first, second = asm.split("atom.acq_rel.gpu.global.or.b32")
    check_publication_inputs(first, barrier.groups(), batches, coordinates)
    store = "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
    refill = "cp.async.bulk.tensor.2d.shared::cluster.global"
    wait = "cp.async.bulk.wait_group 0;"
    assert first.count(store) == second.count(store) == 2
    assert first.rfind(store) < first.index(wait)
    assert second.rfind("fence.proxy.async.global;") < second.index(refill)
    assert second.rfind(store) < second.index(wait)
    assert "ld.global.cg" in second and not re.search(r"\bld\.global\.(?!cg\.)", second)


def check_publication_inputs(asm, arguments, batches, coordinates):
    # The address and both masks have single, unpredicated scalar definitions.
    # Interpret only these dependencies, including the compiler's division
    # and remainder lowering over the bounded CTA IDs.
    definitions = {}
    for line in asm.splitlines():
        inst = line.split("//", 1)[0].strip()
        match = re.fullmatch(r"(?:(@!?%[\w.]+) )?([a-z][\w.]+) (%[\w.]+), (.+);",
                             re.sub(r"\s+", " ", inst))
        if match is None:
            continue
        pred, op, dest, args = match.groups()
        definitions.setdefault(dest, []).append((pred, op, args.split(", ")))

    for round_number in range(1, 6):
        offset = round_number * batches
        for cta in range(batches * len(coordinates)):
            matrix, tile = divmod(cta, len(coordinates))
            r, c = coordinates[tile]
            mask = sum(1 << i for i, (a, b) in enumerate(coordinates)
                       if a in (r, c) or b in (r, c))

            def read(arg):
                if arg == "%ctaid.x":
                    return cta
                if not arg.startswith("%"):
                    return int(arg, 0)
                ((pred, op, args),) = definitions[arg]
                assert pred is None
                if op.startswith("ld.param."):
                    return {"[cooperative_round_param_16]": 0, "[cooperative_round_param_17]": offset}[args[0]]
                values = [read(word) for word in args]
                if op.startswith("mov."):
                    return values[0]
                if op in ("cvt.u32.u64", "cvt.s32.s64"):
                    return values[0] & 0xffffffff
                if op == "bmsk.clamp.b32":
                    start, width = values
                    assert 0 <= start <= start + width <= 32
                    return ((1 << width) - 1) << start
                if op == "neg.s32":
                    return -values[0]
                return integer_op(op, values)

            assert tuple(map(read, arguments)) == (4 * (offset + matrix), 1 << tile, mask)


def diagonal_publishers(ptx, before, matrices, coordinates):
    guard = re.search(r"cp.async.bulk.wait_group 0;[^\n]*\n(?:[^@]*?)@(%p\d+) bra\s+(\$L__BB[\d_]+);", before)
    assert guard and re.search(rf"{re.escape(guard[2])}:\n\s+\.loc[^\n]+\n\s+ret;", ptx)
    coordinates_ptx = re.search(rf"setp.ne.b32\s+{guard[1]}, (%r\d+), (%r\d+);", before)
    assert coordinates_ptx
    first_branch = before.index("setp.eq.b32")
    prefix_ops = (line.strip() for line in before[before.index("mov.u32"):first_branch].splitlines())
    coordinate_ops = list(parse(line for line in prefix_ops if re.match(r"^[a-z]+\.[\w.]+\s", line)))
    diag_ctas = []
    count = matrices * len(coordinates)
    for cta in range(count):
        matrix, local = divmod(cta, len(coordinates))
        row, col = coordinates[local]
        registers = {"%ctaid.x": cta}
        for pred, opcode, args in coordinate_ops:
            assert pred is None
            dst, *src = args
            values = [registers[arg] if arg.startswith("%") else int(arg, 0) for arg in src]
            if opcode.startswith(("mov.", "cvt.")):
                registers[dst] = values[0] & ((1 << 32) - 1) if opcode.startswith("cvt.") else values[0]
            else:
                registers[dst] = integer_op(opcode, values)
        assert tuple(registers[operand] for operand in coordinates_ptx.groups()) == (row, col)
        if row == col:
            diag_ctas.append(cta)
    assert len(diag_ctas) == matrices * sum(row == col for row, col in coordinates)
    return diag_ctas


def check_divisors(ptx, batches, width, coordinates, margin, epsilon):
    check_full_store_wait(ptx, 1)
    tiles = len(coordinates)
    side = width // 128
    atomic = re.search(r"atom\.global\.gpu\.acq_rel\.add\.u32 %r\d+, \[ (%rd\d+) \+ 0 \], (%r\d+);", ptx)
    assert atomic, "Missing last-publisher atomic"
    before = ptx[:atomic.start()]
    assert re.search(rf"mov.b32\s+{atomic[2]}, 1;", before)
    check_counter_address(before, atomic[1], "cooperative_initial_param_12", batches, tiles)
    publishers = diagonal_publishers(ptx, before, batches, coordinates)
    trace = divisor_suffix(ptx, side - 1)
    words = {INPUT + 2 * (matrix * width**2 + row * (width + 1)): ("bf16", matrix, row)
             for matrix in range(batches) for row in range(width)}
    expected = {}
    for matrix in range(batches):
        value = expected_trace(words, width, matrix)
        divisor = ("fma.rn.f32", ("sqrt.rn.f32", value), margin, epsilon)
        expected[OUTPUT + 8 * matrix] = divisor
        expected[OUTPUT + 8 * matrix + 4] = ("mul.f32", divisor, divisor)
    for cta in publishers:
        matrix = cta // tiles
        result = replay_trace(trace, batches * tiles, words, INPUT, OUTPUT, block_ids=(cta,))
        assert result == {addr: expected[addr] for addr in (OUTPUT + 8 * matrix, OUTPUT + 8 * matrix + 4)}
    for ticket in range(side - 1):
        assert not replay_trace(divisor_suffix(ptx, ticket), batches * tiles, words, INPUT, OUTPUT,
                                block_ids=publishers)


def check(baseline, candidate, out):
    check_checkout(baseline, candidate)
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None or triton.__version__ != "3.6.0":
        raise RuntimeError("Use the pinned CPU PyTorch and Triton 3.6.0")
    out.mkdir(parents=True, exist_ok=True)
    polar = load_polar(candidate / POLAR, "matrix_cooperative_polar")
    cascade = load_cascade(candidate / ANVIL, polar)
    rails = cascade.rails_module
    original = (baseline / ANVIL).read_text()
    _, slow_weight = anvil_rail_weight(original)
    _, margin, epsilon = anvil_frobenius(original)
    for label, gram_name, batches, m, k in CASES[:3]:
        tall = m > k
        width = min(m, k)
        side = width // 128
        coordinates = [(r * 128, c * 128) for c in range(side) for r in range(c, side)]
        tiles = len(coordinates)
        n = batches * tiles
        shape = (batches, m, k)
        cells = prod(shape)
        grad = torch.empty(shape, dtype=torch.bfloat16)
        velocity = torch.empty((2, *shape), dtype=torch.float32)
        staged = torch.empty((4,), dtype=torch.float32)
        x = torch.empty_like(grad)
        a = torch.empty((batches, width, width), dtype=torch.bfloat16)
        b = torch.empty_like(a)

        rail = RecordingKernel()
        with patch.object(rails, "_prepare_rails_and_seed", rail):
            counter = rails.prepare_rails_and_seed(grad, velocity, staged[0], staged[2], staged[3], x)
        assert counter.shape == (6 * batches,) and counter.dtype == torch.int32 and counter.stride() == (1,)
        (grid, args, opts), = rail.calls
        assert grid == (cells // 1024,) and args[7:] == (cells, 1024, batches)
        assert [t.data_ptr() for t in args[:7]] == [t.data_ptr() for t in
                       (grad, velocity, staged[0], staged[2], staged[3], x, counter)]
        assert opts == dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
        constants = dict(CELLS=cells, BLOCK=1024, MATRICES=batches)
        sig = dict(Grad="*bf16", Velocity="*fp32", Momentum="*fp32", FastBeta="*fp32",
                   FastWeight="*fp32", X="*bf16", Counter="*i32")
        sig.update(dict.fromkeys(constants, "constexpr"))
        attrs = {(i,): make_backend(HOPPER).parse_attr("D") for i in (0, 1, 2, 5, 6)}
        artifact = triton.compile(GluonASTSource(rails._prepare_rails_and_seed, sig, constants, attrs),
                                  target=HOPPER, options=opts)
        for fmt in ("ptx", "ttgir", "cubin"):
            data = artifact.asm[fmt]
            (out / f"{label}-seed.{fmt}").write_bytes(data if isinstance(data, bytes) else data.encode())
        code, _ = pointwise_instructions(artifact.asm["ptx"])
        roles = ("grad", "velocity", "momentum", "fast_beta", "fast_weight", "prepared", "counter")
        per_matrix = m * k // 1024
        bound = (batch * per_matrix + edge for batch in range(batches) for edge in (0, per_matrix - 1))
        for cta in bound:
            result = pointwise_run(code, roles, 1024, 4, cta, half={"grad", "prepared"})
            expected = expected_cells(cells, 1024, cta, slow_weight)
            if cta == 0:
                expected.update({("counter", matrix): 0 for matrix in range(6 * batches)})
            assert result == expected, (bank, cta)
        print(label, "exact rail stores and zero counters", flush=True)

        initial = RecordingKernel()
        with patch.object(polar, "cooperative_initial", initial):
            divisors = polar.anvil_initial_gram(x, a, counter)
        assert divisors.shape == (batches, 2, 1) and divisors.dtype == torch.float32
        (grid, args, opts), = initial.calls
        assert grid == (n,) and args[2] is a and args[3] is divisors and args[4] is counter
        assert args[5:] == (k, m, tall) and opts == dict(num_warps=8, num_stages=1)

        def checked_desc(controllers, originals):
            result = []
            for controller, original in zip(controllers, originals, strict=True):
                assert controller.base.data_ptr() == original.data_ptr()
                assert controller.block_shape == [128, 128] and controller.padding == "zero"
                dim = original.shape[-1]
                assert tuple(controller.shape) == (original.numel() // dim, dim)
                assert tuple(controller.strides) == (dim, 1)
                assert controller.layout.swizzle_byte_width == 128 and controller.layout.element_bitwidth == 16
                result.append(f"tensordesc<bf16[128,128],{controller.layout}>")
            return result

        xd, ad = checked_desc(args[:2], (x, a))
        constants = dict(K=k, M=m, Tall=tall)
        signature = dict(X=xd, Gram=ad, Gram_ptr="*bf16", Divisors="*fp32", Counter="*i32")
        artifact = compile_launch(polar.cooperative_initial, signature, constants, opts, out / f"{label}-initial")
        native_gram = build_full(polar, gram_name, m, k)
        assert not artifact.metadata.launch_cooperative_grid
        assert mma_trace(artifact.asm["ptx"]) == mma_trace(native_gram.asm["ptx"])
        # The first kernel's tile replay ends when all Gram stores have completed.
        prefix = instructions(artifact.asm["ttgir"])
        before_publish, = [i + 1 for i, line in enumerate(prefix) if "cp.async.bulk.wait_group 0;" in line]
        assert prefix[before_publish:before_publish + 2] == ["scf.if %offdiag {", "} else {"]
        prefix = [line.replace("%X", "%A").replace("%Gram", "%C") if line.startswith("ttng.async_tma") else line for line in prefix[:before_publish]]
        # The compiler hoists the divisor constants outside the last-ticket branch.
        # They have no users in the Gram prefix.
        prefix = [line for line in prefix if not re.search(r"arith.constant (?!0\.000000e\+00)[0-9.]+[eE][+-]\d+ : f32", line)]
        launch = dict(K=k, M=m, c_stride_r=width, tile_count=tiles)
        for pid in range(n):
            matrix, tile = divmod(pid, tiles)
            row, col = coordinates[tile]
            observed = replay([*prefix, "tt.return"], launch, pid, cells, a.numel(), tma=True)
            expected = expected_stores(matrix, row, col, m, k, tall=tall, polynomial=False)
            assert observed[0] == (expected[1:] if row == col else expected), (bank, pid)
        check_divisors(artifact.asm["ptx"], batches, width, coordinates, margin, epsilon)
        print(label, "ordered initial Gram tiles and exact last-diagonal-publisher divisors in ordinary launch", flush=True)

        round_kernel = RecordingKernel()
        with patch.object(polar, "cooperative_round", round_kernel):
            for number in range(1, 6):
                assert polar.anvil_cooperative_round(x, a, b, counter, number, 1.25, -0.5) is None
        assert len(round_kernel.calls) == 5
        for number, (grid, args, opts) in enumerate(round_kernel.calls, 1):
            assert grid == (n,) and opts == dict(num_warps=8, num_stages=1, launch_cooperative_grid=True)
            assert args[3] is a and args[4] is counter
            assert args[5:] == (number * batches, 1.25, -0.5, k, m, tall)
            xd, ad, bd = checked_desc(args[:3], (x, a, b))
        signature = dict(X=xd, Gram=ad, Output=bd, Gram_ptr="*bf16", Counter="*i32",
                         round_offset="i32", alpha="fp32", beta="fp32")
        artifact = compile_launch(polar.cooperative_round, signature, constants, opts, out / f"{label}-round")
        native_poly = build_full(polar, "ba_plus_cAA_kernel", width, width)
        assert artifact.metadata.launch_cooperative_grid
        asm = artifact.asm["ptx"]
        assert mma_trace(asm) == mma_trace(native_gram.asm["ptx"]) + mma_trace(native_poly.asm["ptx"])
        assert polynomial_epilogue(asm) == polynomial_epilogue(native_poly.asm["ptx"])
        check_round_publication(asm, batches, coordinates)
        ttgir = instructions(artifact.asm["ttgir"])
        for pid in range(n):
            matrix, tile = divmod(pid, tiles)
            row, col = coordinates[tile]
            phases = replay(ttgir, dict(launch, round_offset=batches), pid, cells, a.numel(),
                            tma=True, cooperative=True)
            for phase, (r, c, transposed) in enumerate(((m, k, tall), (width, width, False))):
                stores = expected_stores(matrix, row, col, r, c, tall=transposed, polynomial=phase == 1)
                assert phases[phase][0] == (stores[1:] if row == col else stores), (label, pid, phase)
        print(label, "exact ordered Gram and polynomial arithmetic, five matrix-scoped rounds", flush=True)

        inputs = [grad, velocity, staged[0], tall, staged[2], staged[3], True]
        paths = (out / f"{label}-split.py", out / f"{label}-cooperative.py")
        for enabled, path in enumerate(paths):
            capture(cascade.anvil_cascade, path, [*inputs, bool(enabled)], user_triton=True)
        traces = [(*first_gram(path, shape, fused=True)[:4],
                   check_projections(path, shape, cascade.ANVIL_MAPS)) for path in paths]
        match_cascades(*traces, shape)
        print(label, "captured cascade preserves all operands, coefficients and projection outputs", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
