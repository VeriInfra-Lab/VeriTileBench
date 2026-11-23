"""Read the emitted ANVIL wrapper and follow logical tensor dependencies."""
import ast
from collections import Counter
from dataclasses import dataclass
from math import prod


def parse_wrapper(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    kernels = {}
    for entry in tree.body:
        if (isinstance(entry, ast.Assign) and isinstance(entry.value, ast.Call)
                and ast.unparse(entry.value.func) == "async_compile.triton"):
            name, source = map(ast.literal_eval, entry.value.args[:2])
            target, = entry.targets
            kernels[target.id] = name, source
    runner, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Runner"]
    return kernels, runner


def canonical_kernel(name, source):
    tree = ast.parse(source)
    kernel, = (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    kernel.name = "kernel"
    for decorator in kernel.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "inductor_meta":
                for i, k in enumerate(keyword.value.keys):
                    if ast.literal_eval(k) == "kernel_name":
                        keyword.value.values[i] = ast.Constant("kernel")
    return ast.dump(tree)


RAILS_OP = "torch.ops.veritile_track1.anvil_prepare_rails.default"
NORMALIZE_OP = "torch.ops.veritile_track1.anvil_normalize_gram_inputs.default"
RAIL_SEED_OP = "torch.ops.veritile_track1.anvil_prepare_rails_and_seed.default"
INITIAL_OP = "torch.ops.veritile_track1.anvil_initial_gram.default"
ROUND_OP = "torch.ops.veritile_track1.anvil_cooperative_round.default"
DIVISOR_OP = "torch.ops.veritile_track1.anvil_gram_divisors.default"
SPLIT_TRACE = "checked_split_trace"


def cname(stmt):
    return ast.unparse(stmt.value.func) if isinstance(stmt, (ast.Expr, ast.Assign)) and isinstance(stmt.value, ast.Call) else ""


TMA_GRAM = "torch.ops.veritile_track1.anvil_tma_gram.default"
TMA_POLYNOMIAL = "torch.ops.veritile_track1.anvil_tma_polynomial.default"


def first_gram(path, shape, *, fused=False):
    kernels, runner = parse_wrapper(path)
    call, = [fn for fn in runner.body if isinstance(fn, ast.FunctionDef) and fn.name == "call"]
    context, = [stmt for stmt in call.body if isinstance(stmt, ast.With)]
    assert [ast.unparse(s) for s in call.body[:2]] == [
        "arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args", "args.clear()",
    ]
    body = context.body
    start = next(i for i, stmt in enumerate(body) if cname(stmt) in (TMA_GRAM, INITIAL_OP) or cname(stmt).startswith(("XXT_kernel_", "XTX_kernel_")))
    prefix = body[:start]
    common = body[start:]
    x, a = (arg.id for arg in common[0].value.args[:2])
    allocations = [stmt for stmt in prefix if cname(stmt) == "empty_strided_cuda" and stmt.targets[0].id != x]
    assert sum(stmt.targets[0].id == a for stmt in allocations) == 1
    calls = [stmt.value for stmt in prefix if cname(stmt).endswith(".run")]
    assert all(not cname(stmt) or cname(stmt) in
               ("torch.cuda.set_device", "empty_strided_cuda", "get_raw_stream", RAILS_OP,
                RAIL_SEED_OP, "assert_size_stride", "assert_alignment")
               or cname(stmt).endswith(".run") for stmt in prefix)
    parts = []
    pre = []
    if fused:
        assert not calls
        rails, = [stmt for stmt in prefix if cname(stmt) in (RAILS_OP, RAIL_SEED_OP)]
        if cname(rails) == RAIL_SEED_OP:
            result, = rails.targets
            alias, = [stmt for stmt in prefix if isinstance(stmt, ast.Assign)
                      and isinstance(stmt.value, ast.Name) and stmt.value.id == result.id]
            guard, = [stmt for stmt in prefix if cname(stmt) == "assert_size_stride"]
            assert prefix.index(rails) < prefix.index(alias) < prefix.index(guard)
            assert common[0].value.args[2].id == alias.targets[0].id
            pre = [rails, alias, guard]
        assert [arg.id for arg in rails.value.args] == ["arg0_1", "arg1_1", "arg2_1", "arg3_1", "arg4_1", x]
        dest, = [stmt for stmt in prefix if cname(stmt) == "empty_strided_cuda" and stmt.targets[0].id == x]
        assert prefix.index(dest) < prefix.index(rails)
        assert [ast.unparse(arg) for arg in dest.value.args] == [
            str(shape), str((shape[1] * shape[2], shape[2], 1)), "torch.bfloat16",
        ]
        assert sum(cname(stmt) == "empty_strided_cuda" for stmt in prefix) == 2
    else:
        cells = prod(shape)
        rails, copy, look = calls
        assert "lerp_rsub_select" in ast.unparse(rails.func) and "add_lerp_mul" in ast.unparse(look.func)
        assert "fused_copy_" in ast.unparse(copy.func)
        g, beta, vin, vout, n = rails.args
        assert [arg.id for arg in (g, beta, vin)] == ["arg0_1", "arg3_1", "arg2_1"] and ast.literal_eval(n) == 2 * cells
        assert [ast.unparse(arg) for arg in copy.args] == [vout.id, vin.id, str(2 * cells)]
        assert [ast.unparse(arg) for arg in look.args] == ["arg1_1", "arg4_1", vout.id, g.id, x, str(cells)]
        for op in calls:
            assert [(kw.arg, ast.unparse(kw.value)) for kw in op.keywords] == [("stream", "stream0")]
            parts.append(kernels.pop(op.func.value.id))
    returned, = [stmt for stmt in call.body if isinstance(stmt, ast.Return)]
    return kernels, [*pre, *allocations, *common], x, returned.value.elts[0].id, parts


@dataclass
class Buffer:
    shape: tuple
    strides: tuple
    dtype: str
    value: int | tuple[int, ...] | None = None

    def layout(self):
        return self.shape, self.strides, self.dtype

    def snapshot(self):
        assert self.value is not None
        return (*self.layout(), self.value)


def cascade_outputs(kernels, program, x, output, projections, shape, intern, polar_kernels):
    """Track logical tensors through allocations, alias reuse, and kernel outputs."""
    batch, rows, cols = shape
    dim = min(rows, cols)
    matrix = (shape, (rows * cols, cols, 1), "torch.bfloat16")
    gram = ((batch, dim, dim), (dim * dim, dim, 1), "torch.bfloat16")
    trace_layout = ((batch,), (1,), "torch.float32")
    divisor_layout = ((batch, 2, 1), (2, 1, 1), "torch.float32")
    prepared = intern("prepared")
    first_gram = None
    env = {x: Buffer(*matrix, prepared)}
    operations = Counter()
    counter = None
    rounds = 0
    side = dim // 128
    matrix_tiles = side * (side + 1) // 2
    tiles = batch * matrix_tiles

    def emit(op, reads, details=()):
        token = intern(op, tuple(buf.snapshot() for buf in reads), tuple(details))
        operations[token] += 1
        return token

    def tile_operation(left, dest, polynomial, params):
        nonlocal first_gram
        assert left.layout() == (gram if polynomial else matrix)
        assert dest.layout() == gram and left is not dest
        if polynomial:
            alpha, beta = map(ast.literal_eval, params)
            kernel = "ba_plus_cAA_kernel"
            args = (dim, dim * dim, dim, 1, dim * dim, dim, 1, alpha, beta, tiles, 1, 1)
        else:
            assert not params
            kernel = "XTX_kernel" if rows > cols else "XXT_kernel"
            args = (rows, cols, rows * cols, cols, 1, dim * dim, dim, 1, tiles, 1, 1)
        dest.value = emit(kernel, (left,), args)
        if first_gram is None:
            assert not polynomial and left.value == prepared
            first_gram = dest.value

    for stmt in program:
        func = cname(stmt)
        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                del env[target.id]
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name):
            target, = stmt.targets
            env[target.id] = env[stmt.value.id]
            continue
        if func == "empty_strided_cuda":
            target, = stmt.targets
            size, stride, dtype = stmt.value.args
            env[target.id] = Buffer(ast.literal_eval(size), ast.literal_eval(stride), ast.unparse(dtype))
        elif func == RAIL_SEED_OP:
            target, = stmt.targets
            assert target.id not in env and counter is None and not stmt.value.keywords
            counter = env[target.id] = Buffer((6 * batch,), (1,), "torch.int32", (0,) * 6)
        elif func == INITIAL_OP:
            target, = stmt.targets
            src, a, flag = stmt.value.args
            assert not stmt.value.keywords and first_gram is None and target.id not in env
            assert env[flag.id] is counter and counter.layout() == ((6 * batch,), (1,), "torch.int32")
            assert counter.value == (0,) * 6
            tile_operation(env[src.id], env[a.id], False, ())
            counter.value = (side, 0, 0, 0, 0, 0)
            env[target.id] = Buffer(*divisor_layout, emit("first_gram_trace", (env[a.id],)))
        elif func in (SPLIT_TRACE, DIVISOR_OP):
            target, = stmt.targets
            source, = stmt.value.args
            assert env[source.id].layout() == gram and env[source.id].value == first_gram
            table_layout = trace_layout if func == SPLIT_TRACE else divisor_layout
            # The scalar stages are replayed separately against the same trace
            # and denominator terms. The physical table layout can differ here.
            env[target.id] = Buffer(*table_layout, emit("first_gram_trace", (env[source.id],)))
        elif func == NORMALIZE_OP:
            x_buf, a_buf, table = (env[arg.id] for arg in stmt.value.args)
            assert x_buf.layout() == matrix and x_buf.value == prepared
            assert a_buf.layout() == gram and a_buf.value == first_gram and a_buf is not x_buf
            assert table.layout() == divisor_layout
            x_buf.value = emit("prepared_norm", (x_buf,), (table.value,))
            a_buf.value = emit("gram_norm", (a_buf,), (table.value,))
        elif func == "assert_size_stride":
            buf, size, stride, origin = stmt.value.args
            assert ast.literal_eval(origin) in (DIVISOR_OP, RAIL_SEED_OP, INITIAL_OP)
            assert (env[buf.id].shape, env[buf.id].strides) == (ast.literal_eval(size), ast.literal_eval(stride))
            assert counter is env[buf.id] if ast.literal_eval(origin) == RAIL_SEED_OP else env[buf.id].layout() == divisor_layout
        elif func == "assert_alignment":
            buf, align, origin = stmt.value.args
            assert ast.literal_eval(align) == 16
            assert ast.literal_eval(origin) in (DIVISOR_OP, RAIL_SEED_OP, INITIAL_OP)
            assert counter is env[buf.id] if ast.literal_eval(origin) == RAIL_SEED_OP else env[buf.id].layout() == divisor_layout
        elif func == "get_raw_stream":
            assert ast.unparse(stmt) == "stream0 = get_raw_stream(0)"
        elif func in (TMA_GRAM, TMA_POLYNOMIAL):
            assert not stmt.value.keywords
            src, dst, *params = stmt.value.args
            tile_operation(env[src.id], env[dst.id], func == TMA_POLYNOMIAL, params)
        elif func == ROUND_OP:
            assert not stmt.value.keywords and counter is not None and first_gram is not None
            src, a, b, flag, round_number, *params = stmt.value.args
            assert env[flag.id] is counter and counter.value[0] == side
            rounds += 1
            assert ast.literal_eval(round_number) == rounds <= 5
            assert counter.value[rounds] == 0
            x_buf, a_buf, b_buf = (env[arg.id] for arg in (src, a, b))
            assert len({id(buf) for buf in (x_buf, a_buf, b_buf, counter)}) == 4
            tile_operation(x_buf, a_buf, False, ())
            counter.value = (*counter.value[:rounds], (1 << matrix_tiles) - 1, *counter.value[rounds + 1:])
            tile_operation(a_buf, b_buf, True, params)
        elif func.startswith("extern_kernels."):
            assert func in ("extern_kernels.bmm", "extern_kernels.baddbmm")
            src = [env[arg.id] for arg in stmt.value.args]
            kwargs = {kw.arg: kw.value for kw in stmt.value.keywords}
            dest = env[kwargs.pop("out").id]
            assert dest.layout() == matrix and all(dest is not buf for buf in src)
            assert ([b.layout() for b in src] == [matrix, gram] if func.endswith(".bmm") else
                    [b.layout() for b in src] == [matrix, gram, matrix] and src[0] is src[2])
            dest.value = emit(func, src, sorted((key, ast.literal_eval(value)) for key, value in kwargs.items()))
        elif func.endswith(".run"):
            kernel = stmt.value.func.value.id
            code = kernels[kernel]
            args = stmt.value.args
            assert [(kw.arg, ast.unparse(kw.value)) for kw in stmt.value.keywords] == [("stream", "stream0")]
            if "sqrt" in kernel:
                left, table, *out = [env[arg.id] for arg in args[:-1]]
                assert table.layout() == trace_layout
                assert ast.literal_eval(args[-1]) == prod(left.shape)
                assert left.value in (first_gram, prepared)
                kind = "gram_norm" if left.value == first_gram else "prepared_norm"
                assert left.layout() == (gram if kind == "gram_norm" else matrix)
                if kind == "prepared_norm" and not out:
                    out = [left]
                retained = kind == "gram_norm"
                assert len(out) == 1 + int(retained)
                token = emit(kind, (left,), (table.value,))
                for dest in out:
                    assert dest.layout() == left.layout()
                    dest.value = token
            elif kernel.startswith(("XXT_kernel_", "XTX_kernel_", "ba_plus_cAA_kernel_")):
                left, dest = (env[arg.id] for arg in args[:2])
                assert dest.layout() == gram and left.layout() == (gram if kernel.startswith("ba_") else matrix)
                assert left is not dest
                canonical = canonical_kernel(*code)
                dest.value = emit(polar_kernels.get(canonical, canonical), (left,),
                                  (ast.literal_eval(arg) for arg in args[2:]))
                if first_gram is None:
                    assert kernel.startswith(("XXT_kernel_", "XTX_kernel_")) and left.value == prepared
                    first_gram = dest.value
            else:
                assert kernel.startswith("triton_poi_fused_add_mul_")
                bmm, x_input, dest, alpha = projections[kernel]
                inputs = [env[args[i].id] for i in (bmm, x_input)]
                out = env[args[dest].id]
                assert all(buf.layout() == matrix for buf in (*inputs, out))
                assert inputs[0] is not inputs[1]
                assert ast.literal_eval(args[-1]) == prod(shape)
                out.value = emit("rounded_projection", inputs, (alpha,))
        else:
            raise AssertionError(ast.unparse(stmt))
    if counter is not None:
        assert rounds == 5
    return env[output].snapshot(), operations


def match_cascades(reference, candidate, shape, polar_kernels=None):
    terms = {}

    def intern(*node):
        return terms.setdefault(node, len(terms))

    polar_kernels = {} if polar_kernels is None else polar_kernels
    assert (cascade_outputs(*reference, shape, intern, polar_kernels) ==
            cascade_outputs(*candidate, shape, intern, polar_kernels)), shape


