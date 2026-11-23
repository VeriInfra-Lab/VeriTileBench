"""Compare the actual patched kernels against the pinned trainer's original source."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace


from install_candidate import (
    AMAX, MODEL, NGRAM, CROSS_ENTROPY, SAMPLED, KERNEL, WEIGHT_KERNEL, HEAD_KERNEL, GATHER_KERNEL,
    TRANSPOSE, TRANSPOSE_KERNEL, BF16_TRANSPOSE_KERNEL,
    SAMPLED_CE, DENSIFY_KERNEL,
    ATTENTION, CONFIG, LERP, ROPE, TAIL, TAIL_KERNEL, TAIL_NORM_KERNEL, SPARSE_ROW_KERNEL,
    ANVIL, VALUE_PULL, VALUE_ADAM_KERNEL, VALUE_EMBED, VALUE_ACCUM_KERNEL,
    NGRAM_TABLE, NGRAM_ADAM, NGRAM_MOMENT_KERNEL, NGRAM_ENTRIES_KERNEL,
    ROW_SCATTER, ROW_OWNERS, ROW_FETCH_KERNEL, wide_window_sizes,
    PRIVATE_KERNELS, TRACKED_PATCHES, BASELINE, NATIVE_ADAM, git, uses_packed_adam_reference,
)


def load_functions(path, names, triton, tl, *, strip_decorators=False, **globals_):
    """Load selected functions from the trainer with their test dependencies."""
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(tree.body) != len(names):
        raise ValueError(f"Missing function in {path}: {names}")
    if strip_decorators:
        for node in tree.body:
            node.decorator_list.clear()
    env = dict(triton=triton, tl=tl, __name__=str(path), **globals_)
    exec(compile(tree, str(path), "exec"), env)
    return [env[name] for name in names]


def load_candidate_scale(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("candidate_mlp_scale_refresh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def staged_refresh(source, torch):
    """Execute the exact scalar tail of the unmodified GPT method with its original constants."""
    root = ast.parse(source)
    cls = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == "GPT")
    delayed = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_refresh_delayed_scale")
    delayed.decorator_list = []
    refresh = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_quantize_mlp_weights_and_scales")
    first = next(i for i, n in enumerate(refresh.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "torch.mul")
    refresh.body = refresh.body[:2] + refresh.body[first:]
    prefixes = ("NUM_LAYERS", "PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT", "FP8_MLP_X_SCALE",
                "FP8_GRAD_SCALE", "FP8_POST_HEADROOM", "FP8_DPRE_HEADROOM")
    constants = [node for node in root.body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]).strip("()") in prefixes]
    env = dict(torch=torch)
    exec(compile(ast.Module(body=[*constants, delayed, refresh], type_ignores=[]), str(MODEL), "exec"), env)
    return env, env[delayed.name], env[refresh.name]


def use_cuda_rounding_in_cpu_interpreter(torch, tl):
    """Supply the CUDA rounding semantics while replaying Triton on CPU.

    Triton 3.6's interpreter truncates the BF16 cast and mishandles
    E4M3 ties and mantissa carry. The replay kernels clamp before FP8.
    """
    from triton._C.libtriton import ir
    from triton.runtime import interpreter

    old = interpreter.InterpreterBuilder.cast_impl
    old_fp_to_fp = interpreter.InterpreterBuilder.create_fp_to_fp

    def convert(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            values = torch.from_numpy(src.data.copy()).to(torch.bfloat16)
            return interpreter.TensorHandle(values.view(torch.uint16).numpy(), dst_type.scalar)
        if src.dtype.scalar == tl.bfloat16 and dst_type.scalar == tl.float32:
            values = torch.from_numpy(src.data.copy()).view(torch.bfloat16).float()
            return interpreter.TensorHandle(values.numpy(), dst_type.scalar)
        return old(self, src, dst_type)

    interpreter.InterpreterBuilder.cast_impl = convert

    def convert_fp_to_fp(self, src, dst_type, rounding_mode):
        if (src.dtype.scalar == tl.float32 and dst_type.scalar == tl.float8e4nv
                and rounding_mode == ir.ROUNDING_MODE.RTNE):
            values = torch.from_numpy(src.data.copy()).to(torch.float8_e4m3fn)
            return interpreter.TensorHandle(values.view(torch.uint8).numpy(), dst_type.scalar)
        return old_fp_to_fp(self, src, dst_type, rounding_mode)

    interpreter.InterpreterBuilder.create_fp_to_fp = convert_fp_to_fp


def check_scales(source, candidate_module, torch, device):
    env, delayed, baseline = staged_refresh(source, torch)
    gen = torch.Generator(device=device).manual_seed(1)
    names = (
        "_mlp_up_scales", "_mlp_down_scales", "_mlp_post_amax", "_mlp_post_scales",
        "_mlp_dpre_amax", "_mlp_dpre_scales", "post_lambdas", "_mlp_fold_p",
        "_mlp_up_dequant", "_mlp_dq_bwd", "_mlp_down_scales_folded", "_mlp_dq_bwd_folded",
    )
    for case in range(12):
        values = [torch.randn(12, generator=gen, device=device) for _ in names]
        values[2] = values[2].abs().view(12, 1)
        values[4] = values[4].abs().view(12, 1)
        values[6] = torch.randn(11, 2, generator=gen, device=device)
        exceptional = (float("nan"), float("inf"), -float("inf"), -0., +0., 1e-40,
                       -1e-20, 1e-13, 1e-4, 2e12, 1e-7, 1.0)
        values[2][case, 0] = exceptional[case]
        values[4][case, 0] = exceptional[-case - 1]
        reference_values = [v.clone() for v in values]
        model = SimpleNamespace(**dict(zip(names, reference_values)), _refresh_delayed_scale=delayed)
        baseline(model)
        candidate_module.refresh_mlp_scalars(
            *values, parallel_slot=env["PARALLEL_MLP_SLOT"], parallel_layer=env["PARALLEL_MLP_LAYER"],
            x_scale=env["FP8_MLP_X_SCALE"], grad_scale=env["FP8_GRAD_SCALE"],
            post_gain=env["FP8_POST_HEADROOM"] / torch.finfo(torch.float8_e4m3fn).max,
            dpre_gain=env["FP8_DPRE_HEADROOM"] / torch.finfo(torch.float8_e5m2).max,
        )
        for name, actual, expected in zip(names, values, reference_values):
            if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
                raise AssertionError(f"Scale case {case} differs in {name}")
    print("12 scalar cases: all buffers match bitwise")


HASHES = dict(B0=48271, B1=30011, T0=58699, T1=39779, T2=26801)


def check_ngrams(reference, candidate, triton, torch, device):
    gen = torch.Generator(device=device).manual_seed(2)
    for tokens in (1, 2, 3, 4, 5, 7, 17, 64):
        channels, pool_rows = 768, 8192
        inp = torch.randint(0, 50257, (tokens,), device=device, generator=gen, dtype=torch.int32)
        slots = torch.randint(0, 2 * tokens, (2 * tokens,), device=device, generator=gen, dtype=torch.int32)
        cache = torch.randn(2 * tokens, channels, device=device, generator=gen).to(torch.bfloat16)
        pool = torch.randn(pool_rows, channels, device=device, generator=gen).sign().to(torch.bfloat16)
        grad = torch.randn(tokens, channels, device=device, generator=gen).to(torch.bfloat16)
        fwd_ref, fwd = torch.empty_like(grad), torch.empty_like(grad)
        bwd_ref, bwd = torch.empty_like(cache), torch.empty_like(cache)
        n = tokens * channels
        flat_grid = (triton.cdiv(n, 1024),)
        tiled_grid = (triton.cdiv(tokens, 4), triton.cdiv(channels, 256))
        kwargs = dict(D=channels, RMASK=pool_rows - 1, num_warps=4, num_stages=1, **HASHES)
        reference[1][flat_grid](cache, slots, pool, inp, fwd_ref, tokens, n, XBLOCK=1024, **kwargs)
        candidate[1][tiled_grid](cache, slots, pool, inp, fwd, tokens, BT=4, BC=256, **kwargs)
        reference[2][flat_grid](grad, pool, inp, bwd_ref, n, XBLOCK=1024, **kwargs)
        candidate[2][tiled_grid](grad, pool, inp, bwd, tokens, BT=4, BC=256, **kwargs)
        assert torch.equal(fwd.view(torch.int16), fwd_ref.view(torch.int16)), f"forward T={tokens}"
        assert torch.equal(bwd.view(torch.int16), bwd_ref.view(torch.int16)), f"backward T={tokens}"
        if tokens % 16 == 0:
            candidate[1][(tokens // 4,)](cache, slots, pool, inp, fwd, tokens, BT=4, BC=256, FULL_ROWS=True,
                                         **kwargs)
            candidate[2][(tokens // 4,)](grad, pool, inp, bwd, tokens, BT=4, BC=256, FULL_ROWS=True, **kwargs)
            assert torch.equal(fwd.view(torch.int16), fwd_ref.view(torch.int16)), f"full forward T={tokens}"
            assert torch.equal(bwd.view(torch.int16), bwd_ref.view(torch.int16)), f"full backward T={tokens}"
    print("8 n-gram sizes and the full-row launch: forward and backward match bitwise")


# (batch entries, QK rows, V rows) of the three packed caches.
QKV_GROUPS = ((3, 768, 768), (2, 768, 384), (2, 1536, 768))


def check_amax(reference, candidate, torch, device):
    gen = torch.Generator(device=device).manual_seed(3)
    # Also cover unequal trip counts and parts with no work in the QK bank.
    for batch, first_rows, second_rows, cols in (
        *((b, qk, v, 768) for b, qk, v in QKV_GROUPS),
        (2, 3, 7, 2048),
    ):
        data = [torch.randn(batch, n + 1, cols, generator=gen, device=device).to(torch.bfloat16)[:, :n, :]
                for n in (first_rows, second_rows)]
        data[0][0, 0, 0] = -0.0
        data[1][-1, -1, -1] = float("inf")
        expected = torch.empty(batch, 32, device=device)
        actual = torch.empty_like(expected)
        constants = dict(FIRST_ELEMENTS=first_rows * cols, SECOND_ELEMENTS=second_rows * cols,
                         FIRST_STRIDE_B=data[0].stride(0), SECOND_STRIDE_B=data[1].stride(0),
                         NUM_PARTS=32, BLOCK_SIZE=2048)
        reference[(batch, 32)](*data, expected, **constants, num_warps=8, num_stages=1)
        candidate[(batch, 32)](*data, actual, **constants, num_warps=8, num_stages=1)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (first_rows, second_rows, cols)
    print("4 QKV layouts: all per-part FP32 maxima match bitwise")


def check_native_qkv(baseline, candidate, triton, tl, torch, device, hopper, old_scan,
                     native_scan, old_pack, scale_kernel, gluon_pack, grouped_kernels):
    from unittest.mock import MagicMock, patch

    gen = torch.Generator(device=device).manual_seed(9)
    slots = sum(batch for batch, _, _ in QKV_GROUPS)
    qk_bank = torch.randn(slots, 12, 128, 768, dtype=torch.bfloat16, generator=gen, device=device)
    vo_bank = torch.randn(slots, 2, 6, 128, 768, dtype=torch.bfloat16, generator=gen, device=device)
    vo_bank[:, 1].fill_(1024)
    if hopper:
        scan, scale, pack = native_scan, scale_kernel, gluon_pack
    else:
        scan, scale, pack = (MagicMock() for _ in range(3))
    grouped_scan, grouped_scale, grouped_pack = grouped_kernels
    if not hopper:
        grouped_pack = MagicMock()
    _, _, native, grouped = load_functions(
        candidate / AMAX, ("_native_bank_layout", "_native_cache_buffers", "refresh_native_qkv",
                           "refresh_grouped_qkv"), triton, tl, torch=torch, _aligned_packed_amax=scan,
        _qkv_scale_from_partials=scale, _quantize_qkv_tiles_gluon=pack, _grouped_qkv_scan=grouped_scan,
        _grouped_qkv_scale=grouped_scale, _grouped_qkv_pack=grouped_pack,
    )
    original, = load_functions(baseline / AMAX, ("quantize_dual_layout_packed_batched",), triton, tl,
                               torch=torch, _packed_batched_matrix_amax_partial_kernel=old_scan,
                               _quantize_dual_layout_packed_batched_kernel=old_pack)

    start, copied_bytes = 0, 0
    expected, grouped_caches = {}, {}
    for name, (batch, first_rows, second_rows) in zip(("narrow", "half_v", "wide"), QKV_GROUPS):
        qk_width, v_width = first_rows // 12, second_rows // 6
        first = qk_bank[start:start+batch]
        second = vo_bank[start:start+batch, 0]
        start += batch
        first[:, :, qk_width:].fill_(512)
        second[:, :, v_width:].fill_(768)
        first[-1, -1, qk_width - 1, -1] = -7
        second[-1, -1, v_width - 1, -1] = 13
        compact = (first[:, :, :qk_width].reshape(batch, first_rows, 768),
                   second[:, :, :v_width].reshape(batch, second_rows, 768))
        copied_bytes += sum(packed.numel() * packed.element_size()
                            for packed, raw in zip(compact, (first, second))
                            if packed.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr())
        rows = first_rows + second_rows
        expect = SimpleNamespace(
            partial_amax=torch.empty(batch, 32, device=device),
            scales=torch.empty(batch, device=device),
            row=torch.empty(batch, rows, 768, dtype=torch.float8_e4m3fn, device=device),
            col=torch.empty(batch, 768, rows, dtype=torch.float8_e4m3fn, device=device))
        actual = SimpleNamespace(**{key: torch.empty_like(value) for key, value in vars(expect).items()})
        if hopper:
            original(*compact, expect.scales, expect.partial_amax, expect.row, expect.col)
            assert native(actual, first, second, qk_width, v_width)
            for key in vars(expect):
                assert torch.equal(getattr(expect, key).view(torch.uint8), getattr(actual, key).view(torch.uint8)), key
        else:
            assert not native(actual, first, second, qk_width, v_width)
            for launch in (scan, scale, pack):
                launch.__getitem__.assert_not_called()
            constants = dict(FIRST_ELEMENTS=first_rows * 768, SECOND_ELEMENTS=second_rows * 768,
                             NUM_PARTS=32, BLOCK_SIZE=2048, num_warps=8, num_stages=1)
            old_scan[(batch, 32)](
                *compact, expect.partial_amax, FIRST_STRIDE_B=compact[0].stride(0),
                SECOND_STRIDE_B=compact[1].stride(0), **constants)
            native_scan[(batch, 32)](
                first, second, actual.partial_amax, FIRST_STRIDE_B=first.stride(0),
                SECOND_STRIDE_B=second.stride(0), QK_CROP=qk_width == 64, V_CROP=v_width == 64, **constants)
            assert torch.equal(expect.partial_amax.view(torch.int32), actual.partial_amax.view(torch.int32))
            torch.mul(expect.partial_amax.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=expect.scales)
        expected[name] = expect
        grouped_caches[name] = SimpleNamespace(**{key: torch.empty_like(value) for key, value in vars(expect).items()})
    assert copied_bytes == 7077888
    print(f"Native QKV bank stride: bitwise {'full-cache' if hopper else 'amax'} agreement at 3 widths;"
          f" reference reshape materializes {copied_bytes / (1 << 20):.2f} MiB per refresh")
    assert not grouped(grouped_caches, qk_bank[1:], vo_bank[1:, 0])
    if hopper:
        assert grouped(grouped_caches, qk_bank, vo_bank[:, 0])
    else:
        assert not grouped(grouped_caches, qk_bank, vo_bank[:, 0])
        grouped_pack.__getitem__.assert_not_called()
        # Exercise the grouped launcher's argument order with the two Triton
        # stages on this device. Gluon pack bytes are covered by the PTX replay.
        with patch.dict(grouped.__globals__, _native_bank_layout=lambda first, second, batch: batch == 7):
            assert grouped(grouped_caches, qk_bank, vo_bank[:, 0])
        grouped_pack.__getitem__.assert_called_once_with((6, 6, 30))
    keys = ("partial_amax", "scales", "row", "col") if hopper else ("partial_amax", "scales")
    for name, expect in expected.items():
        for key in keys:
            actual = getattr(grouped_caches[name], key)
            assert torch.equal(getattr(expect, key).view(torch.uint8), actual.view(torch.uint8)), (name, key)
    print(f"Grouped QKV refresh: all 224 partials and 7 scales match bitwise"
          f"{', with both FP8 cache layouts' if hopper else ''}")


def check_qkv_scales(kernel, torch, device):
    gen = torch.Generator(device=device).manual_seed(4)
    for rows in (1, 2, 3):
        actual = torch.empty(rows, dtype=torch.float32, device=device)
        expected = torch.empty_like(actual)
        partials = torch.rand(rows, 32, dtype=torch.float32, device=device, generator=gen)
        cases = (
            partials,
            -partials,
            torch.where(partials > 0.5, 0.0, -0.0),
            partials * 1.0e-12,
            partials * 2.0e-12,
            partials * torch.finfo(torch.float32).max,
            torch.cat((partials[:, :31], torch.full_like(partials[:, :1], float("inf"))), dim=1),
            torch.cat((partials[:, :31], torch.full_like(partials[:, :1], float("nan"))), dim=1),
        )
        for case, source in enumerate(cases):
            torch.mul(source.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=expected)
            kernel[(rows,)](source, actual, num_warps=1)
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (rows, case)
    print("24 QKV scale cases: all FP32 scales match bitwise")


def check_qkv_scale_dispatch(path, kernel, triton, tl, torch, device):
    from unittest.mock import MagicMock

    wrapper, = load_functions(path, ("quantize_dual_layout_packed_batched",), triton, tl)
    ignored = MagicMock()
    fused = MagicMock()
    fused.__getitem__.side_effect = lambda grid: kernel[grid]
    wrapper.__globals__.update(
        torch=torch, _packed_batched_matrix_amax_partial_kernel=ignored, _aligned_packed_amax=ignored,
        _qkv_scale_from_partials=fused, _quantize_dual_layout_packed_batched_kernel=ignored,
    )
    batch = 2
    first = torch.empty(batch, 1, 4, device=device, dtype=torch.bfloat16)
    second = torch.empty_like(first)
    row = torch.empty(batch, 2, 4, device=device, dtype=torch.float8_e4m3fn)
    col = torch.empty(batch, 4, 2, device=device, dtype=row.dtype)
    for layout in ("contiguous", "aliased", "strided", "other_parts"):
        count = 33 if layout in ("strided", "other_parts") else 32
        partials = torch.arange(batch * count, dtype=torch.float32, device=device).reshape(batch, count)
        if layout == "strided":
            partials = partials[:, :32]
        scales = partials.view(-1)[1:batch + 1] if layout == "aliased" else torch.empty(batch, device=device)
        expected = partials.amax(dim=1).clamp_(min=1.0e-12) * (1.0 / 448.0)
        fused.reset_mock()
        wrapper(first, second, scales, partials, row, col)
        assert fused.__getitem__.call_count == int(layout == "contiguous"), layout
        assert torch.equal(scales.view(torch.int32), expected.view(torch.int32)), layout
    print("QKV scale launch dispatch: eligible path and three fallback layouts match")


def pack_constants(first_rows, second_rows, cols, first_stride, second_stride):
    output_stride = (first_rows + second_rows) * cols
    return dict(FIRST_ROWS=first_rows, FIRST_STRIDE_B=first_stride, SECOND_STRIDE_B=second_stride,
                ROW_STRIDE_B=output_stride, TRANSPOSED_STRIDE_B=output_stride,
                BLOCK_M=64, BLOCK_N=128)


def check_qkv_pack(reference, candidates, torch, device):
    gen = torch.Generator(device=device).manual_seed(5)
    for width, (batch, first_rows, second_rows) in enumerate(QKV_GROUPS):
        cols, rows = 768, first_rows + second_rows
        v_pitch = second_rows if width == 1 else 2 * second_rows

        def bank(pitch, valid):
            return torch.randn(batch, pitch, cols, generator=gen, device=device).to(torch.bfloat16)[:, :valid]

        first, second = bank(first_rows, first_rows), bank(v_pitch, second_rows)
        edge = torch.tensor([0., -0., float("nan"), float("inf"), float("-inf"), 2**-12, 2**-18],
                            device=device, dtype=torch.bfloat16)
        first[0, -1, :len(edge)] = edge
        second[0, 0, :len(edge)] = edge
        scales = torch.linspace(.0004, .0012, batch, device=device)
        constants = pack_constants(first_rows, second_rows, cols, first.stride(0), second.stride(0))

        def run(kernel):
            row = torch.empty(batch, rows, cols, device=device, dtype=torch.float8_e4m3fn)
            col = torch.empty(batch, cols, rows, device=device, dtype=torch.float8_e4m3fn)
            kernel[(rows // 64, cols // 128, batch)](
                first, second, row, col, scales, rows, cols, **constants, num_warps=8, num_stages=2)
            return row.view(torch.uint8), col.view(torch.uint8)

        expected = run(reference)
        for name, candidate in candidates:
            for actual, ref in zip(run(candidate), expected):
                assert torch.equal(actual, ref), (name, batch, first_rows, second_rows)
    for name, _ in candidates:
        print(f"{name}: both FP8 caches match byte for byte at all 3 widths")


def check_qkv_pack_dispatch(path, triton, tl, torch, device):
    from unittest.mock import MagicMock, patch

    wrapper, = load_functions(path, ("quantize_dual_layout_packed_batched",), triton, tl)
    old_pack, new_pack, explicit_pack, ignored = (MagicMock() for _ in range(4))
    hopper = device == "cuda" and torch.cuda.get_device_capability()[0] == 9
    wrapper.__globals__.update(
        torch=torch, _packed_batched_matrix_amax_partial_kernel=ignored, _aligned_packed_amax=ignored,
        _qkv_scale_from_partials=ignored, _quantize_dual_layout_packed_batched_kernel=old_pack,
        _quantize_aligned_qkv_tiles=new_pack, _quantize_qkv_tiles_gluon=explicit_pack,
    )
    for first_rows, second_rows, cols, aligned in (
        (64, 64, 128, True), (32, 96, 128, False), (64, 32, 128, False), (64, 64, 96, False),
    ):
        first = torch.empty(1, first_rows, cols, device=device, dtype=torch.bfloat16)
        second = torch.empty(1, second_rows, cols, device=device, dtype=first.dtype)
        row = torch.empty(1, first_rows + second_rows, cols, device=device, dtype=torch.float8_e4m3fn)
        col = torch.empty(1, cols, first_rows + second_rows, device=device, dtype=row.dtype)
        old_pack.reset_mock()
        new_pack.reset_mock()
        explicit_pack.reset_mock()
        wrapper(first, second, torch.empty(1, device=device), torch.zeros(1, 32, device=device), row, col)
        assert new_pack.__getitem__.call_count == int(aligned and not hopper)
        assert explicit_pack.__getitem__.call_count == int(aligned and hopper)
        assert old_pack.__getitem__.call_count == int(not aligned)
    if device == "cpu":
        first = torch.empty(1, 64, 128, dtype=torch.bfloat16)
        # Exercise the actual host predicate without submitting a GPU launch.
        fake_bank = SimpleNamespace(shape=first.shape, dtype=first.dtype, stride=first.stride,
                                    is_cuda=True, device=torch.device("cuda", 0))
        row = torch.empty(1, 128, 128, dtype=torch.float8_e4m3fn)
        col = torch.empty_like(row)
        for major in (9, 10):
            new_pack.reset_mock()
            explicit_pack.reset_mock()
            with patch.object(torch.cuda, "get_device_capability", return_value=(major, 0)):
                wrapper(fake_bank, fake_bank, torch.empty(1), torch.zeros(1, 32), row, col)
            assert explicit_pack.__getitem__.call_count == int(major == 9)
            assert new_pack.__getitem__.call_count == int(major != 9)
    print("QKV pack launch dispatch: Hopper, aligned fallback and three partial layouts match")


def check_mlp_weight_bank(path, triton, tl, torch, device, hopper, kernels):
    from unittest.mock import MagicMock, patch

    reference, scale, pack = kernels
    generator = torch.Generator(device=device).manual_seed(27)
    gain = 1.12 / 448.0
    partials = [torch.randn(12, 528, device=device, generator=generator) for _ in range(2)]
    for scratch in partials:
        scratch[0].fill_(-float("inf"))
        scratch[1, -1] = float("inf")
        scratch[2, -1] = float("nan")
        scratch[3].fill_(-0.0)
        scratch[4].fill_(1e-16)
        scratch[5].fill_(8e-10)
    scales = [torch.empty(12, device=device) for _ in partials]
    scale[(24,)](*partials, *scales, GAIN=gain, num_warps=4)
    for scratch, actual in zip(partials, scales):
        expected = torch.clamp(scratch.amax(dim=1) * gain, min=1.0e-12)
        finite = ~expected.isnan()
        assert torch.equal(actual.isnan(), expected.isnan())
        assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
    print("Delayed MLP refresh: 24 scales agree, including the floor and non-finite inputs")

    if not hopper:
        mock_scale, mock_pack = MagicMock(), MagicMock()
        launcher, = load_functions(path, ("refresh_mlp_weight_bank",), triton, tl, torch=torch,
                                   WEIGHT_SCALE_HEADROOM=1.12, _mlp_delayed_scales=mock_scale,
                                   _mlp_paired_pack=mock_pack)

        def tensor(shape, dtype):
            return SimpleNamespace(shape=shape, dtype=dtype, device="cuda:0", is_cuda=True, is_contiguous=lambda: True)

        bank = tensor((12, 2, 2816, 768), torch.bfloat16)
        operands = (tensor((12,), torch.float32), tensor((12, 528), torch.float32),
                    tensor((12, 2816, 768), torch.float8_e4m3fn), tensor((12, 768, 2816), torch.float8_e4m3fn))
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            assert launcher(bank, operands, operands, delayed=False)
            mock_scale.__getitem__.assert_not_called()
            mock_pack.__getitem__.assert_called_once_with((6, 44, 24))
            assert launcher(bank, operands, operands, delayed=True)
            mock_scale.__getitem__.assert_called_once_with((24,))
            assert not launcher(bank, operands, operands[:-1] + (operands[2],), delayed=True)
        return

    launcher, = load_functions(path, ("refresh_mlp_weight_bank",), triton, tl, torch=torch,
                               WEIGHT_SCALE_HEADROOM=1.12, _mlp_delayed_scales=scale, _mlp_paired_pack=pack)
    bank = torch.randn(12, 2, 2816, 768, dtype=torch.bfloat16, device=device, generator=generator)
    # Exercise rounding, saturation, NaN policy and the reduction in the last column half.
    bank[0, 0, 0, :5] = torch.tensor([0., -0., float("nan"), float("inf"), float("-inf")],
                                     dtype=bank.dtype, device=device)
    bank[-1, 1, -64:, -64:].fill_(float("nan"))
    for delayed in (False, True):
        buffers, expected = [], []
        for projection in range(2):
            wt = bank[:, projection]
            scratch = torch.rand(12, 528, device=device, generator=generator)
            scales = torch.full((12,), gain, device=device)
            row = torch.empty(12, 2816, 768, device=device, dtype=torch.float8_e4m3fn)
            col = torch.empty(12, 768, 2816, device=device, dtype=torch.float8_e4m3fn)
            buffers.append((scales.clone(), scratch.clone(), torch.empty_like(row), torch.empty_like(col)))
            if delayed:
                torch.clamp(scratch.amax(dim=1) * gain, min=1.0e-12, out=scales)
            reference[(12, 528)](wt, row, col, scales, scratch, *wt.stride(),
                                   H=2816, D=768, num_tiles_d=12, num_tiles=528,
                                   BLOCK_H=64, BLOCK_D=64, num_warps=4, num_stages=2)
            expected.append((scales, scratch, row, col))
        assert launcher(bank, *buffers, delayed=delayed)
        for observed, original in zip(buffers, expected):
            for actual, ref in zip(observed, original):
                assert torch.equal(actual.view(torch.uint8), ref.view(torch.uint8))
    print("MLP weight refresh: both layouts, delayed scales and 12,672 partials agree bitwise on Hopper")


def check_lm_head(baseline_path, candidate_path, triton, tl, torch, device, hopper, kernels):
    from unittest.mock import MagicMock, patch

    baseline, gluon_pack = kernels
    reference, = load_functions(baseline_path, ("quantize_lm_head_dual",), triton, tl,
                                _quantize_lm_head_dual_kernel=baseline)
    pack = gluon_pack if hopper else MagicMock()
    candidate, = load_functions(candidate_path, ("quantize_lm_head_dual",), triton, tl, torch=torch,
                                 _reference=reference, _quantize_lm_head_gluon=pack)
    shape = (768, 50304) if hopper else (65, 133)
    generator = torch.Generator(device=device).manual_seed(17)
    weights = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    weights[0, :9] = torch.tensor([0., -0., 2**-12, -2**-12, 448., -448., float("inf"), -1.5, .5546875],
                                  dtype=weights.dtype, device=device)
    inverse = (torch.tensor([1.], dtype=torch.float32) / torch.tensor(2 / 448, dtype=torch.float32)).to(device)
    row = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    col = torch.empty(shape[::-1], dtype=row.dtype, device=device).T
    ref_row, ref_col = torch.empty_like(row), torch.empty_like(col)
    gains = (inverse,)
    if hopper:
        all_bf16_bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
        weights.view(torch.int16).flatten()[:65536] = all_bf16_bits
        gains = (inverse, torch.tensor([1.375], dtype=torch.float32, device=device))
    for gain in gains:
        candidate(weights, row, col, gain)
        reference(weights, ref_row, ref_col, gain)
        assert torch.equal(row.view(torch.uint8), ref_row.view(torch.uint8))
        assert torch.equal(col.T.view(torch.uint8), ref_col.T.view(torch.uint8))
    if not hopper:
        pack.__getitem__.assert_not_called()
        expected = (weights.float() * inverse).to(torch.bfloat16).float().clamp(-448, 448).to(row.dtype)
        assert torch.equal(row.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(col.T.view(torch.uint8), expected.T.contiguous().view(torch.uint8))
        device_shape = (768, 50304)
        src = SimpleNamespace(device="cuda:0", is_cuda=True, dtype=torch.bfloat16, shape=device_shape,
                              is_contiguous=lambda: True)
        out = SimpleNamespace(device="cuda:0", dtype=row.dtype, shape=device_shape, is_contiguous=lambda: True)
        transposed = SimpleNamespace(device="cuda:0", dtype=row.dtype, shape=device_shape, stride=lambda: (1, 768))
        inv = SimpleNamespace(device="cuda:0", dtype=torch.float32, shape=(1,))
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            candidate(src, out, transposed, inv)
        pack.__getitem__.assert_called_once_with((393, 12))
    print(f"LM-head FP8 caches: both layouts agree byte for byte on the {'Hopper' if hopper else 'tail fallback'} path")


SAMPLED_COUNTS = (10240, 14336, 24576)


def check_sampled_gather(baseline, candidate, torch, device, hopper, module):
    import math
    from unittest.mock import MagicMock, patch

    launcher = module.gather_sampled_rows

    def load_gather(root):
        path = root / SAMPLED
        tree = ast.parse(path.read_text(), filename=str(path))
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SampledSoftmax")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_gather_rows")
        env = dict(torch=torch, Tensor=torch.Tensor, SampledLoss=SimpleNamespace, gather_sampled_rows=launcher,
                   transpose_copy=lambda src, dst: dst.copy_(src.T))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
        return env["_gather_rows"]

    def outputs(p):
        return SimpleNamespace(rows=torch.empty((p, 768), dtype=torch.float8_e4m3fn, device=device),
                               rows_t=torch.empty((768, p), dtype=torch.float8_e4m3fn, device=device),
                               vocab_pos=torch.empty(50304, dtype=torch.int32, device=device))

    reference, optimized = load_gather(baseline), load_gather(candidate)
    mode = "Hopper" if hopper else "fallback"
    generator = torch.Generator(device=device).manual_seed(707)
    weights = torch.randint(0, 256, (50304, 768), dtype=torch.uint8, device=device, generator=generator)
    cache = weights.view(torch.float8_e4m3fn).T
    for p in SAMPLED_COUNTS:
        permutation = (torch.arange(p, dtype=torch.int64, device=device) * 20011 + 137) % 50304
        holder = SimpleNamespace(candidates=permutation.sort().values,
                                 arange=torch.arange(p, dtype=torch.int32, device=device))
        original, observed = outputs(p), outputs(p)
        reference(holder, cache, p, original)
        observed.vocab_pos.fill_(0x12345)
        optimized(holder, cache, p, observed)
        for key, array in vars(original).items():
            assert torch.equal(array.view(torch.uint8), getattr(observed, key).view(torch.uint8)), (p, key)
        if not hopper:
            # Check the real launcher with Hopper metadata and a mocked pack.
            def metadata(shape, dtype, slot):
                return SimpleNamespace(shape=shape, dtype=dtype, device="cuda:0", is_cuda=True,
                                       numel=lambda: math.prod(shape), element_size=lambda: dtype.itemsize,
                                       data_ptr=lambda: slot << 40, is_contiguous=lambda: True)

            bank = metadata(weights.shape, weights.dtype, 1)
            ids = metadata((p,), torch.int64, 2)
            row, col = metadata((p, 768), torch.uint8, 3), metadata((768, p), torch.uint8, 4)
            inverse = metadata((50304,), torch.int32, 5)
            pack = MagicMock()
            with (patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
                  patch.object(module, "_gather_candidate_rows_and_inverse", pack)):
                assert launcher(bank, ids, row, col, inverse=inverse)
                assert not launcher(bank, ids, row, row)
                inverse.data_ptr = bank.data_ptr
                assert not launcher(bank, ids, row, col, inverse=inverse)
            pack.__getitem__.assert_called_once_with((197 + 6 * (p // 128),))
    print(f"Sampled gather ({mode}): both byte layouts and inverse map agree at all three candidate counts")


def check_densify(baseline, candidate, triton, tl, torch, device, hopper, kernels):
    from unittest.mock import MagicMock, patch

    original, fused = kernels
    pack = fused if hopper else MagicMock()
    fast, = load_functions(candidate / DENSIFY_KERNEL, ("densify_sampled_gradient",), triton, tl,
                           torch=torch, _densify_gluon=pack)

    def load_dispatch(path):
        tree = ast.parse(path.read_text(), filename=str(path))
        dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sampled_densify")
        dispatch.decorator_list = []
        env = dict(torch=torch, triton=triton, _densify_kernel=original, densify_sampled_gradient=fast)
        exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(path), "exec"), env)
        return env["sampled_densify"]

    reference, optimized = (load_dispatch(root / SAMPLED_CE) for root in (baseline, candidate))
    rng = torch.Generator(device=device).manual_seed(711)
    for p in SAMPLED_COUNTS if hopper else (17,):
        r, v = (768, 50304) if hopper else (33, 259)
        src = torch.randint(-32768, 32768, (r, p), dtype=torch.int16, device=device, generator=rng).view(torch.bfloat16)
        pos = torch.full((v,), -1, dtype=torch.int32, device=device)
        ids = torch.randperm(v, generator=rng, device=device)[:p].sort().values
        pos[ids] = torch.arange(p, dtype=torch.int32, device=device)
        old = torch.empty((r, v), dtype=src.dtype, device=device)
        new = torch.empty_like(old)
        reference(src, pos, old)
        optimized(src, pos, new)
        assert torch.equal(old.view(torch.int16), new.view(torch.int16)), p
        assert torch.equal(new.view(torch.int16).index_select(1, ids), src.view(torch.int16)), p
        assert new.view(torch.int16)[:, pos < 0].count_nonzero().item() == 0, p
    if not hopper:
        pack.__getitem__.assert_not_called()

        def meta(shape, dtype, ptr):
            from math import prod

            return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device="cuda:0", is_cuda=True,
                                   is_contiguous=lambda: True, data_ptr=lambda: ptr, numel=lambda: prod(shape))

        slots = meta((50304,), torch.int32, 2**41)
        gradient = meta((768, 50304), torch.bfloat16, 2**42)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            for p in SAMPLED_COUNTS:
                rows = meta((768, p), torch.bfloat16, 2**40)
                assert fast(rows, slots, gradient)
            assert not fast(rows, slots, meta(gradient.shape, gradient.dtype, rows.data_ptr()))
            assert not fast(rows, slots, meta(gradient.shape, gradient.dtype, slots.data_ptr() + 16))
            assert not fast(rows, slots, meta(gradient.shape, gradient.dtype, gradient.data_ptr() + 2))
        assert [call.args for call in pack.__getitem__.call_args_list] == [((393, 12),)] * len(SAMPLED_COUNTS)
        assert [call.kwargs for call in pack.__getitem__.return_value.call_args_list] == [
            dict(P=p, num_warps=8, num_stages=3) for p in SAMPLED_COUNTS
        ]
    print(f"Sampled densify ({'Hopper' if hopper else 'fallback'}): selected BF16 halfwords and zero padding agree")


def check_transpose(baseline, candidate, triton, tl, torch, device, hopper, byte_module, bf16_kernel):
    from unittest.mock import MagicMock, patch

    pack, bf16 = (byte_module._byte_transpose, bf16_kernel) if hopper else (MagicMock(), MagicMock())
    fast, = load_functions(candidate / TRANSPOSE_KERNEL, ("transpose_hopper_bytes",), triton, tl,
                           torch=torch, _byte_transpose=pack)
    fast_bf16, = load_functions(candidate / BF16_TRANSPOSE_KERNEL, ("transpose_hopper_bf16",), triton, tl,
                                torch=torch, _warp_bf16_transpose=bf16)
    copy, add = load_functions(baseline / TRANSPOSE, ("_transpose_copy_kernel", "_transpose_add_kernel"), triton, tl)
    funcs = ("transpose_copy", "transpose_add")
    env = dict(torch=torch, _transpose_copy_kernel=copy, _transpose_add_kernel=add)
    reference, reference_add = load_functions(baseline / TRANSPOSE, funcs, triton, tl, **env)
    optimized, optimized_add, _, transpose_ce, _ = load_functions(
        candidate / TRANSPOSE, (*funcs, "_transpose_copy_fake", "transpose_ce_operands", "_transpose_ce_operands_fake"),
        triton, tl, **env,
        transpose_hopper_bytes=fast, transpose_hopper_bf16=fast_bf16,
        transpose_hopper_ce_operands=byte_module.transpose_hopper_ce_operands,
    )

    def transpose(src):
        dst = torch.empty((src.shape[1], src.shape[0]), device=src.device, dtype=src.dtype)
        optimized(src, dst)
        return dst

    rng = torch.Generator(device=device).manual_seed(709)
    def ce_operands(x, grad):
        xt = torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device)
        gt = torch.empty((grad.shape[1], grad.shape[0]), dtype=grad.dtype, device=grad.device)
        transpose_ce(x, xt, grad, gt)
        return xt, gt

    for m, n in ((128, 10240), (129, 259)):
        x = torch.randint(256, (m, 768), device=device, generator=rng, dtype=torch.uint8).view(torch.float8_e4m3fn)
        grad = torch.randint(256, (m, n), device=device, generator=rng, dtype=torch.uint8).view(torch.float8_e5m2)
        run_ce = ce_operands if hopper else torch.compile(ce_operands, backend="aot_eager", fullgraph=True)
        xt, gt = run_ce(x, grad)
        for src, dst in ((x, xt), (grad, gt)):
            assert torch.equal(src.view(torch.uint8).T, dst.view(torch.uint8)), (m, n, src.dtype)
    print(f"CE backward transposes ({'Hopper' if hopper else 'fallback'}): both FP8 operands match bitwise")

    dtypes = (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2) if hopper else (torch.uint8,)
    for shape in ((256, 384), (129, 131)):
        for dtype in dtypes:
            src = torch.randint(256, shape, device=device, generator=rng, dtype=torch.uint8).view(dtype)
            expected = torch.empty(shape[::-1], dtype=dtype, device=device)
            actual = torch.empty_like(expected)
            reference(src, expected)
            optimized(src, actual)
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), (shape, dtype)
    for accumulating, source_shape, original, changed in (
        (False, (768, 50304), reference, optimized),
        (True, (50304, 768), reference_add, optimized_add),
    ):
        shape = source_shape if hopper else (129, 131)
        src = torch.randint(-32768, 32768, shape, device=device, dtype=torch.int16, generator=rng).view(torch.bfloat16)
        expected = torch.randint(-32768, 32768, shape[::-1], device=device, dtype=torch.int16,
                                 generator=rng).view(torch.bfloat16)
        actual = expected.clone()
        original(src, expected)
        changed(src, actual)
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), ("bf16", accumulating)
    if not hopper:
        pack.__getitem__.assert_not_called()
        src = torch.randint(256, (16, 32), device=device, generator=rng, dtype=torch.uint8)
        actual = torch.compile(transpose, backend="aot_eager", fullgraph=True)(src)
        assert torch.equal(src.T, actual)

        def metadata(dtype, pointer):
            return SimpleNamespace(dtype=dtype, device="cuda:0", is_cuda=True, is_contiguous=lambda: True,
                                   data_ptr=lambda: pointer, view=lambda dtype: dtype)

        row = metadata(torch.float8_e5m2, 2**40)
        col = metadata(torch.float8_e5m2, 2**41)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            assert fast(row, col, 40960, 50304)
            assert not fast(row, col, 49152, 50304)
            assert not fast(row, col, 65536 * 128, 128)
            assert not fast(row, col, 128, 129)
            assert not fast(row, row, 128, 128)
            assert not fast(row, metadata(row.dtype, row.data_ptr() + 16), 128, 128)
            assert not fast(row, metadata(row.dtype, col.data_ptr() + 1), 128, 128)
        pack.__getitem__.assert_called_once_with((393, 320))
        pack.__getitem__.return_value.assert_called_once_with(
            torch.uint8, torch.uint8, 320, 393, num_warps=8, num_stages=2)
        ce_pack = MagicMock()
        x = metadata(torch.float8_e4m3fn, 2**40)
        xt = metadata(torch.float8_e4m3fn, 2**41)
        grad = metadata(torch.float8_e5m2, 2**42)
        gt = metadata(torch.float8_e5m2, 2**43)
        for tensor, shape in ((x, (49152, 768)), (xt, (768, 49152)),
                              (grad, (49152, 24576)), (gt, (24576, 49152))):
            tensor.shape, tensor.ndim = shape, 2
            tensor.numel = lambda shape=shape: shape[0] * shape[1]
        with (patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
              patch.object(byte_module, "_ce_backward_transposes", ce_pack)):
            assert byte_module.transpose_hopper_ce_operands(x, xt, grad, gt)
            gt.data_ptr = grad.data_ptr
            assert not byte_module.transpose_hopper_ce_operands(x, xt, grad, gt)
        ce_pack.__getitem__.assert_called_once_with((198, 384))
        ce_pack.__getitem__.return_value.assert_called_once_with(
            torch.uint8, torch.uint8, torch.uint8, torch.uint8, 384, 192, num_warps=8, num_stages=2)
        bf16.__getitem__.assert_not_called()
        row = metadata(torch.bfloat16, 2**40)
        col = metadata(torch.bfloat16, 2**41)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            assert fast_bf16(row, col, 768, 50304, add=False)
            assert fast_bf16(col, row, 50304, 768, add=True)
            assert not fast_bf16(row, col, 768, 50304, add=True)
            assert not fast_bf16(row, row, 768, 50304, add=False)
            assert not fast_bf16(row, metadata(row.dtype, row.data_ptr() + 50304 * 768), 768, 50304, add=False)
            assert not fast_bf16(row, metadata(row.dtype, col.data_ptr() + 2), 768, 50304, add=False)
        assert [call.args for call in bf16.__getitem__.call_args_list] == [((393, 12),), ((6, 786),)]
        assert [call.kwargs for call in bf16.__getitem__.return_value.call_args_list] == [
            dict(M=768, N=50304, ADD=False, num_warps=8, num_stages=2),
            dict(M=50304, N=768, ADD=True, num_warps=8, num_stages=2),
        ]
    print(f"Transpose ({'Hopper' if hopper else 'fallback'}): FP8 copies and BF16 copies and adds match bitwise")
    return transpose


ROPE_LENGTHS = (16384, 32768, 40960, 49152, 262144)
ROPE_BACKWARD_SHAPES = (("wide", 128, 128, False), ("paired", 64, 128, True), ("half-v", 64, 64, False))


def rope_launch_args(tokens, qk_dim=128, v_dim=128, paired=False):
    layout = rope_backward_args(tokens, qk_dim, v_dim, paired)
    return dict(rows=tokens * 12, num_heads=6, heads2=12, qk_dim=qk_dim, rotary_dim=64,
                stride_qkt=layout["stride_qkt"], stride_qkh=qk_dim, factor_stride_t=layout["factor_stride_t"],
                stride_out_t=layout["stride_grad_q_t"], PAIRED=paired, KEY_OFFSET=qk_dim == 128,
                BLOCK_M=4 if qk_dim == 128 else 8, BLOCK_D=qk_dim)


def rope_backward_args(tokens, qk_dim=128, v_dim=128, paired=False):
    qkv_dim = 12 * qk_dim + 6 * v_dim
    grad_qk_pitch = (3 if paired else 6) * qk_dim
    return dict(
        tokens=tokens, num_heads=6, qk_dim=qk_dim, rotary_dim=64, v_dim=v_dim, qkv_dim=qkv_dim,
        stride_qkt=12 * qk_dim if paired else qkv_dim,
        stride_qkh=qk_dim, factor_stride_t=(2 if paired else 1) * qk_dim,
        stride_grad_q_t=grad_qk_pitch, stride_grad_q_h=qk_dim,
        stride_grad_k_t=grad_qk_pitch, stride_grad_k_h=qk_dim,
        stride_grad_v_t=6 * v_dim, stride_grad_v_h=v_dim, PAIRED=paired, KEY_OFFSET=qk_dim == 128,
        BLOCK_T=32, BLOCK_QK=qk_dim, BLOCK_V=v_dim,
    )


def check_rope_backward(candidate, torch, triton, tl, device, hopper, kernels):
    old, new = kernels
    launcher, _ = load_functions(
        candidate / ROPE, ("_launch_qkv_rope_backward", "_launch_qkv_rope_backward_fake"),
        triton, tl, torch=torch, _qkv_norm_rope_pack_fp8_backward_kernel=new,
    )

    def launch(grad_q, grad_k, grad_v, qk, factor1, factor2, scale, paired=False):
        tokens, _, qk_dim = qk.shape
        qkv_dim = 12 * qk_dim + 6 * grad_v.shape[2]
        row = torch.empty((tokens, qkv_dim), dtype=torch.float8_e4m3fn, device=qk.device)
        transposed = torch.empty((qkv_dim, tokens), dtype=torch.float8_e4m3fn, device=qk.device)
        launcher(grad_q, grad_k, grad_v, qk, factor1, factor2, row, transposed, scale, 6, 64, paired, qk_dim == 128)
        return row, transposed

    execute = launch if hopper else torch.compile(launch, backend="aot_eager", fullgraph=True)
    rng = torch.Generator(device=device).manual_seed(32018)
    scale = torch.tensor([0.125], dtype=torch.float32, device=device)
    for tokens in (16384,) if hopper else (1, 2, 4, 32, 37, 65):
        qk = torch.randn((tokens, 2304), dtype=torch.bfloat16, device=device, generator=rng)
        qk = qk[:, :1536].view(tokens, 12, 128)
        grad_q, grad_k, grad_v = (
            torch.randn((tokens, 6, 128), dtype=torch.bfloat16, device=device, generator=rng) for _ in range(3)
        )
        factors = torch.randn((2, tokens, 128), dtype=torch.bfloat16, device=device, generator=rng)
        grad_q[0, 2, 32] = float("inf")
        grad_k[0, 3, 91] = -0.0
        grad_k[-1, 5, 60] = float("nan")
        outputs = []
        for fn, flags in ((old, {}), (new, dict(TRIM_KEY_GRAD=False)), (new, dict(TRIM_KEY_GRAD=True))):
            row = torch.empty((tokens, 2304), dtype=torch.float8_e4m3fn, device=device)
            transposed = torch.empty((2304, tokens), dtype=torch.float8_e4m3fn, device=device)
            fn[(6, triton.cdiv(tokens, 32))](grad_q, grad_k, grad_v, qk, *factors, row, transposed, scale,
                                            **rope_backward_args(tokens), **flags, num_warps=8)
            outputs.append((row.view(torch.uint8), transposed.view(torch.uint8)))
        assert all(torch.equal(a, b) for pair in outputs[1:] for a, b in zip(outputs[0], pair)), tokens
        if hopper or tokens == 37:
            actual = execute(grad_q, grad_k, grad_v, qk, *factors, scale)
            assert all(torch.equal(a, b.view(torch.uint8)) for a, b in zip(outputs[0], actual)), tokens
    for shape, qk_dim, v_dim, paired in ROPE_BACKWARD_SHAPES[1:]:
        tokens = 16384 if hopper else 37
        layout = rope_backward_args(tokens, qk_dim, v_dim, paired)
        qkv_dim = layout["qkv_dim"]
        qk = torch.randn((tokens, layout["stride_qkt"]), dtype=torch.bfloat16, device=device, generator=rng)
        qk = qk[:, :12 * qk_dim].view(tokens, 12, qk_dim)
        grad_shape = (2 * tokens, 3, qk_dim) if paired else (tokens, 6, qk_dim)
        grad_q, grad_k = [torch.randn(grad_shape, dtype=torch.bfloat16, device=device, generator=rng) for _ in range(2)]
        grad_v = torch.randn((tokens, 6, v_dim), dtype=torch.bfloat16, device=device, generator=rng)
        factors = torch.randn((2, tokens, layout["factor_stride_t"]), dtype=torch.bfloat16, device=device, generator=rng)
        factors[1, 0, -1] = -0.0
        factors[1, 0, -2] = float("nan")
        grad_k[-1, -1, -1] = float("inf")
        row = torch.empty((tokens, qkv_dim), dtype=torch.float8_e4m3fn, device=device)
        col = torch.empty((qkv_dim, tokens), dtype=torch.float8_e4m3fn, device=device)
        old[(6, triton.cdiv(tokens, 32))](grad_q, grad_k, grad_v, qk, *factors, row, col, scale,
                                         **layout, num_warps=4)
        actual = execute(grad_q, grad_k, grad_v, qk, *factors, scale, paired)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip((row, col), actual)), shape
    print(f"RoPE backward ({'Hopper' if hopper else 'interpreter'}): both FP8 layouts match bitwise")
    return launch


def check_yarn_tables(baseline, candidate, triton, tl, torch, device, kernels, rope_forward, rope_backward):
    from itertools import permutations
    import math

    words = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    finite_words = words[(words & 0x7f80) != 0x7f80]
    signed_words = (finite_words.view(torch.bfloat16) * -1).view(torch.int16)
    assert torch.equal(signed_words, finite_words ^ -32768)

    implementations = []
    for root in (baseline, candidate):
        path = root / ATTENTION
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Yarn"]
        env = dict(torch=torch, nn=torch.nn, math=math)
        exec(compile(tree, str(path), "exec"), env)
        implementations.append(env["Yarn"])
    old_fwd, old_bwd = kernels
    original, = load_functions(baseline / ROPE, ("qk_norm_rope_forward",), triton, tl,
                               torch=torch, _qk_norm_rope_forward_kernel=old_fwd)
    if device == "cpu":
        rope_forward = torch.compile(rope_forward, backend="aot_eager", fullgraph=True)
        rope_backward = torch.compile(rope_backward, backend="aot_eager", fullgraph=True)
    rng = torch.Generator(device=device).manual_seed(6427)
    wide_windows = wide_window_sizes((baseline / CONFIG).read_text())
    for shape, qk_dim, v_dim, paired in ROPE_BACKWARD_SHAPES:
        dense, compact = (cls(qk_dim, 67, paired=paired, attn_scale=0.13, device=device)
                          for cls in implementations)
        pointers = tuple(t.data_ptr() for t in (compact.factor1, compact.factor2))
        stationary = (implementations[1](qk_dim, 67, paired=paired, attn_scale=0.13, device=device,
                                         elide_stationary=True)
                      if shape == "wide" else None)
        stationary_ptrs = (stationary.factor1.data_ptr(), stationary.factor2.data_ptr()) if stationary else None

        def check_tables():
            assert dense.valid_rows == compact.valid_rows and dense.attn_scale == compact.attn_scale
            assert tuple(t.data_ptr() for t in (compact.factor1, compact.factor2)) == pointers
            assert dense.factor1.numel() == 2 * compact.factor1.numel()
            expanded_freq = compact.angular_freq.repeat_interleave(2)
            assert torch.equal(dense.angular_freq.view(torch.int32), expanded_freq.view(torch.int32))
            cosine, sine = compact.factor1.view(torch.int16), compact.factor2.view(torch.int16)
            assert torch.equal(dense.factor1.view(torch.int16), cosine.repeat_interleave(2, dim=1))
            expanded_sine = torch.stack((sine, sine ^ -32768), dim=-1).flatten(1)
            assert torch.equal(dense.factor2.view(torch.int16), expanded_sine)
            if stationary:
                assert (stationary.factor1.data_ptr(), stationary.factor2.data_ptr()) == stationary_ptrs
                assert stationary.valid_rows == compact.valid_rows and stationary.attn_scale == compact.attn_scale
                assert stationary.factor1.shape == stationary.factor2.shape == (67, 32)
                assert torch.equal(stationary.angular_freq.view(torch.int32), compact.angular_freq[:32].view(torch.int32))
                assert torch.count_nonzero(compact.angular_freq[32:].view(torch.int32)) == 0
                assert torch.equal(stationary.factor1.view(torch.int16), cosine[:, :32])
                assert torch.equal(stationary.factor2.view(torch.int16), sine[:, :32])
                assert (cosine[:, 32:] == 0x3f80).all()
                assert torch.count_nonzero(sine[:, 32:]) == 0

        check_tables()
        dense.rebuild_rows = compact.rebuild_rows = 37
        if stationary:
            stationary.rebuild_rows = 37
        changes = permutations(wide_windows, 2) if stationary else ((128, 256), (256, 512), (512, 1024))
        for old, new in changes:
            dense.apply(old, new)
            compact.apply(old, new)
            if stationary:
                stationary.apply(old, new)
            check_tables()
            dense.ensure_full()
            compact.ensure_full()
            if stationary:
                stationary.ensure_full()
            check_tables()
        dense.reset()
        compact.reset()
        if stationary:
            stationary.reset()
        check_tables()

        tokens = 37
        layout = rope_backward_args(tokens, qk_dim, v_dim, paired)
        qk = torch.randn((tokens, layout["stride_qkt"]), dtype=torch.bfloat16, device=device, generator=rng)
        qk = qk[:, :12 * qk_dim].view(tokens, 12, qk_dim)
        dense_factors = dense.factor1[:tokens], dense.factor2[:tokens]
        factors = stationary if stationary else compact
        compact_factors = factors.factor1[:tokens], factors.factor2[:tokens]
        expected = original(qk, *dense_factors, 6, 64, paired, qk_dim == 128)
        actual = rope_forward(qk, *compact_factors, 6, 64, paired, qk_dim == 128)
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, actual)), shape

        grad_q, grad_k = (torch.randn_like(t) for t in expected)
        grad_v = torch.randn((tokens, 6, v_dim), dtype=torch.bfloat16, device=device, generator=rng)
        row = torch.empty((tokens, layout["qkv_dim"]), dtype=torch.float8_e4m3fn, device=device)
        col = torch.empty(row.shape[::-1], dtype=row.dtype, device=device)
        args = (grad_q, grad_k, grad_v, qk)
        scale = torch.tensor([0.125], dtype=torch.float32, device=device)
        old_bwd[(6, triton.cdiv(tokens, 32))](*args, *dense_factors, row, col, scale, **layout,
                                             num_warps=8 if qk_dim == 128 else 4)
        actual = rope_backward(*args, *compact_factors, scale, paired)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip((row, col), actual)), shape
    dense_fallback, actual_fallback = (cls(96, 11, paired=True, attn_scale=0.13, device=device)
                                       for cls in implementations)
    dense_fallback.apply(128, 256)
    actual_fallback.apply(128, 256)
    for name in ("factor1", "factor2"):
        expected = getattr(dense_fallback, name).view(torch.int16)
        assert torch.equal(expected, getattr(actual_fallback, name).view(torch.int16))
    print("Compact YaRN: reset, partial rebuild, completion, BF16 forward and both FP8 gradients match bitwise")


def check_rope_shift(baseline, candidate, triton, tl, torch, device, hopper, kernels):
    from unittest.mock import MagicMock, patch

    old, new = kernels
    original, = load_functions(baseline / ROPE, ("qk_norm_rope_forward",), triton, tl,
                               torch=torch, _qk_norm_rope_forward_kernel=old)
    optimized, *_ = load_functions(
        candidate / ROPE,
        ("qk_norm_rope_forward", "_launch_qk_norm_rope", "_launch_qk_norm_rope_fake"),
        triton, tl, torch=torch, _qk_norm_rope_forward_kernel=new,
    )

    rng = torch.Generator(device=device).manual_seed(32017)
    for tokens in (16384,) if hopper else (1, 2, 4, 16, 37):
        qk = torch.randn((tokens, 2304), dtype=torch.bfloat16, device=device, generator=rng)
        qk = qk[:, :1536].view(tokens, 12, 128)
        factors = torch.randn((2, tokens, 128), dtype=torch.bfloat16, device=device, generator=rng)
        qk[0, 6, 65] = -0.0
        qk[-1, 11, 127] = float("inf")
        expected = original(qk, *factors, 6, 64, False, True)
        observed = optimized(qk, *factors, 6, 64, False, True)
        if not hopper:
            assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, observed))
            if tokens == 37:
                compiled = torch.compile(optimized, backend="aot_eager", fullgraph=True)
                actual = compiled(qk, *factors, 6, 64, False, True)
                assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, actual))
            # Exercise the same constexpr path at short lengths in the interpreter.
            new[(tokens * 3,)](qk, *factors, *observed, **rope_launch_args(tokens), SHIFT_STORE=True, num_warps=2)
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(expected, observed)), tokens
    if not hopper:
        launch = MagicMock()
        launcher, = load_functions(candidate / ROPE, ("_launch_qk_norm_rope",), triton, tl,
                                   strip_decorators=True, torch=torch, _qk_norm_rope_forward_kernel=launch)

        def metadata(shape, strides, pointer=2**40):
            return SimpleNamespace(shape=shape, device="cuda:0", dtype=torch.bfloat16, is_cuda=True,
                                   stride=lambda dim=None: strides if dim is None else strides[dim],
                                   data_ptr=lambda: pointer)

        outputs = metadata((16384, 6, 128), (768, 128, 1))
        qk = metadata((16384, 12, 128), (2304, 128, 1))
        factor = metadata((16384, 128), (128, 1))
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            launcher(qk, factor, factor, outputs, outputs, 6, 64, False, True, 4, 2)
            launcher(qk, factor, factor, outputs, outputs, 6, 64, False, False, 8, 2)
            launcher(qk, factor, factor, outputs, outputs, 6, 96, False, True, 4, 2)
            unaligned = metadata(factor.shape, (128, 1), 2**40 + 2)
            launcher(qk, factor, unaligned, outputs, outputs, 6, 64, False, True, 4, 2)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            launcher(qk, factor, factor, outputs, outputs, 6, 64, False, True, 4, 2)
        assert [call.kwargs["SHIFT_STORE"] for call in launch.__getitem__.return_value.call_args_list] == [
            True, False, False, False, False,
        ]
    print(f"Shifted RoPE ({'Hopper' if hopper else 'interpreter'}): both Q and K match bitwise")
    return optimized


def check_fullgraph_cuda_metadata(torch, rope_forward, rope_backward, transpose):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    mode = FakeTensorMode()
    device = torch.device("cuda:0")

    def fake(shape, dtype, stride=None):
        metadata = (torch.empty(shape, device="meta", dtype=dtype) if stride is None
                    else torch.empty_strided(shape, stride, device="meta", dtype=dtype))
        return FakeTensor(mode, metadata, device)

    tokens = 16384
    qk = fake((tokens, 12, 128), torch.bfloat16, (2304, 128, 1))
    f1, f2 = (fake((tokens, 32), torch.bfloat16) for _ in range(2))
    grads = [fake((tokens, 6, 128), torch.bfloat16) for _ in range(3)]
    scale = fake((1,), torch.float32)
    src = fake((40960, 50304), torch.float8_e5m2)

    def trace(function, name, *args):
        def backend(graph, _):
            custom_op = getattr(torch.ops.nanogpt, name).default
            assert any(node.target == custom_op for node in graph.graph.nodes), name
            return graph.forward

        return torch.compile(function, backend=backend, fullgraph=True)(*args)

    with mode:
        q, k = trace(rope_forward, "track1_qk_norm_rope", qk, f1, f2, 6, 64, False, True)
        row, col = trace(rope_backward, "track1_qkv_rope_backward", *grads, qk, f1, f2, scale)
        dst = trace(transpose, "track1_transpose_copy", src)
        assert q.shape == k.shape == (tokens, 6, 128) and q.dtype == k.dtype == torch.bfloat16
        assert row.shape == (tokens, 2304) and col.shape == (2304, tokens)
        assert row.dtype == col.dtype == torch.float8_e4m3fn
        assert dst.shape == (50304, 40960) and dst.dtype == src.dtype
        assert all(t.device == device for t in (q, k, row, col, dst))
    print("Full-graph compile: RoPE forward, RoPE backward and transpose trace at Track 1 CUDA shapes")


def executable_ptx(ptx, *, labels=None):
    index = 0
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        if line.startswith(".section"):
            return
        if labels is not None and line.startswith("$") and line.endswith(":"):
            labels[line[:-1]] = index
        if line.startswith("{ ") and line.endswith(" }"):
            for op in line[2:-2].split(";"):
                if op.strip():
                    yield op.strip() + ";"
                    index += 1
        if line and not line.startswith(".") and line.endswith(";"):
            yield line
            index += 1


def load_tail_class(root, name, **runtime):
    from sys import modules
    from types import ModuleType

    path = root / TAIL
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom) or not node.module.startswith("track_1_short.")]
    module = ModuleType(name)
    modules[name] = module
    env = vars(module)
    exec(compile(tree, str(path), "exec"), env)
    env.update(NGRAM_ADAM_PERIOD4_START=0, **runtime)
    return env["TailAverages"]


def check_tail_schedule(baseline, candidate, torch, paired_allowed):
    labels = ("lm_head", "embed", "qk_bank", "vo_bank", "mlp_bank", "value_embeds")
    initial = {label: torch.arange(8, dtype=torch.bfloat16).reshape(1, 8) + i
               for i, label in enumerate(labels)}
    calls = []

    def single(dst, src, rate):
        dst.lerp_(src.float(), rate)

    def paired(first, second, src, rate_first, rate_second):
        if not paired_allowed:
            return False
        single(first, src, rate_first)
        single(second, src, rate_second)
        if src.numel() == 8:  # Ignore the warmup scratch launch.
            calls.append(step)
        return True

    def load(root, name, params):
        constructor = load_tail_class(root, name, lerp_upcast_=single, lerp_shared_source_=paired,
                                      warm_tail_norms=lambda _: None)
        optimizer = SimpleNamespace(_param_by_label=params, param_cfgs={
            p: SimpleNamespace(optim="anvil" if label.endswith("_bank") else "adam",
                               reshape=p.shape, chunk_size=1) for label, p in params.items()
        })
        return constructor(optimizer, 0, lambda s: s % 4 == 1, total)

    for total in (1192, 1193, 1194, 1195):
        weights = {label: value.clone() for label, value in initial.items()}
        old = load(baseline, "baseline_tail_test", weights)
        new = load(candidate, "candidate_tail_test", weights)
        calls.clear()
        for step in range(total - 298, total):
            for label, tensor in weights.items():
                tensor.copy_(initial[label] + (step % 127))
            old.tick(step)
            new.tick(step)
            for a, b in zip(old.accumulators, new.accumulators, strict=True):
                assert a.seeded == b.seeded
                for label in a.seeded:
                    assert torch.equal(a.bufs[label].view(torch.int32), b.bufs[label].view(torch.int32)), (step, label)
        if total == 1194:
            assert len(calls) == (124 if paired_allowed else 0)


def check_tail_lerp(baseline, candidate, triton, tl, torch, mode, hopper):
    from itertools import pairwise
    from unittest.mock import MagicMock, patch

    check_tail_schedule(baseline, candidate, torch, True)
    check_tail_schedule(baseline, candidate, torch, False)
    launch = MagicMock()
    dispatcher, = load_functions(candidate / TAIL_KERNEL, ("lerp_shared_source_",), triton, tl,
                                 torch=torch, pairwise=pairwise, BLOCK=4096, WARPS=8,
                                 _tail_lerp_pair=launch)

    def metadata(dtype, pointer, elements=4096, contiguous=True):
        return SimpleNamespace(dtype=dtype, data_ptr=lambda: pointer, numel=lambda: elements,
                               element_size=lambda: 2 if dtype == torch.bfloat16 else 4,
                               is_contiguous=lambda: contiguous, is_cuda=True, device="cuda:0")

    src = metadata(torch.bfloat16, 1 << 20)
    first = metadata(torch.float32, 2 << 20)
    second = metadata(torch.float32, 3 << 20)
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        assert dispatcher(first, second, src, 0.04, 0.02)
        assert not dispatcher(first, first, src, 0.04, 0.02)
        assert not dispatcher(first, metadata(torch.float32, (2 << 20) + 16), src, 0.04, 0.02)
        assert not dispatcher(first, second, metadata(torch.bfloat16, 1 << 20, 4095), 0.04, 0.02)
        assert not dispatcher(first, second, metadata(torch.bfloat16, (1 << 20) + 2), 0.04, 0.02)
        assert not dispatcher(first, second, metadata(torch.bfloat16, 1 << 20, contiguous=False), 0.04, 0.02)
    with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
        assert not dispatcher(first, second, src, 0.04, 0.02)
    launch.__getitem__.assert_called_once_with((1,))
    if hopper and mode == "cuda":
        module = load_candidate_scale(candidate / TAIL_KERNEL)
        single, _ = load_functions(baseline / LERP, ("lerp_upcast_", "_lerp_upcast_kernel"), triton, tl, torch=torch)
        torch.manual_seed(223)
        for elements in (2 * 768**2, 3 * 2816 * 768):
            src = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
            first, second = [torch.randn(elements, device="cuda") for _ in range(2)]
            edges = torch.tensor([0.0, -0.0, float("inf"), -float("inf")], device="cuda")
            src[:edges.numel()].copy_(edges)
            first[:edges.numel()].copy_(edges.flip(0))
            second[:edges.numel()].copy_(edges)
            exact_first, exact_second = first.clone(), second.clone()
            single(exact_first, src, 4.0 / 53.0)
            single(exact_second, src, 2.0 / 75.0)
            assert module.lerp_shared_source_(first, second, src, 4.0 / 53.0, 2.0 / 75.0)
            assert torch.equal(first.view(torch.int32), exact_first.view(torch.int32))
            assert torch.equal(second.view(torch.int32), exact_second.view(torch.int32))
    print("Tail averaging: seed and update schedules agree, including dispatch and fallbacks.")


def check_tail_norm(baseline, candidate, torch, mode, hopper):
    from itertools import accumulate
    from unittest.mock import patch

    module = load_candidate_scale(candidate / TAIL_NORM_KERNEL)
    if torch.version.git_version == module.ATEN_REVISION:
        headers = Path(torch.__file__).parent / "include/ATen/native"
        checks = {
            "cuda/Reduce.cuh": (
                "int vt0=4, int input_vec_size=vt0",
                "const auto values_vec = memory::load_vector<input_vec_size>(data, idx)",
                "idx += stride", "value_list[0] = ops.combine(value_list[0], value_list[i])",
            ),
            "SharedReduceOps.h": ("return acc + data_ * data_", "return device_sqrt(a)"),
            "ReduceOpsUtils.h": ("gpu_lowp_to_f32 ? self.scalar_type()",),
            "cuda/Loops.cuh": ("iter.scalar_value<opmath_arg_t>(2)", "return f(a, b)"),
        }
        for header, clauses in checks.items():
            text = (headers / header).read_text()
            assert all(clause in text for clause in clauses), header

    def metadata(size, pointer):
        return SimpleNamespace(dtype=torch.bfloat16, data_ptr=lambda: pointer, numel=lambda: size,
                               is_contiguous=lambda: True, is_cuda=True, device="cuda:0")

    meta = tuple(metadata(size, (i + 1) << 40) for i, size in enumerate(module.SIZES))
    with patch.object(torch.version, "cuda", "12.8"), patch.object(torch.version, "git_version", module.ATEN_REVISION):
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            assert module.can_restore_tail_norms(meta)
            assert not module.can_restore_tail_norms((meta[0], meta[0], *meta[2:]))
            assert not module.can_restore_tail_norms((metadata(4096, 16), *meta[1:]))
            assert not module.can_restore_tail_norms((metadata(module.SIZES[0], 18), *meta[1:]))
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            assert not module.can_restore_tail_norms(meta)

    if mode == "cpu":
        labels = ("lm_head", "embed", "vo_bank", "mlp_bank", "value_embeds", "qk_bank")
        initial = {label: (torch.arange(128, dtype=torch.float32).reshape(2, 64) / (97 + i)).bfloat16()
                   for i, label in enumerate(labels)}

        def run(root, direct):
            params = {label: tensor.clone() for label, tensor in initial.items()}
            order = []

            def gather(output, shard):
                label = next(name for name, p in params.items() if p.data_ptr() == output.data_ptr())
                order.append(label)
                # The synthetic peer sends independent data. Rank zero's chunk is already in place.
                assert shard.data_ptr() == output.data_ptr()
                if label == "vo_bank":
                    output[1].zero_()
                else:
                    output[1].copy_(initial[label][1] + 1)

            def restore(weights, ratios):
                ends = tuple(accumulate(t.numel() // 64 for t in weights))
                module._restore_tail_norms[(ends[-1],)](*weights, ratios, ends, 64, num_warps=4)

            muted = lambda *_: None
            constructor = load_tail_class(
                root, root.name + "_ship", lerp_upcast_=muted, lerp_shared_source_=muted,
                warm_tail_norms=muted, can_restore_tail_norms=lambda _: direct,
                tail_norms=module.tail_norms, restore_tail_norms_=restore,
                dist=SimpleNamespace(all_gather_into_tensor=gather),
            )
            optimizer = SimpleNamespace(_param_by_label=params, param_cfgs={
                p: SimpleNamespace(optim="anvil" if label.endswith("_bank") else "adam",
                                   reshape=p.shape, chunk_size=1) for label, p in params.items()
            })
            average = constructor(optimizer, 0, lambda _: True, 1194)
            for acc_i, acc in enumerate(average.accumulators):
                for label in acc.labels:
                    if label == "vo_bank":
                        acc.bufs[label].zero_()
                    else:
                        acc.bufs[label].copy_(initial[label][:1].float() + (acc_i + 1) / 17.0)
                    acc.seeded.add(label)
            average.ship()
            assert order == list(labels)
            return params

        expected = run(baseline, False)
        for direct in (False, True):
            candidate_weights = run(candidate, direct)
            for label in labels:
                assert torch.equal(expected[label].view(torch.int16), candidate_weights[label].view(torch.int16)), label
    elif hopper and torch.version.git_version == module.ATEN_REVISION:
        torch.manual_seed(412)
        weights = tuple(torch.randn(n, device="cuda", dtype=torch.bfloat16) for n in module.SIZES)
        assert module.can_restore_tail_norms(weights)
        expected = module.tail_norms(weights, False).view(torch.int32)
        assert torch.equal(expected, module.tail_norms(weights, True).view(torch.int32))
        ratios = torch.tensor([0.97021, 1.02739, 0.99761, 1.0139, 1.00762], device="cuda")
        originals = tuple(w.clone() for w in weights)
        for w, ratio in zip(originals, ratios.tolist()):
            w.mul_(ratio)
        module.restore_tail_norms_(weights, ratios)
        assert all(torch.equal(w.view(torch.int16), old.view(torch.int16)) for w, old in zip(weights, originals))
        # A half-ULP FP32 square exposes unequal FMA contraction near underflow.
        tiny = 2.0**-75
        for weight in weights:
            weight.fill_(tiny)
            weight[:2048].fill_(1.5 * tiny)
        expected = module.tail_norms(weights, False).view(torch.int32)
        assert torch.equal(expected, module.tail_norms(weights, True).view(torch.int32))
    print("Tail ship: verified guarded layouts and the available norm restoration checks.")


def load_sparse_method(root, path, owner, name, *, strip_decorators=False, **runtime):
    tree = ast.parse((root / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if strip_decorators:
        fn.decorator_list.clear()
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), runtime)
    return runtime[name]


def replay_sparse_host(root, torch, packer, *, online=False):
    width = 768
    source = (torch.arange(13 * width).reshape(13, width) % 1024).half()
    want = torch.tensor([1, 3, 4, 6, 8, 11], dtype=torch.int32)
    pull = SimpleNamespace(want=want, lo=1, hi=3, num_rows=want.numel(), peer_want=None)
    events = []

    def send(label):
        def exchange(_pull, wire):
            events.append(label)
            return wire.clone(), label
        return SimpleNamespace(send_grads=exchange)

    ve = SimpleNamespace(grad_accum=source.clone(), param=SimpleNamespace(grad=None),
                         live=pull, owners=send("value"))
    ng = SimpleNamespace(grad_per_slot=torch.empty_like(source), live=pull, owners=send("ngram"), device=source.device)
    event_grad = torch.arange(want.numel()), source[:want.numel()]
    if online:
        ng.ngram_event_grad, ng.ngram_sum_started = event_grad, False
        ng.grad_per_slot.zero_()
    else:
        ng.pending = [event_grad]
    value_send = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "send_grads",
                                    torch=torch, pack_value_gradients=packer.pack_value_gradients)
    value_clear = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "zero_sent_grads",
                                     torch=torch, clear_value_want_=packer.clear_value_want_)
    ngram_send = load_sparse_method(
        root, NGRAM_TABLE, "NgramTable", "send_grads", torch=torch, pack_ngram_gradients=packer.pack_ngram_gradients,
        NGRAM_DIM=width, GRAD_WIRE_DTYPE=torch.bfloat16,
        scatter_add_rows=lambda src, slots, dst: dst.index_add_(0, slots, src))

    def clear():
        events.append("clear")
        value_clear(ve)

    objects = SimpleNamespace(value_embeds=SimpleNamespace(send_grads=lambda: value_send(ve), zero_sent_grads=clear),
                              ngram=SimpleNamespace(send_grads=lambda: ngram_send(ng)))
    launch = load_sparse_method(root, Path("track_1_short/perf/row_prefetch.py"), "CycleEvent", "launch")
    launch(SimpleNamespace(prefetch=objects))
    assert events == ["value", "ngram", "clear"]
    assert (ng.ngram_event_grad is None and not ng.ngram_sum_started) if online else not ng.pending
    return [ve.grad_accum, *[x for x in ve.grads_in_flight if isinstance(x, torch.Tensor)],
            *[x for x in ng.grads_in_flight if isinstance(x, torch.Tensor)]]


def check_sparse_packets(baseline, candidate, torch, tl, mode, hopper):
    from contextlib import ExitStack
    from unittest.mock import patch

    module = load_candidate_scale(candidate / SPARSE_ROW_KERNEL)
    assert module.pack_value_gradients(torch.empty(1, 767), torch.empty(1, dtype=torch.int32), 0, 1) is None
    if mode != "cpu" and not hopper:
        return
    device = "cpu" if mode == "cpu" else "cuda"
    bits = torch.arange(86 * 768, dtype=torch.int32, device=device).to(torch.int16)
    source = bits.view(torch.float16).reshape(86, 768)
    ids = torch.arange(85, -1, -1, dtype=torch.int32, device=device)
    with ExitStack() as stack:
        if mode == "cpu":
            from triton.runtime import interpreter

            def cast_rne(_builder, asm, constraints, values, _types, pure, pack):
                assert asm == "cvt.rn.bf16.f32 $0, $1;" and constraints == "=h,f" and pure and pack == 1
                words = torch.from_numpy(values[0].data.copy()).bfloat16().view(torch.uint16)
                result = interpreter.TensorHandle(words.numpy(), tl.bfloat16)
                return SimpleNamespace(get_result=lambda _: result)

            stack.enter_context(patch.object(module, "_supported", return_value=True))
            stack.enter_context(patch.object(interpreter.InterpreterBuilder, "create_inline_asm", cast_rne))
        for live, lo, hi in ((0, 0, 0), (86, 0, 0), (86, 0, 86), (37, 6, 17), (86, 84, 86)):
            want = ids[:live]
            own, wire = module.pack_value_gradients(source, want, lo, hi)
            gathered = source.index_select(0, want)
            assert torch.equal(own.view(torch.int16), gathered[lo:hi].view(torch.int16))
            assert torch.equal(wire.view(torch.int16), torch.cat((gathered[:lo], gathered[hi:])).view(torch.int16))
            omitted, peer_only = module.pack_value_gradients(source, want, lo, hi, pack_own=False)
            assert omitted is None and torch.equal(peer_only.view(torch.int16), wire.view(torch.int16))
            if torch.version.git_version == module.ATEN_REVISION:
                converted = module.pack_ngram_gradients(source[:live], lo, hi)
                expected = torch.empty((live - (hi - lo), 768), dtype=torch.bfloat16, device=device)
                expected[:lo].copy_(source[:lo])
                expected[lo:].copy_(source[hi:live])
                if mode == "cpu":
                    # CPU low-precision copy may choose a different NaN encoding.
                    nan = expected.isnan()
                    assert torch.equal(converted.isnan(), nan)
                    assert torch.equal(converted.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])
                else:
                    assert torch.equal(converted.view(torch.int16), expected.view(torch.int16))
                scratch = source.clone()
                released = module.pack_ngram_gradients(scratch[:live], lo, hi, clear_source=True)
                assert torch.equal(released.view(torch.int16), converted.view(torch.int16))
                assert torch.count_nonzero(scratch[:lo].view(torch.int16)).item() == 0
                assert torch.count_nonzero(scratch[hi:live].view(torch.int16)).item() == 0
                assert torch.equal(scratch[lo:hi].view(torch.int16), source[lo:hi].view(torch.int16))
                assert torch.equal(scratch[live:].view(torch.int16), source[live:].view(torch.int16))
        for listed in ([], [85], [4, 0, 79], [5, 1, 83, 22], [42, 81, 0, 76, 16, 59, 11]):
            want = torch.tensor(sorted(listed), dtype=torch.int32, device=device)
            cleared = source.clone()
            expected = source.clone().index_fill_(0, want.long(), 0)
            assert module.clear_value_want_(cleared, want)
            assert torch.equal(cleared.view(torch.int16), expected.view(torch.int16))
        if mode == "cpu":
            assert module.pack_value_gradients(source, ids[1:], 0, 1) is None
            assert not module.clear_value_want_(source, ids[1:])
            assert not module.clear_value_want_(source, source.view(torch.int32).flatten()[:4])
            expected = replay_sparse_host(baseline, torch, module)
            actual = replay_sparse_host(candidate, torch, module, online=True)
            assert all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                       for a, b in zip(actual, expected, strict=True))
    print("Sparse packets: checked local and wire order and bitwise clearing of sent value rows.")


def check_ngram_accumulation_host(baseline, candidate, torch):
    width = 768
    slots = torch.tensor([2, 1, 2, 3], dtype=torch.int32)
    old_graphs = load_sparse_method(baseline, Path("track_1_short/perf/cuda_graphs/step_graphs.py"),
                                   "StepGraphs", "hold_ngram_grad", Tensor=torch.Tensor)

    def run(root, online):
        submitted = []
        outputs = []
        graph = torch.empty((4, width), dtype=torch.bfloat16)
        ring = SimpleNamespace(ngram_grad_ring=[torch.empty_like(graph).flatten() for _ in range(4)])
        scratch = torch.full((9, width), 0x7e05, dtype=torch.int16).view(torch.float16)
        merge = None

        def scatter(src, idx, out, *, row_map=None, clear_source=False):
            assert src.is_contiguous()
            if row_map is None:
                assert idx is slots and out.shape == (pull.num_rows, width)
                submitted.append("scatter")
                out.index_add_(0, idx, src.half())
            else:
                out.index_add_(0, row_map[idx.long()].long(), src.to(out.dtype))
            if clear_source:
                assert row_map is not None and src.dtype == torch.float16
                src.zero_()

        def exchange(p, wire):
            assert p is pull
            submitted.append("wire")
            return wire.clone(), "work"

        def pack(source, lo, hi, *, clear_source):
            assert clear_source
            if fallback:
                return None
            wire = torch.cat((source[:lo], source[hi:])).bfloat16()
            source[:lo].zero_()
            source[hi:].zero_()
            return wire

        def claim(entries, row_map):
            for i, row in enumerate(entries.tolist()):
                row_map[row] = i

        def adam(_p, _v, _last, _history, grad, rows, _map, _event, **_):
            nonlocal merge
            merge = grad.view(torch.int16).clone()

        methods = dict(torch=torch, Tensor=torch.Tensor, scatter_add_rows=scatter,
                       pack_ngram_gradients=pack, NGRAM_DIM=width, GRAD_WIRE_DTYPE=torch.bfloat16)
        add = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "accumulate_grad", **methods)
        send = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "send_grads", **methods)
        start = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "start_cycle", RowPull=object)
        update = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "adam_update", **methods,
                                    pack_adam_entries=lambda *_, **__: None, claim_rows=claim, adam_rows_=adam,
                                    adam_beta2_and_wd_mul=lambda _: (0.95, 1), NGRAM_LR_MUL=1,
                                    GRAD_MERGE_DTYPE=torch.bfloat16)
        table = SimpleNamespace(device=graph.device, grad_per_slot=scratch, grads_in_flight=None,
                                owners=SimpleNamespace(send_grads=exchange), first_row=0, adam_events=0,
                                world_size=8, entry_of_row=torch.empty(8, dtype=torch.int32),
                                beta2_history=torch.empty(6), shard=None, exp_avg_sq=None, last_event=None)
        if online:
            table.ngram_event_grad = None
            table.ngram_sum_started = False
            scratch[:6].zero_()
        else:
            table.pending = []
        for values, rows, lo, hi, fallback in [((-2,), 4, 0, 0, False), ((7, -2), 6, 2, 5, True),
                                                ((2048, 1, -2048, -0.0), 5, 1, 4, False), ((16, 8, 4), 6, 0, 6, False)]:
            pull = SimpleNamespace(num_rows=rows, lo=lo, hi=hi, want=torch.arange(rows, dtype=torch.int32),
                                   asked=torch.full((rows - (hi - lo),), lo, dtype=torch.int32))
            table.cached = pull
            start(table, pull)
            for j, value in enumerate(values):
                event = j + 1 == len(values)
                graph.copy_(torch.tensor([value, -value, 0.0, 0.125])[:, None].expand_as(graph))
                # The next backward rewrites the same storage. Exercise an
                # earlier noncontiguous graph result as well as the alias.
                result = graph.T.contiguous().T if j == 0 and not event and len(values) == 3 else graph
                if online:
                    add(table, slots, result, event)
                else:
                    add(table, slots, old_graphs(ring, result, len(table.pending), event))
            submitted.append("value")
            send(table)
            expected = (["scatter"] * (len(values) - 1) + ["value", "scatter", "wire"]
                        if online else ["value"] + ["scatter"] * len(values) + ["wire"])
            assert submitted == expected and (scratch[6:].view(torch.int16) == 0x7e05).all()
            submitted.clear()
            own, peer, _, wire = table.grads_in_flight[:4]
            payload = [part.view(torch.int16).clone() for part in (own, peer, wire)]
            work = SimpleNamespace(wait=lambda: None)
            table.grads_in_flight = (own, peer, work, wire, *table.grads_in_flight[4:])
            update(table, 1, 0.01, 0.1, 1e-8)
            assert table.grads_in_flight is None and (scratch[6:].view(torch.int16) == 0x7e05).all()
            if online:
                assert (scratch[:6].view(torch.int16) == 0).all()
            outputs.append([*payload, merge])
        return outputs

    for old, new in zip(run(baseline, False), run(candidate, True), strict=True):
        assert all(torch.equal(left, right) for left, right in zip(old, new, strict=True))
    print("N-gram accumulation: installed methods agree across graph-buffer reuse, duplicate slots and fused/fallback resets.")


def check_row_routes(baseline, candidate, torch):
    import numpy as np

    world = 8
    for total, dtype in [(84602880, np.int32), (84602880, np.int64), (2**31, np.int32)]:
        size = total // world
        cases = [np.empty(0, dtype=dtype), np.array([0, size - 1, size, total - 1], dtype=dtype),
                 np.array([i * size + j for i in range(world) for j in (0, 1, size - 1)], dtype=dtype)]
        for root in (baseline, candidate):
            initialize = load_sparse_method(root, ROW_OWNERS, "RowOwners", "__init__", np=np)
            for rank in range(world):
                owner = SimpleNamespace()
                initialize(owner, total, rank, world)
                assert owner.bounds.dtype == np.int64
                key_dtype = np.int32 if root == candidate and total < 2**31 and dtype == np.int32 else np.int64
                for host in cases:
                    want = torch.from_numpy(host)
                    expected = [int(((host >= i * size) & (host < (i + 1) * size)).sum()) for i in range(world)]
                    lo = sum(expected[:rank])
                    hi = lo + expected[rank]
                    expected[rank] = 0
                    progress = object()

                    def exchange(recv, send, *, async_op):
                        assert async_op and recv.dtype == send.dtype == torch.int64
                        assert send.tolist() == expected
                        return progress

                    def searchsorted(values, keys):
                        assert values is host and keys.dtype == key_dtype
                        return np.searchsorted(values, keys)

                    request = load_sparse_method(
                        root, ROW_OWNERS, "RowOwners", "request", torch=torch, Tensor=torch.Tensor,
                        RowPull=SimpleNamespace, dist=SimpleNamespace(all_to_all_single=exchange),
                        np=SimpleNamespace(ndarray=np.ndarray, int32=np.int32, int64=np.int64,
                                           diff=np.diff, searchsorted=searchsorted),
                    )
                    pull = request(owner, want, host)
                    assert pull.want is want and (pull.lo, pull.hi, pull.send_counts) == (lo, hi, expected)
                    assert pull.counts_exchange[0] is progress and pull.counts_exchange[2].tolist() == expected
    print("Sparse row routes: all eight ranks preserve owner boundaries and outgoing counts.")


def check_sparse_fetch(baseline, candidate, torch, mode, hopper):
    module = load_candidate_scale(candidate / ROW_FETCH_KERNEL)
    device = "cpu" if mode == "cpu" else "cuda"
    first = 64
    width = module.WIDTH

    def words(rows, bias):
        bits = (torch.arange(rows * width, dtype=torch.int32, device=device) * 31 + bias).to(torch.int16)
        return bits.view(torch.bfloat16).reshape(rows, width)

    shard = words(19, 0x8000)
    for count, lo, hi in [(0, 0, 0), (7, 0, 0), (7, 0, 7), (11, 2, 5), (13, 9, 13)]:
        actual_ids = torch.arange(count, dtype=torch.int32, device=device) + first
        want = torch.cat((actual_ids[:lo] - first, actual_ids[lo:hi], actual_ids[hi:] + first))
        wire = words(count - (hi - lo), 0x7f80)
        initial = words(count + 5, 0x9f80)
        expected_served = shard.index_select(0, actual_ids - first)
        if hopper:
            assert module._layout(shard, want, first)

        def run(root):
            events = []
            asked = torch.full_like(actual_ids, -1)
            pull = SimpleNamespace(want=want, lo=lo, hi=hi, num_rows=count, asked=asked,
                                   send_counts=[lo, 0, count - hi], recv_counts=[0, 0, count])

            def wait_ids():
                events.append("ids")
                asked.copy_(actual_ids)

            pull.ids_exchange = SimpleNamespace(wait=wait_ids)

            def exchange(served, inputs, outputs, *, async_op):
                assert pull.ids_exchange is None and events == ["ids"]
                assert inputs is pull.recv_counts and outputs is pull.send_counts and async_op
                assert torch.equal(served.view(torch.int16), expected_served.view(torch.int16))
                events.append("exchange")
                received = torch.empty_like(wire)

                def wait_rows():
                    assert table.landing is None
                    events.append("rows")
                    received.copy_(wire)

                return received, SimpleNamespace(wait=wait_rows)

            def serve(*args):
                result = module.serve_rows(*args)
                assert (result is not None) == hopper
                return result

            def land(*args):
                result = module.land_ngram_(*args)
                assert result == hopper
                return result

            start = load_sparse_method(root, ROW_OWNERS, "RowOwners", "start_fetch", torch=torch,
                                       Tensor=torch.Tensor, RowPull=object, all_to_all_rows=exchange, serve_rows=serve)
            finish = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "complete_land",
                                        torch=torch, land_ngram_=land)
            owner = SimpleNamespace(first_row=first)
            cache = initial.clone()
            table = SimpleNamespace(cache=cache, shard=shard, first_row=first, landing=None, cached=None)
            table.landing = (pull, *start(owner, pull, shard))
            finish(table)
            assert events == ["ids", "exchange", "rows"]
            assert table.cached is pull and table.cache is cache
            assert torch.equal(cache[count:].view(torch.int16), initial[count:].view(torch.int16))
            return cache.view(torch.int16)

        assert torch.equal(run(candidate), run(baseline))
    print("Sparse row fetch: checked peer split counts, wait order, cache identity and untouched cache suffix.")


def check_moment_replay(baseline, candidate, triton, tl, torch, mode, hopper, kernels, module):
    from unittest.mock import patch

    device = "cpu" if mode == "cpu" else "cuda"
    first = 71
    size = 221
    shard = torch.empty((size, 768), dtype=torch.bfloat16, device=device)
    dispatched = []

    def launch(*args, **kwargs):
        assert kwargs == {"positive_history": True}
        result = module.replay_next_moments_(*args, **kwargs)
        dispatched.append(result)
        return result

    def method(root, kernel):
        check, = load_functions(root / NGRAM_ADAM, ("_check_state",), triton, tl, torch=torch)
        bring, = load_functions(root / NGRAM_ADAM, ("bring_rows_current",), triton, tl,
                               torch=torch, _check_state=check, _adam_rows_kernel=kernel)
        return load_sparse_method(root, NGRAM_TABLE, "NgramTable", "bring_current", torch=torch,
                                  RowPull=object, bring_rows_current=bring, replay_next_moments_=launch)

    base, fast = method(baseline, kernels[0]), method(candidate, kernels[1])
    cases = [(3, 0, 0, False), (0, 0, 1, False), (2, 13, 5, True), (17, 1, 1, False),
             (19, 36, 19, False), (9, 69, 67, False), (32, 33, 382, False)]
    bits = torch.tensor([0, 1, -0x80000000, 0x3fa57a12, 0x007fffff, 0x00800000, 0x7f800000, 0x7fc01234],
                        dtype=torch.int32, device=device)
    initial = bits[torch.arange(size, device=device) % len(bits)].view(torch.float32)
    for lo, count, event, fresh in cases:
        selected = torch.arange(count, dtype=torch.int32, device=device) * 3
        low = torch.arange(lo, dtype=torch.int32, device=device)
        high = torch.arange(5, dtype=torch.int32, device=device) + first + size
        want = torch.cat((low, selected + first, high))
        steps = torch.arange(size, dtype=torch.int32, device=device)
        stamps = torch.full_like(steps, event) if fresh else steps * 17 % (event + 1)
        history = torch.where(torch.arange(event + 1, device=device) % 2 == 0, 0.95, 0.95 ** 2)
        pull = SimpleNamespace(want=want, lo=lo, hi=lo + count)
        touched = torch.zeros(size, dtype=torch.bool, device=device)
        if event:
            touched[selected.long()] = True

        def run(impl):
            table = SimpleNamespace(shard=shard, exp_avg_sq=initial.clone(), last_event=stamps.clone(),
                                    beta2_history=history, first_row=first, adam_events=event)
            impl(table, pull)
            assert torch.equal(table.exp_avg_sq[~touched].view(torch.int32), initial[~touched].view(torch.int32))
            assert torch.equal(table.last_event[~touched], stamps[~touched])
            return table.exp_avg_sq.view(torch.int32), table.last_event

        expected = run(base)
        dispatched.clear()
        if mode == "cpu":
            with patch.object(module, "_hopper", return_value=True):
                actual = run(fast)
            assert dispatched == ([True] if event else [])
            dispatched.clear()
            with patch.object(module, "_hopper", return_value=False):
                fallback = run(fast)
            assert dispatched == ([False] if event else [])
            assert all(torch.equal(a, b) for a, b in zip(fallback, expected, strict=True))
        else:
            actual = run(fast)
            assert dispatched == ([hopper] if event else [])
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), (lo, count, event)
    print("N-gram moment replay: compared the actual host methods and every FP32 word and event stamp.")


def check_adam_zero_rows(baseline, candidate, triton, tl, torch, mode, kernels):
    device = "cpu" if mode == "cpu" else "cuda"
    launchers = []
    for root, kernel in zip((baseline, candidate), kernels, strict=True):
        check, = load_functions(root / NGRAM_ADAM, ("_check_state",), triton, tl, torch=torch)
        update, = load_functions(root / NGRAM_ADAM, ("adam_rows_",), triton, tl,
                                  torch=torch, _check_state=check, _adam_rows_kernel=kernel)
        launchers.append(update)
    rows = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.int32, device=device)
    claims = torch.tensor([5, 4, 2, 3, -1], dtype=torch.int32, device=device)
    g = (torch.arange(rows.numel() * 768, device=device) % 17 - 8).reshape(-1, 768).to(torch.bfloat16) / 16
    p = torch.full((claims.numel(), 768), 0.75, dtype=torch.bfloat16, device=device)
    v = torch.tensor([0.0, 0.125, -0.0, 0.25, 1.0], dtype=torch.float32, device=device)
    stamps = torch.tensor([0, 377, 2, 381, 0], dtype=torch.int32, device=device)
    history = torch.where(torch.arange(383, device=device) % 2 == 0, 0.95, 0.95 ** 2)

    def run(update, **extra):
        weight, moment, stamp = p.clone(), v.clone(), stamps.clone()
        update(weight, moment, stamp, history, g, rows, claims, 382, beta2=0.95 ** 2,
               eps=1e-10, step_size=0.01, decay=0.0001, grad_mul=0.125,
               sq_mul=(1 - 0.95 ** 2) / 64, **extra)
        return weight.view(torch.int16), moment.view(torch.int32), stamp

    expected = run(launchers[0])
    for use_fast in (False, True):
        actual = run(launchers[1], positive_history=use_fast)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert actual[-1].tolist() == [382, 382, 382, 382, 0]
    print("N-gram Adam rows: compared the actual launchers with stale positive zero, negative zero and duplicate entries.")


def check_adam_entries(baseline, candidate, triton, tl, torch, mode, hopper, module):
    from unittest.mock import patch

    device = "cpu" if mode == "cpu" else "cuda"
    constants = {}
    names = {"NGRAM_LR_MUL", "NGRAM_ADAM_BETA2", "NGRAM_WD_MUL", "NGRAM_WD_MUL_PERIOD4",
             "NGRAM_ADAM_PERIOD4_START"}
    for node in ast.parse((baseline / NGRAM_TABLE).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    schedule, = load_functions(baseline / NGRAM_TABLE, ("adam_beta2_and_wd_mul",), triton, tl, **constants)
    gwidth = 768
    first = 211
    if mode == "cpu" or hopper:
        shared = torch.arange(16, dtype=torch.int32, device=device)
        with patch.object(module, "_hopper", return_value=True):
            assert module.pack_adam_entries(shared, shared[:0], 0, 1, 0, entry_of_row=shared) is None
    for own, peers, lo in [(0, 0, 0), (0, 7, 2), (9, 0, 1), (1, 8, 0), (127, 131, 3), (128, 5, 6)]:
        local = torch.arange(own, dtype=torch.int32, device=device)
        received_ids = torch.arange(peers, dtype=torch.int32, device=device) % 4
        want = torch.cat((torch.zeros(lo, dtype=torch.int32, device=device), local + first))
        pull = SimpleNamespace(want=want, asked=received_ids + first, lo=lo, hi=lo + own)
        own_values = torch.full((own, gwidth), 0.125, dtype=torch.float16, device=device)
        peer_values = torch.full((peers, gwidth), 0.25, dtype=torch.bfloat16, device=device)

        def run(root, expect_fused):
            trace = []
            params = None
            captured = None
            accumulation = own_values.new_zeros((lo + own, gwidth))
            own_input = accumulation[lo:].copy_(own_values)
            pull.num_rows = accumulation.shape[0]

            def wait():
                assert table.grads_in_flight is None
                trace.append("wait")

            def pack(*args, **kwargs):
                result = module.pack_adam_entries(*args, **kwargs)
                assert (result is not None) == expect_fused
                if result is not None:
                    trace.append("prepare")
                    assert all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in result)
                    assert torch.count_nonzero(result[-1].view(torch.int16)).item() == 0
                return result

            def claim(ids, row_map):
                trace.append("claim")
                assert ids.tolist() == local.tolist() + received_ids.tolist()
                assert table.adam_events == 17 and table.beta2_history[17].item() != 0
                for i, row in enumerate(ids.tolist()):
                    row_map[row] = i

            def scatter(src, ids, target, *, row_map, clear_source=False):
                label = "own" if src is own_input else "peer"
                trace.append(label)
                assert torch.equal(ids, local if label == "own" else received_ids)
                target.index_add_(0, row_map[ids.long()].long(), src.to(target.dtype))
                if clear_source:
                    assert label == "own" and expect_fused
                    src.zero_()

            def adam(_p, _v, _last, beta_history, grad, ids, row_map, event, **launch):
                nonlocal params, captured
                trace.append("adam")
                assert launch.pop("positive_history", False) == (root == candidate)
                params = (event, launch)
                touched = torch.unique(ids).long()
                owners = row_map[touched].long()
                assert torch.equal(ids[owners].long(), touched)
                absent = torch.ones_like(row_map, dtype=torch.bool)
                absent[touched] = False
                assert (row_map[absent] == -1).all()
                captured = (beta_history, grad[owners], ids)

            impl = load_sparse_method(root, NGRAM_TABLE, "NgramTable", "adam_update", torch=torch,
                                      NGRAM_LR_MUL=constants["NGRAM_LR_MUL"], adam_beta2_and_wd_mul=schedule,
                                      NGRAM_DIM=gwidth, GRAD_MERGE_DTYPE=torch.bfloat16,
                                      pack_adam_entries=pack, claim_rows=claim, scatter_add_rows=scatter, adam_rows_=adam)
            table = SimpleNamespace(live=pull, first_row=first, adam_events=16, world_size=8,
                                    device=own_values.device, shard=None, exp_avg_sq=None, last_event=None,
                                    grad_per_slot=accumulation,
                                    beta2_history=torch.zeros(18, dtype=torch.float32, device=device),
                                    entry_of_row=torch.full((max(4, own),), -1, dtype=torch.int32, device=device),
                                    grads_in_flight=(own_input, peer_values, SimpleNamespace(wait=wait), None))
            if root == candidate:
                table.grads_in_flight += (not expect_fused,)
            impl(table, pull.hi + constants["NGRAM_ADAM_PERIOD4_START"], 0.001, 0.11, 1e-10)
            assert trace == ["wait", "prepare" if expect_fused else "claim", "own", "peer", "adam"]
            if root == candidate:
                assert torch.count_nonzero(accumulation.view(torch.int16)).item() == 0
            return params, captured

        baseline_args, baseline_outputs = run(baseline, False)
        if mode == "cpu":
            with patch.object(module, "_hopper", return_value=True):
                args, results = run(candidate, True)
            with patch.object(module, "_hopper", return_value=False):
                fallback_args, fallback = run(candidate, False)
            assert fallback_args == baseline_args
            assert all(torch.equal(a, b) for a, b in zip(fallback, baseline_outputs, strict=True))
        else:
            args, results = run(candidate, hopper)
        assert args == baseline_args
        assert all(torch.equal(a, b) for a, b in zip(results, baseline_outputs, strict=True))
    print("N-gram Adam entries: verified both scatter inputs, duplicate claims, gradient merge and launch args.")


def check_value_adam(baseline, candidate, triton, tl, torch, mode, hopper, module, row_scatter):
    from unittest.mock import patch

    row_tile, = load_functions(baseline / ROW_SCATTER, ("_row_tile",), triton, tl, torch=torch, MAX_BLOCK=1024)
    scatter, = load_functions(baseline / ROW_SCATTER, ("scatter_add_rows",), triton, tl,
                              torch=torch, _row_tile=row_tile, _scatter_add_rows_kernel=row_scatter)
    update_step = load_sparse_method(baseline, ANVIL, "AnvilAndAdam", "_adam_update_step", strip_decorators=True)
    optimizer_types = dict(nn=torch.nn, Tensor=torch.Tensor, ParamConfig=object, torch=torch)
    update = load_sparse_method(baseline, ANVIL, "AnvilAndAdam", "_adam_update",
                                AnvilAndAdam=SimpleNamespace(_adam_update_step=update_step), **optimizer_types)
    update_shard = load_sparse_method(baseline, ANVIL, "AnvilAndAdam", "adam_update_shard", **optimizer_types)

    device = "cpu" if mode == "cpu" else "cuda"
    packer = load_candidate_scale(candidate / SPARSE_ROW_KERNEL)
    rows, width, world, rank = 37, 768, 8, 3
    first = rows * rank
    supported = (mode == "cpu" or hopper) and torch.version.git_version == module.ATEN_REVISION

    def scale_eighth_cpu(value):
        return (value.to(tl.float32) * 0.125).to(tl.float16)

    bits = torch.tensor([4, 12, -32767, 0x0400, 0x3000, -0x4c00, 0x3c00], dtype=torch.int16, device=device)
    payload = bits[torch.arange(width, device=device) % len(bits)].view(torch.float16)

    def run(root):
        param = torch.full((world * rows, width), 0.25, dtype=torch.bfloat16, device=device)
        state = dict(step=16, exp_avg=torch.full((rows, width), 0.03125, device=device),
                     exp_avg_sq=torch.full((rows, width), 0.0625, device=device))
        cfg = SimpleNamespace(optim="adam", comms="sharded", chunk_size=rows, adam_betas=(0.75, 0.95),
                              lr=0.001, lr_mul=70.0, weight_decay=0.11, wd_mul=5.0, eps=1e-10)
        optimizer = SimpleNamespace(param_cfgs={param: cfg}, param_states={param: state}, rank=rank,
                                    _step_size_t=torch.zeros([], device=device),
                                    _eff_wd_t=torch.zeros([], device=device))
        optimizer._adam_update = lambda p, g, config: update(optimizer, p, g, config)
        optimizer.adam_update_shard = lambda p, g: update_shard(optimizer, p, g)
        waits = []
        launched = []

        def wait():
            assert table.grads_in_flight is None
            waits.append(True)

        def scatter_fused(*args, **kwargs):
            assert kwargs == {"direct_own": direct_own, "grouped_local": True}
            if force_fallback:
                return False
            result = module.scatter_value_gradients_(*args, **kwargs)
            assert result == (mode == "cpu" or hopper)
            return result

        dense = torch.full((rows, width), 9.0, dtype=torch.float16, device=device)
        storage, touched = None, None
        if supported:
            storage = torch.full((rows * (width + 2),), 9.0, dtype=torch.float16, device=device)
            dense = storage[:rows * width].view(rows, width)
            touched = storage[rows * width:].view(torch.int32)
            touched.fill_(7)

        def merge_sparse(*args, **kwargs):
            assert kwargs == {"direct_own": direct_own, "scale_read": False, "grouped_local": True}
            if not supported or not previous_sparse:
                assert args[-1] is None
            elif previous_pull.hi != previous_pull.lo or previous_pull.asked.numel():
                before = args[-1]
                assert before[0] is previous_pull.want and before[1] is previous_pull.asked
                assert before[2:] == (previous_pull.lo, previous_pull.hi)
            else:
                assert args[-1] == ()
            result = module.merge_sparse_value_gradients_(*args, **kwargs)
            assert result == (supported and 2 * (own + peers) <= rows)
            if result:
                expected = torch.zeros_like(touched)
                destinations = torch.cat((pull.want[lo:lo + own], pull.asked)).long() - first
                tickets = torch.arange(1, own + peers + 1, dtype=torch.int32, device=device)
                expected.scatter_reduce_(0, destinations, tickets, reduce="amax", include_self=True)
                assert torch.equal(touched, expected)
            return result

        def send(_pull, wire):
            expected = payload.repeat(peer_want.numel(), 1)
            assert torch.equal(wire.view(torch.int16), expected.view(torch.int16))
            launched.append(True)
            return payload.repeat(peers, 1), SimpleNamespace(wait=wait)

        table = SimpleNamespace(param=param, optimizer=optimizer, owners=SimpleNamespace(first_row=first, send_grads=send),
                                grad_accum=torch.zeros_like(param, dtype=torch.float16), own_rows=slice(first, first + rows),
                                grad_shard=dense, grad_workspace=storage, grad_touched=touched,
                                world_size=world, grad_previous=None, landing=None,
                                # This fixture uses 37 rows. Neither GPU tile is eligible.
                                _current_scale_read=lambda grad: (None, None),
                                _remember_scale_read=lambda grad, key: None)
        reset = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "reset")
        start = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "send_grads", torch=torch,
                                   pack_value_gradients=packer.pack_value_gradients)
        release = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "zero_sent_grads", torch=torch,
                                     clear_value_want_=packer.clear_value_want_)
        step = load_sparse_method(root, VALUE_PULL, "ValueEmbedPull", "adam_update", torch=torch,
                                  scatter_add_rows=scatter, scatter_value_gradients_=scatter_fused,
                                  merge_sparse_value_gradients_=merge_sparse)
        outputs = []
        events = [(0, 0, 0, 0), (0, 5, 2, 8), (1, 0, 3, 36), (9, 9, 17, 11),
                  (9, 10, 17, 6), (31, 15, 4, 0), (0, 5, 4, 2), (0, 0, 0, 0)]
        previous_sparse = False
        previous_pull = None
        for event, (own, peers, lo, offset) in enumerate(events, 1):
            want = torch.cat((torch.arange(lo, dtype=torch.int32, device=device),
                              torch.arange(own, dtype=torch.int32, device=device) + first + offset,
                              torch.arange(2, dtype=torch.int32, device=device) + first + rows))
            asked = first + offset + torch.arange(peers, dtype=torch.int32, device=device) % 4
            peer_want = torch.cat((want[:lo], want[lo + own:]))
            pull = SimpleNamespace(want=want, asked=asked, lo=lo, hi=lo + own,
                                   peer_want=None if event == 4 else peer_want)
            table.live = pull
            table.grad_accum[want.long()] = payload
            table.grad_accum[-1] = 1.5
            start(table)
            direct_own = root == candidate and (mode == "cpu" or hopper) and pull.peer_want is not None
            assert (table.grads_in_flight[0] is None) == direct_own
            force_fallback = event == 6 and direct_own
            release(table)
            assert not torch.count_nonzero(table.grad_accum[peer_want.long()].view(torch.int16))
            owned = table.grad_accum[want[lo:lo + own].long()].view(torch.int16)
            assert torch.equal(owned, payload.repeat(own, 1).view(torch.int16) if direct_own else torch.zeros_like(owned))
            step(table)
            previous_sparse = 2 * (own + peers) <= rows
            previous_pull = pull
            assert len(launched) == len(waits) == event and state["step"] == 16 + event
            assert not torch.count_nonzero(table.grad_accum[:-1].view(torch.int16))
            assert (table.grad_accum[-1] == 1.5).all()
            assert (param[:first] == 0.25).all() and (param[first+rows:] == 0.25).all()
            if own == 1 and peers == 0:
                assert table.grad_shard[offset, :3].view(torch.int16).tolist() == [0, 2, -32768]
            outputs.append(tuple(x.clone() for x in (table.grad_shard.view(torch.int16), param.view(torch.int16),
                                   state["exp_avg"].view(torch.int32), state["exp_avg_sq"].view(torch.int32))))
            if event == 3:
                previous = table.grad_previous
                reset(table)
                assert table.grad_previous is previous
        return outputs

    expected = run(baseline)
    if mode == "cpu":
        with (patch.object(module, "_hopper", return_value=True),
              patch.object(packer, "_supported", return_value=True),
              patch.object(module, "_scale_half_eighth", scale_eighth_cpu)):
            actual = run(candidate)
    else:
        actual = run(candidate)
    for event, (new, old) in enumerate(zip(actual, expected, strict=True), 1):
        assert all(torch.equal(x, y) for x, y in zip(new, old, strict=True)), event

    if supported:
        rows = triton.cdiv(65536, width) + 1
        bits = torch.arange(rows * width, dtype=torch.int32, device=device).to(torch.int16)
        values = bits.view(torch.float16).reshape(rows, width)
        expected = values * 0.125
        expected[-1] = values[-1]
        active = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        active[-1] = 0
        own = rows // 2
        want = torch.arange(own, dtype=torch.int32, device=device)
        asked = torch.arange(own, rows, dtype=torch.int32, device=device)

        def scale():
            module._scale_claimed[(triton.cdiv(rows, module.ROWS),)](
                values, active, want, asked, own, rows, 0, 0, module.ROWS, module.COLS, num_warps=4, num_stages=1)

        if mode == "cpu":
            with patch.object(module, "_scale_half_eighth", scale_eighth_cpu):
                scale()
            nan = torch.isnan(expected)
            assert torch.equal(torch.isnan(values), nan)
            assert torch.equal(values.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])
            assert torch.equal(values[-1].view(torch.int16), expected[-1].view(torch.int16))
        else:
            scale()
            assert torch.equal(values.view(torch.int16), expected.view(torch.int16))
    print("Value Adam: compared dense and marked scaling, duplicate markers, FP16 bits, weights and both moments.")


def check_value_embed_accum(baseline, candidate, triton, tl, torch, mode, hopper, reference, module):
    from unittest.mock import patch

    device = "cpu" if mode == "cpu" else "cuda"
    calls = []

    class Recorded:
        def __getitem__(self, grid):
            def launch(src, token_ids, out, **settings):
                calls.append((grid, settings))
                reference[grid](src, token_ids, out, **settings)
            return launch

    class Fused:
        def __getitem__(self, grid):
            def launch(g0, g1, g2, g3, token_ids, out, *, V, num_warps, num_stages):
                calls.append((grid, {"V": V}))
                for plane, gradient in enumerate((g0, g1, g2, g3)):
                    reference[grid](gradient, token_ids, out[plane * V:(plane + 1) * V],
                                    V=V, D=768, SEG=256, NSEG=3, num_warps=num_warps, num_stages=num_stages)
            return launch

    def dispatcher(root):
        return load_functions(root / VALUE_EMBED, ("value_embed_grad_accumulate",), triton, tl,
                              strip_decorators=True, torch=torch, GRAD_DTYPE=torch.float16, SEGMENT=256,
                              NUM_WARPS=4, _row_scatter_add_kernel=Recorded(),
                              accumulate_four_planes_=module.accumulate_four_planes_)[0]

    def run(dispatch, gradients):
        out = torch.full((len(gradients) * vocab, 768), 0.5, device=device, dtype=torch.float16)
        calls.clear()
        dispatch(tokens, gradients, out)
        return out.view(torch.int16)

    vocab = 53
    tokens = torch.tensor([52, 0, 17, 52], dtype=torch.int32, device=device)
    # Keep the four sources distinguishable and every duplicate sum exact.
    gradients = [torch.full((tokens.numel(), 1536), (i + 1) / 8, device=device, dtype=torch.bfloat16)[:, ::2]
                 for i in range(4)]
    original, patched = dispatcher(baseline), dispatcher(candidate)
    expected = run(original, gradients)
    assert len(calls) == 4
    if mode == "cpu":
        with (patch.object(module, "_hopper", return_value=True),
              patch.object(module, "_scatter_four_planes", Fused())):
            assert torch.equal(run(patched, gradients), expected) and len(calls) == 1
            expected_two = run(original, gradients[:2])
            assert torch.equal(run(patched, gradients[:2]), expected_two) and len(calls) == 2
    else:
        assert torch.equal(run(patched, gradients), expected) and len(calls) == (0 if hopper else 4)
    print("Value embedding backward: exercised the installed four-plane dispatch and two-plane fallback.")


def compile_sm90(scale, reference, candidate, old_amax, new_amax, qkv_scale, old_pack, new_pack,
                 gluon_pack, grouped_kernels, mlp_kernels, head_kernels, sampled_gather, transpose_kernels,
                 densify_kernels, rope_kernels, tail_kernels, sparse, fetch, moments, adam, value_embed,
                 triton, ir_dir=None):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.experimental.gluon._runtime import GluonASTSource
    from ngram_embed_replay import compile_and_replay
    from densify_ptx_replay import verify_densify_tiles
    from moment_ptx_replay import (
        verify_adam_entries, verify_adam_preparation, verify_adam_row, verify_adam_scatter, verify_moment_replay,
        verify_ngram_accumulation, verify_tiled_adam_scatter, verify_value_clear, verify_value_normalization,
        verify_value_scatter, verify_value_embed_accum,
    )
    from rope_ptx_replay import verify_compact_factors, verify_rope_backward, verify_rope_shift
    from tail_lerp_replay import verify_pair
    from tail_norm_replay import verify_tail_norm
    from sparse_row_replay import verify_sent_clear, verify_sparse_fetch, verify_sparse_packets
    from qkv_scan_replay import verify_grouped_scan
    from qkv_ptx_replay import (
        verify_bf16_transpose_tiles, verify_byte_transpose_tiles, verify_ce_transpose_tiles, verify_grouped_tiles,
        verify_lm_head_tiles,
        verify_mlp_tiles, verify_sampled_inverse, verify_sampled_tiles, verify_tiles,
    )

    target = GPUTarget("cuda", 90, 32)

    def build(kernel, constants, types, *, aligned_args=False, **options):
        defaults = {"CLEAR_SOURCE": False, "PACK_OWN": True, "DIRECT_OWN": False}
        constants = {**{k: v for k, v in defaults.items() if k in kernel.arg_names}, **constants}
        sig = {a: "constexpr" if a in constants else types[a] for a in kernel.arg_names}
        attrs = {}
        if aligned_args:
            # Match the launcher for 16-byte-aligned pointers and multiples of 16.
            attrs = {(kernel.arg_names.index(name),): make_backend(target).parse_attr("D")
                     for name, ty in types.items() if ty != "fp32" and name not in constants
                     and name not in getattr(kernel, "do_not_specialize", ())}
        source = GluonASTSource if kernel.is_gluon() else ASTSource
        binary = triton.compile(source(fn=kernel, signature=sig, constexprs=constants, attrs=attrs),
                                target=target, options=options)
        print(f"sm90 {kernel.__name__}: {len(binary.asm['cubin'])} cubin bytes")
        return binary

    sizes = dict(ROWS=sparse.ROWS, COLS=sparse.COLS)
    value = build(sparse._value_gradient_packets, sizes,
                  dict(src="*u16", ids="*i32", own="*u16", wire="*u16",
                       own_rows="i32", peer_rows="i32", lo="i32", source_rows="i32"),
                  aligned_args=True, num_warps=4, debug=True, sanitize_overflow=False)
    value_wire = build(sparse._value_gradient_packets, dict(sizes, PACK_OWN=False),
                       dict(src="*u16", ids="*i32", own="*u16", wire="*u16",
                            own_rows="i32", peer_rows="i32", lo="i32", source_rows="i32"),
                       aligned_args=True, num_warps=4, debug=True, sanitize_overflow=False)
    ngram = build(sparse._ngram_gradient_packet, sizes,
                  dict(src="*fp16", wire="*bf16", peer_rows="i32", lo="i32", own_rows="i32"),
                  aligned_args=True, num_warps=4)
    ngram_clear = build(sparse._ngram_gradient_packet, dict(sizes, CLEAR_SOURCE=True),
                        dict(src="*fp16", wire="*bf16", peer_rows="i32", lo="i32", own_rows="i32"),
                        aligned_args=True, num_warps=4)
    clear = build(sparse._clear_value_want, sizes, dict(source="*fp16", ids="*i32", count="i32"),
                  aligned_args=True, num_warps=4)
    assert clear.metadata.shared == 0
    verify_sparse_packets(value.asm["ptx"], ngram.asm["ptx"], ngram_clear.asm["ptx"], value_wire.asm["ptx"])
    verify_sent_clear(clear.asm["ptx"])
    print("Sparse packets: validated both destinations, BF16 conversion and zero stores by int32 row ID.")
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)
        for kind, kernel in (("value", value), ("value-wire", value_wire), ("ngram", ngram), ("value-clear", clear)):
            for form in ("ttgir", "ptx"):
                (ir_dir / f"sparse-{kind}.{form}").write_text(kernel.asm[form])
        for form in ("ttgir", "ptx"):
            (ir_dir / f"ngram-scratch-peer-clear.{form}").write_text(ngram_clear.asm[form])

    sizes = dict(ROWS=fetch.ROWS, COLS=fetch.COLS)
    serve = build(fetch._serve_rows, sizes, dict(shard="*bf16", asked="*i32", served="*bf16",
                                               count="i32", first="i32", shard_rows="i32"),
                  aligned_args=True, num_warps=4, debug=True, sanitize_overflow=False)
    land = build(fetch._land_ngram, sizes, dict(shard="*bf16", received="*bf16", want="*i32", cache="*bf16",
                                              count="i32", lo="i32", hi="i32", first="i32", shard_rows="i32"),
                 aligned_args=True, num_warps=4, debug=True, sanitize_overflow=False)
    verify_sparse_fetch(serve.asm["ptx"], land.asm["ptx"])
    for kind, binary in (("serve", serve), ("land", land)):
        assert binary.metadata.shared == 0 and binary.asm["ptx"].count("ld.global.b32") == 1
        assert binary.asm["ptx"].count("ld.global.v4.b32") == binary.asm["ptx"].count("st.global.v4.b32") == 3
        if ir_dir is not None:
            for form in ("ttgir", "ptx"):
                (ir_dir / f"rows-{kind}.{form}").write_text(binary.asm[form])
    print("Sparse row fetch: verified raw BF16 routing, partial tiles and signed 64-bit byte addresses.")

    scalar_moments, guarded_adam, moment_kernel = moments
    scalar_types = dict(P="*bf16", V="*fp32", LAST_EVENT="*i32", BETA2_HISTORY="*fp32",
                        G="*bf16", ROWS="*i32", ENTRY_OF_ROW="*i32", event="i32", beta2="fp32",
                        eps="fp32", step_size="fp32", decay="fp32", grad_mul="fp32", sq_mul="fp32")
    scalar_options = dict(D=768, HAS_GRAD=False, BLOCK=1024)
    reference_replay = build(scalar_moments, scalar_options, scalar_types, aligned_args=True, num_warps=4)
    fallback_replay = build(guarded_adam, dict(scalar_options, POSITIVE_HISTORY=False), scalar_types,
                            aligned_args=True, num_warps=4)
    replays = [("reference", reference_replay), ("fallback", fallback_replay)]
    for label, use_fast in (("grouped", False), ("zero", True)):
        grouped_replay = build(moment_kernel._replay_moments, dict(BLOCK=moment_kernel.ROWS,
                               POSITIVE_HISTORY=use_fast),
                               dict(moment="*fp32", stamps="*i32", history="*fp32", want="*i32",
                                    count="i32", offset="i32", first="i32", event="i32"),
                               aligned_args=True, num_warps=1)
        assert grouped_replay.metadata.shared == 0 and grouped_replay.asm["ptx"].count("redux.sync.min.s32") == 1
        verify_moment_replay(reference_replay.asm["ptx"], grouped_replay.asm["ptx"], skip_zero=use_fast)
        verify_moment_replay(fallback_replay.asm["ptx"], grouped_replay.asm["ptx"], skip_zero=use_fast)
        replays.append((label, grouped_replay))
    if ir_dir is not None:
        for kind, binary in replays:
            for form in ("ttgir", "ptx"):
                (ir_dir / f"ngram-replay-{kind}.{form}").write_text(binary.asm[form])
    print("N-gram moment replay: verified ordered FP32 updates and no history loads for stale positive-zero tiles.")

    entries_kernel, row_scatter, mapped_scatter, value_adam = adam
    packed = build(entries_kernel._pack_entries, dict(BLOCK=entries_kernel.BLOCK),
                   dict(want="*i32", asked="*i32", entries="*i32", peer_rows="*i32",
                        own="i32", count="i32", lo="i32", first="i32"),
                   aligned_args=True, num_warps=4)
    assert packed.metadata.shared == 0 and packed.asm["ptx"].count("ld.global.b32") == 1
    prepared = build(entries_kernel._prepare_entries,
                     dict(ROWS=entries_kernel.ROWS, D=entries_kernel.CHANNELS, TILE=entries_kernel.TILE),
                     dict(want="*i32", asked="*i32", entries="*i32", peer_rows="*i32",
                          entry_of_row="*i32", grad="*bf16", own="i32", count="i32", lo="i32", first="i32"),
                     aligned_args=True, num_warps=4, num_stages=1)
    assert prepared.metadata.shared == 0
    verify_adam_preparation(prepared.asm["ptx"])
    update_constants = dict(D=768, HAS_GRAD=True, BLOCK=1024)
    update = build(scalar_moments, update_constants, scalar_types, aligned_args=True, num_warps=4)
    verify_adam_entries(packed.asm["ptx"])
    verify_adam_row(update.asm["ptx"])
    artifacts = [("pack", packed), ("prepare", prepared), ("update", update)]
    for label, use_fast in (("guarded", False), ("zero", True)):
        binary = build(guarded_adam, dict(update_constants, POSITIVE_HISTORY=use_fast), scalar_types,
                       aligned_args=True, num_warps=4)
        verify_adam_row(binary.asm["ptx"], guarded=True, skip_zero=use_fast)
        artifacts.append(("update-" + label, binary))
    for label, dtype in (("scatter-own", "*fp16"), ("scatter-peer", "*bf16")):
        binary = build(row_scatter, dict(D=768, BLOCK=256, NCHUNK=3, HAS_MAP=True),
                       dict(SRC=dtype, IDX="*i32", DST="*bf16", ROW_MAP="*i32"),
                       aligned_args=True, num_warps=4)
        assert binary.asm["ptx"].count("atom.global.gpu.relaxed.add.noftz.v2.bf16") == 3
        verify_adam_scatter(binary.asm["ptx"], dtype == "*fp16")
        artifacts.append((label, binary))
    # The installed launcher specializes the source type and the last-read clear together.
    for local, incoming, name in ((True, "*fp16", "ngram-scratch-own-clear"),
                                  (False, "*bf16", "ngram-adam-scatter-peer-tiled")):
        binary = build(mapped_scatter, dict(CLEAR_SOURCE=local),
                       dict(source=incoming, indices="*i32", dest="*bf16", row_map="*i32", count="i32"),
                       aligned_args=True, num_warps=4, num_stages=1)
        assert binary.asm["ptx"].count("atom.global.gpu.relaxed.add.noftz.v8.bf16") == 3
        verify_tiled_adam_scatter(binary.asm["ptx"], local=local)
        if ir_dir is not None:
            for form in ("ttgir", "ptx"):
                (ir_dir / f"{name}.{form}").write_text(binary.asm[form])
    if ir_dir is not None:
        for label, binary in artifacts:
            for form in ("ttgir", "ptx"):
                (ir_dir / f"ngram-adam-{label}.{form}").write_text(binary.asm[form])
    print("N-gram scratch: verified exact gradients and the added zero stores in both consumers.")
    print("N-gram Adam: replayed index routing, the 768 weight expressions and the conditional moment/stamp stores.")

    sink_accum = build(row_scatter, dict(D=768, BLOCK=256, NCHUNK=3, HAS_MAP=False),
                       dict(SRC="*bf16", IDX="*i32", DST="*fp16", ROW_MAP="*i32"),
                       aligned_args=True, num_warps=4, num_stages=1)
    verify_ngram_accumulation(sink_accum.asm["ptx"])
    if ir_dir is not None:
        for form in ("ttgir", "ptx"):
            (ir_dir / f"ngram-accumulate.{form}").write_text(sink_accum.asm[form])
    print("N-gram accumulation: replayed all BF16-to-FP16 atomics, including repeated slots.")

    base_value = build(row_scatter, dict(D=768, BLOCK=256, NCHUNK=3, HAS_MAP=False),
                      dict(SRC="*fp16", IDX="*i32", DST="*fp16", ROW_MAP="*i32"),
                      aligned_args=True, num_warps=4, num_stages=1)
    value_args = dict(src="*fp16", ids="*i32", dst="*fp16", offset="i32", first="i32", touched="*i32",
                      claim_start="i32")
    direct_value, marked_value = [
        build(value_adam._scatter_value_rows, dict(TILE=value_adam.TILE, MARK=mark), value_args,
              aligned_args=True, num_warps=4, num_stages=1)
        for mark in (False, True)
    ]
    own_updates = [(mark, build(value_adam._scatter_value_rows,
                               dict(TILE=value_adam.TILE, MARK=mark, DIRECT_OWN=True), value_args,
                               aligned_args=True, num_warps=4, num_stages=1)) for mark in (False, True)]
    claims = dict(grad="*fp16", touched="*i32", want="*i32", asked="*i32",
                  own="i32", count="i32", lo="i32", first="i32")
    sparse_scale, sparse_clear = [
        build(kernel, dict(ROWS=value_adam.ROWS, COLS=value_adam.COLS), claims,
              aligned_args=True, num_warps=4, num_stages=1)
        for kernel in (value_adam._scale_claimed, value_adam._clear_claimed)
    ]
    assert sparse_clear.metadata.shared == 0
    verify_value_scatter(base_value.asm["ptx"], direct_value.asm["ptx"], marked_value.asm["ptx"],
                         [(mark, binary.asm["ptx"]) for mark, binary in own_updates])
    verify_value_normalization(sparse_scale.asm["ptx"])
    verify_value_clear(sparse_clear.asm["ptx"])
    if ir_dir is not None:
        for label, binary in (("reference", base_value), ("direct", direct_value),
                              ("marked", marked_value), ("scale", sparse_scale), ("clear", sparse_clear),
                              ("own", own_updates[0][1]), ("own-marked", own_updates[1][1])):
            for form in ("ttgir", "ptx"):
                (ir_dir / f"value-adam-{label}.{form}").write_text(binary.asm[form])
    print("Value Adam: verified ordered FP16 atomics, row flags, masked clear and FP32-to-FP16 scale.")

    value_reference, value_four = value_embed
    for debug in (False, True):
        options = dict(aligned_args=True, num_warps=4, num_stages=1, debug=debug, sanitize_overflow=False)
        old = build(value_reference, dict(V=50304, D=768, SEG=256, NSEG=3),
                    dict(src="*bf16", token_ids="*i32", out="*fp16"), **options)
        new = build(value_four._scatter_four_planes, dict(V=50304),
                    dict(g0="*bf16", g1="*bf16", g2="*bf16", g3="*bf16", token_ids="*i32", out="*fp16"),
                    **options)
        assert old.metadata.shared == new.metadata.shared == 0
        verify_value_embed_accum(old.asm["ptx"], new.asm["ptx"], debug=debug)
        if ir_dir is not None:
            for label, binary in (("reference", old), ("four", new)):
                suffix = "-debug" if debug else ""
                for form in ("ttgir", "ptx"):
                    (ir_dir / f"value-embed-{label}{suffix}.{form}").write_text(binary.asm[form])
    print("Value embedding backward: replayed every destination, exact FP16 atomics, repeated rows and debug asserts.")

    single, merged, norm = tail_kernels
    restoration = build(norm._restore_tail_norms, dict(ENDS=norm.ENDS, BLOCK=norm.BLOCK),
                        {**dict.fromkeys(("embed", "head", "mlp", "qk", "vo"), "*bf16"), "ratios": "*fp32"},
                        aligned_args=True, num_warps=8)
    verify_tail_norm(restoration.asm["ptx"], norm.ENDS, norm.BLOCK)
    assert restoration.metadata.shared == 0
    print("Tail ship: verified five-bank dispatch and both boundary stores for each bank.")
    single = build(single, dict(BLOCK=4096), dict(SRC="*bf16", DST="*fp32", n_elements="i32", rate="fp32"),
                   aligned_args=True, num_warps=8, num_stages=2)
    merged = build(merged, dict(BLOCK=4096, WARPS=8),
                   dict(src="*bf16", first="*fp32", second="*fp32", rate_first="fp32", rate_second="fp32"),
                   aligned_args=True, num_warps=8, num_stages=2)
    ref_ops, fused_ops = [list(executable_ptx(k.asm["ptx"])) for k in (single, merged)]
    verify_pair(ref_ops, fused_ops)
    assert merged.metadata.shared == 0 and len(fused_ops) < 2 * len(ref_ops)
    print(f"Tail averaging: {2 * len(ref_ops)} → {len(fused_ops)} PTX instructions, "
          f"{single.metadata.shared} → 0 shared bytes per CTA, 16 → 8 KiB of BF16 loads per tile.")
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)
        for name, kernel in (("single", single), ("merged", merged)):
            for ir_kind in ("ttgir", "ptx"):
                (ir_dir / f"tail-lerp-{name}.{ir_kind}").write_text(kernel.asm[ir_kind])
        for ir_kind in ("ttgir", "ptx"):
            (ir_dir / f"tail-norm.{ir_kind}").write_text(restoration.asm[ir_kind])

    constants = dict(SLOTS=12, LAYERS=11, PARALLEL_SLOT=11, PARALLEL_LAYER=8, BLOCK=16,
                     X_SCALE=2.0**-4, GRAD_SCALE=2.0**-6, POST_GAIN=1.03/448., DPRE_GAIN=1.25/57344.)
    ptx = build(scale._refresh_mlp_scalars, constants, dict.fromkeys(scale._refresh_mlp_scalars.arg_names, "*fp32"),
                num_warps=1, enable_fp_fusion=False).asm["ptx"]
    assert "fma.rn.f32" not in ptx and "cvt.rn.bf16.f32" in ptx
    compile_and_replay(reference, candidate, build, ir_dir)

    amax_types = dict(first="*bf16", second="*bf16", partial_amax="*fp32")
    width_names = ("narrow", "halfValue", "wide")
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)
    old_rope, new_rope, old_rope_backward, new_rope_backward = rope_kernels
    dense_factors = dict(COMPACT_FACTORS=False, FORCE_SIGN_BITS=False, IMPLICIT_STATIONARY=False)
    for tokens in ROPE_LENGTHS:
        constants = rope_launch_args(tokens)
        types = dict.fromkeys(("qk", "factor1", "factor2", "out_q", "out_k"), "*bf16")
        original = build(old_rope, constants, types, aligned_args=True, num_warps=2)
        shifted = build(new_rope, dict(constants, SHIFT_STORE=True, **dense_factors), types,
                        aligned_args=True, num_warps=2)
        old_ops, new_ops = (list(executable_ptx(item.asm["ptx"])) for item in (original, shifted))
        assert sum("rsqrt.approx.ftz.f32" in op for op in old_ops) == 2
        assert sum("rsqrt.approx.ftz.f32" in op for op in new_ops) == 1
        verify_rope_shift(old_ops, new_ops, tokens)
        print(f"Shifted RoPE {tokens}: {len(old_ops)} → {len(new_ops)} PTX instructions, "
              "exact Q and K expression trees checked")
        if ir_dir is not None:
            for side, binary in (("reference", original), ("shifted", shifted)):
                for form in ("ttgir", "ptx"):
                    (ir_dir / f"rope-{side}-{tokens}.{form}").write_text(binary.asm[form])
    for shape, qk_dim, v_dim, paired in ROPE_BACKWARD_SHAPES:
        for tokens in ROPE_LENGTHS[:-1]:
            constants = rope_backward_args(tokens, qk_dim, v_dim, paired)
            types = dict.fromkeys(("grad_q", "grad_k", "grad_v", "qk", "factor1", "factor2"), "*bf16")
            types.update(grad_row="*fp8e4nv", grad_transposed="*fp8e4nv", grad_scale_ptr="*fp32")
            options = dict(aligned_args=True, num_warps=8 if qk_dim == 128 else 4)
            original = build(old_rope_backward, constants, types, **options)
            optimized = build(new_rope_backward, dict(constants, TRIM_KEY_GRAD=qk_dim == 128, **dense_factors),
                              types, **options)
            old_ops, new_ops = (list(executable_ptx(item.asm["ptx"])) for item in (original, optimized))
            loads = verify_rope_backward(old_ops, new_ops, constants)
            assert original.metadata.shared == optimized.metadata.shared
            old_scalar, new_scalar = (sum("ld.global.b16" in op for op in ops) for ops in (old_ops, new_ops))
            assert (old_scalar, new_scalar) == (16, 0), (shape, tokens, old_scalar, new_scalar)
            traffic = f", K-gradient bytes {loads[0]} → {loads[1]} per interior CTA" if loads else ""
            print(f"RoPE backward {shape} {tokens}: both FP8 expression trees checked "
                  f"({len(old_ops)} → {len(new_ops)} PTX instructions, "
                  f"{old_scalar} → {new_scalar} scalar BF16 loads{traffic})")
            if ir_dir is not None:
                prefix, side = ("rope-backward", "trimmed") if shape == "wide" else (f"rope-backward-{shape}", "vector")
                for version, binary in (("reference", original), (side, optimized)):
                    for form in ("ttgir", "ptx"):
                        (ir_dir / f"{prefix}-{version}-{tokens}.{form}").write_text(binary.asm[form])
    for shape, qk_dim, v_dim, paired in ROPE_BACKWARD_SHAPES:
        for tokens in ROPE_LENGTHS:
            fwd = dict(rope_launch_args(tokens, qk_dim, v_dim, paired), SHIFT_STORE=qk_dim == 128)
            fwd_types = dict.fromkeys(("qk", "factor1", "factor2", "out_q", "out_k"), "*bf16")
            variants = [("forward", new_rope, fwd, fwd_types, 2)]
            if tokens != ROPE_LENGTHS[-1]:
                bwd = dict(rope_backward_args(tokens, qk_dim, v_dim, paired), TRIM_KEY_GRAD=qk_dim == 128)
                bwd_types = dict.fromkeys(("grad_q", "grad_k", "grad_v", "qk", "factor1", "factor2"), "*bf16")
                bwd_types.update(grad_row="*fp8e4nv", grad_transposed="*fp8e4nv", grad_scale_ptr="*fp32")
                variants.append(("backward", new_rope_backward, bwd, bwd_types, 8 if qk_dim == 128 else 4))
            for stage, fn, constants, types, warps in variants:
                dense = build(fn, dict(constants, **dense_factors), types, aligned_args=True, num_warps=warps)
                compact = build(fn, dict(constants, COMPACT_FACTORS=True, FORCE_SIGN_BITS=True, IMPLICIT_STATIONARY=False,
                                         factor_stride_t=constants["factor_stride_t"] // 2),
                                types, aligned_args=True, num_warps=warps)
                old_ops, new_ops = (list(executable_ptx(item.asm["ptx"])) for item in (dense, compact))
                loads = verify_compact_factors(old_ops, new_ops, tokens, qk_dim, backward=stage == "backward")
                assert compact.metadata.shared <= dense.metadata.shared
                assert len(new_ops) <= len(old_ops), (shape, tokens, stage, len(old_ops), len(new_ops))
                print(f"Compact YaRN {stage} {shape} {tokens}: {loads[0]} → {loads[1]} factor bytes per sampled CTA, "
                      f"{len(old_ops)} → {len(new_ops)} PTX instructions, exact output expressions checked")
                if ir_dir is not None:
                    for form in ("ttgir", "ptx"):
                        (ir_dir / f"rope-compact-{stage}-{shape}-{tokens}.{form}").write_text(compact.asm[form])
                if shape == "wide":
                    stationary = build(fn, dict(constants, COMPACT_FACTORS=True, FORCE_SIGN_BITS=True,
                                                IMPLICIT_STATIONARY=True, factor_stride_t=32),
                                       types, aligned_args=True, num_warps=warps)
                    stationary_ops = list(executable_ptx(stationary.asm["ptx"]))
                    old_bytes, new_bytes = verify_compact_factors(new_ops, stationary_ops, tokens, qk_dim,
                                                                   backward=stage == "backward", stationary=True)
                    assert stationary.metadata.shared <= compact.metadata.shared
                    print(f"Stationary YaRN {stage} {tokens}: {old_bytes} → {new_bytes} factor bytes per sampled CTA, "
                          f"{len(new_ops)} → {len(stationary_ops)} PTX instructions, exact output expressions checked")
                    if ir_dir is not None:
                        for form in ("ttgir", "ptx"):
                            (ir_dir / f"rope-stationary-{stage}-{tokens}.{form}").write_text(stationary.asm[form])
    old_transpose, old_add, byte_module, new_bf16 = transpose_kernels
    original = build(old_transpose, dict(BLOCK_M=128, BLOCK_N=128, src_stride_n=1, dst_stride_1=1),
                     dict(src_ptr="*u8", dst_ptr="*u8", M="i32", N="i32", src_stride_m="i32", dst_stride_0="i32"),
                     aligned_args=True, num_warps=8, num_stages=2)
    hopper = build(byte_module._byte_transpose, {}, dict(src="*u8", dst="*u8", M_tiles="i32", N_tiles="i32"),
                   aligned_args=True, num_warps=8, num_stages=2)
    original_ops, hopper_ops = (list(executable_ptx(item.asm["ptx"])) for item in (original, hopper))
    assert hopper.metadata.shared == 4096
    assert sum(i.startswith("ldmatrix.sync.aligned.m8n8.x4.shared.b16") for i in hopper_ops) == 4
    assert sum(i.startswith("bar.warp.sync") for i in hopper_ops) == 7
    assert not any(i.startswith(("bar.sync", "@")) for i in hopper_ops)
    verify_byte_transpose_tiles(hopper_ops)
    print(f"Byte transpose: {len(original_ops)} → {len(hopper_ops)} PTX instructions, "
          f"{original.metadata.shared} → {hopper.metadata.shared} shared bytes, boundary tiles checked")
    if ir_dir is not None:
        for name, binary, stage in (("reference", original, "ttir"), ("hopper", hopper, "ttgir")):
            for form in (stage, "ptx"):
                (ir_dir / f"byte-transpose-{name}.{form}").write_text(binary.asm[form])
    paired = build(byte_module._ce_backward_transposes, {},
                   dict(x="*u8", xt="*u8", grad="*u8", gt="*u8", M_tiles="i32", N_tiles="i32"),
                   aligned_args=True, num_warps=8, num_stages=2)
    pair_ops = list(executable_ptx(paired.asm["ptx"]))
    assert paired.metadata.shared == 8192
    assert sum("ld.global.v4.b32" in i for i in pair_ops) == 4
    assert sum(i.startswith("ldmatrix.sync.aligned.m8n8.x4.shared.b16") for i in pair_ops) == 4
    assert sum(i.startswith("bar.warp.sync") for i in pair_ops) == 3
    assert not any(i.startswith(("bar.sync", "@")) for i in pair_ops)
    verify_ce_transpose_tiles(pair_ops)
    print("CE backward transposes: one launch with disjoint byte layouts, both operand intervals checked")
    if ir_dir is not None:
        for form in ("ttgir", "ptx"):
            (ir_dir / f"ce-transpose-pair.{form}").write_text(paired.asm[form])
    for name, add, m, n, kernel in (("copy", False, 768, 50304, old_transpose),
                                    ("add", True, 50304, 768, old_add)):
        original = build(kernel, dict(BLOCK_M=64, BLOCK_N=128, src_stride_n=1, dst_stride_1=1),
                         dict(src_ptr="*bf16", dst_ptr="*bf16", M="i32", N="i32",
                              src_stride_m="i32", dst_stride_0="i32"),
                         aligned_args=True, num_warps=8, num_stages=2)
        hopper = build(new_bf16, dict(M=m, N=n, ADD=add), dict(src="*bf16", dst="*bf16"),
                       aligned_args=True, num_warps=8, num_stages=2)
        old_ops, new_ops = (list(executable_ptx(item.asm["ptx"])) for item in (original, hopper))
        assert hopper.metadata.shared == 4096
        assert sum(i.startswith("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16") for i in new_ops) == 4
        assert sum(i.startswith("ld.shared.v4.b32") for i in new_ops) == 4
        assert sum(i.startswith("ld.global.v4.b32") for i in new_ops) == (8 if add else 4)
        assert sum(i.startswith("bar.warp.sync") for i in new_ops) == 7
        assert not any(i.startswith(("bar.sync", "@")) for i in new_ops)
        assert all(sum(i.startswith("add.bf16x2") for i in ops) == (16 if add else 0)
                   for ops in (old_ops, new_ops))
        verify_bf16_transpose_tiles(old_ops, new_ops, add)
        print(f"BF16 transpose {name}: {len(old_ops)} → {len(new_ops)} PTX instructions, "
              f"{original.metadata.shared} → {hopper.metadata.shared} shared bytes, PTX operands checked")
        if ir_dir is not None:
            for side, binary, stage in (("reference", original, "ttir"), ("hopper", hopper, "ttgir")):
                for form in (stage, "ptx"):
                    (ir_dir / f"bf16-transpose-{name}-{side}.{form}").write_text(binary.asm[form])
    gather_types = dict(weights="*u8", candidates="*i64", rows="*u8", transposed="*u8")
    old_densify, new_densify = densify_kernels
    reference_densify = build(
        old_densify, dict(BLOCK_R=32, BLOCK_V=256),
        dict(SRC="*bf16", POS="*i32", DST="*bf16", R="i32", V="i32", src_stride_r="i32", dst_stride_r="i32"),
        aligned_args=True, num_warps=4, num_stages=3,
    )
    reference_ops = list(executable_ptx(reference_densify.asm["ptx"]))
    if ir_dir is not None:
        for form in ("ttir", "ptx"):
            (ir_dir / f"densify-reference.{form}").write_text(reference_densify.asm[form])
    for count in SAMPLED_COUNTS:
        binary = build(sampled_gather._gather_candidate_rows, {"P": count}, gather_types,
                       aligned_args=True, num_warps=8, num_stages=2)
        hot = list(executable_ptx(binary.asm["ptx"]))
        assert binary.metadata.shared == 4096
        assert sum(i.startswith("ldmatrix.sync.aligned.m8n8.x4.shared.b16") for i in hot) == 4
        assert sum(i.startswith("bar.warp.sync") for i in hot) == 7
        assert not any(i.startswith(("bar.sync", "@")) for i in hot)
        assert sum(i.startswith("ld.global.v4.b32 ") for i in hot) == 4
        assert sum(i.startswith("st.global.v4.b32 ") for i in hot) == 8
        verify_sampled_tiles(hot, count)
        print(f"Sampled gather {count}: {len(hot)} PTX instructions, 4 KiB shared, both layouts checked")
        if ir_dir is not None:
            for artifact in ("ttgir", "ptx"):
                (ir_dir / f"sampled-gather-{count}.{artifact}").write_text(binary.asm[artifact])
        with_inverse = build(sampled_gather._gather_candidate_rows_and_inverse, {"P": count},
                             dict(gather_types, inverse="*i32"), aligned_args=True, num_warps=8, num_stages=2)
        assert with_inverse.metadata.shared == binary.metadata.shared
        inverse_ops = list(executable_ptx(with_inverse.asm["ptx"]))
        assert sum(i.startswith("redux.sync.add.s32 ") for i in inverse_ops) == 1
        assert sum(i.startswith("ld.global.v4.b32 ") for i in inverse_ops) == 4
        assert sum(i.startswith("st.global.v4.b32 ") for i in inverse_ops) == 8
        verify_sampled_inverse(with_inverse.asm["ptx"], count)
        if ir_dir is not None:
            for artifact in ("ttgir", "ptx"):
                (ir_dir / f"sampled-inverse-{count}.{artifact}").write_text(with_inverse.asm[artifact])
        print(f"Sampled inverse {count}: checked the block search, warp count, both byte layouts and index stores.")
        dense = build(new_densify, {"P": count}, dict(src="*bf16", pos="*i32", dst="*bf16"),
                      aligned_args=True, num_warps=8, num_stages=3)
        hot_dense = list(executable_ptx(dense.asm["ptx"]))
        assert dense.metadata.shared == 0
        assert not any(op.startswith("bar.") for op in hot_dense)
        assert sum(op.startswith("st.global.v4.b32") for op in hot_dense) == 4
        verify_densify_tiles(hot_dense, count)
        print(f"Sampled densify {count}: {len(reference_ops)} → {len(hot_dense)} PTX instructions, "
              f"{reference_densify.metadata.shared} → 0 shared bytes, route and zero fill checked")
        if ir_dir is not None:
            for form in ("ttgir", "ptx"):
                (ir_dir / f"densify-hopper-{count}.{form}").write_text(dense.asm[form])
    old_head, new_head = head_kernels
    head_types = dict(W="*bf16", ROW="*fp8e4nv", COL="*fp8e4nv", INV="*fp32")
    original_head = build(old_head, dict(BLOCK_M=64, BLOCK_N=128), dict(head_types, M="i32", N="i32"),
                          aligned_args=True, num_warps=8, num_stages=2)
    hopper_head = build(new_head, {}, head_types, aligned_args=True, num_warps=8, num_stages=2)
    old_head_ops = list(executable_ptx(original_head.asm["ptx"]))
    head_labels = {}
    new_head_ops = list(executable_ptx(hopper_head.asm["ptx"], labels=head_labels))
    assert hopper_head.metadata.shared == 4096
    assert not any(i.startswith("bar.sync") or (i.startswith("@") and "global" in i) for i in new_head_ops)
    assert sum(i.startswith("bar.warp.sync") for i in new_head_ops) == 6
    for opcode, expected in (("cvt.rn.bf16x2.f32", 16), ("mul.rn.bf16x2", 16),
                             ("min.bf16x2", 32), ("cvt.rn.f16.bf16", 64),
                             ("cvt.rn.satfinite.e4m3x2.f16x2", 32),
                             ("st.global.v4.b32", 4), ("st.global.v2.b32", 8)):
        assert sum(i.startswith(opcode + " ") for i in new_head_ops) == expected
    assert not any(i.startswith(("cvt.rn.bf16.f32 ", "min.f32 ", "max.f32 ", "max.bf16x2 "))
                   for i in new_head_ops)
    verify_lm_head_tiles(new_head_ops, head_labels)
    print(f"LM-head pack: {len(old_head_ops)} → {len(new_head_ops)} PTX instructions, "
          f"{original_head.metadata.shared} → {hopper_head.metadata.shared} shared bytes, both layouts checked")
    if ir_dir is not None:
        for side, binary, stage in (("reference", original_head, "ttir"), ("hopper", hopper_head, "ttgir")):
            for artifact in (stage, "ptx"):
                (ir_dir / f"lm-head-{side}.{artifact}").write_text(binary.asm[artifact])
    old_mlp, mlp_scale, mlp_pack = mlp_kernels
    base_types = dict(w_ptr="*bf16", row_ptr="*fp8e4nv", col_ptr="*fp8e4nv", scale_ptr="*fp32",
                      partial_amax_ptr="*fp32", w_stride_l="i32", w_stride_h="i32")
    base_constants = dict(w_stride_d=1, H=2816, D=768, num_tiles_d=12, num_tiles=528, BLOCK_H=64, BLOCK_D=64)
    base = build(old_mlp, base_constants, base_types, aligned_args=True, num_warps=4, num_stages=2)
    delayed = build(mlp_scale, {"GAIN": 1.12 / 448.0}, dict.fromkeys(mlp_scale.arg_names[:-1], "*fp32"),
                    aligned_args=True, num_warps=4)
    packed_types = {arg: "*bf16" if arg == "bank" else "*fp8e4nv" if arg.startswith(("row", "col")) else "*fp32"
                    for arg in mlp_pack.arg_names}
    paired = build(mlp_pack, {}, packed_types, aligned_args=True, num_warps=8, num_stages=2)
    assert paired.metadata.shared == 4160
    base_ops, paired_ops = [list(executable_ptx(item.asm["ptx"])) for item in (base, paired)]
    assert sum(op.startswith("bar.sync") for op in paired_ops) == 1
    verify_mlp_tiles(paired_ops)
    print(f"MLP weight pack: {len(base_ops)} → {len(paired_ops)} PTX instructions, "
          f"{base.metadata.shared} → {paired.metadata.shared} shared bytes, 12,672 → 6,336 CTAs")
    if ir_dir is not None:
        for name, binary, stage in (("reference", base, "ttir"), ("scale", delayed, "ttir"),
                                    ("pack", paired, "ttgir")):
            for artifact in (stage, "ptx"):
                (ir_dir / f"mlp-weight-{name}.{artifact}").write_text(binary.asm[artifact])
    scale_asm = build(qkv_scale, {}, dict(partial_amax="*fp32", scales="*fp32"),
                      aligned_args=True, num_warps=1).asm
    if ir_dir is not None:
        for stage in ("ttir", "ptx"):
            (ir_dir / f"qkv-scale.{stage}").write_text(scale_asm[stage])
    for name, kernel, warps, stages in zip(("scan", "scale", "pack"), grouped_kernels, (8, 1, 8), (1, 3, 2)):
        types = {arg: "*bf16" if arg in ("first", "second") else
                 "*fp8e4nv" if arg.startswith(("row", "col")) else "*fp32" for arg in kernel.arg_names}
        binary = build(kernel, {}, types, aligned_args=True, num_warps=warps, num_stages=stages)
        asm = binary.asm
        if name == "scan":
            verify_grouped_scan(asm["ptx"])
            print("Grouped QKV scan: verified ordered vector loads at all 224 launch points.")
        if name == "pack":
            hot = list(executable_ptx(asm["ptx"]))
            assert binary.metadata.shared == 4096
            assert sum(i.startswith("ldmatrix.sync.aligned.m8n8.x4.shared.b16") for i in hot) == 2
            assert sum(i.startswith("bar.warp.sync") for i in hot) == 3
            assert not any(i.startswith(("bar.sync", "div.s32", "div.u32")) for i in hot)
            assert sum(i.startswith("st.global.v4.b32") for i in hot) == 2
            assert sum(i.startswith("st.global.v2.b32") for i in hot) == 4
            verify_grouped_tiles(hot, QKV_GROUPS)
            print(f"Grouped QKV pack: {len(hot)} PTX instructions, 4 KiB shared, "
                  "both FP8 output planes match at 12 boundary tiles")
        if ir_dir is not None:
            for stage in ("ttgir" if name == "pack" else "ttir", "ptx"):
                (ir_dir / f"qkv-grouped-{name}.{stage}").write_text(asm[stage])
    for width, (batch, first_rows, second_rows) in zip(width_names, QKV_GROUPS):
        v_stride = second_rows * 768 if width == "halfValue" else 2 * second_rows * 768
        amax_constants = dict(FIRST_ELEMENTS=first_rows * 768, SECOND_ELEMENTS=second_rows * 768,
                              FIRST_STRIDE_B=first_rows * 768, SECOND_STRIDE_B=v_stride,
                              NUM_PARTS=32, BLOCK_SIZE=2048)
        ref_ir = build(old_amax, amax_constants, amax_types, aligned_args=True, num_warps=8, num_stages=1).asm
        new_ir = build(new_amax, dict(amax_constants, QK_CROP=False, V_CROP=False), amax_types,
                       aligned_args=True, num_warps=8, num_stages=1).asm
        head_crop = dict(QK_CROP=first_rows == 12 * 64, V_CROP=second_rows == 6 * 64)
        raw_pitch = 12 * 128 * 768
        raw_amax = dict(amax_constants, FIRST_STRIDE_B=raw_pitch, SECOND_STRIDE_B=raw_pitch, **head_crop)
        raw_ir = build(new_amax, raw_amax, amax_types, aligned_args=True, num_warps=8, num_stages=1).asm
        if ir_dir is not None:
            for side, artifact in (("reference", ref_ir), ("aligned", new_ir), ("native", raw_ir)):
                (ir_dir / f"{width}-{side}.ttir").write_text(artifact["ttir"])
                (ir_dir / f"{width}-{side}.ptx").write_text(artifact["ptx"])

        pack = pack_constants(first_rows, second_rows, 768, first_rows * 768, v_stride)
        pack_types = dict(first="*bf16", second="*bf16", row="*fp8e4nv", transposed="*fp8e4nv",
                          scale_ptr="*fp32", M="i32", N="i32")
        old = build(old_pack, pack, pack_types, aligned_args=True, num_warps=8, num_stages=2)
        pack.update(M=first_rows + second_rows, N=768)
        new = build(new_pack, pack, pack_types, aligned_args=True, num_warps=8, num_stages=2)
        raw_pack = dict(pack, FIRST_STRIDE_B=raw_pitch, SECOND_STRIDE_B=raw_pitch, **head_crop)
        explicit = build(gluon_pack, raw_pack, pack_types, aligned_args=True, num_warps=8, num_stages=2)
        old_asm, new_asm, explicit_asm = old.asm, new.asm, explicit.asm
        scalar_ops = ("div.full.f32", "min.f32", "max.f32", "cvt.rn.satfinite.e4m3x2.f32")
        old_instructions = list(executable_ptx(old_asm["ptx"]))
        new_instructions = list(executable_ptx(new_asm["ptx"]))
        explicit_instructions = list(executable_ptx(explicit_asm["ptx"]))
        for op in scalar_ops:
            base = sum(i.startswith(op) for i in old_instructions)
            assert sum(i.startswith(op) for i in new_instructions) == base
            assert 2 * sum(i.startswith(op) for i in explicit_instructions) == base
        for instructions in (new_instructions, explicit_instructions):
            assert not any(i.startswith("@") and ("ld.global" in i or "st.global" in i) for i in instructions)
        assert explicit.metadata.shared == 4096
        assert sum(i.startswith("ldmatrix.sync.aligned.m8n8.x4.shared.b16")
                   for i in explicit_instructions) == 2
        assert sum(i.startswith("bar.warp.sync") for i in explicit_instructions) == 3
        assert not any(i.startswith("bar.sync") for i in explicit_instructions)
        print(f"{width} QKV pack PTX instructions: {len(old_instructions)} → {len(explicit_instructions)}"
              f" (aligned Triton: {len(new_instructions)})")
        print(f"{width} QKV pack shared memory: {old.metadata.shared} → {explicit.metadata.shared} bytes")
        verify_tiles(explicit_instructions, raw_pack, batch)
        print(f"{width} QKV pack PTX byte routing: both output layouts match for four boundary tiles")
        if ir_dir is not None:
            for side, asm in (("reference", old_asm), ("aligned", new_asm)):
                for stage in ("ttir", "ptx"):
                    (ir_dir / f"pack-{width}-{side}.{stage}").write_text(asm[stage])
            for stage in ("ttgir", "ptx"):
                (ir_dir / f"pack-{width}-gluon.{stage}").write_text(explicit_asm[stage])


def check_checkout(baseline, candidate):
    for path in (baseline, candidate):
        if git(path, "rev-parse", "HEAD").strip() != BASELINE:
            raise ValueError(f"Expected pinned checkout {BASELINE} at {path}")
    if git(baseline, "diff", "--name-only", "HEAD").strip():
        raise ValueError("Expected a pristine tracked baseline")
    changed = set(git(candidate, "diff", "--name-only", "HEAD").splitlines())
    if changed != set(map(str, TRACKED_PATCHES)):
        raise ValueError("Candidate must contain only the tracked source replacements")
    for path, allowed in ((baseline, set()), (candidate, set(map(str, PRIVATE_KERNELS)))):
        other_sources = {
            name for name in git(path, "ls-files", "--others", "--exclude-standard").splitlines()
            if Path(name).suffix in (".py", ".pyi", ".pyc", ".so", ".cuh", ".cu", ".cpp", ".sh")
        }
        if other_sources != allowed:
            raise ValueError(f"Unexpected runtime source files in {path}: {sorted(other_sources ^ allowed)}")
    originals = {path: git(baseline, "show", f"HEAD:{path}") for path in TRACKED_PATCHES}
    for path, patch in TRACKED_PATCHES.items():
        if (baseline / path).read_text() != originals[path] or (candidate / path).read_text() != patch(originals[path]):
            raise ValueError(f"Source does not match the pinned transformation: {path}")
    for path in PRIVATE_KERNELS:
        if path == NATIVE_ADAM:
            uses_packed_adam_reference(candidate)
        elif (candidate / path).read_bytes() != Path(__file__).with_name(path.name).read_bytes():
            raise ValueError(f"Candidate kernel differs from the checked-in source: {path}")
    return originals[MODEL]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", choices=("offline", "cpu", "cuda"), required=True)
    parser.add_argument("--ir-dir", type=Path, help="Export offline QKV TTIR, TTGIR and PTX")
    args = parser.parse_args()
    model = check_checkout(args.baseline, args.candidate)
    if args.mode == "cpu":
        import os
        os.environ["TRITON_INTERPRET"] = "1"
    import triton
    import triton.language as tl

    original_moments, = load_functions(args.baseline / NGRAM_ADAM, ("_adam_rows_kernel",), triton, tl)
    guarded_adam, = load_functions(args.candidate / NGRAM_ADAM, ("_adam_rows_kernel",), triton, tl)
    moment_kernel = load_candidate_scale(args.candidate / NGRAM_MOMENT_KERNEL)
    entries_kernel = load_candidate_scale(args.candidate / NGRAM_ENTRIES_KERNEL)
    value_adam_kernel = load_candidate_scale(args.candidate / VALUE_ADAM_KERNEL)
    value_reference, = load_functions(args.baseline / VALUE_EMBED, ("_row_scatter_add_kernel",), triton, tl)
    value_four = load_candidate_scale(args.candidate / VALUE_ACCUM_KERNEL)
    row_scatter, = load_functions(args.candidate / ROW_SCATTER, ("_scatter_add_rows_kernel",), triton, tl)

    ngram_names = ("_sign_rows", "_ngram_embed_fwd_kernel", "_ngram_embed_bwd_kernel")
    ref = load_functions(args.baseline / NGRAM, ngram_names, triton, tl)
    new = load_functions(args.candidate / NGRAM, ngram_names, triton, tl)
    old_amax, = load_functions(args.baseline / AMAX, ("_packed_batched_matrix_amax_partial_kernel",), triton, tl)
    new_amax, = load_functions(args.candidate / AMAX, ("_aligned_packed_amax",), triton, tl)
    _, qkv_scale, _, grouped_scan, grouped_scale = load_functions(
        args.candidate / AMAX, ("_qkv_scale_combine", "_qkv_scale_from_partials",
                               "_qkv_scan_bank", "_grouped_qkv_scan", "_grouped_qkv_scale"), triton, tl)
    old_pack, = load_functions(args.baseline / AMAX, ("_quantize_dual_layout_packed_batched_kernel",), triton, tl)
    new_pack, = load_functions(args.candidate / AMAX, ("_quantize_aligned_qkv_tiles",), triton, tl)
    scale = load_candidate_scale(args.candidate / KERNEL)
    mlp_path = args.candidate / WEIGHT_KERNEL
    old_head_path = args.baseline / CROSS_ENTROPY
    head_path = args.candidate / HEAD_KERNEL
    old_head, = load_functions(old_head_path, ("_quantize_lm_head_dual_kernel",), triton, tl)
    old_transpose, old_transpose_add = load_functions(
        args.baseline / TRANSPOSE, ("_transpose_copy_kernel", "_transpose_add_kernel"), triton, tl)
    byte_module = load_candidate_scale(args.candidate / TRANSPOSE_KERNEL)
    old_densify, = load_functions(args.baseline / SAMPLED_CE, ("_densify_kernel",), triton, tl)
    rope_names = ("_qk_norm_rope_forward_kernel", "_qkv_norm_rope_pack_fp8_backward_kernel")
    old_rope, old_rope_backward = load_functions(args.baseline / ROPE, rope_names, triton, tl)
    new_rope, new_rope_backward, _ = load_functions(
        args.candidate / ROPE, (*rope_names, "_rope_pair_factors"), triton, tl,
    )
    mlp_reference, = load_functions(args.baseline / "track_1_short/perf/kernels/mlp.py",
                                     ("_quantize_weights_dual_kernel",), triton, tl)
    _, mlp_delayed = load_functions(mlp_path, ("_mlp_scale_max", "_mlp_delayed_scales"), triton, tl)
    if args.mode == "offline":
        hopper = True
    else:
        import torch
        hopper = args.mode == "cuda" and torch.cuda.get_device_capability()[0] == 9
    if hopper:
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl

        gluon_pack, grouped_pack = load_functions(
            args.candidate / AMAX, ("_quantize_qkv_tiles_gluon", "_grouped_qkv_pack"),
            triton, tl, gluon=gluon, gl=gl)
        mlp_paired, = load_functions(mlp_path, ("_mlp_paired_pack",), triton, tl, gluon=gluon, gl=gl)
        _, new_head = load_functions(head_path, ("_pack_lm_head_tile", "_quantize_lm_head_gluon"),
                                     triton, tl, gluon=gluon, gl=gl)
        bf16_transpose, = load_functions(args.candidate / BF16_TRANSPOSE_KERNEL, ("_warp_bf16_transpose",),
                                          triton, tl, gluon=gluon, gl=gl)
        densify, = load_functions(args.candidate / DENSIFY_KERNEL, ("_densify_gluon",),
                                  triton, tl, gluon=gluon, gl=gl)
    else:
        gluon_pack, grouped_pack = None, None
        mlp_paired = None
        new_head = None
        bf16_transpose = None
        densify = None
    sampled_gather = load_candidate_scale(args.candidate / GATHER_KERNEL)
    grouped = (grouped_scan, grouped_scale, grouped_pack)
    mlp_weights = (mlp_reference, mlp_delayed, mlp_paired)
    lm_head = (old_head, new_head)
    if args.mode == "offline":
        single, = load_functions(args.baseline / LERP, ("_lerp_upcast_kernel",), triton, tl)
        merged, = load_functions(args.candidate / TAIL_KERNEL, ("_tail_lerp_pair",), triton, tl, gluon=gluon, gl=gl)
        norm = load_candidate_scale(args.candidate / TAIL_NORM_KERNEL)
        mapped_scatter, = load_functions(args.candidate / ROW_SCATTER, ("_scatter_mapped_rows_kernel",),
                                         triton, tl, gluon=gluon, gl=gl)
        sparse = load_candidate_scale(args.candidate / SPARSE_ROW_KERNEL)
        fetch = load_candidate_scale(args.candidate / ROW_FETCH_KERNEL)
        compile_sm90(scale, ref, new, old_amax, new_amax, qkv_scale, old_pack, new_pack, gluon_pack,
                     grouped, mlp_weights, lm_head, sampled_gather,
                     (old_transpose, old_transpose_add, byte_module, bf16_transpose),
                     (old_densify, densify), (old_rope, new_rope, old_rope_backward, new_rope_backward),
                     (single, merged, norm), sparse, fetch, (original_moments, guarded_adam, moment_kernel),
                     (entries_kernel, row_scatter, mapped_scatter, value_adam_kernel), (value_reference, value_four), triton, args.ir_dir)
        from check_fused_slot_stage import check_artifacts as check_graph_slots, load_graph_stage

        check_graph_slots(load_graph_stage(args.candidate), args.ir_dir)
    else:

        if args.mode == "cpu":
            use_cuda_rounding_in_cpu_interpreter(torch, tl)
        check_tail_lerp(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper)
        check_tail_norm(args.baseline, args.candidate, torch, args.mode, hopper)
        check_sparse_packets(args.baseline, args.candidate, torch, tl, args.mode, hopper)
        check_ngram_accumulation_host(args.baseline, args.candidate, torch)
        check_row_routes(args.baseline, args.candidate, torch)
        check_sparse_fetch(args.baseline, args.candidate, torch, args.mode, hopper)
        check_moment_replay(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                            (original_moments, guarded_adam), moment_kernel)
        check_adam_zero_rows(args.baseline, args.candidate, triton, tl, torch, args.mode,
                             (original_moments, guarded_adam))
        check_adam_entries(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper, entries_kernel)
        check_value_adam(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                         value_adam_kernel, row_scatter)
        check_value_embed_accum(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                                value_reference, value_four)
        check_scales(model, scale, torch, args.mode)
        check_mlp_weight_bank(mlp_path, triton, tl, torch, args.mode, hopper, mlp_weights)
        check_lm_head(old_head_path, head_path, triton, tl, torch, args.mode, hopper, lm_head)
        check_sampled_gather(args.baseline, args.candidate, torch, args.mode, hopper, sampled_gather)
        check_densify(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper, (old_densify, densify))
        rope_forward = check_rope_shift(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                                        (old_rope, new_rope))
        rope_backward = check_rope_backward(args.candidate, torch, triton, tl, args.mode, hopper,
                                            (old_rope_backward, new_rope_backward))
        check_yarn_tables(args.baseline, args.candidate, triton, tl, torch, args.mode,
                          (old_rope, old_rope_backward), rope_forward, rope_backward)
        transpose = check_transpose(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                                    byte_module, bf16_transpose)
        if args.mode == "cpu":
            check_fullgraph_cuda_metadata(torch, rope_forward, rope_backward, transpose)
        check_ngrams(ref, new, triton, torch, args.mode)
        check_qkv_scales(qkv_scale, torch, args.mode)
        check_qkv_scale_dispatch(args.candidate / AMAX, qkv_scale, triton, tl, torch, args.mode)
        check_amax(old_amax, new_amax, torch, args.mode)
        check_native_qkv(args.baseline, args.candidate, triton, tl, torch, args.mode, hopper,
                         old_amax, new_amax, old_pack, qkv_scale, gluon_pack, grouped)
        packers = [("aligned Triton", new_pack)]
        if hopper:
            packers.append(("explicit Hopper layout", gluon_pack))
        check_qkv_pack(old_pack, packers, torch, args.mode)
        check_qkv_pack_dispatch(args.candidate / AMAX, triton, tl, torch, args.mode)
