"""Capture Track 1 value Adam with CPU AOT, then lower and replay for Hopper."""
import argparse
import ast
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from importlib.util import module_from_spec, spec_from_file_location
from unittest.mock import patch

import torch
from torch._dynamo.device_interface import CudaInterface, caching_worker_device_properties
from torch._dynamo.exc import BackendCompilerFailed
from torch._guards import TracingContext, tracing
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.graph import GraphLowering
from torch._inductor.runtime import triton_heuristics
from torch._subclasses import fake_tensor
from torch.utils._triton import has_triton
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from install_candidate import NATIVE_ADAM, PACKED_ADAM_REFERENCE, uses_packed_adam_reference
from value_adam_replay import ELEMENTS, check_tile
from verify_candidate import ANVIL, load_sparse_method


TORCH_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
HOPPER = GPUTarget("cuda", 90, 32)


class CaptureComplete(Exception):
    pass


def capture(fn, artifact, args, *, user_triton=False, cpu_scalar_arguments=False):
    """Retarget AOT inputs and tensor factories, then stop before loading GPU code."""
    props = SimpleNamespace(name="Hopper", major=9, minor=0, multi_processor_count=132,
                            regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048,
                            max_threads_per_block=1024, warp_size=32, total_memory=80 << 30)
    driver = SimpleNamespace(get_current_target=lambda: HOPPER, get_current_device=lambda: 0,
                             get_current_stream=lambda _: 0)
    tensor_copy = torch.Tensor.copy_

    def copy_with_fake_cuda(dst, src, *args, **kwargs):
        # Reinplace checks a view copy on fake CUDA tensors. The CPU wheel has
        # no CUDA copy kernel, and for this shape-preserving copy the fake
        # destination already describes the result and its alias.
        if (isinstance(dst, fake_tensor.FakeTensor) and isinstance(src, fake_tensor.FakeTensor)
                and dst.device.type == "cuda" and dst.device == src.device
                and dst.shape == src.shape and dst.dtype == src.dtype):
            return dst
        return tensor_copy(dst, src, *args, **kwargs)

    def stop_at_codegen(lowering):
        wrapper, kernel = lowering.codegen()
        artifact.write_text(wrapper.value + "\n" + kernel.value)
        raise CaptureComplete()

    def lower_for_hopper(graph, inputs, **kwargs):
        mode = fake_tensor.FakeTensorMode()
        storage = {}

        def retarget(obj):
            if isinstance(obj, torch.Tensor):
                original = obj.untyped_storage()
                bytes_ = original.nbytes()
                if original._cdata not in storage:
                    storage[original._cdata] = torch.empty(bytes_, device="meta", dtype=torch.uint8)
                base = storage[original._cdata]
                size = bytes_ - bytes_ % obj.element_size()
                metadata = base[:size].view(obj.dtype).as_strided(obj.shape, obj.stride(), obj.storage_offset())
                device = torch.device("cuda", 0)
                if cpu_scalar_arguments and obj.ndim == 0:
                    device = torch.device("cpu")
                return fake_tensor.FakeTensor(mode, metadata, device)
            if isinstance(obj, (tuple, list)):
                return type(obj)(map(retarget, obj))
            if isinstance(obj, dict):
                return type(obj)((key, retarget(value)) for key, value in obj.items())
            return obj

        for node in graph.graph.nodes:
            if node.kwargs.get("device") == torch.device("cpu"):
                node.kwargs = {**node.kwargs, "device": torch.device("cuda", 0)}
            for key in ("val", "example_value", "eager_input_vals"):
                if key in node.meta:
                    node.meta[key] = retarget(node.meta[key])
        with ExitStack() as stack:
            stack.enter_context(tracing(TracingContext(mode)))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            if user_triton:
                stack.enter_context(patch.object(torch.Tensor, "copy_", copy_with_fake_cuda))
            has_triton.cache_clear()
            try:
                return compile_fx_inner(graph, retarget(inputs), **kwargs)
            finally:
                has_triton.cache_clear()

    def backend(graph, inputs):
        # The AOT graph before retargeting still has CPU inputs and factories.
        device_scope = patch.object(torch.cuda, "is_available", return_value=False) if user_triton else nullcontext()
        with device_scope:
            return compile_fx(graph, inputs, inner_compile=lower_for_hopper)

    torch._dynamo.reset()
    with ExitStack() as stack:
        overrides = [
            (torch.version, "cuda", "12.8"),
            (torch.cuda, "is_initialized", lambda: True),
            (torch.cuda, "get_device_properties", lambda *_: props),
            (torch.cuda, "get_device_capability", lambda *_: (9, 0)),
            (torch.cuda, "device_count", lambda: 1),
            (torch.cuda, "current_device", lambda: 0),
            (torch.cuda, "_exchange_device", lambda *_: 0),
            (torch.cuda, "_maybe_exchange_device", lambda *_: 0),
            (torch.cuda, "get_rng_state", lambda *_: torch.get_rng_state()),
            (torch.cuda, "set_rng_state", lambda *_: None),
            (CudaInterface, "is_bf16_supported", lambda *_, **__: True),
            (CudaInterface, "get_device_properties", lambda *_: props),
            (CudaInterface, "current_device", lambda: 0),
            (triton.runtime.driver, "_active", driver),
            (GraphLowering, "compile_to_module", stop_at_codegen),
            (fake_tensor, "init_gpu_context", lambda *_: None),
        ]
        for obj, attr, setting in overrides:
            stack.enter_context(patch.object(obj, attr, setting))
        stack.enter_context(patch.dict(caching_worker_device_properties, {"cuda": [props]}))
        stack.enter_context(torch._inductor.config.patch(force_disable_caches=True))
        if user_triton:
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            has_triton.cache_clear()
        CudaInterface.Worker.set_device(0)
        try:
            torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)(*args)
        except CaptureComplete:
            pass
        except BackendCompilerFailed as failure:
            if not isinstance(failure.inner_exception, CaptureComplete):
                raise
        else:
            raise RuntimeError("Inductor did not reach the CUDA code-capture hook")
        finally:
            if user_triton:
                has_triton.cache_clear()


def build(artifact, input_names):
    """Compile every advertised default configuration without executing the wrapper."""
    tree = ast.parse(artifact.read_text())
    call, = (n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "async_compile.triton")
    name, source = map(ast.literal_eval, call.args[:2])
    launch, = (n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == name + ".run")
    assert ast.literal_eval(launch.args[-1]) == ELEMENTS
    def arg_name(arg):
        if isinstance(arg, ast.Call):
            assert isinstance(arg.func, ast.Attribute) and arg.func.attr == "item"
            assert not arg.args and not arg.keywords
            arg = arg.func.value
        assert isinstance(arg, ast.Name)
        return input_names[int(arg.id.removeprefix("arg").removesuffix("_1"))]
    order = tuple(map(arg_name, launch.args[:-1]))

    fn, = (n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    heuristics, jit = fn.decorator_list
    assert ast.unparse(heuristics.func) == "triton_heuristics.pointwise"
    assert ast.unparse(jit) == "triton.jit"
    kwargs = {k.arg: k.value for k in heuristics.keywords}
    metadata_dict = kwargs.pop("triton_meta")
    metadata = {ast.literal_eval(k): ast.literal_eval(v)
                for k, v in zip(metadata_dict.keys, metadata_dict.values, strict=True)
                if ast.literal_eval(k) != "device"}
    kwargs = {k: ast.literal_eval(v) for k, v in kwargs.items() if k != "filename"}
    with patch.object(triton_heuristics, "cached_autotune", side_effect=lambda _, configs, **kw: configs):
        configs = triton_heuristics.pointwise(triton_meta={"device": None}, **kwargs)

    fn.decorator_list = [jit]
    kernel_src = artifact.with_name(artifact.stem + "-triton.py")
    kernel_src.write_text("import triton\nimport triton.language as tl\n\n" + ast.unparse(fn) + "\n")
    module_spec = spec_from_file_location(kernel_src.stem, kernel_src)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, name)
    signature = metadata["signature"]
    assert len(order) + 2 == len(signature)
    for config in configs:
        options = {"num_warps": config.num_warps, "num_stages": config.num_stages,
                   "enable_fp_fusion": metadata["enable_fp_fusion"], "sanitize_overflow": False,
                   "debug": kwargs["inductor_meta"]["assert_indirect_indexing"]}
        code = triton.compile(ASTSource(kernel, signature, config.kwargs, metadata["configs"][0]),
                              target=HOPPER, options=options)
        block = config.kwargs["XBLOCK"]
        ptx = code.asm["ptx"]
        artifact.with_suffix(f".{block}.ptx").write_text(ptx)
        artifact.with_suffix(f".{block}.json").write_text(json.dumps(code.metadata._asdict(), default=vars))
        artifact.with_suffix(f".{block}.ttgir").write_text(code.asm["ttgir"])
        artifact.with_suffix(f".{block}.cubin").write_bytes(code.asm["cubin"])
        yield block, config.num_warps, order, ptx


def build_installed_candidate(root, out):
    build_consumer(root / NATIVE_ADAM, out)


def build_consumer(kernel_path, out):
    from triton.experimental.gluon._runtime import GluonASTSource

    out.mkdir(parents=True, exist_ok=True)
    spec = spec_from_file_location("value_adam_read_scale", kernel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module._update_with_scaled_read
    args = "*fp32 *fp16 *fp32 fp32 *bf16 fp32 *fp32 *fp32 *bf16"
    args += " constexpr constexpr constexpr constexpr constexpr constexpr"
    signature = dict(zip(kernel.arg_names, args.split(), strict=True))
    aligned = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 4, 6, 7, 8)}
    for period, (beta1, beta2) in module.BETAS.items():
        for tile, warps in module.PROFILE_WARPS.items():
            for clear_all, mode in ((False, "masked"), (True, "dense")):
                constants = dict(BLOCK=tile, BETA1=beta1, BETA2=beta2, FIRST_MIX=1.0 - beta1,
                                 SECOND_MIX=1.0 - beta2, CLEAR_ALL=clear_all)
                native = GluonASTSource(kernel, signature, constants, aligned)
                code = triton.compile(native, target=HOPPER,
                                      options=dict(num_warps=warps, num_stages=1, enable_fp_fusion=False))
                artifact = out / f"consume-{mode}-{period}-{tile}"
                artifact.with_suffix(".ptx").write_text(code.asm["ptx"])
                artifact.with_suffix(".ttgir").write_text(code.asm["ttgir"])
                artifact.with_suffix(".cubin").write_bytes(code.asm["cubin"])
                artifact.with_suffix(".json").write_text(json.dumps(code.metadata._asdict(), default=vars))
                print(f"Compiled installed {mode} consumer, period {period}, tile {tile}.")


def build_installed_scatter(root, out):
    from triton.experimental.gluon._runtime import GluonASTSource

    kernel_path = root / "track_1_short/perf/kernels/value_adam_scatter.py"
    spec = spec_from_file_location("installed_value_adam_scatter", kernel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.local_rows
    arguments = "*fp16 *i32 *fp16 i32 i32 i32 *i32 constexpr"
    signature = dict(zip(kernel.arg_names, arguments.split(), strict=True))
    attributes = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 6)}
    for direct in (False, True):
        for marked in (False, True):
            constants = dict(ASSEMBLY=module._LOCAL_ASSEMBLY[direct, marked])
            artifact = triton.compile(GluonASTSource(kernel, signature, constants, attributes),
                                      target=HOPPER, options=dict(num_warps=4, num_stages=1))
            mode = "direct" if direct else "staged"
            stem = out / f"value-local-{mode}-{int(marked)}"
            for ext in ("ptx", "ttgir"):
                stem.with_suffix("." + ext).write_text(artifact.asm[ext])
            stem.with_suffix(".cubin").write_bytes(artifact.asm["cubin"])
            stem.with_suffix(".json").write_text(json.dumps(artifact.metadata._asdict(), default=vars))
            print(f"Compiled installed {mode} local scatter, markers {marked}.")


def check(root, out):
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None:
        raise RuntimeError("Use the pinned CPU build of Torch 2.10.0 for offline AOT capture")
    if triton.__version__ != "3.6.0":
        raise RuntimeError("Use Triton 3.6.0 to reproduce the Hopper artifacts")
    reference = load_sparse_method(root, ANVIL, "AnvilAndAdam", "_adam_update_step", strip_decorators=True)

    def fused_scale(p, g, m, v, beta1, beta2, eps, step, wd):
        rounded = (g.to(torch.float32) * 0.125).to(torch.float16)
        # A dtype view keeps the binary16 rounding alive in default codegen.
        restored = torch.bitwise_or(rounded.view(torch.int16), 0).view(torch.float16)
        g.copy_(rounded)
        reference(p, restored, m, v, beta1, beta2, eps, step, wd)

    out.mkdir(parents=True, exist_ok=True)
    for period in (2, 4):
        for tag, fn, names in (("dense", reference, "p m g v step wd"),
                               ("fused-scale", fused_scale, "g p m v step wd")):
            args = [torch.empty((25152, 768), dtype=dt) for dt in
                    (torch.bfloat16, torch.float16, torch.float32, torch.float32)]
            args += [0.75 ** (period // 2), 0.95 ** (period // 2), 1e-10,
                     torch.empty((), dtype=torch.float32), torch.empty((), dtype=torch.float32)]
            artifact = out / f"{tag}-period{period}.py"
            capture(fn, artifact, args, cpu_scalar_arguments=True)
            checked = set()
            for block, warps, order, ptx in build(artifact, names.split()):
                expected_warps = {512: 8, 1024: 4}[block]
                assert warps == expected_warps and ELEMENTS % block == 0
                expected = ("decay",) * block
                for cta in (0, 1, ELEMENTS // block - 1):
                    measured = check_tile(ptx, order, period, block, warps, cta, tag == "fused-scale", by_value=True)
                    assert measured == expected, (tag, period, block, cta)
                checked.add(block)
                print(f"{tag} period {period} tile {block}, {warps} warps: decay FMA in all lanes")
            assert checked == {512, 1024}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", type=Path, help="compile the installed value Adam consumer")
    args = parser.parse_args()
    check(args.baseline, args.output_dir)
    if args.candidate is not None:
        if uses_packed_adam_reference(args.candidate):
            build_installed_candidate(args.candidate, args.output_dir)
        else:
            build_consumer(PACKED_ADAM_REFERENCE, args.output_dir)
            build_installed_candidate(args.candidate, args.output_dir / "value-adam-guarded")
        build_installed_scatter(args.candidate, args.output_dir)
