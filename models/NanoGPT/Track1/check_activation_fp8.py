"""Compile the installed training-head quantizer and replay its Hopper stores."""
import argparse
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from export_source_link import lm_head_scales
from install_candidate import ACTIVATION_KERNEL, CROSS_ENTROPY, MODEL, SAMPLED_CE
from qkv_ptx_replay import INT32_OPS, operands_in, parse
from verify_candidate import check_checkout

BASE_X, BASE_Y = 1 << 32, 2 << 32
WORD_MASK = np.uint64(2**32 - 1)
FP8 = np.arange(127, dtype=np.uint8)
LEVELS = torch.from_numpy(FP8).view(torch.float8_e4m3fn).float().numpy()
MIDPOINTS = (LEVELS[:-1] + LEVELS[1:]) * np.float32(0.5)


def widen(word):
    return (np.asarray(word, dtype=np.uint32) << 16).view(np.float32)


def fpword(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)


def narrow(word):
    bits = np.asarray(word, dtype=np.uint32)
    nan = (bits & 0x7fffffff) > 0x7f800000
    rounded = ((bits + 0x7fff + (bits >> 16 & 1)) >> 16).astype(np.uint16)
    # Model the common PTX narrowing of NaNs in both programs.
    return np.where(nan, 0x7fff, rounded).astype(np.uint64)


def saturate(word):
    x = np.asarray(word, dtype=np.uint32).view(np.float32)
    mag = np.minimum(np.abs(x), np.float32(448))
    k = np.searchsorted(MIDPOINTS, mag, side="left")
    tie = (k < len(MIDPOINTS)) & (MIDPOINTS[np.minimum(k, len(MIDPOINTS) - 1)] == mag)
    k += tie & (k % 2 == 1)
    return (k.astype(np.uint64) | np.signbit(x).astype(np.uint64) << 7)


def permute(a, b, sel):
    pair = a | (b << np.uint64(32))
    return sum(
        (pair >> np.uint64(8 * (sel >> (i * 4) & 7)) & np.uint64(255)) << np.uint64(8 * i)
        for i in range(4)
    )


def program(path):
    instruction_lines = []
    inside_function = False
    for line in path.read_text().splitlines():
        code = " ".join(line.partition("//")[0].split())
        if code == ".reqntid 128":
            inside_function = True
        if inside_function and code.endswith(";") and not code.startswith("."):
            instruction_lines.append(code)
        if inside_function and code == "}":
            break
    return list(parse(instruction_lines))


def replay(path, inputs):
    lanes = np.arange(len(inputs) // 4, dtype=np.uint64)
    out = np.empty(len(inputs), dtype=np.uint8)
    writes = np.zeros(len(inputs), dtype=np.uint8)
    regs = {"%tid.x": lanes % 128, "%ctaid.x": lanes // 128}
    def val(name):
        if name.startswith("%"):
            return regs[name]
        if name.startswith("0f"):
            return np.uint64(int(name[2:], 16))
        return np.uint64(int(name.removesuffix("U"), 0))
    def ptr(token):
        inside = token[1:-1].replace(" ", "").split("+")
        return val(inside[0]) + np.uint64(inside[1] if len(inside) > 1 else "0")
    for pred, op, args in program(path):
        assert pred is None
        if op == "ret":
            break
        dst, *src = args
        if op == "ld.param.b64":
            base = {"[_activation_fp8_param_0]": BASE_X, "[_activation_fp8_param_1]": BASE_Y}
            regs[dst] = np.uint64(base[src[0]])
        elif op in INT32_OPS:
            regs[dst] = INT32_OPS[op](*map(val, src)) & WORD_MASK
        elif op == "mov.b32":
            lo, hi = operands_in(dst)
            x = val(src[0])
            regs[lo] = x & np.uint64(65535)
            regs[hi] = x >> np.uint64(16)
        elif op in ("mov.u32", "cvt.s64.s32"):
            regs[dst] = val(src[0])
        elif op == "mad.wide.s32":
            regs[dst] = val(src[0]) * val(src[1]) + val(src[2])
        elif op == "add.s64":
            regs[dst] = val(src[0]) + val(src[1])
        elif op == "ld.global.v2.b32":
            first = (ptr(src[0]) - np.uint64(BASE_X)) // np.uint64(2)
            assert np.all(first % 4 == 0)
            lo, hi = operands_in(dst)
            regs[lo] = inputs[first].astype(np.uint64) | (inputs[first + 1].astype(np.uint64) << 16)
            regs[hi] = inputs[first + 2].astype(np.uint64) | (inputs[first + 3].astype(np.uint64) << 16)
        elif op == "st.global.b32":
            first = ptr(dst) - np.uint64(BASE_Y)
            x = val(operands_in(src[0])[0])
            for i in range(4):
                out[first + i] = ((x >> (i * 8)) & 255).astype(np.uint8)
                np.add.at(writes, first + i, 1)
        elif op == "cvt.f32.bf16":
            regs[dst] = fpword(widen(val(src[0])))
        elif op == "mul.f32":
            a, b = (np.asarray(val(t), dtype=np.uint32).view(np.float32) for t in src)
            with np.errstate(over="ignore", invalid="ignore"):
                regs[dst] = fpword(a * b)
        elif op == "cvt.rn.bf16.f32":
            regs[dst] = narrow(val(src[0]))
        elif op == "cvt.rn.satfinite.e4m3x2.f32":
            regs[dst] = (saturate(val(src[0])) << 8) | saturate(val(src[1]))
        elif op == "prmt.b32":
            regs[dst] = permute(val(src[0]), val(src[1]), int(val(src[2])))
        elif op == "abs.f32":
            regs[dst] = val(src[0]) & np.uint64(0x7fffffff)
        elif op == "setp.le.f32":
            regs[dst] = np.asarray(val(src[0]), dtype=np.uint32).view(np.float32) <= (
                np.asarray(val(src[1]), dtype=np.uint32).view(np.float32))
        elif op == "shr.s16":
            regs[dst] = (val(src[0]).astype(np.int16) >> int(val(src[1]))).astype(np.uint16).astype(np.uint64)
        elif op == "cvt.u32.u16":
            regs[dst] = val(src[0]) & np.uint64(65535)
        elif op == "selp.b32":
            regs[dst] = np.where(val(src[2]), val(src[0]), val(src[1]))
        else:
            raise RuntimeError((op, args))
    assert np.all(writes == 1)
    return out


def check_launch(source, scale):
    tree = ast.parse(source.read_text(), filename=str(source))
    launcher, = (node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "quantize_activation")
    launches = []
    eager = []
    encoded = SimpleNamespace(view=lambda dtype: ("bytes", dtype))

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))

    class Tensor:
        def __init__(self, size, **changes):
            self.size = size
            self.dtype = torch.bfloat16
            self.device = "cuda:0"
            self.is_cuda = True
            self.requires_grad = True
            self.contiguous = True
            self.negative = False
            self.address = 16
            vars(self).update(changes)

        def numel(self):
            return self.size

        def is_contiguous(self):
            return self.contiguous

        def is_neg(self):
            return self.negative

        def data_ptr(self):
            return self.address

        def div(self, value):
            eager.append(("div", value))
            return self

        def to(self, dtype):
            eager.append(("cast", dtype))
            return "eager"

    mock = SimpleNamespace(
        bfloat16=torch.bfloat16, float8_e4m3fn=torch.float8_e4m3fn, uint8=torch.uint8,
        compiler=SimpleNamespace(is_compiling=lambda: False),
        cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0)),
        is_grad_enabled=lambda: False,
        empty_like=lambda x, dtype: encoded if dtype == torch.float8_e4m3fn else None,
    )
    environment = dict(torch=mock, ACTIVATION_SCALE=scale, BLOCK=512, _activation_fp8=Kernel())
    exec(compile(ast.Module(body=[launcher], type_ignores=[]), str(source), "exec"), environment)
    quantize = environment["quantize_activation"]
    for tokens in (16384, 32768, 40960, 49152):
        x = Tensor(tokens * 768)
        assert quantize(x, scale) is encoded
        grid, args, options = launches.pop()
        assert grid == (tokens * 768 // 512,)
        assert args == (x, ("bytes", torch.uint8), 1 / scale, 512)
        assert options == dict(num_warps=4, num_stages=1)
    for tensor, scalar in (
        (Tensor(512), np.nextafter(scale, 1)), (Tensor(512, dtype=torch.float32), scale),
        (Tensor(512, is_cuda=False), scale), (Tensor(512, contiguous=False), scale),
        (Tensor(512, negative=True), scale), (Tensor(512, address=18), scale),
        (Tensor(0), scale), (Tensor(513), scale), (Tensor(2**31), scale),
    ):
        assert quantize(tensor, scalar) == "eager"
        assert eager[-2:] == [("div", scalar), ("cast", torch.float8_e4m3fn)]
    mock.is_grad_enabled = lambda: True
    assert quantize(Tensor(512), scale) == "eager"
    mock.is_grad_enabled = lambda: False
    mock.cuda.get_device_capability = lambda device: (8, 0)
    assert quantize(Tensor(512), scale) == "eager"
    mock.compiler.is_compiling = lambda: True
    mock.cuda.get_device_capability = lambda device: (9, 0)

    class CompiledTensor(Tensor):
        def numel(self):
            raise AssertionError("Inductor must receive the two original ATen operations")

    assert quantize(CompiledTensor(512), scale) == "eager"
    assert eager[-2:] == [("div", scale), ("cast", torch.float8_e4m3fn)]
    assert not launches


def load_quantizer(source):
    spec = importlib.util.spec_from_file_location("activation_fp8", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_quantizer(source, output, scale):
    module = load_quantizer(source)
    assert module.ACTIVATION_SCALE == scale and module.BLOCK == 512
    signature = {"X": "*bf16", "Y": "*u8", "GAIN": "constexpr", "TILE": "constexpr"}
    aligned = {(i,): [["tt.divisibility", 16]] for i in range(2)}
    src = ASTSource(module._activation_fp8, signature, dict(GAIN=1 / scale, TILE=module.BLOCK), attrs=aligned)
    code = triton.compile(src, target=GPUTarget("cuda", 90, 32), options=dict(num_warps=4, num_stages=1))
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("ttgir", "ptx", "cubin"):
        value = code.asm[extension]
        (output / f"activation-fp8.{extension}").write_bytes(value if isinstance(value, bytes) else value.encode())
    return output / "activation-fp8.ptx"


def check_inductor(baseline, candidate, output, scale):
    from check_value_adam_inductor import TORCH_REVISION, capture

    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None:
        raise RuntimeError("Use the pinned CPU build of Torch 2.10.0 for offline Inductor capture")

    @torch.library.custom_op("veritile_probe::head_input", mutates_args=())
    def head_input(x: torch.Tensor) -> torch.Tensor:
        return x.new_empty(x.shape[:-1], dtype=torch.float32)

    @head_input.register_fake
    def head_input_fake(x):
        return x.new_empty(x.shape[:-1], dtype=torch.float32)

    def capture_wrapper(checkout, path, cls, tag, tokens, autograd):
        tree = ast.parse((checkout / path).read_text())
        wrapper, = (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        forward, = (node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        quantize, = (node for node in forward.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "x_f8" for target in node.targets))
        layers = (checkout / MODEL).parent / "layers.py"
        norm, = (node for node in ast.parse(layers.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "norm")
        scope = dict(torch=torch, Tensor=torch.Tensor, F=torch.nn.functional, x_s=scale,
                     quantize_activation=load_quantizer(candidate / ACTIVATION_KERNEL).quantize_activation,
                     head_input=head_input)
        # Keep the FP8 store and saved RMS statistic live without loading cuBLAS.
        cast = ast.unparse(quantize)
        if autograd:
            probe = ast.parse(f"""
class Head(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        {cast}
        ctx.save_for_backward(x_f8)
        return head_input(x_f8)

    @staticmethod
    def backward(ctx, upstream):
        x_f8, = ctx.saved_tensors
        return x_f8.to(torch.bfloat16) * upstream.reshape(-1, 1).to(torch.bfloat16)

def probe(hidden):
    return Head.apply(norm(hidden).view(-1, 768)).sum()
""").body
        else:
            probe = ast.parse(f"""
def probe(hidden):
    x = norm(hidden).view(-1, 768)
    {cast}
    return head_input(x_f8)
""").body
        exec(compile(ast.Module([norm, *probe], []), str(layers), "exec"), scope)
        artifact = output / f"{tag}-{tokens}.py"
        hidden = torch.empty((1, tokens, 768), dtype=torch.bfloat16, requires_grad=autograd)
        capture(scope["probe"], artifact, [hidden])
        return [ast.literal_eval(node.args[1]) for node in ast.walk(ast.parse(artifact.read_text()))
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "async_compile.triton"]

    for path, cls, tag in ((CROSS_ENTROPY, "FusedSoftcappedCrossEntropy", "full"),
                           (SAMPLED_CE, "SampledSoftcappedCrossEntropy", "sampled")):
        for tokens in (16384, 32768, 40960, 49152):
            for autograd in (False, True):
                mode = f"{tag}-{'grad' if autograd else 'forward'}"
                reference = capture_wrapper(baseline, path, cls, f"{mode}-reference", tokens, autograd)
                optimized = capture_wrapper(candidate, path, cls, f"{mode}-candidate", tokens, autograd)
                assert reference == optimized and optimized
                assert any("tl.sum" in k and "libdevice.rsqrt" in k for k in optimized)
                assert any("tl.float8e4nv" in k for k in optimized)
    print("Both head wrappers preserve Inductor's RMSNorm and FP8 stores at all four token counts, with and without autograd.")


def check_cast(ptx, gain):
    inputs = np.random.default_rng(2026).permutation(np.arange(65536, dtype=np.uint16))
    reference = torch.from_numpy(inputs.view(np.int16)).view(torch.bfloat16)
    reference = (reference.float() * np.float32(gain)).to(torch.bfloat16)
    reference = reference.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    out = replay(ptx, inputs)
    # On CPU, C10 may canonicalize the scaled NaN differently from
    # cvt.rn.bf16.f32. Both CUDA wrappers call that same narrowing.
    nan = (inputs & 0x7fff) > 0x7f80
    assert np.array_equal(out[~nan], reference[~nan]), inputs[(out != reference) & ~nan][:20]
    same_cuda_narrowing = reference.copy()
    same_cuda_narrowing[nan] = torch.tensor(0x7fff, dtype=torch.int16).view(torch.bfloat16).to(
        torch.float8_e4m3fn).view(torch.uint8).item()
    assert np.array_equal(out, same_cuda_narrowing)
    print("Compiled vector instructions store each byte once and match 65282 non-NaN BF16 inputs.")
    print("The remaining 254 input encodings agree when both use the common CUDA BF16 narrowing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check-inductor", action="store_true")
    args = parser.parse_args()
    check_checkout(args.baseline, args.candidate)
    _, scales = lm_head_scales((args.baseline / MODEL).read_text())
    source = args.candidate / ACTIVATION_KERNEL
    check_launch(source, scales["x_s"])
    ptx = compile_quantizer(source, args.output, scales["x_s"])
    check_cast(ptx, 1 / scales["x_s"])
    if args.check_inductor:
        check_inductor(args.baseline, args.candidate, args.output, scales["x_s"])
