"""Compile and replay the installed training cross-entropy without a GPU."""
import argparse
import ast
import ctypes
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from export_source_link import lm_head_gradient_scale
from install_candidate import CONFIG, CROSS_ENTROPY, MODEL, SAMPLED, SAMPLED_CE
from training_ce_prediction_replay import DefaultSoftcapPrefix, Program, verify_specialized_ce
from training_ce_replay import FACTOR_SYMBOL, split, verify_training_ce
from verify_candidate import check_checkout

WIDTHS = (10240, 14336, 24576, 50304)
FIXED = ((10240, 2), (10240, 3), (14336, 1), (24576, 1), (50304, 1))
# PyTorch 2.10's _get_gpu_rtc_compatible_flags removes relaxed constexpr.
TORCH_NVRTC_FLAGS = (
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
)
SASS_INSTRUCTION = re.compile(r"\s*/\*[0-9a-f]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_]*)\b")


def capture_source(path, width, predictions=None):
    """Run just the pinned Python compile wrapper with a recording backend."""
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = {"CE_KERNEL_BLOCK_SIZE", "CE_KERNEL_VOCAB_SIZE", "CE_FIXED_COUNTS", "CE_KERNEL_SOURCE"}
    seen = {}

    class Compiler:
        def _compile_kernel(self, source, name, *, compute_capability, cuda_include_dirs, nvcc_options):
            seen.update(source=source, name=name, arch=compute_capability, options=nvcc_options)
            return self

        def set_shared_memory_config(self, extent):
            seen["shared_bytes"] = extent

    class Torch:
        cuda = Compiler()

    env = dict(torch=Torch, CUDA_INCLUDE_DIRS=[])
    tree.body = [node for node in tree.body
                 if (isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))
                 or (isinstance(node, ast.FunctionDef) and node.name == "compile_ce_kernel")]
    exec(compile(tree, str(path), "exec"), env)
    if predictions is None:
        env["compile_ce_kernel"](width)
    else:
        env["compile_ce_kernel"](width, predictions)
    table = (width, predictions) in FIXED
    assert ("#define CE_USE_LOGIT_TABLE\n" in seen["source"]) == table
    assert ("#define CE_CACHE_MTP_TARGETS\n" in seen["source"]) == (table and width == 10240)
    assert ("#define CE_DEFAULT_BC\n" in seen["source"]) == table
    assert ("#define CE_DEFAULT_A\n" in seen["source"]) == (table and width != WIDTHS[-1])
    assert seen["shared_bytes"] == width * (1 if table else 2)
    return seen


def emit_ptx(lib, launch, includes):
    name = launch["name"].encode()
    program = ctypes.c_void_p()

    def check(result, call):
        if result:
            raise RuntimeError(f"NVRTC {call} failed with status {result}")

    check(lib.nvrtcCreateProgram(ctypes.byref(program), launch["source"].encode(),
                                name + b".cu", 0, None, None), "create")
    try:
        check(lib.nvrtcAddNameExpression(program, name), "name")
        options = [f"--gpu-architecture=sm_{launch['arch']}", *(f"-I{p}" for p in includes),
                   *launch["options"], *TORCH_NVRTC_FLAGS]
        flags = (ctypes.c_char_p * len(options))(*(opt.encode() for opt in options))
        status = lib.nvrtcCompileProgram(program, len(options), flags)
        if status:
            count = ctypes.c_size_t()
            check(lib.nvrtcGetProgramLogSize(program, ctypes.byref(count)), "log size")
            log = ctypes.create_string_buffer(count.value)
            check(lib.nvrtcGetProgramLog(program, log), "log")
            raise RuntimeError(log.value.decode())
        count = ctypes.c_size_t()
        check(lib.nvrtcGetPTXSize(program, ctypes.byref(count)), "PTX size")
        ptx = ctypes.create_string_buffer(count.value)
        check(lib.nvrtcGetPTX(program, ptx), "PTX")
        return ptx.value.decode()
    finally:
        check(lib.nvrtcDestroyProgram(ctypes.byref(program)), "destroy")


def check_dispatch(candidate):
    """Exercise the installed eager custom-op bodies with recorded launches."""
    launches = []

    def compile_ce_kernel(width, predictions=None):
        def kernel(grid, block, params, *, shared_mem):
            launches.append((width, predictions, grid, block, params, shared_mem))
        return kernel

    env = dict(torch=SimpleNamespace(Tensor=object), compile_ce_kernel=compile_ce_kernel,
               ALL_CANDIDATE_COUNTS=WIDTHS[:-1])
    for path, names, functions in (
        (CROSS_ENTROPY, {"CE_KERNEL_BLOCK_SIZE", "CE_KERNEL_VOCAB_SIZE", "CE_FIXED_COUNTS",
                        "ce_fwd_bwd_kernel", "CE_FIXED_KERNELS", "FULL_CE_FIXED_KERNELS", "SOFTCAP_A",
                        "LM_HEAD_GRAD_SCALE"}, {"ce_fwd_bwd", "ce_reuse_logits", "_ce_forward_buffers"}),
        (SAMPLED_CE, {"SAMPLED_CE_KERNELS", "SAMPLED_CE_FIXED_KERNELS"}, {"sampled_ce_fwd_bwd"}),
    ):
        source = candidate / path
        module = ast.parse(source.read_text(), filename=str(source))
        module.body = [node for node in module.body
                       if (isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id in names
                                   or isinstance(t, ast.Tuple)
                                   and any(isinstance(x, ast.Name) and x.id in names for x in t.elts)
                                   for t in node.targets))
                       or isinstance(node, ast.FunctionDef) and node.name in functions]
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
        exec(compile(module, str(source), "exec"), env)
    defaults = (23.0, 5.0, 7.5)
    assert tuple(env[name] for name in ("SOFTCAP_A", "SOFTCAP_B", "SOFTCAP_C")) == defaults
    _, grad_scale = lm_head_gradient_scale((candidate / MODEL).read_text())
    assert env["LM_HEAD_GRAD_SCALE"] == grad_scale
    scales = (grad_scale, math.nextafter(grad_scale, math.inf), grad_scale * 2, 0.0, math.nan, math.inf)
    cases = (defaults, (22.0, 5.0, 7.5), (23.0, 5.125, 7.5), (23.0, 5.0, 7.625),
             (23.0, float("nan"), 7.5), (float("inf"), 5.0, 7.5), (23.0, 5.0, 7.500000000000001))
    for width in WIDTHS:
        dispatch = env["ce_fwd_bwd" if width == WIDTHS[-1] else "sampled_ce_fwd_bwd"]
        for count in range(5):
            buffers = (SimpleNamespace(shape=(17, width)), *(object() for _ in range(6)))
            for softcap in cases:
                for scale in scales:
                    arguments = (*buffers, 17, count, *softcap, scale)
                    dispatch(*arguments)
                    default_bc = softcap[1:] == defaults[1:]
                    accepted = (default_bc and (width == WIDTHS[-1] or softcap[0] == defaults[0])
                                and scale == grad_scale)
                    fixed = count if (width, count) in FIXED and accepted else None
                    shared = width if fixed is not None else width * 2
                    assert launches.pop() == (width, fixed, (17, 1, 1), (256, 1, 1), arguments, shared)


    class Buffer:
        def __init__(self, shape, dtype, storage=None):
            self.shape = (shape,) if isinstance(shape, int) else shape
            self.dtype = dtype
            self.device = "cuda:0"
            self.storage = object() if storage is None else storage

        def contiguous(self):
            return self

        def reshape(self, *shape):
            return Buffer(shape, self.dtype, self.storage)

        def view(self, dtype):
            return Buffer(self.shape, dtype, self.storage)

        def to(self, dtype):
            return self if self.dtype == dtype else Buffer(self.shape, dtype)

    env["torch"].float32 = "f32"
    env["torch"].float8_e5m2 = "e5m2"
    env["torch"].empty = lambda shape, dtype, device: Buffer(shape, dtype)
    for width, count in FIXED:
        fallback = env["ce_fwd_bwd" if width == WIDTHS[-1] else "sampled_ce_fwd_bwd"]
        for rows, scale, reuse in ((17, grad_scale, True), (17, grad_scale * 2, False),
                                   (((1 << 31) + width - 1) // width, grad_scale, False)):
            logits = Buffer((rows, width), "e4m3")
            targets = Buffer(rows, "i64")
            weights = Buffer(count, "f32")
            prefix = Buffer(rows, "i64")
            loss, gradient = env["_ce_forward_buffers"](
                logits, targets, weights, prefix, Buffer((), "f32"), scale, fallback,
            )
            _, _, grid, block, parameters, _ = launches.pop()
            assert loss.shape == (rows,) and loss.dtype == "f32"
            assert gradient.shape == logits.shape and gradient.dtype == "e5m2"
            assert grid == (rows, 1, 1) and block == (256, 1, 1)
            assert parameters[0] is logits and parameters[5] is loss
            assert parameters[6] is (logits if reuse else gradient)
            assert (gradient.storage is logits.storage) == reuse


def check_default_plan(candidate):
    def assignment(path, name):
        source = candidate / path
        value, = (node.value for node in ast.parse(source.read_text(), filename=str(source)).body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        return value

    widths = ast.literal_eval(assignment(SAMPLED, "CANDIDATES_BY_STAGE"))
    stages = assignment(CONFIG, "TRAINING_STAGES")
    observed = set()
    for stage, call in enumerate(stages.elts):
        weights = {arg.arg: arg.value for arg in call.keywords}
        count = len(ast.literal_eval(weights["mtp_weights_start"]))
        assert count == len(ast.literal_eval(weights["mtp_weights_end"]))
        observed.update((width, count) for width in widths.get(stage, (WIDTHS[-1],)))
    assert observed == set(FIXED)


def build_artifact(args, lib, root, stem, width, predictions=None, *, direct=False):
    launch = capture_source(root / CROSS_ENTROPY, width, predictions)
    if predictions is not None:
        min_blocks = 6 if width == WIDTHS[-1] else 8
        old_unroll, old_gradient, old_blocks = {
            10240: (1, 1, 8), 14336: (1, 1, 7), 24576: (2, 4, 2), 50304: (25, 4, 2),
        }[width]
        for macro, before, after in (
            ("SIGMOID_UNROLL", 2, old_unroll),
            ("GRADIENT_UNROLL", 4, old_gradient),
            ("MIN_BLOCKS_PER_SM", min_blocks, old_blocks),
        ):
            old = f"#define CE_{macro} {before}\n"
            assert launch["source"].count(old) == 1
            if direct:
                launch["source"] = launch["source"].replace(old, f"#define CE_{macro} {after}\n")
        if direct:
            for macro in ("USE_LOGIT_TABLE", "DEFAULT_A", "DEFAULT_BC", "CACHE_MTP_TARGETS"):
                launch["source"] = launch["source"].replace(f"#define CE_{macro}\n", "")
    ptx = emit_ptx(lib, launch, args.include)
    path = args.output / (stem + ".ptx")
    path.write_text(ptx)
    count = None
    registers = None
    if args.ptxas:
        result = subprocess.run([args.ptxas, "-v", f"-arch=sm_{launch['arch']}", path,
                                 "-o", path.with_suffix(".cubin")], check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path.with_suffix(".ptxas.log").write_text(result.stdout)
        registers = int(re.search(r"Used (\d+) registers", result.stdout)[1])
        if args.nvdisasm:
            sass = subprocess.check_output([args.nvdisasm, "-c", path.with_suffix(".cubin")], text=True)
            path.with_suffix(".sass").write_text(sass)
            count = sum(m[1] != "NOP" for line in sass.splitlines() if (m := SASS_INSTRUCTION.match(line)))
    return ptx, count, registers


def check_table_initialization(ptx, width):
    # A table read is valid only after all lanes have published the reference
    # halfword and contribution for their own exact byte encodings.
    bad_barrier, replaced = re.subn(r"\bbar\.sync\s+0;", "", ptx, count=1)
    assert replaced == 1
    wrong_exp, replaced = re.subn("0f3fb8aa3b", "0f3F800000", ptx, count=1, flags=re.IGNORECASE)
    assert replaced == 1
    wrong_factor, replaced = re.subn(rf"(st\.shared\.f32\s+\[{FACTOR_SYMBOL})\]", r"\g<1>+4]", ptx)
    assert replaced == 1
    wrong_scale, replaced = re.subn("0f395b6db7", "0f395B6DB8", ptx, count=1, flags=re.IGNORECASE)
    assert replaced == 1
    for changed in (bad_barrier, wrong_exp, wrong_factor, wrong_scale):
        try:
            Program(changed, width, DefaultSoftcapPrefix(width))
        except AssertionError:
            continue
        raise AssertionError("CE table replay accepted an invalid initialization")


def check_correction_owner(ptx, width, predictions):
    # A store one packed vector ahead belongs to a different lane.
    shifted, replaced = re.subn(r"(st\.global\.u8\s+\[%rd\d+)\]", r"\g<1>+8]", ptx)
    assert replaced > 0
    program = Program(shifted, width, DefaultSoftcapPrefix(width))
    try:
        program.execute(predictions, 0, 16384, (0,) * predictions, -1, False)
    except AssertionError:
        return
    raise AssertionError("CE replay accepted a correction in another lane's packed store")


def check_probability_publication(ptx, width, predictions):
    shuffle = ptx.index("shfl.sync.down.b32")
    changed, replaced = re.subn("0f3fb8aa3b", "0f3F800000", ptx[shuffle:], count=1, flags=re.IGNORECASE)
    assert replaced == 1
    wrong_pair, replaced = re.subn(
        r"(st\.shared\.v2\.f32\s+\[[^\]]+\],\s+\{)(%f\d+),\s*(%f\d+)\}",
        r"\g<1>\g<3>, \g<2>}", ptx[shuffle:], count=1)
    assert replaced == 1
    for tail in (changed, wrong_pair):
        try:
            Program(ptx[:shuffle] + tail, width, DefaultSoftcapPrefix(width)).execute(
                predictions, 0, 16384, (0,) * predictions, -1, False)
        except AssertionError:
            continue
        raise AssertionError("CE replay accepted an incorrect published sigmoid or probability")


def check_base_gradient_publication(ptx, width, predictions):
    shifted, replaced = re.subn(r"(st\.shared\.u8\s+\[%rd\d+)\]", r"\g<1>+1]", ptx)
    assert replaced == 1
    try:
        Program(shifted, width, DefaultSoftcapPrefix(width)).execute(predictions, 0, 16384,
                                                                   (0,) * predictions, -1, False)
    except AssertionError:
        return
    raise AssertionError("CE replay accepted a base gradient published at the wrong byte")


def check_target_cache(args, lib, ptx, width, predictions):
    # The fixed uncached path checks both bounds before dereferencing a
    # sparse target. Use it for invalid inputs to the narrowing conversion.
    launch = capture_source(args.candidate / CROSS_ENTROPY, width, predictions)
    launch["source"] = launch["source"].replace("#define CE_CACHE_MTP_TARGETS\n", "")
    uncached = emit_ptx(lib, launch, args.include)
    prefix = DefaultSoftcapPrefix(width)
    reference = Program(uncached, width, prefix)
    candidate = Program(ptx, width, prefix)
    assert reference.cache_slots == 0 and candidate.cache_slots == predictions
    invalid = (-1, (1 << 32) + 5, width)[:predictions]
    inputs = (predictions, 1, 16384, invalid, 0, True)
    assert candidate.execute(*inputs) == reference.execute(*inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nvrtc", type=Path, required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--ptxas", type=Path)
    parser.add_argument("--nvdisasm", type=Path)
    args = parser.parse_args()

    check_checkout(args.baseline, args.candidate)
    check_default_plan(args.candidate)
    check_dispatch(args.candidate)
    lib = ctypes.CDLL(str(args.nvrtc))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.nvdisasm and not args.ptxas:
        parser.error("--nvdisasm requires --ptxas")
    generic = {}
    for width in WIDTHS:
        programs = []
        instruction_counts = []
        for side, root in (("reference", args.baseline), ("paired", args.candidate)):
            ptx, count, _ = build_artifact(args, lib, root, f"{side}-{width}", width)
            if count is not None:
                instruction_counts.append(count)
            programs.append(ptx)
            if side == "paired":
                generic[width] = ptx, count
        live, old, new = verify_training_ce(*programs, width)
        decodes = [ptx.count("cvt.rn.f16x2.e4m3x2") for ptx in programs]
        assert decodes == [8 * ((width + 2047) // 2048), 4 * ((width + 2047) // 2048)]
        print(f"{width}: {decodes[0]} → {decodes[1]} FP8 conversions, "
              f"{old} → {new} prefix instructions, {live} live registers and all halfwords checked", flush=True)
        if instruction_counts:
            print(f"{width}: {instruction_counts[0]} → {instruction_counts[1]} non-NOP SASS instructions")
        if width == WIDTHS[0]:
            for old, new in (("mov.b32 {low, high}, halves", "mov.b32 {high, low}, halves"),
                             ("lg2.approx.ftz.f32", "lg2.approx.f32")):
                changed = programs[1].replace(old, new, 1)
                assert changed != programs[1]
                try:
                    verify_training_ce(programs[0], changed, width)
                except AssertionError:
                    continue
                raise AssertionError(f"CE replay accepted mutation of {old}")
    for width, predictions in FIXED:
        ptx, count, regs = build_artifact(
            args, lib, args.candidate, f"fixed-{width}-{predictions}", width, predictions)
        dynamic, dynamic_count = generic[width]
        cases = verify_specialized_ce(dynamic, ptx, width, predictions)
        _, _, suffix = split(ptx)
        assert all("cvt.f32.f16" not in line for line in suffix)
        assert all("rcp.approx.ftz.f32" not in line for line in suffix)
        assert sum("ex2.approx.ftz.f32" in line for line in suffix) == 1
        assert any("ld.shared.v2.f32" in line for line in suffix)
        print(f"fixed {width}/{predictions}: {cases} in-place CTA traces match loss and E5M2 stores", flush=True)
        check_table_initialization(ptx, width)
        if width == 10240:
            check_target_cache(args, lib, ptx, width, predictions)
            print(f"fixed {width}/{predictions}: cached narrowing preserves invalid-target handling", flush=True)
        if width == WIDTHS[0] and predictions == 2:
            check_correction_owner(ptx, width, predictions)
            check_probability_publication(ptx, width, predictions)
            check_base_gradient_publication(ptx, width, predictions)
        direct, _, direct_regs = build_artifact(args, lib, args.candidate, f"direct-{width}-{predictions}",
                                                width, predictions, direct=True)
        if args.ptxas:
            print(f"fixed {width}/{predictions}: {direct_regs} → {regs} registers", flush=True)
        tanh = [body.count("tanh.approx.f32") for body in (direct, ptx)]
        assert tanh[1] == 1
        print(f"fixed {width}/{predictions}: {tanh[0]} → {tanh[1]} static tanh instructions", flush=True)
        if count is not None:
            print(f"fixed {width}/{predictions}: {dynamic_count} → {count} encoded non-NOP SASS instructions")


if __name__ == "__main__":
    main()
