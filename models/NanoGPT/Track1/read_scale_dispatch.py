"""Select the fused value Adam kernel using the reference launch already chosen by Inductor."""
import ast
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

ELEMENTS = 25152 * 768
KERNEL = "triton_poi_fused__to_copy_add_copy__div_gt_mul_sqrt_0"
LAUNCH = KERNEL + ".run"
TILES = {512: (8, 0), 1024: (4, 4096)}
ARGS = ("arg0_1", "arg1_1", "arg2_1", "arg3_1", "arg4_1.item()", "arg5_1.item()",
        "arg1_1", "arg3_1", "arg0_1", str(ELEMENTS))


@lru_cache(maxsize=16)
def ptx_digest(ptx):
    instructions = []
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        if line.startswith(".section"):
            break
        if line and not line.startswith((".file ", ".file\t", ".loc ", ".loc\t")):
            instructions.append(" ".join(line.split()))
    return sha256("\n".join(instructions).encode()).hexdigest()


@lru_cache(maxsize=16)
def wrapper_function_digest(source):
    code = ast.parse(source)
    compiled = [node for node in ast.walk(code) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == "async_compile.triton"]
    if len(compiled) != 1:
        return None
    kernel = compiled[0]
    if (len(kernel.args) < 2 or not isinstance(kernel.args[0], ast.Constant)
            or kernel.args[0].value != KERNEL or not isinstance(kernel.args[1], ast.Constant)
            or not isinstance(kernel.args[1].value, str)):
        return None
    declarations = [node for node in ast.parse(kernel.args[1].value).body
                    if isinstance(node, ast.FunctionDef) and node.name == KERNEL]
    if len(declarations) != 1:
        return None
    fn, = declarations
    function = ast.FunctionDef(name=fn.name, args=fn.args, body=fn.body, decorator_list=[])
    return sha256(ast.dump(function).encode()).hexdigest()


@lru_cache(maxsize=16)
def matches_reference_wrapper(source):
    code = ast.parse(source)
    launches = [node for node in ast.walk(code) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
    if len(launches) != 1:
        return False
    launch, = launches
    return (ast.unparse(launch.func) == LAUNCH
            and tuple(map(ast.unparse, launch.args)) == ARGS
            and len(launch.keywords) == 1 and launch.keywords[0].arg == "stream")


def reference_profile(tuner, period, device, exact_ptx, cache_ptx):
    if (len(tuner.launchers) != 1 or tuner.triton_interpret or tuner.dump_launch_params
            or tuner.inductor_meta.get("coordinate_descent_tuning", False)
            or tuner.triton_meta["device"].index != device):
        return None
    launcher, = tuner.launchers
    config = launcher.config
    block = config.kwargs.get("XBLOCK")
    if (block not in TILES or config.kwargs != {"XBLOCK": block}
            or (config.num_warps, launcher.shared) != TILES[block] or config.num_ctas != 1 or config.num_stages != 1 or config.maxnreg is not None
            or config.pre_hook is not None or getattr(config, "ir_override", None)
            or launcher.store_cubin):
        return None
    ptx = cache_ptx(launcher)
    if ptx is None or ptx_digest(ptx) != exact_ptx.get((period, block)):
        return None
    return block, config.num_warps


def select_value_adam(modules, tuner_type, period, device, exact_wrapper, exact_ptx, cache_ptx):
    profiles = set()
    # Cache modules are immutable after load. Copy the list before reading.
    for module in tuple(modules):
        tuner = vars(module).get(KERNEL)
        if not isinstance(tuner, tuner_type) or tuner.triton_meta["device"].index != device:
            continue
        try:
            source = Path(module.__file__).read_text()
            if matches_reference_wrapper(source) and wrapper_function_digest(source) == exact_wrapper[period]:
                profiles.add(reference_profile(tuner, period, device, exact_ptx, cache_ptx))
        except (AttributeError, OSError, SyntaxError):
            return None
    return next(iter(profiles)) if len(profiles) == 1 else None


EXACT_WRAPPER = {2: 'b363959382bf980ddc9e462c52b692e25e6c1709d3c0e18332a809df17fcdfb2', 4: 'cd93db4a529ec1923c5a8f741f4595e1a38cb93ad18c61c15814e94a5f962b26'}
EXACT_PTX = {(2, 512): '61d3c1c50e75b500245409550c3ec2b398ca6684aa0faa3dda317a2bedcfcc9b', (2, 1024): '4bbb85f4d8a5585f64520224d5e9c87c1db534f7c44132ec460e595c06780fce', (4, 512): '21bd59b25bf351e471f58477c9277cb2358eb5662cfa263500e545f36cdb7a7f', (4, 1024): '3ad9b651505b80b78a87dc17abaf04b2faaa577b11774ffcc27856aa0f5c7b12'}

def find_installed_launch(period, device):
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton import knobs

    def cache_ptx(launcher):
        if not launcher.cache_hash.isalnum():
            return None
        try:
            return (Path(knobs.cache.dir) / launcher.cache_hash / (KERNEL + ".ptx")).read_text()
        except (OSError, UnicodeError):
            return None

    return select_value_adam(PyCodeCache.modules, CachingAutotuner, period, device,
                             EXACT_WRAPPER, EXACT_PTX, cache_ptx)
