"""Publish ANVIL Gram divisors and normalize the two BF16 inputs."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


TORCH_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"


@gluon.jit
def _seed(x):
    # Inductor starts both halves of this pair at FP32 positive zero.
    return gl.inline_asm_elementwise("add.f32 $0, $1, 0f00000000;", "=f,f", [x],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _butterfly(x, MASK: gl.constexpr):
    peer = gl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 31, -1;", "=f,f,n",
                                    [x, MASK], dtype=gl.float32, is_pure=True, pack=1)
    return gl.inline_asm_elementwise("add.f32 $0, $1, $2;", "=f,f,f", [x, peer],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def trace_divisors(Input, Divisors, batch, DIM: gl.constexpr, CHUNKS: gl.constexpr,
                   WARPS: gl.constexpr, READ_L2: gl.constexpr):
    # One warp per 128 diagonal entries. Each lane reconstructs the
    # two pairs assigned to a thread in the original reduction.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [WARPS, 1], [1, 0])
    chunk = gl.arange(0, CHUNKS, layout=gl.SliceLayout(1, layout))
    lane = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    diagonal = chunk[:, None] * 128 + lane[None, :]
    ptr = Input + batch * DIM * DIM + diagonal * (DIM + 1)
    active = chunk[:, None] < DIM // 128
    x0 = gl.load(ptr, active, other=0, cache_modifier=".cg" if READ_L2 else "").to(gl.float32)
    x32 = gl.load(ptr + 32 * (DIM + 1), active, other=0, cache_modifier=".cg" if READ_L2 else "").to(gl.float32)
    x64 = gl.load(ptr + 64 * (DIM + 1), active, other=0, cache_modifier=".cg" if READ_L2 else "").to(gl.float32)
    x96 = gl.load(ptr + 96 * (DIM + 1), active, other=0, cache_modifier=".cg" if READ_L2 else "").to(gl.float32)
    partial = (_seed(x0) + x64) + (_seed(x32) + x96)
    for bit in gl.static_range(0, 5):
        # The last exchange corresponds to the original cross-warp add.
        partial = _butterfly(partial, (8, 4, 2, 1, 16)[bit])
    sums = gl.gather(partial, gl.full((CHUNKS, 1), 0, gl.int32, layout), axis=1)
    sums = gl.reshape(sums, (CHUNKS,))
    sums = gl.convert_layout(sums, gl.BlockedLayout([1], [32], [WARPS], [0]))
    trace = gl.sum(sums, axis=0)
    # Same rounded sqrt/FMA that Inductor emits for the pinned margin and epsilon.
    denominator = gl.fma(gl.sqrt_rn(trace), 1.05, 1e-6)
    gl.store(Divisors + 2 * batch, denominator)
    gl.store(Divisors + 2 * batch + 1, denominator * denominator)


@gluon.jit
def _gram_divisors(Input, Divisors, DIM: gl.constexpr, CHUNKS: gl.constexpr):
    trace_divisors(Input, Divisors, gl.program_id(0), DIM, CHUNKS, CHUNKS, False)


@torch.library.custom_op("veritile_track1::anvil_gram_divisors", mutates_args=())
def gram_divisors(a: torch.Tensor) -> torch.Tensor:
    batch, dim, _ = a.shape
    divisors = torch.empty((batch, 2, 1), dtype=torch.float32, device=a.device)
    chunks = 2 if dim == 256 else 8
    _gram_divisors[(batch,)](a, divisors, dim, chunks, num_warps=chunks, num_stages=1)
    return divisors


@gram_divisors.register_fake
def _fake_gram_divisors(a):
    return a.new_empty((a.shape[0], 2, 1), dtype=torch.float32)


@gluon.jit
def _normalize_gram_inputs(X, A, Divisors, X_TILES: gl.constexpr,
                           A_TILES: gl.constexpr, BLOCK: gl.constexpr):
    tile = gl.program_id(0)
    matrix = gl.program_id(1)
    prepared = tile < X_TILES
    selector = (tile >= X_TILES).to(gl.int32)
    stride = X_TILES + selector * (A_TILES - X_TILES)
    divisor = gl.load(Divisors + 2 * matrix + selector)
    local_tile = tile - selector * X_TILES
    offset = matrix * (stride * BLOCK) + local_tile * BLOCK
    base = gl.where(prepared, X, A) + offset
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    ptr = base + gl.arange(0, BLOCK, layout=layout)
    x = gl.load(ptr).to(gl.float32)
    # Preserve Inductor's full divide and the following BF16 rounding.
    gl.store(ptr, (x / divisor).to(gl.bfloat16))


@torch.library.custom_op("veritile_track1::anvil_normalize_gram_inputs", mutates_args={"x", "a"})
def normalize_gram_inputs(x: torch.Tensor, a: torch.Tensor, divisors: torch.Tensor) -> None:
    batch, rows, cols = x.shape
    block = 2048
    x_tiles = rows * cols // block
    a_tiles = a.size(-1) ** 2 // block
    _normalize_gram_inputs[(x_tiles + a_tiles, batch)](
        x, a, divisors, x_tiles, a_tiles, block, num_warps=4, num_stages=1
    )


@normalize_gram_inputs.register_fake
def _fake_normalize_gram_inputs(x, a, divisors):
    return None


def can_fuse_gram_trace(shape, dtype, device):
    # The tree matches the pinned TorchInductor heuristics on the default H100 banks.
    if (torch.version.git_version != TORCH_REVISION or triton.__version__ != "3.6.0"
            or device.type != "cuda" or dtype != torch.bfloat16
            or shape not in ((6, 256, 768), (2, 768, 768), (3, 2816, 768))):
        return False
    props = torch.cuda.get_device_properties(device)
    from torch._inductor import config as cfg

    return ((props.major, props.minor, props.multi_processor_count) == (9, 0, 132)
            and not (cfg.max_autotune or cfg.max_autotune_pointwise or cfg.coordinate_descent_tuning
                     or cfg.deterministic or torch.are_deterministic_algorithms_enabled()
                     or cfg.triton.cooperative_reductions or cfg.triton.force_cooperative_reductions)
            and cfg.split_reductions and cfg.unroll_reductions_threshold == 8)


def supports_cooperative_anvil(device):
    # The pinned reductions already restrict this path to a full H100.
    # Query its cooperative-launch attribute before tracing the optimizer.
    import ctypes

    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        return False
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        result = ctypes.c_int()
        handle = ctypes.c_int()
        ordinal = device.index if device.index is not None else torch.cuda.current_device()
        # CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH in the CUDA driver ABI.
        return (driver.cuDeviceGet(ctypes.byref(handle), ordinal) == 0
                and driver.cuDeviceGetAttribute(ctypes.byref(result), 95, handle) == 0
                and result.value == 1)
    except (OSError, AttributeError):
        return False

