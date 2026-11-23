namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

def candidateNativeAdam : String := r#""""Fuse value Adam gradient scaling, the update and workspace clearing."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

ELEMENTS = 25152 * 768
PROFILE_WARPS = {512: 8, 1024: 4}
BETAS = {period: (0.75 ** (period // 2), 0.95 ** (period // 2)) for period in (2, 4)}


@gluon.jit(do_not_specialize=["step_size", "decay"])
def _update_with_scaled_read(
    first, gradient, second, step_size, weight, decay,
    first_out, second_out, weight_out,
    BLOCK: gl.constexpr, BETA1: gl.constexpr, BETA2: gl.constexpr,
    FIRST_MIX: gl.constexpr, SECOND_MIX: gl.constexpr, CLEAR_ALL: gl.constexpr,
):
    width: gl.constexpr = 8 if BLOCK == 1024 else 2
    warps: gl.constexpr = 4 if BLOCK == 1024 else 8
    layout: gl.constexpr = gl.BlockedLayout([width], [32], [warps], [0])
    lane = gl.arange(0, BLOCK, layout=layout)
    i = gl.program_id(0) * BLOCK + lane

    m = gl.load(first + i)
    half = gl.load(gradient + i)
    v = gl.load(second + i)
    p = gl.load(weight + i).to(gl.float32)

    # A warp with a NaN preserves the scalar conversion and its payload behavior.
    if BLOCK == 1024 and not CLEAR_ALL:
        scaled = gl.inline_asm_elementwise(
            """        {
        .reg .b32 %isnan<4>, %flags, %gain;
        .reg .pred %nanLane, %slow;
        .reg .b16 %a, %b;
        .reg .f32 %lo, %hi;
        mov.b32 %gain, 0x30003000;
        max.NaN.f16x2 %isnan0, $4, $5;
        max.NaN.f16x2 %isnan1, $6, $7;
        max.NaN.f16x2 %flags, %isnan0, %isnan1;
        set.nan.f16x2.f16x2 %flags, %flags, %flags;
        setp.ne.b32 %nanLane, %flags, 0;
        vote.sync.any.pred %slow, %nanLane, 0xffffffff;
        @%slow bra.uni L__half_product_slow;
        mul.rn.f16x2 $0, $4, %gain;
        mul.rn.f16x2 $1, $5, %gain;
        mul.rn.f16x2 $2, $6, %gain;
        mul.rn.f16x2 $3, $7, %gain;
        bra.uni L__half_product_done;
        L__half_product_slow:
        mov.b32 {%a, %b}, $4;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $0, {%a, %b};
        mov.b32 {%a, %b}, $5;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $1, {%a, %b};
        mov.b32 {%a, %b}, $6;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $2, {%a, %b};
        mov.b32 {%a, %b}, $7;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $3, {%a, %b};
        L__half_product_done:
        }""",
            constraints="=r,=r,=r,=r,r,r,r,r", args=[half], dtype=gl.float16, is_pure=True, pack=8,
        )
    else:
        raw = half.to(gl.float32)
        scaled = gl.inline_asm_elementwise(
            """{
        .reg .f32 %scaled;
        mul.f32 %scaled, $1, 0f3e000000;
        cvt.rn.f16.f32 $0, %scaled;
    }""",
            constraints="=h,f", args=[raw], dtype=gl.float16, is_pure=True, pack=1,
        )
    g = scaled.to(gl.float32)
    if CLEAR_ALL:
        gl.store(gradient + i, 0.0)
    elif BLOCK == 1024:
        gl.inline_asm_elementwise(
            """{
        .reg .b32 %dirty;
        .reg .pred %write;
        lop3.b32 %dirty, $10, $11, $12, 0xfe;
        or.b32 %dirty, %dirty, $13;
        setp.ne.b32 %write, %dirty, 0;
        mov.b32 $0, 0;
        mov.b32 $1, 0;
        @%write st.global.v4.b32 [$2], {$0, $0, $0, $0};
    }""",
            constraints="=r,=r,l,l,l,l,l,l,l,l,r,r,r,r",
            args=[(gradient + i).to(gl.uint64), half], dtype=gl.uint8, is_pure=False, pack=8,
        )
    else:
        gl.inline_asm_elementwise(
            """{
        .reg .pred %write;
        setp.ne.b32 %write, $3, 0;
        mov.b32 $0, 0;
        @%write st.global.b32 [$1], $0;
    }""",
            constraints="=r,l,l,r", args=[(gradient + i).to(gl.uint64), half],
            dtype=gl.uint16, is_pure=False, pack=2,
        )
    m_new = gl.fma(m, BETA1, g * FIRST_MIX)
    v_new = gl.fma(v, BETA2, (g * g) * SECOND_MIX)

    direction = m_new / (gl.sqrt_rn(v_new) + 1e-10)
    applied_step = step_size * direction
    active = (applied_step * p > 0.0).to(gl.float32)
    masked = p * active
    update = gl.fma(decay, masked, applied_step)
    p_new = p - update

    gl.store(first_out + i, m_new)
    gl.store(second_out + i, v_new)
    gl.store(weight_out + i, p_new)


def launch_scaled_read(weight, raw_gradient, first, second, step_size, decay, period, block, clear_all):
    beta1, beta2 = BETAS[period]
    _update_with_scaled_read[(ELEMENTS // block,)](
        first, raw_gradient, second, step_size.item(), weight, decay.item(), first, second, weight,
        block, beta1, beta2, 1.0 - beta1, 1.0 - beta2, clear_all, num_warps=PROFILE_WARPS[block], num_stages=1,
        enable_fp_fusion=False,
    )


def probe_scaled_read(optimizer, param, raw_gradient, world):
    import torch

    if (world != 8 or raw_gradient.device.type != "cuda"
            or tuple(raw_gradient.shape) != (25152, 768)
            or torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
            or triton.__version__ != "3.6.0"):
        return None
    cfg = optimizer.param_cfgs[param]
    if cfg.eps != 1e-10 or cfg.comms != "sharded" or cfg.chunk_size != 25152:
        return None
    period = next((tick for tick, betas in BETAS.items() if betas == tuple(cfg.adam_betas)), None)
    if period is None:
        return None
    p = param[optimizer.rank * cfg.chunk_size:(optimizer.rank + 1) * cfg.chunk_size]
    state = optimizer.param_states[param]
    operands = (
        (p, torch.bfloat16, (25152, 768)),
        (raw_gradient, torch.float16, (25152, 768)),
        (state["exp_avg"], torch.float32, (25152, 768)),
        (state["exp_avg_sq"], torch.float32, (25152, 768)),
    )
    if not all(x.device == p.device and x.dtype == dtype and tuple(x.shape) == shape
               and x.is_contiguous() and x.data_ptr() % 16 == 0 for x, dtype, shape in operands):
        return None
    if not all(x.device.type == "cpu" and x.dtype == torch.float32 and x.shape == ()
               for x in (optimizer._step_size_t, optimizer._eff_wd_t)):
        return None
    props = torch.cuda.get_device_properties(p.device)
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, 132):
        return None
    from track_1_short.perf.kernels.read_scale_dispatch import find_installed_launch

    profile = find_installed_launch(period, p.device.index)
    return None if profile is None else (period, profile[0])
"#

def candidateNativeDispatch : String := r#""""Select the fused value Adam kernel using the reference launch already chosen by Inductor."""
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
"#

def candidateNativeScatter : String := r#""""Scatter value-embedding gradients and scale only rows that can be nonzero."""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

WIDTH = 768
LOCAL_ROWS = 4
TILE = 256
ROWS = 4
COLS = 256
ATEN_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"


@triton.jit(do_not_specialize=["offset", "first", "claim_start"])
def _scatter_value_rows(src, ids, dst, offset, first, touched, claim_start, TILE: tl.constexpr, MARK: tl.constexpr,
                        DIRECT_OWN: tl.constexpr = False):
    slot = tl.program_id(0)
    row = tl.load(ids + offset + slot) - first
    col = tl.arange(0, TILE).to(tl.int64)
    src_base = (row if DIRECT_OWN else slot).to(tl.int64) * 768
    dst_base = row.to(tl.int64) * 768
    for stripe in tl.static_range(3):
        index = stripe * TILE + col
        val = tl.load(src + src_base + index).to(tl.float32)
        tl.atomic_add(dst + dst_base + index, val, sem="relaxed")
        if DIRECT_OWN:
            tl.store(src + src_base + index, 0.0)
    if MARK:
        # Reserve zero for untouched rows. Duplicate entries keep the largest position.
        tl.atomic_max(touched + row, claim_start + slot + 1, sem="relaxed")


def _local_row_assembly(direct, mark):
    """One warp owns a row, so its tail predicate is uniform."""
    start = """
    .reg .pred %inactive, %leader;
    .reg .b32 %row, %relative, %zero, %ticket, %idx, %w, %srcWord;
    .reg .b64 %lookup, %source, %target, %mark;
    .reg .b32 %b<4>;
    .reg .b16 %h<8>, %old<8>;
    .reg .b32 %oldTicket;
    setp.ge.s32 %inactive, $7, $6;
    @%inactive bra.uni L__local_rows_end;
    add.s32 %idx, $7, $9;
    mad.wide.s32 %lookup, %idx, 4, $2;
    ld.global.b32 %row, [%lookup];
    sub.s32 %relative, %row, $5;
    mad.lo.s32 %w, %relative, 768, $8;
    SRCWORD
    mad.wide.s32 %source, SRCVALUE, 2, $1;
    mad.wide.s32 %target, %w, 2, $3;
    """
    start = start.replace("SRCWORD", "" if direct else "mad.lo.s32 %srcWord, $7, 768, $8;")
    start = start.replace("SRCVALUE", "%w" if direct else "%srcWord")
    if direct:
        start += "mov.b32 %zero, 0;\n"
    parts = [start]
    for byte in (0, 512, 1024):
        addr = f"+{byte}" if byte else ""
        parts.append(f"""
        ld.global.v4.b32 {{%b0, %b1, %b2, %b3}}, [%source{addr}];
        mov.b32 {{%h0, %h1}}, %b0;
        mov.b32 {{%h2, %h3}}, %b1;
        mov.b32 {{%h4, %h5}}, %b2;
        mov.b32 {{%h6, %h7}}, %b3;
        atom.global.gpu.relaxed.add.noftz.v8.f16 {{%old0, %old1, %old2, %old3, %old4, %old5, %old6, %old7}}, [%target{addr}], {{%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7}};
        """)
        if direct:
            parts.append(f"st.global.v4.b32 [%source{addr}], {{%zero, %zero, %zero, %zero}};\n")
    if mark:
        parts.append("""
        setp.eq.b32 %leader, $8, 0;
        mad.wide.s32 %mark, %relative, 4, $4;
        add.s32 %ticket, $7, 1;
        @%leader atom.global.gpu.relaxed.max.s32 %oldTicket, [%mark], %ticket;
        """)
    parts.append("L__local_rows_end:\nmov.b32 $0, 0;\n")
    return "{\n" + "".join(parts) + "}"


_LOCAL_ASSEMBLY = {(direct, mark): _local_row_assembly(direct, mark)
                   for direct in (False, True) for mark in (False, True)}


@gluon.jit(do_not_specialize=["offset", "first", "count"])
def local_rows(src, ids, dst, offset, first, count, marker, ASSEMBLY: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    thread = gl.arange(0, 128, layout=layout)
    slot = gl.program_id(0) * 4 + thread // 32
    col = thread % 32 * 8
    gl.inline_asm_elementwise(
        ASSEMBLY, constraints="=r,l,l,l,l,r,r,r,r,r",
        args=[src.to(gl.uint64), ids.to(gl.uint64), dst.to(gl.uint64),
              marker.to(gl.uint64), first, count, slot, col, offset],
        dtype=gl.uint32, is_pure=False, pack=1,
    )


@triton.jit
def _scale_half_eighth(value):
    # Keep the scalar FP32 multiply and independent FP16 store used by ATen.
    return tl.inline_asm_elementwise(
        "{ .reg .f32 wide; cvt.f32.f16 wide, $1; "
        "mul.rn.f32 wide, wide, 0f3e000000; cvt.rn.f16.f32 $0, wide; }",
        constraints="=h,h", args=[value], dtype=tl.float16, is_pure=True, pack=1,
    )


@triton.jit
def _claim_rows(touched, want, asked, own, count, lo, first, ROWS: tl.constexpr):
    slot = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = slot < count
    local = slot < own
    ids = tl.where(local, want, asked)
    index = tl.where(local, lo + slot, slot - own)
    row = tl.load(ids + index, mask=live, other=0) - first
    claim = tl.load(touched + row, mask=live, other=0)
    use = live & (claim == slot + 1)
    return row, use


@triton.jit(do_not_specialize=["own", "count", "lo", "first"])
def _scale_claimed(grad, touched, want, asked, own, count, lo, first, ROWS: tl.constexpr, COLS: tl.constexpr):
    row, use = _claim_rows(touched, want, asked, own, count, lo, first, ROWS)
    cols = tl.arange(0, COLS)
    for stripe in tl.static_range(3):
        offsets = row[:, None] * 768 + cols[None, :] + stripe * COLS
        value = tl.load(grad + offsets, mask=use[:, None], other=0.0)
        tl.store(grad + offsets, _scale_half_eighth(value), mask=use[:, None])


@triton.jit(do_not_specialize=["own", "count", "lo", "first"])
def _clear_claimed(grad, touched, want, asked, own, count, lo, first, ROWS: tl.constexpr, COLS: tl.constexpr):
    row, use = _claim_rows(touched, want, asked, own, count, lo, first, ROWS)
    cols = tl.arange(0, COLS)
    for stripe in tl.static_range(3):
        offsets = row[:, None] * 768 + cols[None, :] + stripe * COLS
        tl.store(grad + offsets, 0.0, mask=use[:, None])
    tl.debug_barrier()
    tl.store(touched + row, 0, mask=use)


def _hopper(tensor):
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 9


def _layout(t, device, dtype, ndim):
    return (t.device == device and t.dtype == dtype and t.ndim == ndim
            and t.is_contiguous() and t.data_ptr() % 16 == 0)


def value_gradient_workspace(rows, width, dtype, device, world):
    device = torch.device(device)
    if (device.type == "cuda" and torch.version.git_version == ATEN_REVISION
            and torch.cuda.get_device_capability(device)[0] == 9
            and dtype == torch.float16 and width == WIDTH and world == 8
            and 0 < rows and (rows + ROWS) * WIDTH < 2**31):
        storage = torch.zeros(rows * (WIDTH + 2), dtype=dtype, device=device)
        grad = storage[:rows * WIDTH].view(rows, WIDTH)
        touched = storage[rows * WIDTH:].view(torch.int32)
        # Both density branches can be reached after the timed run begins.
        for mark in (False, True):
            for direct in (False, True):
                _scatter_value_rows.warmup(grad, touched, grad, 0, 0, touched, 0, TILE, mark, direct,
                                           num_warps=4, num_stages=1, grid=(1,))
                if rows == 25152:
                    local_rows.warmup(grad, touched, grad, 0, 0, 0, touched,
                                      _LOCAL_ASSEMBLY[direct, mark], num_warps=4, num_stages=1, grid=(1,))
        _scale_claimed.warmup(grad, touched, touched, touched, 0, 0, 0, 0, ROWS, COLS,
                              num_warps=4, num_stages=1, grid=(1,))
        _clear_claimed.warmup(grad, touched, touched, touched, 0, 0, 0, 0, ROWS, COLS,
                              num_warps=4, num_stages=1, grid=(1,))
        return grad, storage, touched
    return torch.zeros((rows, width), dtype=dtype, device=device), None, None


def _scatter_supported(grad, own, received, want, asked, lo, hi, first, storage=None, *, direct_own=False):
    if not _hopper(grad):
        return False
    device = grad.device
    if (not all(_layout(t, device, torch.float16, 2) and t.shape[1] == WIDTH
                for t in (grad, own, received))
            or not all(_layout(t, device, torch.int32, 1) for t in (want, asked))
            or not 0 <= lo <= hi <= want.numel() < 2**31
            or not 0 <= first <= first + grad.shape[0] < 2**31
            or own.shape[0] != (grad.shape[0] if direct_own else hi - lo)
            or received.shape[0] != asked.numel()
            or max(own.numel(), received.numel()) >= 2**31):
        return False
    destination = grad if storage is None else storage
    inputs = (received, want, asked)
    return (_disjoint(destination, (own, *inputs)) and (not direct_own or _disjoint(own, inputs)))


def _disjoint(destination, inputs):
    start = destination.data_ptr()
    end = start + destination.numel() * destination.element_size()
    return all(not tensor.numel() or end <= tensor.data_ptr()
               or tensor.data_ptr() + tensor.numel() * tensor.element_size() <= start for tensor in inputs)


def _scatter(grad, own, received, want, asked, lo, hi, first, touched=None, *,
             direct_own=False, grouped_local=False):
    own_count = hi - lo
    marker = want if touched is None else touched
    if own_count:
        if (grouped_local and grad.shape[0] == 25152 and LOCAL_ROWS <= own_count <= 25152
                and want.numel() < 2**20 and first + 25152 <= 2**20):
            local_rows[(triton.cdiv(own_count, LOCAL_ROWS),)](
                own, want, grad, lo, first, own_count, marker,
                _LOCAL_ASSEMBLY[direct_own, touched is not None],
                num_warps=4, num_stages=1,
            )
        else:
            _scatter_value_rows[(own_count,)](
                own, want, grad, lo, first, marker, 0, TILE, touched is not None,
                DIRECT_OWN=direct_own, num_warps=4, num_stages=1,
            )
    # Peers can repeat a destination. Keep their original launch and order
    # that whole launch after the local owner's writes.
    peer_count = received.shape[0]
    if peer_count:
        marker = asked if touched is None else touched
        _scatter_value_rows[(peer_count,)](
            received, asked, grad, 0, first, marker, own_count, TILE, touched is not None,
            DIRECT_OWN=False, num_warps=4, num_stages=1,
        )

def scatter_value_gradients_(grad, own, received, want, asked, lo, hi, first, *, direct_own=False, grouped_local=False):
    """Apply the two scatter operations in their original order."""
    if not _scatter_supported(grad, own, received, want, asked, lo, hi, first, direct_own=direct_own):
        return False
    _scatter(grad, own, received, want, asked, lo, hi, first, direct_own=direct_own, grouped_local=grouped_local)
    return True


def merge_sparse_value_gradients_(grad, storage, touched, own, received, want, asked, lo, hi, first, world,
                                  previous, *, direct_own=False, scale_read=False, grouped_local=False):
    if (storage is None or touched is None or world != 8
            or grad.ndim != 2 or own.ndim != 2 or received.ndim != 2):
        return False
    rows = grad.shape[0]
    entries = hi - lo + received.shape[0]
    # This upper bound leaves at least half of the rows untouched by scaling.
    if (not 0 < rows or 2 * entries > rows or (rows + ROWS) * WIDTH >= 2**31
            or torch.version.git_version != ATEN_REVISION
            or not _layout(storage, grad.device, torch.float16, 1)
            or not _layout(touched, grad.device, torch.int32, 1)
            or storage.numel() != rows * (WIDTH + 2) or touched.numel() != rows
            or grad.data_ptr() != storage.data_ptr()
            or touched.data_ptr() != grad.data_ptr() + rows * WIDTH * 2
            or not _scatter_supported(grad, own, received, want, asked, lo, hi, first, storage,
                                      direct_own=direct_own)):
        return False
    if previous is None:
        storage.zero_()
    elif previous:
        prior_want, prior_asked, prior_lo, prior_hi = previous
        prior_own = prior_hi - prior_lo
        prior_entries = prior_own + prior_asked.numel()
        if prior_entries:
            _clear_claimed[(triton.cdiv(prior_entries, ROWS),)](
                grad, touched, prior_want, prior_asked, prior_own, prior_entries, prior_lo, first, ROWS, COLS,
                num_warps=4, num_stages=1,
            )
    # A consuming update restores the FP16 payload to zero. Only a
    # materialized gradient needs to record the next clear's row positions.
    markers = None if scale_read else touched
    _scatter(grad, own, received, want, asked, lo, hi, first, markers, direct_own=direct_own, grouped_local=grouped_local)
    if entries and not scale_read:
        _scale_claimed[(triton.cdiv(entries, ROWS),)](
            grad, touched, want, asked, hi - lo, entries, lo, first, ROWS, COLS,
            num_warps=4, num_stages=1,
        )
    return True
"#

def nativeValuePullReplacements : List (String × String) := [
  (r#"                    full.index_fill_(0, own_rows.long(), 0)
                scatter_add_rows(own_grad, own_rows, grad)
                scatter_add_rows(received, pull.asked - first, grad)
            if profile is None:
                grad.mul_(1.0 / self.world_size)
        if profile is None:
            self.optimizer.adam_update_shard(self.param, grad)
            self._remember_scale_read(grad, profile_key)
        else:
            self.optimizer.adam_update_shard(self.param, grad, scale_read=(*profile, not sparse))
        self.grad_previous = next_previous
"#, r#"                    full.index_fill_(0, own_rows.long(), 0)
                scatter_add_rows(own_grad, own_rows, grad)
                scatter_add_rows(received, pull.asked - first, grad)
            grad.mul_(1.0 / self.world_size)
        self.optimizer.adam_update_shard(self.param, grad)
"#),
  (r#"        work.wait()
        pull, first = self.live, self.owners.first_row
        grad = self.grad_shard
        profile_key, profile = self._current_scale_read(grad)
        direct_own = own_grad is None
        if direct_own:
            own_grad = self.grad_accum[self.own_rows]
        previous = self.grad_previous
        self.grad_previous = None
        next_previous = None
        sparse = merge_sparse_value_gradients_(grad, self.grad_workspace, self.grad_touched, own_grad, received,
                                        pull.want, pull.asked, pull.lo, pull.hi, first, self.world_size, previous,
                                        direct_own=direct_own, scale_read=profile is not None, grouped_local=True)
        if sparse:
            if profile is None and (pull.lo != pull.hi or pull.asked.numel()):
                next_previous = (pull.want, pull.asked, pull.lo, pull.hi)
            else:
                next_previous = ()
        else:
            if profile is None or previous is None or previous:
                grad.zero_()
            else:
                # The previous consuming launch left both gradient and
                # marker planes clear. The fallback leaves markers alone.
                next_previous = ()
            if not scatter_value_gradients_(grad, own_grad, received, pull.want, pull.asked, pull.lo, pull.hi, first,
                                            direct_own=direct_own, grouped_local=True):
                own_rows = pull.want[pull.lo:pull.hi] - first
                if direct_own:
                    full = own_grad
"#, r#"        work.wait()
        pull, first = self.live, self.owners.first_row
        grad = self.grad_shard
        direct_own = own_grad is None
        if direct_own:
            own_grad = self.grad_accum[self.own_rows]
        previous = self.grad_previous
        self.grad_previous = None
        if merge_sparse_value_gradients_(grad, self.grad_workspace, self.grad_touched, own_grad, received,
                                        pull.want, pull.asked, pull.lo, pull.hi, first, self.world_size, previous,
                                        direct_own=direct_own):
            if pull.lo != pull.hi or pull.asked.numel():
                self.grad_previous = (pull.want, pull.asked, pull.lo, pull.hi)
            else:
                self.grad_previous = ()
        else:
            grad.zero_()
            if not scatter_value_gradients_(grad, own_grad, received, pull.want, pull.asked, pull.lo, pull.hi, first,
                                            direct_own=direct_own):
                own_rows = pull.want[pull.lo:pull.hi] - first
                if direct_own:
                    full = own_grad
"#),
  (r#"        if not clear_value_want_(self.grad_accum, ids):
            self.grad_accum.index_fill_(0, ids.long(), 0)

    def _current_scale_read(self, grad):
        if grad.device.type != "cuda":
            return None, None
        from torch._inductor.codecache import PyCodeCache

        cfg = self.optimizer.param_cfgs[self.param]
        signature = (tuple(cfg.adam_betas), len(PyCodeCache.modules))
        return signature, self.scale_read_profiles.get(signature)

    def _remember_scale_read(self, grad, signature):
        if signature is None:
            return
        from torch._inductor.codecache import PyCodeCache
        from track_1_short.perf.kernels.value_adam_read_scale import probe_scaled_read

        profile = probe_scaled_read(self.optimizer, self.param, grad, self.world_size)
        if profile is not None:
            self.scale_read_profiles[(signature[0], len(PyCodeCache.modules))] = profile

    @torch.no_grad()
    def adam_update(self):
        """Event, part 2 (collective with send_grads): rebuild this rank's dense gradient shard and run
"#, r#"        if not clear_value_want_(self.grad_accum, ids):
            self.grad_accum.index_fill_(0, ids.long(), 0)

    @torch.no_grad()
    def adam_update(self):
        """Event, part 2 (collective with send_grads): rebuild this rank's dense gradient shard and run
"#),
  (r#"        )
        # Empty tuple denotes a clean workspace. None selects a full clear.
        self.grad_previous = ()
        self.scale_read_profiles = {}
        self.landing = None  # a started land: (pull, received, work, served), until complete_land
        self.reset()

"#, r#"        )
        # Empty tuple denotes a clean workspace. None selects a full clear.
        self.grad_previous = ()
        self.landing = None  # a started land: (pull, received, work, served), until complete_land
        self.reset()

"#)
]

def nativeValueAnvilReplacements : List (String × String) := [
  (r#"        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        if scale_read is None:
            AnvilAndAdam._adam_update_step(
                p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
            )
        else:
            from track_1_short.perf.kernels.value_adam_read_scale import launch_scaled_read

            launch_scaled_read(p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
                               self._step_size_t, self._eff_wd_t, *scale_read)

        return p_slice

    @torch.no_grad()
    def adam_update_shard(self, param: nn.Parameter, grad_chunk: Tensor, *, scale_read=None):
        """Adam on this rank's shard of a sharded parameter whose gradient was reduced outside step()
        (value_embeds: perf/value_embed_pull.py). Same update, state and config as in step()."""
        p_cfg = self.param_cfgs[param]
        assert p_cfg.optim == "adam" and p_cfg.comms == "sharded" and param.grad is None
        self._adam_update(param, grad_chunk, p_cfg, scale_read=scale_read)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
"#, r#"        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        AnvilAndAdam._adam_update_step(
            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )

        return p_slice

    @torch.no_grad()
    def adam_update_shard(self, param: nn.Parameter, grad_chunk: Tensor):
        """Adam on this rank's shard of a sharded parameter whose gradient was reduced outside step()
        (value_embeds: perf/value_embed_pull.py). Same update, state and config as in step()."""
        p_cfg = self.param_cfgs[param]
        assert p_cfg.optim == "adam" and p_cfg.comms == "sharded" and param.grad is None
        self._adam_update(param, grad_chunk, p_cfg)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
"#),
  (r#"    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, *, scale_read=None) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul
"#, r#"    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul
"#)
]

namespace DispatchMemo

variable {ε α : Type}

/-- The three content caches keep successful results of the corresponding
source or PTX inspection. They do not cache file reads or launch choices. -/
structure Memo (inspect : String → Except ε α) where
  private mk ::
  private entries : List (String × α)
  private correct : ∀ entry ∈ entries, inspect entry.1 = .ok entry.2

variable {inspect : String → Except ε α}

def Memo.empty (inspect : String → Except ε α) : Memo inspect :=
  ⟨[], by simp⟩

private theorem Memo.found (memo : Memo inspect) (source : String) (entry : String × α)
    (hit : memo.entries.find? (fun item => item.1 == source) = some entry) :
    inspect source = .ok entry.2 := by
  have matched := List.find?_some (p := fun item : String × α => item.1 == source) hit
  have same : entry.1 = source := beq_iff_eq.mp matched
  rw [← same]
  exact memo.correct entry (List.mem_of_find?_eq_some hit)

private def Memo.retain (memo : Memo inspect) (source : String) (value : α)
    (computed : inspect source = .ok value) : Memo inspect where
  entries := ((source, value) :: memo.entries.filter (fun entry => entry.1 != source)).take 16
  correct := by
    intro entry member
    have original := List.mem_of_mem_take member
    rcases List.mem_cons.mp original with equal | previous
    · simpa [equal] using computed
    · exact memo.correct entry ((List.mem_filter.mp previous).1)

def Memo.access (memo : Memo inspect) (source : String) : Except ε α × Memo inspect :=
  match hit : memo.entries.find? (fun entry => entry.1 == source) with
  | some entry => (.ok entry.2, memo.retain source entry.2 (memo.found source entry hit))
  | none =>
    match computed : inspect source with
    | .error reason => (.error reason, memo)
    | .ok result => (.ok result, memo.retain source result computed)

theorem Memo.access_exact (memo : Memo inspect) (source : String) :
    (memo.access source).1 = inspect source := by
  unfold Memo.access
  split
  · rename_i entry hit
    exact (memo.found source entry hit).symm
  · split
    · rename_i reason computed
      exact computed.symm
    · rename_i value computed
      exact computed.symm

def Memo.trace (memo : Memo inspect) : List String → List (Except ε α)
  | [] => []
  | source :: rest =>
    let (result, next) := memo.access source
    result :: next.trace rest

theorem Memo.trace_exact (memo : Memo inspect) (reads : List String) :
    memo.trace reads = reads.map inspect := by
  induction reads generalizing memo with
  | nil => rfl
  | cons source rest ih =>
    change (memo.access source).1 :: (memo.access source).2.trace rest =
      inspect source :: rest.map inspect
    rw [Memo.access_exact, ih]

#print axioms Memo.access_exact
#print axioms Memo.trace_exact

end DispatchMemo

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
