"""Fuse the post-MUDD launches in a pinned Inductor wrapper."""
import ast
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock, get_ident
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

TORCH_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
_COMPILE_LOCK = RLock()
ROWS = (16384, 32768, 40960, 49152)
_FULL_VOCAB_ROWS = (16384, 40960)
_SAMPLED_ROWS = (16384, 32768, 49152)
_REFERENCE_SOURCE = '''
def kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = {size}
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 12
    x0 = xindex % 12
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ({mu}), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + {bias}).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr2 + ({delta}), None).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp4
    tmp8 = tmp5 + tmp7
    tl.store(out_ptr0 + x2, tmp8, None)
'''

def _kernel_body(kernel):
    return tuple(ast.dump(n, annotate_fields=False) for n in (kernel.args, *kernel.body))

def _reference_body(rows, plane):
    return _kernel_body(ast.parse(_REFERENCE_SOURCE.format(
        size=rows * 12,
        mu=f"{plane} + 10*x1" if plane else "10*x1",
        bias=14 + plane,
        delta=f"{plane*12} + x0 + 120*x1" if plane else "x0 + 120*x1",
    )).body[0])

_REFERENCE_KERNELS = {_reference_body(rows, plane): (rows, plane) for rows in ROWS for plane in range(10)}

def _call(stmt, symbol, args=None):
    if not isinstance(stmt, ast.Call) or ast.unparse(stmt.func) != symbol:
        return False
    return args is None or len(stmt.args) == args

def _metadata_value(d, field):
    if isinstance(d, ast.Dict):
        return next((v for k, v in zip(d.keys, d.values)
                     if isinstance(k, ast.Constant) and k.value == field), None)


def _supported_metadata(kernel):
    for decorator in kernel.decorator_list:
        if not _call(decorator, "triton_heuristics.pointwise"):
            continue
        data = next((kw.value for kw in decorator.keywords if kw.arg == "triton_meta"), None)
        fusion = _metadata_value(data, "enable_fp_fusion")
        signature = _metadata_value(data, "signature")
        expected = {"in_ptr0": "*bf16", "in_ptr1": "*bf16", "in_ptr2": "*bf16",
                    "out_ptr0": "*bf16", "xnumel": "i32", "XBLOCK": "constexpr"}
        return (isinstance(fusion, ast.Constant) and fusion.value is True and
                isinstance(signature, ast.Dict) and ast.literal_eval(signature) == expected)
    return False


def _candidate_assignments(module):
    found, norms = {}, {}
    for expr in module.body:
        if not isinstance(expr, ast.Assign) or not _call(expr.value, "async_compile.triton", 2):
            continue
        if len(expr.targets) != 1:
            continue
        target, = expr.targets
        if not isinstance(target, ast.Name):
            continue
        if not all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in expr.value.args):
            continue
        kernel_name, text = (a.value for a in expr.value.args)
        if "'enable_fp_fusion': True" not in text:
            continue
        fn = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
        if len(fn) != 1 or target.id != kernel_name:
            continue
        kernel = fn[0]
        body = _kernel_body(kernel)
        if (descriptor := _REFERENCE_KERNELS.get(body)) and _supported_metadata(kernel):
            found[target.id] = descriptor
        elif (rows := _NORM_KERNELS.get(body)) and _supported_norm_metadata(kernel, rows):
            norms[target.id] = rows
    return found, norms

def _identify_region(kernels, statements, start, device):
    rows = None
    names = {}
    allocations = []
    for i in range(10):
        allocation, get_stream, launch = statements[start + 3*i:start + 3*i + 3]
        if not isinstance(allocation, ast.Assign) or not isinstance(get_stream, ast.Assign):
            return
        if (len(get_stream.targets) != 1 or not isinstance(get_stream.targets[0], ast.Name)
                or ast.unparse(get_stream.value) != f"get_raw_stream({device})"):
            return
        stream = get_stream.targets[0].id
        if not isinstance(launch, ast.Expr) or not isinstance(launch.value, ast.Call):
            return
        op = launch.value
        fn = ast.unparse(op.func)
        if not fn.endswith(".run") or fn[:-4] not in kernels:
            return
        n, plane = kernels[fn[:-4]]
        test_shape = f"empty_strided_cuda((1, {n}, 12), ({n*12}, 12, 1), torch.bfloat16)"
        if len(allocation.targets) != 1 or ast.unparse(allocation.value) != test_shape:
            return
        name = ast.unparse(allocation.targets[0])
        if not isinstance(allocation.targets[0], ast.Name) or name in names.values() or plane in names:
            return
        if len(op.args) != 5 or not all(isinstance(arg, ast.Name) for arg in op.args[:4]):
            return
        size = op.args[-1]
        if not isinstance(size, ast.Constant) or size.value != n*12 or name != op.args[-2].id:
            return
        if [ast.unparse(kw) for kw in op.keywords] != [f"stream={stream}"]:
            return
        args = tuple(ast.unparse(arg) for arg in op.args[:3])
        if rows is None:
            rows, inputs, stream_read = n, args, ast.unparse(get_stream)
        if n != rows or args != inputs or name in inputs or ast.unparse(get_stream) != stream_read:
            return
        names[plane] = name
        allocations.append(allocation)
    if stream in (*inputs, *names.values()):
        return
    return rows, inputs, names, allocations, stream_read


_REFERENCE_NORM_SOURCE = """
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = {rows}
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = xindex
    r0_2 = r0_index // 64
    tmp0 = tl.load(in_ptr0 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr6 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr7 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr8 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr9 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr10 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr11 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr12 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr13 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr14 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr15 + (r0_2 + 12 * x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr16 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr17 + (12 * x0 + r0_3 // 64), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr18 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr19 + (12 * x0 + r0_3 // 64), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr20 + (r0_3 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 + tmp19
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp31 = tmp29 * tmp30
    tmp32 = tmp28 + tmp31
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 + tmp35
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, R0_BLOCK])
    tmp45 = tl.where(r0_mask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None].to(tl.float32)
    tmp47 = 768.0
    tmp48 = tmp46 / tmp47
    tmp49 = 1.1920928955078125e-07
    tmp50 = tmp48 + tmp49
    tmp51 = libdevice.rsqrt(tmp50)
    tmp52 = tmp41 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = 4.4799999999999995
    tmp55 = tmp53 * tmp54
    tmp56 = tmp55.to(tl.float8e4nv)
    tl.store(out_ptr0 + (r0_3 + 768 * x0), tmp41, r0_mask)
    tl.store(out_ptr2 + (r0_3 + 768 * x0), tmp56, r0_mask)
    tl.store(out_ptr1 + x0, tmp46, None)
"""


def _reference_norm_body(rows):
    return _kernel_body(ast.parse(_REFERENCE_NORM_SOURCE.format(rows=rows)).body[0])


_NORM_KERNELS = {_reference_norm_body(rows): rows for rows in ROWS}


def _keyword_map(call):
    return {kw.arg: kw.value for kw in call.keywords}


def _literal_eq(expr, value):
    try:
        return ast.literal_eval(expr) == value
    except (TypeError, ValueError):
        return False


def _supported_norm_metadata(kernel, rows):
    signature = {**{f"in_ptr{i}": "*bf16" for i in range(21)}, "out_ptr0": "*fp32", "out_ptr1": "*fp32",
                 "out_ptr2": "*fp8e4nv"}
    return _supported_reduction_metadata(kernel, rows, signature, [])


def _supported_reduction_metadata(kernel, rows, pointers, mutated):
    for decorator in kernel.decorator_list:
        if not _call(decorator, "triton_heuristics.persistent_reduction", 0):
            continue
        kw = _keyword_map(decorator)
        if (not _literal_eq(kw.get("size_hints"), {"x": 1 << (rows - 1).bit_length(), "r0_": 1024})
                or "reduction_hint" not in kw
                or ast.unparse(kw["reduction_hint"]) != "ReductionHint.INNER"):
            return False
        triton_meta = kw.get("triton_meta")
        inductor_meta = kw.get("inductor_meta")
        # Admit only the default NVIDIA reduction cfg: XBLOCK=1, one warp.
        if (not isinstance(triton_meta, ast.Dict) or not isinstance(inductor_meta, ast.Dict)
                or any(not isinstance(k, ast.Constant) for k in (*triton_meta.keys, *inductor_meta.keys))):
            return False
        signature = {**pointers, "xnumel": "i32", "r0_numel": "i32", "XBLOCK": "constexpr"}
        config = [{(i,): [["tt.divisibility", 16]] for i in range(len(pointers) + 2)}]
        metadata = {"signature": signature, "native_matmul": False, "enable_fp_fusion": True,
                    "constants": {}, "configs": config}
        if not all(_literal_eq(_metadata_value(triton_meta, k), v) for k, v in metadata.items()):
            return False
        defaults = {"grid_type": "Grid1D", "max_autotune": False,
                    "max_autotune_pointwise": False, "no_x_dim": None,
                    "mutated_arg_names": mutated}
        if not all(_literal_eq(_metadata_value(inductor_meta, k), v) for k, v in defaults.items()):
            return False
        unexpected = ("RSPLIT_SIZE", "tma_info", "enable_warp_specialization", "coordinate_descent_tuning")
        return not any(_metadata_value(inductor_meta, field) is not None for field in unexpected)
    return False


def _parse_launch(stmt):
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None, None
    call = stmt.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "run" and isinstance(call.func.value, ast.Name):
        return call.func.value.id, call
    return None, None


def _assigns(stmt, expected):
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and ast.unparse(stmt.value) == expected)


def _identify_norm(kernels, statements, start, coeff_region):
    if len(statements) < start + 8:
        return
    rows, coef_inputs, coef_outputs, _, stream_read = coeff_region
    deletions = statements[start:start + 3]
    start += 3
    if not all(isinstance(s, ast.Delete) and len(s.targets) == 1 for s in deletions):
        return
    if {ast.unparse(s.targets[0]) for s in deletions} != set(coef_inputs):
        return
    # Inductor may allocate the later CE transpose destination here. Keep
    # its original position relative to the RMSNorm allocations.
    transpose_layout = f"empty_strided_cuda((768, {rows}), ({rows}, 1), torch.float8_e4m3fn)"
    early_allocations = []
    if _assigns(statements[start], transpose_layout):
        early_allocations.append(statements[start])
        start += 1
    if len(statements) < start + 5:
        return
    allocations = statements[start:start + 3]
    stream_stmt, launch = statements[start + 3:start + 5]
    layouts = (
        f"empty_strided_cuda((1, {rows}, 768), ({rows*768}, 768, 1), torch.float32)",
        f"empty_strided_cuda((1, {rows}, 1), ({rows}, 1, {rows}), torch.float32)",
        f"empty_strided_cuda(({rows}, 768), (768, 1), torch.float8_e4m3fn)",
    )
    if not all(_assigns(stmt, layout) for stmt, layout in zip(allocations, layouts, strict=True)):
        return
    norm_outputs = tuple(stmt.targets[0].id for stmt in allocations)
    name, op = _parse_launch(launch)
    if kernels.get(name) != rows or ast.unparse(stream_stmt) != stream_read:
        return
    if len(op.args) != 26 or not all(isinstance(a, ast.Name) for a in op.args[:24]):
        return
    operands = tuple(a.id for a in op.args[:24])
    if (operands[21:] != norm_outputs
            or operands[1:21:2] != tuple(coef_outputs[i] for i in range(10))
            or not _literal_eq(op.args[-2], rows) or not _literal_eq(op.args[-1], 768)
            or [ast.unparse(kw) for kw in op.keywords] != [f"stream={stream_stmt.targets[0].id}"]):
        return
    output_allocations = {*norm_outputs, *coef_outputs.values(), *(a.targets[0].id for a in early_allocations)}
    readers = {*coef_inputs, *operands[:21:2]}
    if len(output_allocations) != 13 + len(early_allocations) or output_allocations & readers:
        return
    mixed, square_sum, fp8 = norm_outputs
    return [*early_allocations, *allocations], (*coef_inputs, mixed, square_sum, *operands[:21], fp8), deletions, launch


_BACKWARD_GRADIENT_ONLY = """
def kernel(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = {rows}
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (r0_1 + 768 * x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = 768.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.1920928955078125e-07
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp1 * tmp13
    tmp15 = -0.5
    tmp16 = tmp7 * tmp15
    tmp17 = tmp13 * tmp13
    tmp18 = tmp17 * tmp13
    tmp19 = tmp16 * tmp18
    tmp20 = 0.0013020833333333333
    tmp21 = tmp19 * tmp20
    tmp22 = 2.0
    tmp23 = tmp2 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp14 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768 * x0), tmp26, r0_mask)
"""


_BACKWARD_WITH_ENCODING = """
def kernel(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = {rows}
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768 * x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (r0_1 + 768 * x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = 768.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1.1920928955078125e-07
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = 4.4799999999999995
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float8e4nv)
    tmp19 = tmp1 * tmp13
    tmp20 = -0.5
    tmp21 = tmp7 * tmp20
    tmp22 = tmp13 * tmp13
    tmp23 = tmp22 * tmp13
    tmp24 = tmp21 * tmp23
    tmp25 = 0.0013020833333333333
    tmp26 = tmp24 * tmp25
    tmp27 = 2.0
    tmp28 = tmp2 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp19 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 768 * x0), tmp18, r0_mask)
    tl.store(in_out_ptr0 + (r0_1 + 768 * x0), tmp31, r0_mask)
"""


def _backward_assignments(module):
    expected = {}
    for rows in ROWS:
        for encoded, template in ((True, _BACKWARD_WITH_ENCODING), (False, _BACKWARD_GRADIENT_ONLY)):
            kernel = ast.parse(template.format(rows=rows)).body[0]
            expected[_kernel_body(kernel)] = (rows, encoded)
    matched = {}
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign) or not _call(stmt.value, "async_compile.triton", 2):
            continue
        if not all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in stmt.value.args):
            continue
        name, source = (x.value for x in stmt.value.args)
        definitions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        if len(definitions) != 1 or len(stmt.targets) != 1 or ast.unparse(stmt.targets[0]) != name:
            continue
        function, = definitions
        layout = expected.get(_kernel_body(function))
        if layout is None:
            continue
        rows, encoded = layout
        runtime_signature = {"in_out_ptr0": "*bf16", "in_ptr0": "*fp32", "in_ptr1": "*fp32"}
        if encoded:
            runtime_signature["out_ptr1"] = "*fp8e4nv"
        if _supported_reduction_metadata(function, rows, runtime_signature, ["in_out_ptr0"]):
            matched[name] = layout
    return matched


def _identify_cached_backward(tree, candidates, statements, norm_index, rows, stream_read, norm):
    allocations, arguments, _, forward_launch = norm
    mixed, statistic = arguments[3:5]
    for i in range(norm_index + 2, len(statements) - 2):
        launch = statements[i]
        name, call = _parse_launch(launch)
        layout = candidates.get(name)
        if layout is None or layout[0] != rows or ast.unparse(statements[i - 1]) != stream_read:
            continue
        ports = 4 if layout[1] else 3
        if len(call.args) != ports + 2 or not all(isinstance(a, ast.Name) for a in call.args[:ports]):
            continue
        stream, _ = stream_read.split(" = ", 1)
        if ([ast.unparse(kw) for kw in call.keywords] != [f"stream={stream}"]
                or not _literal_eq(call.args[-2], rows) or not _literal_eq(call.args[-1], 768)):
            continue
        gradient, residual, raw_statistic, *auxiliary = (arg.id for arg in call.args[:ports])
        if (residual, raw_statistic) != (mixed, statistic):
            continue
        if len({gradient, mixed, statistic, *auxiliary}) != ports:
            continue
        deletion = statements[i + 2]
        if not (isinstance(deletion, ast.Delete) and len(deletion.targets) == 1
                and isinstance(deletion.targets[0], ast.Name) and deletion.targets[0].id == statistic):
            continue
        # The original FP32 allocation becomes private to this producer/consumer pair.
        actual_uses = {node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == statistic}
        expected_uses = {allocations[-2].targets[0], forward_launch.value.args[22], call.args[2],
                         deletion.targets[0]}
        if actual_uses != expected_uses:
            continue
        return launch, (gradient, mixed, statistic, *auxiliary)


def _identify_retained_activation(tree, statements, rows, norm, cached):
    allocations, _, _, forward = norm
    backward, ports = cached
    if len(ports) != 4:
        return
    original = forward.value.args[23]
    rematerialized = backward.value.args[3]
    start, stop = statements.index(forward), statements.index(backward)
    layout = f"empty_strided_cuda(({rows}, 768), (768, 1), torch.float8_e4m3fn)"
    producers = [s for s in statements[start + 1:stop]
                 if _assigns(s, layout) and s.targets[0].id == rematerialized.id]
    deaths = [(i, s) for i, s in enumerate(statements[start + 1:stop], start + 1)
              if _deletes(s, original.id)]
    if len(producers) != 1 or len(deaths) != 1:
        return
    producer, = producers
    (i, original_delete), = deaths
    if not start + 1 < i:
        return
    logits = statements[i - 1]
    if not (isinstance(logits, ast.Expr) and _call(logits.value, "extern_kernels._scaled_mm", 4)
            and ast.unparse(logits.value.args[0]) == original.id
            and any(k.arg == "out_dtype" and ast.unparse(k.value) == "torch.float8_e4m3fn"
                    for k in logits.value.keywords)):
        return
    consumers = [(i, s.value) for i, s in enumerate(statements[stop + 1:], stop + 1)
                 if isinstance(s, ast.Expr)
                 and _call(s.value, "torch.ops.nanogpt.track1_ce_transpose_operands.default", 4)
                 and ast.unparse(s.value.args[0]) == rematerialized.id]
    if len(consumers) != 1:
        return
    (i, transpose), = consumers
    final_deletions = [s for s in statements[i + 1:] if _deletes(s, rematerialized.id)]
    if len(final_deletions) != 1:
        return
    final_delete, = final_deletions
    for stmt in (producer, original_delete, final_delete):
        if stmt.lineno != stmt.end_lineno or sum(s.lineno == stmt.lineno for s in statements) != 1:
            return
    original_uses = {allocations[-1].targets[0], original, logits.value.args[0], original_delete.targets[0]}
    remat_uses = {producer.targets[0], rematerialized, transpose.args[0], final_delete.targets[0]}
    for name, expected in ((original.id, original_uses), (rematerialized.id, remat_uses)):
        if {n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name} != expected:
            return
    return producer, original_delete, final_delete, original.id


def _deletes(stmt, name):
    return (isinstance(stmt, ast.Delete) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name)


_TRANSPOSE_PAIR = "torch.ops.nanogpt.track1_ce_transpose_operands.default"



def _symbol(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _simple_target(node):
    return node.targets[0].id if (
        isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    ) else None


def _dtype(signature, name):
    return next((value.value for key, value in zip(signature.keys, signature.values, strict=True)
                 if _literal_eq(key, name) and isinstance(value, ast.Constant)), None)


def _same_assignments(body, expected):
    for name, expression in expected.items():
        matches = [node for node in body if _simple_target(node) == name]
        if len(matches) != 1 or ast.unparse(matches[0].value) != expression:
            return False
    return True


def _complete_activation_store(fn, loop, store):
    rows = next((n.value.value for n in fn.body if _simple_target(n) == "xnumel"
                 and isinstance(n.value, ast.Constant)), 0)
    if type(rows) is not int or rows not in ROWS or rows & (rows - 1):
        return None
    # The one-dimensional Inductor launch grid tiles the full row extent.
    # A power-of-two extent is divisible by its power-of-two row tile.
    if not _same_assignments(fn.body, {
        "xnumel": str(rows), "r0_numel": "768", "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:, None]", "x0": "xindex",
        "xmask": "tl.full([XBLOCK], True, tl.int1)[:, None]",
        "r0_base": "tl.arange(0, R0_BLOCK)[None, :]",
    }):
        return None
    if (not _symbol(loop.target, "r0_offset") or ast.unparse(loop.iter) != "range(0, r0_numel, R0_BLOCK)"
            or not _same_assignments(loop.body, {"r0_index": "r0_offset + r0_base", "r0_3": "r0_index",
                                                 "r0_mask": "r0_index < r0_numel"})):
        return None
    if ast.unparse(store.args[0].right) != "r0_3 + 768 * x0" or not _symbol(store.args[2], "r0_mask"):
        return None
    return rows


def _reduction_kernels(wrapper, fragment):
    for stmt in wrapper.body:
        if not isinstance(stmt, ast.Assign) or not _call(stmt.value, "async_compile.triton", 2):
            continue
        symbol, text = stmt.value.args
        if (not isinstance(symbol, ast.Constant) or not isinstance(symbol.value, str)
                or _simple_target(stmt) != symbol.value or not isinstance(text, ast.Constant)
                or not isinstance(text.value, str) or "rsqrt" not in symbol.value or fragment not in symbol.value):
            continue
        kernel_ast = ast.parse(text.value)
        for fn in kernel_ast.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name != symbol.value:
                continue
            reductions = [d for d in fn.decorator_list if _call(d, "triton_heuristics.reduction", 0)]
            if len(reductions) != 1:
                continue
            grids = [kw.value for kw in reductions[0].keywords if kw.arg == "inductor_meta"]
            if (len(grids) != 1 or not isinstance(grids[0], ast.Dict)
                    or not _literal_eq(_metadata_value(grids[0], "grid_type"), "Grid1D")
                    or not _literal_eq(_metadata_value(grids[0], "no_x_dim"), False)):
                continue
            metadata = [kw.value for kw in reductions[0].keywords if kw.arg == "triton_meta"]
            if len(metadata) != 1 or not isinstance(metadata[0], ast.Dict):
                continue
            signature = next((v for k, v in zip(metadata[0].keys, metadata[0].values, strict=True)
                              if isinstance(k, ast.Constant) and k.value == "signature"), None)
            if not isinstance(signature, ast.Dict):
                continue
            yield symbol.value, text, fn, signature


def _duplicated_stores(wrapper):
    for symbol, text, fn, signature in _reduction_kernels(wrapper, "track1_ce"):
        positions = {port.arg: i for i, port in enumerate(fn.args.args)}
        for loop in fn.body:
            if not isinstance(loop, ast.For) or len(loop.body) < 2:
                continue
            stores = loop.body[-2:]
            if not all(isinstance(n, ast.Expr) and _call(n.value, "tl.store", 3)
                       and isinstance(n.value.args[0], ast.BinOp) and isinstance(n.value.args[0].op, ast.Add)
                       and isinstance(n.value.args[0].left, ast.Name) and not n.value.keywords for n in stores):
                continue
            first, second = [n.value for n in stores]
            names = (first.args[0].left, second.args[0].left)
            if names[0].id == names[1].id or any(n.id not in positions or _dtype(signature, n.id) != "*fp8e4nv" for n in names):
                continue
            if ([ast.dump(x) for x in (first.args[0].right, *first.args[1:])]
                    != [ast.dump(x) for x in (second.args[0].right, *second.args[1:])]):
                continue
            if any({n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == p.id} != {p} for p in names):
                continue
            rows = _complete_activation_store(fn, loop, first)
            if rows:
                yield symbol, tuple(positions[n.id] for n in names), rows, text, stores[1]


def _origin(name, body, start, layout):
    """Follow only Inductor's move-and-delete chains back to a fresh allocation."""
    for index in range(start - 1, -1, -1):
        stmt = body[index]
        if _simple_target(stmt) != name:
            continue
        if ast.unparse(stmt.value) == layout:
            return stmt
        if (not isinstance(stmt.value, ast.Name) or index + 1 >= start
                or not _deletes(body[index + 1], stmt.value.id)):
            return None
        name = stmt.value.id
        start = index
    return None


def _find_swap(wrapper, body, symbol, ports, rows):
    for call_pos, stmt in enumerate(body):
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        launch = stmt.value
        if ast.unparse(launch.func) != f"{symbol}.run" or len(launch.args) <= max(ports):
            continue
        primary, duplicate = (launch.args[i] for i in ports)
        if not isinstance(primary, ast.Name) or not isinstance(duplicate, ast.Name) or primary.id == duplicate.id:
            continue
        if ast.unparse(launch.args[-2]) != str(rows) or ast.unparse(launch.args[-1]) != "768":
            continue
        layout = f"empty_strided_cuda(({rows}, 768), (768, 1), torch.float8_e4m3fn)"
        a, b = (_origin(p.id, body, call_pos, layout) for p in (primary, duplicate))
        if a is None or b is None or a is b:
            continue
        relative = body[call_pos + 1:]
        views = [(i, s) for i, s in enumerate(relative) if _simple_target(s) is not None
                 and ast.unparse(s.value) == f"reinterpret_tensor({primary.id}, (768, {rows}), ({rows}, 1), 0)"]
        if len(views) != 1:
            continue
        (v, view), = views
        transposes = [(i, s.value) for i, s in enumerate(relative) if isinstance(s, ast.Expr)
                      and _call(s.value, _TRANSPOSE_PAIR, 4) and _symbol(s.value.args[0], duplicate.id)
                      and _symbol(s.value.args[1], _simple_target(view))]
        if len(transposes) != 1:
            continue
        (t, transpose), = transposes
        logits = [s.value for s in relative[:v] if isinstance(s, ast.Expr)
                  and _call(s.value, "extern_kernels._scaled_mm", 4) and _symbol(s.value.args[0], primary.id)
                  and any(k.arg == "out_dtype" and ast.unparse(k.value) == "torch.float8_e4m3fn"
                          for k in s.value.keywords)]
        if len(logits) != 1 or not v + 1 < t or not _deletes(relative[v + 1], primary.id):
            continue
        deletes = [s for s in relative[t + 1:] if _deletes(s, duplicate.id)]
        if len(deletes) != 1:
            continue
        delete, = deletes
        liveness = ((primary.id, {primary, view.value.args[0], relative[v + 1].targets[0], logits[0].args[0]}),
                    (duplicate.id, {duplicate, transpose.args[0], delete.targets[0]}))
        if any({n for n in ast.walk(wrapper) if isinstance(n, ast.Name) and n.id == name
                and not isinstance(n.ctx, ast.Store)} != expected for name, expected in liveness):
            continue
        if any(sum(_simple_target(n) == name for n in body) != 1 for name, _ in liveness):
            continue
        if any(n.lineno != n.end_lineno for n in (view, relative[v + 1], transpose, delete)):
            continue
        yield view, relative[v + 1], transpose, delete, primary.id, duplicate.id


def _patch_duplicate_activation(source, tree):
    for symbol, ports, rows, kernel, store in _duplicated_stores(tree):
        for node in ast.walk(tree):
            if not isinstance(node, ast.With) or len(node.items) != 1:
                continue
            guard = node.items[0].context_expr
            if not _call(guard, "torch.cuda._DeviceGuard", 1):
                continue
            for view, first_delete, transpose, final_delete, primary, duplicate in _find_swap(
                    tree, node.body, symbol, ports, rows):
                lines = source.splitlines(keepends=True)
                store_index = kernel.lineno + store.lineno - 2
                old_store = ast.get_source_segment(kernel.value, store)
                if not kernel.lineno - 1 < store_index < kernel.end_lineno - 1:
                    continue
                if lines[store_index].strip() != old_store or store.lineno != store.end_lineno:
                    continue
                touched = {s.lineno for s in (view, first_delete, transpose, final_delete)}
                allowed = {view, first_delete, final_delete, *(s for s in node.body
                           if isinstance(s, ast.Expr) and s.value is transpose)}
                if any(s.lineno in touched and s not in allowed for s in node.body):
                    continue
                lines[store_index] = "\n"
                view.value.args[0].id = duplicate
                first_delete.targets[0].id = duplicate
                transpose.args[0].id = primary
                final_delete.targets[0].id = primary
                replacements = {view.lineno: [ast.unparse(view)], transpose.lineno: [ast.unparse(transpose)],
                                final_delete.lineno: [ast.unparse(final_delete)]}
                replacements.setdefault(first_delete.lineno, []).append(ast.unparse(first_delete))
                for lineno, statements in replacements.items():
                    line = lines[lineno - 1]
                    indent = line[:len(line) - len(line.lstrip())]
                    lines[lineno - 1] = indent + "; ".join(statements) + "\n"
                return "".join(lines), 1
    return source, 0


_RESIDUAL_EPILOGUE = ast.parse("""
r0_index = r0_offset + r0_base
r0_mask = r0_index < r0_numel
roffset = r0_offset
rindex = r0_index
r0_3 = r0_index
tmp135 = tl.load(in_out_ptr0 + (r0_3 + 768 * x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
tmp136 = tmp135.to(tl.float32)
tmp137 = 768.0
tmp138 = tmp133 / tmp137
tmp139 = 1.1920928955078125e-07
tmp140 = tmp138 + tmp139
tmp141 = libdevice.rsqrt(tmp140)
tmp142 = tmp136 * tmp141
tmp143 = tmp142.to(tl.float32)
tmp144 = 4.4799999999999995
tmp145 = tmp143 * tmp144
tmp146 = tmp145.to(tl.float8e4nv)
tl.store(out_ptr1 + (r0_3 + 768 * x0), tmp146, r0_mask)
tl.store(out_ptr2 + (r0_3 + 768 * x0), tmp146, r0_mask)
""").body
_RESIDUAL_EPILOGUE = tuple(map(ast.dump, _RESIDUAL_EPILOGUE))


def _residual_edits(fn, signature):
    if _dtype(signature, "in_out_ptr0") != "*bf16" or _dtype(signature, "out_ptr0") != "*fp32":
        return
    rows = next((n.value.value for n in fn.body if _simple_target(n) == "xnumel"
                 and isinstance(n.value, ast.Constant)), None)
    if rows not in ROWS or not _same_assignments(fn.body, {
        "xnumel": str(rows), "r0_numel": "768", "xoffset": "tl.program_id(0) * XBLOCK",
        "xindex": "xoffset + tl.arange(0, XBLOCK)[:, None]", "x0": "xindex",
        "r0_base": "tl.arange(0, R0_BLOCK)[None, :]",
    }):
        return
    if any(isinstance(n, ast.Name) and n.id == "residual_register" for n in ast.walk(fn)):
        return
    loops = [n for n in fn.body if isinstance(n, ast.For)]
    if len(loops) != 2 or any(not _symbol(loop.target, "r0_offset") or loop.orelse
                              or ast.unparse(loop.iter) != "range(0, r0_numel, R0_BLOCK)" for loop in loops):
        return
    first, second = loops
    epilogue = tuple(map(ast.dump, second.body))
    if epilogue not in (_RESIDUAL_EPILOGUE[:-1], _RESIDUAL_EPILOGUE) or second is not fn.body[-1]:
        return
    at = fn.body.index(first)
    if at == 0 or ast.unparse(fn.body[at - 1]) != "_tmp133 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)":
        return
    if [ast.unparse(n) for n in fn.body[at + 1:-1]] != [
        "tmp133 = tl.sum(_tmp133, 1)[:, None]", "tl.store(out_ptr0 + x0, tmp133, None)"
    ]:
        return
    if not first.body or not _same_assignments(first.body, {
        "r0_index": "r0_offset + r0_base", "r0_3": "r0_index", "r0_mask": "r0_index < r0_numel",
    }):
        return
    store, reload = first.body[-1], second.body[5]
    if ast.unparse(store) != "tl.store(in_out_ptr0 + (r0_3 + 768 * x0), tmp129, r0_mask)":
        return
    if {n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "in_out_ptr0"} != {
            store.value.args[0].left, reload.value.func.value.args[0].left}:
        return
    # The tail of the second loop only uses masked stores. This keeps
    # the BF16 rounding of the write while skipping its matching load.
    return rows, (
        (fn.body[at - 1], "; residual_register = tl.full([XBLOCK, R0_BLOCK], 0, tl.bfloat16)"),
        (store, "; residual_register = tmp129.to(tl.bfloat16) if R0_BLOCK >= 768 else residual_register"),
        (reload, "= residual_register.to(tl.float32) if R0_BLOCK >= 768 else "),
    )


def _residual_rewrites(wrapper):
    for symbol, kernel, fn, signature in _reduction_kernels(wrapper, "mean_mul_permute_pow_rsqrt"):
        match = _residual_edits(fn, signature)
        if match is None:
            continue
        rows, edits = match
        ports = {arg.arg: i for i, arg in enumerate(fn.args.args)}
        if not {"in_out_ptr0", "out_ptr0"} <= ports.keys():
            continue
        for node in ast.walk(wrapper):
            if not isinstance(node, ast.With) or len(node.items) != 1 or not _call(
                    node.items[0].context_expr, "torch.cuda._DeviceGuard", 1):
                continue
            for i, stmt in enumerate(node.body):
                if not isinstance(stmt, ast.Expr) or not _call(stmt.value, f"{symbol}.run", len(fn.args.args) - 2):
                    continue
                args = stmt.value.args
                if not _literal_eq(args[-2], rows) or not _literal_eq(args[-1], 768):
                    continue
                residual, statistic = (args[ports[name]] for name in ("in_out_ptr0", "out_ptr0"))
                if not all(isinstance(arg, ast.Name) for arg in (residual, statistic)):
                    continue
                left = _origin(residual.id, node.body, i,
                               f"empty_strided_cuda((1, {rows}, 12, 64), ({rows * 768}, 768, 64, 1), torch.bfloat16)")
                right = _origin(statistic.id, node.body, i,
                                f"empty_strided_cuda((1, {rows}, 1), ({rows}, 1, {rows}), torch.float32)")
                if left is not None and right is not None and left is not right:
                    yield symbol, rows, kernel, edits
                    break


def _patch_residual_reuse(source, tree):
    lines = source.splitlines(keepends=True)
    count = 0
    for _, _, kernel, edits in _residual_rewrites(tree):
        updated = {}
        for stmt, replacement in edits:
            i = kernel.lineno + stmt.lineno - 2
            if not (kernel.lineno - 1 < i < kernel.end_lineno - 1 and stmt.lineno == stmt.end_lineno
                    and lines[i].strip() == ast.get_source_segment(kernel.value, stmt)):
                break
            text = lines[i].rstrip("\n")
            if replacement.startswith(";"):
                updated[i] = text + replacement + "\n"
            else:
                updated[i] = " " * stmt.col_offset + "tmp135 " + replacement + ast.unparse(stmt.value) + "\n"
        else:
            for index, line in updated.items():
                lines[index] = line
            count += 1
    return ("".join(lines), count) if count else (source, 0)


_COLUMN_GROUPS = ((9, 1), (6, 4), (3, 1))
_COLUMN_REDUCTION_BODY = '''
xnumel = ROWS
r0_numel = 12
R0_BLOCK: tl.constexpr = 16
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
r0_index = tl.arange(0, R0_BLOCK)[None, :]
r0_offset = 0
r0_mask = r0_index < r0_numel
roffset = r0_offset
rindex = r0_index
r0_1 = r0_index
x0 = xindex
tmp0 = tl.load(in_ptr0 + (r0_1 + 12*x0), r0_mask, other=0.0).to(tl.float32)
tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
tmp3 = tl.where(r0_mask, tmp1, 0)
tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
tl.store(out_ptr0 + (10*x0), tmp4, None)
'''


@dataclass
class _ColumnReduction:
    index: int
    src: str
    base: str
    column: int
    kernel: str
    rows: int
    stream: str


@dataclass
class _GradientPacking:
    kernel: str
    index: int
    base: str
    column_zero_input: str


def _column_reduction_kernels(tree, *, aligned_output=False):
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not _call(stmt.value, 'async_compile.triton', 2):
            continue
        symbol, source = stmt.value.args
        if not isinstance(symbol, ast.Constant) or not isinstance(symbol.value, str) or not symbol.value.startswith(
                'triton_per_fused_squeeze_sum_') or _simple_target(stmt) != symbol.value:
            continue
        if not isinstance(source, ast.Constant) or not isinstance(source.value, str):
            continue
        functions = [node for node in ast.parse(source.value).body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            continue
        fn, = functions
        signature = 'in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr'
        if fn.name != symbol.value or ast.unparse(fn.args) != signature or len(fn.decorator_list) != 2:
            continue
        heuristics, jit = fn.decorator_list
        if not _call(heuristics, 'triton_heuristics.persistent_reduction', 0) or ast.unparse(jit) != 'triton.jit':
            continue
        meta = {kw.arg: kw.value for kw in heuristics.keywords}
        info, compiler = meta.get('inductor_meta'), meta.get('triton_meta')
        if not isinstance(info, ast.Dict) or not isinstance(compiler, ast.Dict):
            continue
        if (not _literal_eq(_metadata_value(info, 'grid_type'), 'Grid1D') or
                not _literal_eq(_metadata_value(info, 'no_x_dim'), None) or
                not _literal_eq(_metadata_value(compiler, 'enable_fp_fusion'), True) or
                not _literal_eq(_metadata_value(compiler, 'signature'), {
                    'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32',
                    'r0_numel': 'i32', 'XBLOCK': 'constexpr'})):
            continue
        config = {(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}
        if aligned_output:
            config[(1,)] = [['tt.divisibility', 16]]
        if (_literal_eq(_metadata_value(info, 'coordinate_descent_tuning'), True) or
                not _literal_eq(_metadata_value(compiler, 'configs'), [config])):
            continue
        rows = next((r for r in ROWS if ast.dump(ast.Module(body=fn.body, type_ignores=[])) ==
                     ast.dump(ast.parse(_COLUMN_REDUCTION_BODY.replace('ROWS', str(r))))), None)
        if rows is not None:
            yield symbol.value, rows


def _column_reduction_launch(body, i, rows_by_kernel, device, *, column_zero=False):
    stmt = body[i]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return
    call = stmt.value
    callee = ast.unparse(call.func)
    kernel = callee.removesuffix('.run')
    if kernel == callee or kernel not in rows_by_kernel or len(call.args) != 4:
        return
    rows = rows_by_kernel[kernel]
    src, dest, r, width = call.args
    if not (isinstance(src, ast.Name) and isinstance(dest, ast.Name) and
            _literal_eq(r, rows) and _literal_eq(width, 12) and len(call.keywords) == 1):
        return
    stream = _simple_target(body[i - 1])
    kw = call.keywords[0]
    if (stream is None or kw.arg != 'stream' or not _symbol(kw.value, stream) or
            ast.unparse(body[i - 1].value) != f'get_raw_stream({device})'):
        return
    alias = body[i - 2]
    if _simple_target(alias) != dest.id or not _call(alias.value, 'reinterpret_tensor', 4) or alias.value.keywords:
        return
    base, size, stride, column = alias.value.args
    if not (isinstance(base, ast.Name) and _literal_eq(size, (1, rows, 1)) and
            _literal_eq(stride, (rows * 10, 10, 1)) and isinstance(column, ast.Constant) and
            type(column.value) is int and (column.value == 0 if column_zero else 0 < column.value < 10)):
        return
    return _ColumnReduction(i, src.id, base.id, column.value, kernel, rows, stream)


def _consecutive_column_reductions(body, rows, device):
    group = []
    for i in range(2, len(body)):
        entry = _column_reduction_launch(body, i, rows, device)
        if entry is None:
            continue
        if group:
            prev = group[-1]
            between = body[prev.index + 1:i - 2]
            if ((entry.rows, entry.kernel, entry.base, entry.stream, entry.column + 1) !=
                    (prev.rows, prev.kernel, prev.base, prev.stream, prev.column) or
                    (between and (len(between) != 1 or not _deletes(between[0], prev.src)))):
                yield group
                group = []
        group.append(entry)
    if group:
        yield group


def _independent_column_buffers(body, group):
    first = group[0]
    prefix = body[:first.index - 2]
    output_aliases = [node for node in prefix if _simple_target(node) == first.base]
    if len(output_aliases) != 1:
        return False
    alias, = output_aliases
    names = tuple(g.src for g in group)
    if len(set(names)) != len(group) or first.base in names or not _call(alias.value, 'reinterpret_tensor', 4):
        return False
    out, size, stride, offset = alias.value.args
    rows = first.rows
    if not (isinstance(out, ast.Name) and _literal_eq(size, (1, rows, 10)) and
            _literal_eq(stride, (rows * 10, 10, 1)) and _literal_eq(offset, 0)):
        return False
    base = _origin(out.id, prefix, prefix.index(alias), f'empty_strided_cuda(({rows}, 10), (10, 1), torch.bfloat16)')
    if base is None:
        return False
    source_layout = f'empty_strided_cuda((1, {rows}, 12, 1), ({rows * 12}, 12, 1, 1), torch.bfloat16)'
    srcs = [_origin(name, prefix, len(prefix), source_layout) for name in names]
    return all(s is not None for s in srcs) and len(set(srcs)) == len(group)


def _gradient_packing_kernels(tree):
    """Identify the packing launch by its read-only planes and sole write port."""
    inputs = tuple(f'in_ptr{i}' for i in range(10))
    ports = ('in_out_ptr0', *inputs)
    signature = {**dict.fromkeys(ports, '*bf16'), 'xnumel': 'i32', 'XBLOCK': 'constexpr'}
    loads = {f"tl.load({name} + (x0 + 12 * x2), None, eviction_policy='evict_last')" for name in inputs}
    pure_calls = {'tl.program_id', 'tl.arange', 'tl.full', 'tl.where'}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not _call(stmt.value, 'async_compile.triton', 2):
            continue
        symbol, source = stmt.value.args
        if (not isinstance(symbol, ast.Constant) or not isinstance(symbol.value, str)
                or not symbol.value.startswith('triton_poi_fused_add_mul_select_backward_squeeze_')
                or _simple_target(stmt) != symbol.value or not isinstance(source, ast.Constant)
                or not isinstance(source.value, str)):
            continue
        functions = [node for node in ast.parse(source.value).body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            continue
        fn, = functions
        args = ', '.join((*ports, 'xnumel', 'XBLOCK: tl.constexpr'))
        if fn.name != symbol.value or ast.unparse(fn.args) != args or len(fn.decorator_list) != 2:
            continue
        heuristics, jit = fn.decorator_list
        if not _call(heuristics, 'triton_heuristics.pointwise', 0) or ast.unparse(jit) != 'triton.jit':
            continue
        meta = {kw.arg: kw.value for kw in heuristics.keywords}
        info, compiler = meta.get('inductor_meta'), meta.get('triton_meta')
        if not (_literal_eq(_metadata_value(compiler, 'signature'), signature)
                and _literal_eq(_metadata_value(info, 'grid_type'), 'Grid1D')
                and _literal_eq(_metadata_value(info, 'mutated_arg_names'), ['in_out_ptr0'])):
            continue
        targets = [_simple_target(s) for s in fn.body[:-1]]
        if None in targets or len(targets) != len(set(targets)) or set(ports).intersection(targets):
            continue
        end = fn.body[-1]
        if not isinstance(end, ast.Expr) or not _call(end.value, 'tl.store', 3) or end.value.keywords:
            continue
        if ast.unparse(end.value.args[0]) != 'in_out_ptr0 + x3' or not _literal_eq(end.value.args[2], None):
            continue
        allowed = pure_calls | {'tl.load', 'tl.store'}
        calls = [n for n in ast.walk(ast.Module(body=fn.body, type_ignores=[])) if isinstance(n, ast.Call)]
        if any(ast.unparse(c.func) not in allowed and not (
                isinstance(c.func, ast.Attribute) and c.func.attr == 'to'
                and ast.unparse(c.func.value) in loads and len(c.args) == 1 and not c.keywords
                and ast.unparse(c.args[0]) == 'tl.float32') for c in calls):
            continue
        touched = [ast.unparse(c) for c in calls if _call(c, 'tl.load')]
        stores = [c for c in calls if _call(c, 'tl.store')]
        pointer_uses = [n.id for s in fn.body for n in ast.walk(s) if isinstance(n, ast.Name) and n.id in ports]
        if set(touched) != loads or len(touched) != len(loads) or stores != [end.value] or sorted(pointer_uses) != sorted(ports):
            continue
        for rows in ROWS:
            if _same_assignments(fn.body, {
                'xnumel': str(rows * 120), 'xoffset': 'tl.program_id(0) * XBLOCK',
                'xindex': 'xoffset + tl.arange(0, XBLOCK)[:]', 'x0': 'xindex % 12',
                'x2': 'xindex // 120', 'x3': 'xindex',
            }):
                yield symbol.value, rows


def _hoistable_packing(body, early, late, packing, device):
    end, start = early[-1], late[0]
    if ([g.column for g in early] != list(range(9, 3, -1)) or
            [g.column for g in late] != [3, 2, 1] or
            (end.rows, end.kernel, end.base, end.stream) != (start.rows, start.kernel, start.base, start.stream)):
        return False
    rows = end.rows
    middle = body[end.index + 1:start.index - 2]
    if len(middle) != 12:
        return False
    view, del_base, move, del_view, stream, launch, *deletes = middle
    if not isinstance(launch, ast.Expr) or not isinstance(launch.value, ast.Call):
        return False
    call = launch.value
    callee = ast.unparse(call.func)
    symbol = callee.removesuffix('.run')
    if callee == symbol or packing.get(symbol) != rows or len(call.args) != 12 or len(call.keywords) != 1:
        return False
    pointers = call.args[:11]
    if not all(isinstance(arg, ast.Name) for arg in pointers) or not _literal_eq(call.args[-1], rows * 120):
        return False
    output, *inputs = (arg.id for arg in pointers)
    wanted = [g.src for g in (*early, *late)]
    if inputs[:9] != wanted or len(set(inputs)) != 10 or output in inputs or output == end.base or end.base in inputs:
        return False
    if not all(_deletes(stmt, g.src) for stmt, g in zip(deletes, early, strict=True)):
        return False
    kw = call.keywords[0]
    if (kw.arg != 'stream' or not _symbol(kw.value, end.stream) or _simple_target(stream) != end.stream
            or ast.unparse(stream.value) != f'get_raw_stream({device})'):
        return False
    view_name = _simple_target(view)
    if (view_name is None or not _call(view.value, 'reinterpret_tensor', 4) or view.value.keywords
            or not _simple_target(move) == output or not _symbol(move.value, view_name) or not _deletes(del_view, view_name)):
        return False
    base, shape, strides, offset = view.value.args
    if (not isinstance(base, ast.Name) or not _deletes(del_base, base.id) or not _literal_eq(offset, 0)
            or not _literal_eq(shape, (1, rows, 10, 12)) or not _literal_eq(strides, (rows * 120, 120, 12, 1))):
        return False
    # These roots predate the first reduction. The intervening view and kernel
    # only touch other allocations, and the late input planes are already live.
    prefix = body[:early[0].index - 2]
    source_layout = f'empty_strided_cuda((1, {rows}, 12, 1), ({rows * 12}, 12, 1, 1), torch.bfloat16)'
    output_layout = f'empty_strided_cuda((1, {rows}, 120), ({rows * 120}, 120, 1), torch.bfloat16)'
    if (_origin(base.id, prefix, len(prefix), output_layout) is not None
            and _origin(inputs[-1], prefix, len(prefix), source_layout) is not None
            and _independent_column_buffers(body, early + late)):
        return _GradientPacking(symbol, end.index + 6, base.id, inputs[-1])


def _column_reduction_groups(tree, *, hoist=True):
    kernels = dict(_column_reduction_kernels(tree))
    packing = dict(_gradient_packing_kernels(tree)) if hoist else {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        guard = node.items[0].context_expr
        if not _call(guard, 'torch.cuda._DeviceGuard', 1) or not isinstance(guard.args[0], ast.Constant):
            continue
        device = guard.args[0].value
        pending = None
        for group in _consecutive_column_reductions(node.body, kernels, device):
            if pending is not None:
                packer = _hoistable_packing(node.body, pending, group, packing, device) if packing else None
                if packer:
                    yield node, pending + group, packer
                    pending = None
                    continue
                yield node, pending, None
                pending = None
            if ((len(group), group[-1].column) in _COLUMN_GROUPS
                    and _independent_column_buffers(node.body, group)):
                pending = group
        if pending is not None:
            yield node, pending, None


def _packing_load_edits(kernel):
    fn, = (node for node in ast.parse(kernel.value).body if isinstance(node, ast.FunctionDef))
    bindings = {_simple_target(stmt): stmt for stmt in fn.body[:-1]}
    if not _same_assignments(fn.body, {'x1': 'xindex // 12 % 10'}):
        return
    load_sites = []
    for port in range(10):
        expected = f"tl.load(in_ptr{port} + (x0 + 12 * x2), None, eviction_policy='evict_last').to(tl.float32)"
        loads = [stmt for stmt in bindings.values() if ast.unparse(stmt.value) == expected]
        if len(loads) != 1:
            return
        load, = loads
        target = _simple_target(load)
        uses = [n for n in ast.walk(fn) if _symbol(n, target)]
        consumers = [stmt.value for stmt in bindings.values()
                     if _call(stmt.value, 'tl.where', 3) and _symbol(stmt.value.args[1], target)]
        if len(uses) != 2 or len(consumers) != 1 or consumers[0].keywords:
            return
        predicate, value, _ = consumers[0].args
        if not isinstance(predicate, ast.Name) or predicate.id not in bindings or value not in uses:
            return
        check = bindings[predicate.id].value
        if not (isinstance(check, ast.Compare) and len(check.ops) == 1 and isinstance(check.ops[0], ast.Eq)):
            return
        plane, column = check.left, check.comparators[0]
        if not (isinstance(plane, ast.Name) and isinstance(column, ast.Name)
                and plane.id in bindings and column.id in bindings
                and _symbol(bindings[plane.id].value, 'x1')
                and ast.unparse(bindings[column.id].value) == f'tl.full([1], {9 - port}, tl.int32)'):
            return
        load_sites.append(load)
    first, *rest = load_sites
    if [s.lineno for s in load_sites] != list(range(first.lineno, first.lineno + 10)):
        return
    pointer = 'in_ptr9'
    for column in range(1, 10):
        pointer = f'tl.where(x1 == {column}, in_ptr{9 - column}, {pointer})'
    loaded = f"tl.load({pointer} + (x0 + 12*x2), None, eviction_policy='evict_last').to(tl.float32)"
    return [(first, loaded), *((stmt, _simple_target(first)) for stmt in rest)]


def _packing_column_zero(body, group, packing, aligned_kernels, device):
    last = group[-1]
    index = last.index + 4
    if index >= len(body) or not _deletes(body[last.index + 1], last.src):
        return
    zero = _column_reduction_launch(body, index, aligned_kernels, device, column_zero=True)
    if zero and (zero.src, zero.rows, zero.base, zero.stream) == (
            packing.column_zero_input, last.rows, last.base, last.stream):
        return zero


def _matches_packing_arithmetic(source, rows):
    fn, = (node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    expected = [
        f'xnumel = {rows * 120}', 'xoffset = tl.program_id(0) * XBLOCK',
        'xindex = xoffset + tl.arange(0, XBLOCK)[:]',
        'xmask = tl.full([XBLOCK], True, tl.int1)[:]',
        'x1 = xindex // 12 % 10', 'x0 = xindex % 12', 'x2 = xindex // 120', 'x3 = xindex',
    ]
    for i in range(10):
        expected.append(f"tmp{5*i + 3} = tl.load(in_ptr{i} + (x0 + 12*x2), None, "
                        "eviction_policy='evict_last').to(tl.float32)")
    expected.extend(['tmp0 = x1', 'tmp1 = tl.full([1], 9, tl.int32)', 'tmp2 = tmp0 == tmp1',
                     'tmp4 = 0.0', 'tmp5 = tl.where(tmp2, tmp3, tmp4)'])
    for i in range(1, 10):
        n = 5*i
        expected.extend([f'tmp{n + 1} = tl.full([1], {9 - i}, tl.int32)',
                         f'tmp{n + 2} = tmp0 == tmp{n + 1}',
                         f'tmp{n + 4} = tl.where(tmp{n + 2}, tmp{n + 3}, tmp4)',
                         f'tmp{n + 5} = tmp{n} + tmp{n + 4}'])
    expected.extend(['tmp51 = 0.1', 'tmp52 = tmp50 * tmp51', 'tl.store(in_out_ptr0 + x3, tmp52, None)'])
    return ast.dump(ast.Module(body=fn.body, type_ignores=[])) == ast.dump(ast.parse('\n'.join(expected)))


def _full_packer(tree, with_node, group, aligned_kernels):
    """Return the packing and scaling ports after the ten full-vocab reductions."""
    if [g.column for g in group] != list(range(9, 0, -1)):
        return
    stmt = with_node.body
    device = with_node.items[0].context_expr.args[0].value
    start, last = group[0], group[-1]
    rows = start.rows
    if rows not in _FULL_VOCAB_ROWS:
        return
    length = rows * 10
    after = last.index + 1
    commands = stmt[after:after + 24]
    if len(commands) != 24:
        return
    zero = _column_reduction_launch(stmt, after + 2, aligned_kernels, device, column_zero=True)
    if zero is None or (last.base, last.rows, last.stream) != (zero.base, zero.rows, zero.stream):
        return
    args = [*group, zero]
    ports = [g.src for g in args]
    if len(set(ports)) != 10 or not _independent_column_buffers(stmt, args):
        return

    scale_alloc, scale_stream, scale_launch = commands[3:6]
    releases = commands[6:16]
    mm_alloc, mm_launch = commands[16:18]
    view, del_packed, alias, del_view, stream, launch = commands[18:]
    scaled = _simple_target(scale_alloc)
    mm_out = _simple_target(mm_alloc)
    if scaled is None or mm_out is None:
        return
    if ast.unparse(scale_alloc.value) != f'empty_strided_cuda(({rows}, 10), (10, 1), torch.bfloat16)':
        return
    if ast.unparse(mm_alloc.value) != f'empty_strided_cuda(({rows}, 64), (64, 1), torch.bfloat16)':
        return
    raw = f'{start.stream} = get_raw_stream({device})'
    if raw != ast.unparse(scale_stream) or raw != ast.unparse(stream):
        return
    if not isinstance(scale_launch, ast.Expr) or not isinstance(scale_launch.value, ast.Call):
        return
    scaling_func = ast.unparse(scale_launch.value.func).removesuffix('.run')
    if not scaling_func.startswith('triton_poi_fused_mul_view_'):
        return
    args_expected = f'{scaling_func}.run({start.base}, {scaled}, {length}, stream={start.stream})'
    if ast.unparse(scale_launch) != args_expected:
        return
    scale_sources = [n.value.args[1].value for n in tree.body
                     if _simple_target(n) == scaling_func and _call(n.value, 'async_compile.triton', 2)
                     and isinstance(n.value.args[1], ast.Constant) and isinstance(n.value.args[1].value, str)]
    if len(scale_sources) != 1:
        return
    functions = [node for node in ast.parse(scale_sources[0]).body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return
    fn, = functions
    if (ast.unparse(fn.args) != 'in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr'
            or len(fn.decorator_list) != 2):
        return
    heuristics, jit = fn.decorator_list
    if not _call(heuristics, 'triton_heuristics.pointwise', 0) or ast.unparse(jit) != 'triton.jit':
        return
    metadata = {kw.arg: kw.value for kw in heuristics.keywords}
    compiler, wrapper = metadata.get('triton_meta'), metadata.get('inductor_meta')
    signature = {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}
    if not (_literal_eq(_metadata_value(compiler, 'signature'), signature)
            and _literal_eq(_metadata_value(wrapper, 'grid_type'), 'Grid1D')
            and _literal_eq(_metadata_value(wrapper, 'mutated_arg_names'), [])):
        return
    arithmetic = f'''xnumel = {length}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
x0 = xindex
tmp0 = tl.load(in_ptr0 + x0, None).to(tl.float32)
tmp1 = 0.1
tmp2 = tmp0 * tmp1
tl.store(out_ptr0 + x0, tmp2, None)'''
    if fn.name != scaling_func or ast.unparse(ast.Module(body=fn.body, type_ignores=[])) != arithmetic:
        return
    release_names = [stmt[g.index].value.args[1].id for g in args]
    if not all(_deletes(statement, name) for statement, name in zip(releases, release_names, strict=True)):
        return
    if not isinstance(mm_launch, ast.Expr) or not _call(mm_launch.value, 'extern_kernels.mm', 2):
        return
    mm = mm_launch.value
    weight = mm.args[1]
    if (not _symbol(mm.args[0], scaled) or len(mm.keywords) != 1 or mm.keywords[0].arg != 'out'
            or not _symbol(mm.keywords[0].value, mm_out)
            or not _call(weight, 'reinterpret_tensor', 4) or weight.keywords):
        return
    w, *layout = weight.args
    if not isinstance(w, ast.Name) or not w.id.startswith('primals_'):
        return
    if not all(_literal_eq(value, expected) for value, expected in zip(layout, ((10, 64), (64, 1), 896), strict=True)):
        return

    view_name, out = _simple_target(view), _simple_target(alias)
    if view_name is None or out is None or not _call(view.value, 'reinterpret_tensor', 4):
        return
    packed, *layout = view.value.args
    if (not isinstance(packed, ast.Name) or view.value.keywords
            or not _deletes(del_packed, packed.id) or not _deletes(del_view, view_name)
            or not _symbol(alias.value, view_name)):
        return
    expected_view = ((1, rows, 10, 12), (rows * 120, 120, 12, 1), 0)
    if not all(_literal_eq(value, expected) for value, expected in zip(layout, expected_view, strict=True)):
        return
    if not isinstance(launch, ast.Expr) or not isinstance(launch.value, ast.Call):
        return
    packer = ast.unparse(launch.value.func).removesuffix('.run')
    packed_symbols = dict(_gradient_packing_kernels(tree))
    if packed_symbols.get(packer) != rows:
        return
    pack_sources = [stmt.value.args[1].value for stmt in tree.body
                    if _simple_target(stmt) == packer and _call(stmt.value, 'async_compile.triton', 2)
                    and isinstance(stmt.value.args[1], ast.Constant) and isinstance(stmt.value.args[1].value, str)]
    if len(pack_sources) != 1 or not _matches_packing_arithmetic(pack_sources[0], rows):
        return
    call = f'{packer}.run({out}, {", ".join(ports)}, {rows * 120}, stream={start.stream})'
    if ast.unparse(launch) != call:
        return
    prefix = stmt[:start.index - 2]
    allocation = f'empty_strided_cuda((1, {rows}, 120), ({rows * 120}, 120, 1), torch.bfloat16)'
    if _origin(packed.id, prefix, len(prefix), allocation) is None:
        return
    names = [packed.id, view_name, out, scaled, mm_out, w.id, start.base, *ports, *release_names]
    if len(set(names)) != len(names):
        return
    return zero, _GradientPacking(packer, after + 23, packed.id, zero.src), scaled, (scale_alloc, scale_launch)


def _patch_column_reductions(source, tree, *, route_packing=True, fuse_packing=True):
    lines = source.splitlines(keepends=True)
    aligned_kernels = dict(_column_reduction_kernels(tree, aligned_output=True)) if fuse_packing else {}
    found = 0
    for node, group, packer in _column_reduction_groups(tree):
        launches = [node.body[g.index] for g in group]
        current = node.body[group[0].index - 1]
        if not all(s.lineno == s.end_lineno for s in [*launches, current]):
            continue
        line = current.lineno - 2
        if not lines[line].lstrip().startswith('# Topologically Sorted Source Nodes: '):
            continue
        start, end = group[0], group[-1]
        flag = f'_mudd_column_reduction_{found}'
        # Preserve line offsets for the Inductor wrapper's source-to-node map.
        if flag in source:
            continue
        indent = ' ' * current.col_offset
        inputs = ', '.join(g.src for g in reversed(group))
        call = (f'try_grouped_gradients({start.kernel}, {start.rows}, {end.column}, '
                f'{start.base}, ({inputs}), {start.stream})')
        helpers = 'try_grouped_gradients'
        launch_setup = f'; {flag} = {call}'
        if packer:
            kernel, = (stmt.value.args[1] for stmt in tree.body if isinstance(stmt, ast.Assign)
                       and _simple_target(stmt) == packer.kernel and _call(stmt.value, 'async_compile.triton', 2))
        if packer and aligned_kernels and _matches_packing_arithmetic(kernel.value, start.rows):
            device = node.items[0].context_expr.args[0].value
            zero = _packing_column_zero(node.body, group, packer, aligned_kernels, device)
            if zero:
                pack_launch, zero_launch = node.body[packer.index], node.body[zero.index]
                fused = f'_mudd_fused_packing_{found}'
                if fused not in source and all(s.lineno == s.end_lineno for s in (pack_launch, zero_launch)):
                    helpers += ', try_fused_gradients'
                    fused_call = (f'try_fused_gradients({start.kernel}, {zero.kernel}, {start.rows}, '
                                  f'{start.base}, {packer.base}, ({zero.src}, {inputs}), {start.stream})')
                    launch_setup = f'; {fused} = {fused_call}; {flag} = {fused} or {call}'
                    for stmt in (pack_launch, zero_launch):
                        lineno = stmt.lineno - 1
                        lines[lineno] = indent + f'if not {fused}: ' + lines[lineno].lstrip()
        if packer is None and fuse_packing and (full := _full_packer(tree, node, group, aligned_kernels)):
            zero, full_packing, scaled, scale_lines = full
            protected = [node.body[zero.index], *scale_lines, node.body[full_packing.index]]
            fused = f'_mudd_fused_packing_{found}'
            if fused not in source and all(stmt.lineno == stmt.end_lineno for stmt in protected):
                helpers += ', try_fused_full_gradients'
                fused_call = (f'try_fused_full_gradients({start.kernel}, {zero.kernel}, {start.rows}, '
                              f'{start.base}, {full_packing.base}, ({zero.src}, {inputs}), {start.stream})')
                launch_setup = f'; {scaled} = {fused_call}; {fused} = {scaled} is not None; {flag} = {fused} or {call}'
                for stmt in protected:
                    lineno = stmt.lineno - 1
                    lines[lineno] = indent + f'if not {fused}: ' + lines[lineno].lstrip()
        lines[line] = indent + f'from track_1_short.perf.kernels.mudd_coefficients import {helpers}\n'
        lines[current.lineno - 1] = lines[current.lineno - 1].rstrip('\n') + launch_setup + '\n'
        for stmt in launches:
            lineno = stmt.lineno - 1
            lines[lineno] = indent + f'if not {flag}: ' + lines[lineno].lstrip()
        if packer and route_packing:
            edits = _packing_load_edits(kernel)
            if edits and all(stmt.lineno == stmt.end_lineno and
                             lines[kernel.lineno + stmt.lineno - 2].strip() == ast.get_source_segment(kernel.value, stmt)
                             for stmt, _ in edits):
                for stmt, value in edits:
                    lines[kernel.lineno + stmt.lineno - 2] = f"{' ' * stmt.col_offset}{_simple_target(stmt)} = {value}\n"
        found += 1
    return ''.join(lines), found


def patch_wrapper(source, *, fuse_norm=True, cache_backward=True, retain_activation=True,
                  retain_residual=True, group_gradients=True, fuse_gradient_packing=True):
    separated_coefficients = "evict_last" in source and "tmp4 = 0.1" in source
    full_graph = fuse_norm and _TRANSPOSE_PAIR in source
    if not separated_coefficients and not full_graph:
        return source, 0
    tree = ast.parse(source)
    if full_graph:
        rewritten, count = _patch_duplicate_activation(source, tree)
        if retain_residual:
            rewritten, reloads = _patch_residual_reuse(rewritten, ast.parse(rewritten) if count else tree)
            count = max(count, reloads)
        if group_gradients:
            rewritten, groups = _patch_column_reductions(rewritten, tree, fuse_packing=fuse_gradient_packing)
            count += groups
        if count:
            return rewritten, count
    if not separated_coefficients:
        return source, 0
    kernels, norms = _candidate_assignments(tree)
    backward = _backward_assignments(tree) if fuse_norm and cache_backward else {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        guard = node.items[0].context_expr
        if (not _call(guard, "torch.cuda._DeviceGuard", 1)
                or not isinstance(guard.args[0], ast.Constant) or type(guard.args[0].value) is not int
                or guard.args[0].value < 0):
            continue
        device = guard.args[0].value
        for i in range(len(node.body) - 29):
            match = _identify_region(kernels, node.body, i, device)
            if match is None:
                continue
            rows, inputs, names, allocations, stream_read = match
            # Preserve the ten original allocations, including their tensor identity
            # and storage independence for the unchanged forward and backward.
            arguments = (*inputs, *(names[plane] for plane in range(10)))
            generated = [
                "from track_1_short.perf.kernels.mudd_coefficients import launch_mudd_coefficients",
                *(ast.unparse(allocation) for allocation in allocations),
                stream_read,
                f"launch_mudd_coefficients({rows}, {', '.join(arguments)})",
            ]
            first, last = node.body[i], node.body[i + 29]
            removed = len(names)
            cached = None
            retained = None
            if fuse_norm and (norm := _identify_norm(norms, node.body, i + 30, match)):
                extra, args, deletions, last = norm
                removed += 1
                cached = _identify_cached_backward(tree, backward, node.body, node.body.index(last), rows, stream_read, norm)
                launcher = "launch_mudd_norm_cached" if cached else "launch_mudd_norm"
                generated = [
                    f"from track_1_short.perf.kernels.mudd_coefficients import {launcher}",
                    *(ast.unparse(allocation) for allocation in (*allocations, *extra)),
                    stream_read,
                    f"{launcher}({rows}, {', '.join(args)})",
                    *(ast.unparse(delete) for delete in deletions),
                ]
                if cached:
                    if retain_activation:
                        retained = _identify_retained_activation(tree, node.body, rows, norm, cached)
                    backward_launcher = "launch_mudd_backward_retained" if retained else "launch_mudd_backward_cached"
                    generated.insert(1, f"from track_1_short.perf.kernels.mudd_backward import {backward_launcher}")
            before, after = first.lineno - 1, last.end_lineno
            lines = source.splitlines(keepends=True)
            indent = " " * first.col_offset
            statements = [indent + row + "\n" for row in generated]
            if len(statements) > after - before:
                continue
            rewritten = lines[:before] + statements + ["\n"] * (after - before - len(statements)) + lines[after:]
            if cached:
                launch, operands = cached
                site = launch.lineno - 1
                if retained:
                    allocation, original_delete, final_delete, fp8 = retained
                    rewritten[allocation.lineno - 1] = indent + f"{operands[-1]} = {fp8}\n"
                    rewritten[original_delete.lineno - 1] = "\n"
                    rewritten[final_delete.lineno - 1] = indent + f"del {operands[-1]}, {fp8}\n"
                    operands = operands[:-1]
                rewritten[site] = indent + f"{backward_launcher}({rows}, {', '.join(operands)})\n"
                rewritten[site + 1:launch.end_lineno] = ["\n"] * (launch.end_lineno - site - 1)
                removed += 1
            return "".join(rewritten), removed
    return source, 0

@triton.jit
def mudd_coefficients(MU, BIAS, DELTA, OUT0, OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8, OUT9,
                      TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    k = tl.program_id(1)
    row = i.to(tl.uint32) // 12
    bias = tl.load(BIAS + 14 + k).to(tl.float32)
    mu = tl.load(MU + (row * 10 + k)).to(tl.float32)
    # i is row * 12 + group. The other nine planes supply the stride difference.
    delta = tl.load(DELTA + (row * 108 + k * 12 + i)).to(tl.float32)
    mixed = tl.fma(bias + mu, 0.10000000149011612, delta * 0.10000000149011612)
    outputs = (OUT0, OUT1, OUT2, OUT3, OUT4, OUT5, OUT6, OUT7, OUT8, OUT9)
    output = outputs[0]
    for plane in tl.static_range(1, 10):
        output = tl.where(k == plane, outputs[plane], output)
    tl.store(output + i, mixed)


def launch_mudd_coefficients(rows, mu, bias, delta, *outputs):
    mudd_coefficients[(rows * 12 // 512, 10)](
        mu, bias, delta, *outputs, 512, num_warps=4, num_stages=1, enable_fp_fusion=False,
    )


@triton.jit
def mudd_gradient_groups(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, OUT,
               XBLOCK: tl.constexpr, PLANES: tl.constexpr, FIRST: tl.constexpr):
    plane = tl.program_id(1)
    inputs = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8)
    in_ptr0 = inputs[0]
    for index in tl.static_range(1, PLANES):
        in_ptr0 = tl.where(plane == index, inputs[index], in_ptr0)
    out_ptr0 = OUT + (FIRST + plane)
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 12*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (10*x0), tmp4, None)


@triton.jit
def mudd_gradient_and_pack(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, OUT, PACK,
                    XBLOCK: tl.constexpr, PLANES: tl.constexpr, FIRST: tl.constexpr):
    # Keep the y coordinate in the independent column-zero schedule.
    plane = tl.program_id(1)
    inputs = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9)
    in_ptr0 = inputs[0]
    for index in tl.static_range(1, PLANES):
        in_ptr0 = tl.where(plane == index, inputs[index], in_ptr0)
    out_ptr0 = OUT + (FIRST + plane)
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 12*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (10*x0), tmp4, None)
    column = FIRST + plane
    # The reference adds one selected value and nine positive zeros.
    # Add positive zero once to retain its signed-zero behavior.
    packed = tmp1 + 0.0
    tl.store(PACK + (120*x0 + column*12 + r0_1), packed*0.1, r0_mask)


@gluon.jit
def _write_gradient_tile(source, OUT, PACK, SCALED, column,
                         XBLOCK: gl.constexpr, layout: gl.constexpr, row_start=None):
    xoffset = gl.program_id(0) * XBLOCK if row_start is None else row_start
    rows = xoffset + gl.arange(0, XBLOCK, layout=gl.SliceLayout(1, layout))
    row = rows[:, None]
    lane = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))[None, :]
    live = lane < 12
    value = gl.load(source + (lane + 12 * row), live, other=0.0).to(gl.float32)
    reduced = gl.sum(gl.where(live, value, 0), 1)
    rounded = reduced.to(gl.bfloat16)
    gl.store(OUT + column + 10 * rows, rounded)
    if SCALED is not None:
        # The projection operand widens the already rounded column sum.
        gl.store(SCALED + column + 10 * rows, rounded.to(gl.float32) * 0.1)
    packed = gl.fma(value, 0.1, 0.0)
    gl.store(PACK + (120 * row + column * 12 + lane), packed, live)



@gluon.jit
def _select_gradient_plane(column, inputs, skip_zero: gl.constexpr = False):
    # Reuse each column-bit predicate for every selector on that bit.
    middle = (column & 4) != 0
    high = (column & 8) != 0
    high_pair = (column & 2) != 0
    odd = (column & 1) != 0
    first_pair = inputs[1] if skip_zero else gl.where(odd, inputs[1], inputs[0])
    next_pair = gl.where(odd, inputs[3], inputs[2])
    first_four = gl.where(high_pair, next_pair, first_pair)
    last_pair = gl.where(odd, inputs[9], inputs[8])
    outer = gl.where(high, last_pair, first_four)
    low_middle = gl.where(odd, inputs[5], inputs[4])
    high_middle = gl.where(odd, inputs[7], inputs[6])
    middle_four = gl.where(high_pair, high_middle, low_middle)
    return gl.where(middle, middle_four, outer)


@gluon.jit
def _select_gradient_plane_by_parity(column, inputs):
    # The smaller matched tiles use fewer registers with parity selected last.
    odd = (column & 1) != 0
    high = (column & 8) != 0
    middle = (column & 4) != 0
    high_pair = (column & 2) != 0
    low_even = gl.where(high_pair, inputs[2], inputs[0])
    high_even = gl.where(high_pair, inputs[6], inputs[4])
    even_first_eight = gl.where(middle, high_even, low_even)
    even = gl.where(high, inputs[8], even_first_eight)
    low_odd = gl.where(high_pair, inputs[3], inputs[1])
    high_odd = gl.where(high_pair, inputs[7], inputs[5])
    odd_first_eight = gl.where(middle, high_odd, low_odd)
    odd_input = gl.where(high, inputs[9], odd_first_eight)
    return gl.where(odd, odd_input, even)


@gluon.jit
def mudd_gradient_pack_direct(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, OUT, PACK, SCALED,
                              XBLOCK: gl.constexpr, PLANES: gl.constexpr, FIRST: gl.constexpr):
    if XBLOCK == 4:
        # Two independent 16-lane reductions per warp. The four shuffle distances stay within each row.
        layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [2, 1], [1, 0])
    elif XBLOCK == 8:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [2, 1], [1, 0])
    elif XBLOCK == 32:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0])
    else:
        gl.static_assert(XBLOCK == 128)
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [8, 1], [1, 0])
    plane = gl.program_id(1)
    ptrs = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9)
    if PLANES == 10:
        if XBLOCK == 8 or XBLOCK == 32:
            source = _select_gradient_plane_by_parity(plane, ptrs)
        else:
            source = _select_gradient_plane(plane, ptrs)
    else:
        source = ptrs[0]
        for i in gl.static_range(1, PLANES):
            source = gl.where(plane == i, ptrs[i], source)
    _write_gradient_tile(source, OUT, PACK, SCALED, FIRST + plane, XBLOCK, layout)


@gluon.jit
def _select_warp_row_group(values, second_group, layout: gl.constexpr):
    pairs = gl.permute(gl.reshape(values, (8, 2, 8)), (0, 2, 1))
    first, second = gl.split(pairs)
    first = gl.convert_layout(first, gl.SliceLayout(2, layout), assert_trivial=True)
    second = gl.convert_layout(second, gl.SliceLayout(2, layout), assert_trivial=True)
    return gl.where(second_group, second[:, :, None], first[:, :, None])


@gluon.jit
def mudd_gradient_pack_warp(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, OUT, PACK,
                            SCALED: gl.constexpr):
    # Each warp owns 16 consecutive rows. Lane bit zero chooses the
    # eight-row register group when packing those sums into one store.
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [8, 0]], [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]],
        [[16, 0], [32, 0], [64, 0]], [], [128, 16],
    )
    plane = gl.program_id(1)
    ptrs = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9)
    source = _select_gradient_plane(plane, ptrs)
    gl.static_assert(SCALED is None)
    xoffset = gl.program_id(0) * 128
    rows = xoffset + gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    row = rows[:, None]
    lane = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))[None, :]
    live = lane < 12
    value = gl.load(source + (lane + 12 * row), live, other=0.0).to(gl.float32)
    packed = gl.fma(value, 0.1, 0.0)
    gl.store(PACK + (120 * row + plane * 12 + lane), packed, live)
    reduced = gl.sum(gl.where(live, value, 0), 1)

    # Split the two register groups into separate values in the same lane.
    # Lanes zero and one computed the same reduction with the outer add
    # operands exchanged, so each can write one group without a shuffle.
    select_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 8, 4], [8, 1, 1], [2, 1, 0])
    subgroup_lane = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, select_layout)))
    subgroup_lane = subgroup_lane[None, None, :]
    second_group = subgroup_lane == 1
    sum_row = _select_warp_row_group(rows, second_group, select_layout)
    sum_value = _select_warp_row_group(reduced, second_group, select_layout)
    gl.store(OUT + plane + 10 * sum_row, sum_value.to(gl.bfloat16), subgroup_lane < 2)


@gluon.jit
def mudd_mixed_gradients(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, OUT, PACK, SCALED,
                         GROUP_TILE: gl.constexpr, ZERO_TILE: gl.constexpr):
    if GROUP_TILE == 128 or ZERO_TILE == 128:
        XBLOCK: gl.constexpr = 64 if GROUP_TILE == 1 else 128
        WARPS: gl.constexpr = 8
    elif GROUP_TILE == 32 or ZERO_TILE == 32:
        XBLOCK: gl.constexpr = 32
        WARPS: gl.constexpr = 4
    else:
        XBLOCK: gl.constexpr = 8
        WARPS: gl.constexpr = 2
    half_warp: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [WARPS, 1], [1, 0])
    paired_lanes: gl.constexpr = gl.BlockedLayout([1, 2], [4, 8], [WARPS, 1], [1, 0])
    four_values: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [WARPS, 1], [1, 0])
    zero_layout: gl.constexpr = half_warp if ZERO_TILE == 1 else paired_lanes if ZERO_TILE == 8 else four_values
    group_layout: gl.constexpr = half_warp if GROUP_TILE == 1 else paired_lanes if GROUP_TILE == 8 else four_values
    column = gl.program_id(1)
    if column == 0:
        _write_gradient_tile(IN0, OUT, PACK, SCALED, column, XBLOCK, zero_layout)
    else:
        pointers = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9)
        source = _select_gradient_plane(column, pointers, skip_zero=True)
        _write_gradient_tile(source, OUT, PACK, SCALED, column, XBLOCK, group_layout)


@gluon.jit
def mudd_banded_gradients(IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9, OUT, PACK, SCALED):
    half_warp: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [8, 1], [1, 0])
    four_values: gl.constexpr = gl.BlockedLayout([1, 4], [8, 4], [8, 1], [1, 0])
    row_start = gl.program_id(0) * 128
    band = gl.program_id(1)
    if band < 8:
        _write_gradient_tile(IN0, OUT, PACK, SCALED, 0, 16, half_warp, row_start + band * 16)
    else:
        column = band - 7
        pointers = (IN0, IN1, IN2, IN3, IN4, IN5, IN6, IN7, IN8, IN9)
        source = _select_gradient_plane(column, pointers, skip_zero=True)
        _write_gradient_tile(source, OUT, PACK, SCALED, column, 128, four_values, row_start)


_LAUNCH_CONFIG = {(1, 2), (8, 2), (32, 4), (128, 8)}


def _selected_column_launch(autotuner, rows):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    if (not isinstance(autotuner, CachingAutotuner) or len(autotuner.launchers) != 1
            or autotuner.triton_interpret or autotuner.dump_launch_params
            or autotuner.inductor_meta.get('coordinate_descent_tuning', False)):
        return None
    launcher, = autotuner.launchers
    config = launcher.config
    block = config.kwargs.get('XBLOCK')
    if (config.kwargs.keys() != {'XBLOCK'} or config.pre_hook is not None or launcher.store_cubin
            or (block, config.num_warps) not in _LAUNCH_CONFIG or config.num_stages != 1
            or config.num_ctas != 1 or config.maxnreg is not None or rows % block != 0):
        return None
    return block, config.num_warps


def _direct_gradient_tile(tile):
    return 4 if tile == 1 else tile


def _launch_gradient_bank(rows, schedule, output, packed, inputs, planes, first, scaled):
    block, warps = schedule
    if scaled is None and (rows not in _SAMPLED_ROWS or block == 128):
        mudd_gradient_and_pack[(rows // block, planes)](
            *inputs, output, packed, block, planes, first, num_warps=warps, num_stages=1,
        )
    else:
        block = _direct_gradient_tile(block)
        mudd_gradient_pack_direct[(rows // block, planes)](
            *inputs, output, packed, scaled, block, planes, first, num_warps=warps, num_stages=1,
        )


def _launch_fused_gradients(rows, group, zero, output, packed, inputs, scaled=None):
    if group != zero and (scaled is not None or rows in _SAMPLED_ROWS):
        tiles = group[0], zero[0]
        if tiles == (128, 1):
            mudd_banded_gradients[(rows // 128, 17)](
                *inputs, output, packed, scaled, num_warps=8, num_stages=1,
            )
        else:
            warps = max(group[1], zero[1])
            tile = 64 if tiles == (1, 128) else {2: 8, 4: 32, 8: 128}[warps]
            mudd_mixed_gradients[(rows // tile, 10)](
                *inputs, output, packed, scaled, group[0], zero[0], num_warps=warps, num_stages=1,
            )
        return
    if group == zero and group == (128, 8) and scaled is None and rows in _SAMPLED_ROWS:
        mudd_gradient_pack_warp[(rows // 128, 10)](
            *inputs, output, packed, None, num_warps=8, num_stages=1,
        )
    elif group == zero:
        _launch_gradient_bank(rows, group, output, packed, inputs, 10, 0, scaled)
    else:
        _launch_gradient_bank(rows, group, output, packed, (*inputs[1:], inputs[1]), 9, 1, scaled)
        _launch_gradient_bank(rows, zero, output, packed, (inputs[0],) * 10, 1, 0, scaled)


def try_fused_full_gradients(group_tuner, zero_tuner, rows, output, packed, inputs, raw_stream):
    if rows not in _FULL_VOCAB_ROWS:
        return None
    group = _selected_column_launch(group_tuner, rows)
    zero = _selected_column_launch(zero_tuner, rows)
    if group is None or zero is None or torch.cuda.current_stream(output.device).cuda_stream != raw_stream:
        return None

    scaled = torch.empty((rows, 10), dtype=torch.bfloat16, device=output.device)
    _launch_fused_gradients(rows, group, zero, output, packed, inputs, scaled)
    return scaled


def try_grouped_gradients(autotuner, rows, first, output, inputs, raw_stream):
    """Reuse the launch selected by the original reduction after warmup."""
    schedule = _selected_column_launch(autotuner, rows)
    if schedule is None or torch.cuda.current_stream(output.device).cuda_stream != raw_stream:
        return False
    block, warps = schedule
    count = len(inputs)
    if (count, first) not in _COLUMN_GROUPS:
        return False
    operands = tuple(inputs) + (inputs[0],) * (9 - count)
    mudd_gradient_groups[(rows // block, count)](
        *operands, output, block, count, first, num_warps=warps, num_stages=1,
    )
    return True


def try_fused_gradients(group_tuner, zero_tuner, rows, output, packed, inputs, raw_stream):
    """Write both gradients with the schedules selected by the original reductions."""
    group = _selected_column_launch(group_tuner, rows)
    zero = _selected_column_launch(zero_tuner, rows)
    if group is None or zero is None or torch.cuda.current_stream(output.device).cuda_stream != raw_stream:
        return False
    _launch_fused_gradients(rows, group, zero, output, packed, inputs)
    return True


@gluon.jit
def mudd_final_norm(MU, BIAS, DELTA, MIXED, SUM, X, C0, S0, C1, S1, C2, S2, C3, S3, C4, S4,
                    C5, S5, C6, S6, C7, S7, C8, S8, C9, S9, FP8, CACHE_INVERSE: gl.constexpr):
    # Four adjacent lanes share one coefficient. The projected layout
    # puts one BF16 store per group in the lane selected by Gluon.
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [1, 1], [1, 0])
    groups = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    channels = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    col = groups[:, None] * 64 + channels[None, :]
    row = gl.program_id(0)
    base = row * 768 + col
    group = row * 12 + groups
    group_live = groups < 12
    live = col < 768
    mixed = gl.load(X + base, live, other=0).to(gl.float32)
    coefficients = (C0, C1, C2, C3, C4, C5, C6, C7, C8, C9)
    activations = (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9)
    scale: gl.constexpr = 0.10000000149011612
    # The two adjacent BF16 words form one aligned 32-bit load.
    pair_layout: gl.constexpr = gl.SliceLayout(0, gl.BlockedLayout([1, 2], [32, 1], [1, 1], [1, 0]))
    pair_lanes = gl.arange(0, 2, layout=pair_layout)
    for pair in gl.static_range(5):
        bias_values = gl.load(BIAS + (14 + pair * 2) + pair_lanes)
        bias_pair = gl.split(bias_values)
        mu_values = gl.load(MU + (row * 10 + pair * 2) + pair_lanes)
        mu_pair = gl.split(mu_values)
        for plane in gl.static_range(2 * pair, 2 * pair + 2):
            bias = bias_pair[plane % 2].to(gl.float32)
            mu = mu_pair[plane % 2].to(gl.float32)
            delta = gl.load(DELTA + row * 120 + plane * 12 + groups, group_live, other=0).to(gl.float32)
            m = gl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f",
                                         [bias, mu], dtype=gl.float32, is_pure=True, pack=1)
            rounded = gl.fma(m, scale, delta * scale).to(gl.bfloat16)
            gl.store(coefficients[plane] + group, rounded, group_live)
            weight = gl.broadcast(rounded.to(gl.float32)[:, None], mixed)[0]
            value = gl.load(activations[plane] + base, live, other=0).to(gl.float32)
            mixed = gl.fma(weight, value, mixed)
    # Flatten in the same order as the original CUDA persistent reduction.
    total = gl.sum(gl.reshape(gl.where(live, mixed * mixed, 0.0), (1024,)), 0)
    variance = gl.inline_asm_elementwise("div.full.f32 $0, $1, 0f44400000;", "=f,f",
                                         [total], dtype=gl.float32, is_pure=True, pack=1)
    biased = variance + 1.1920928955078125e-7
    rstd = gl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=f,f",
                                     [biased], dtype=gl.float32, is_pure=True, pack=1)
    if CACHE_INVERSE:
        gl.store(SUM + row, rstd)
    else:
        gl.store(SUM + row, total)
    gl.store(MIXED + base, mixed, live)
    normalized = gl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f",
                                          [mixed, rstd], dtype=gl.float32, is_pure=True, pack=1)
    gl.store(FP8 + base, (normalized * 4.48).to(gl.float8e4nv), live)

def launch_mudd_norm(rows, *operands):
    mudd_final_norm[(rows,)](*operands, False, num_warps=1, num_stages=1, enable_fp_fusion=True)


def launch_mudd_norm_cached(rows, *operands):
    mudd_final_norm[(rows,)](*operands, True, num_warps=1, num_stages=1, enable_fp_fusion=True)


class _WrapperCacheKey(CustomGraphPass):
    """Keep the rewritten wrappers in a separate Inductor cache entry."""

    def __call__(self, graph):
        pass

    def uuid(self):
        return get_hash_for_files((__file__, str(Path(__file__).with_name("mudd_backward.py"))))


def _compile_with_coefficients(graph, inputs):
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.graph import GraphLowering

    config = torch._inductor.config
    if config.post_grad_custom_post_pass is not None:
        return compile_fx(graph, inputs)
    fuse_norm = not (config.max_autotune or config.max_autotune_pointwise or config.coordinate_descent_tuning
                     or config.deterministic or torch.are_deterministic_algorithms_enabled())

    with _COMPILE_LOCK:
        original = GraphLowering._compile_to_module_lines
        owner = get_ident()

        def compile_lines(lowering, wrapper):
            if get_ident() == owner:
                text, count = patch_wrapper(wrapper.value, fuse_norm=fuse_norm)
                if count:
                    wrapper = replace(wrapper, value=text)
            return original(lowering, wrapper)

        with patch.object(GraphLowering, "_compile_to_module_lines", compile_lines):
            return compile_fx(graph, inputs, config_patches={"post_grad_custom_post_pass": _WrapperCacheKey()})


def compile_training_model(model, device):
    options = dict(dynamic=False, fullgraph=True)
    if (torch.device(device).type == "cuda" and torch.version.git_version == TORCH_REVISION
            and triton.__version__ == "3.6.0" and torch.cuda.get_device_capability(device) == (9, 0)):
        options["backend"] = _compile_with_coefficients
    return torch.compile(model, **options)
