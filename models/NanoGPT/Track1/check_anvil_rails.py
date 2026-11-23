"""Replay the ANVIL rail stores and the emitted cascade's data dependencies."""
import argparse
import ast
from math import prod
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from triton.experimental.gluon._runtime import GluonASTSource

from check_anvil_epilogue import artifact as compile_pointwise, run as replay
from check_anvil_gram import CASES, load_cascade, load_polar
from anvil_cascade_replay import first_gram
from check_anvil_trace import capture_cascades
from check_replicated_adam import term
from check_value_adam_inductor import HOPPER, TORCH_REVISION
from export_source_link import anvil_rail_weight, function_span
from install_candidate import ANVIL, POLAR
from tail_norm_replay import instructions
from verify_candidate import check_checkout


HALF = {"grad", "prepared"}


def expected_cells(cells, tile, cta, slow_weight):
    one, half, neg_one = 0x3f800000, 0x3f000000, 0xbf800000

    def lerp(a, b, w):
        high = term("setp.ge.f32", term("abs.f32", w), half)
        scale = term("selp.f32", term("add.f32", w, neg_one), w, high)
        base = term("selp.f32", b, a, high)
        return term("fma.rn.f32", scale, term("sub.f32", b, a), base)

    result = {}
    for i in range(cta * tile, (cta + 1) * tile):
        grad = term("cvt.f32.bf16", ("grad", i))
        fast = lerp(("velocity", i), grad, term("sub.f32", one, ("fast_beta", 0)))
        prior = ("velocity", cells + i)
        slow = term("fma.rn.f32", slow_weight, term("sub.f32", grad, prior), prior)
        weight = ("fast_weight", 0)
        blend = term("fma.rn.f32", weight, fast, term("mul.f32", term("sub.f32", one, weight), slow))
        result[("velocity", i)] = fast
        result[("velocity", cells + i)] = slow
        result[("prepared", i)] = term("cvt.rn.bf16.f32", lerp(grad, blend, ("momentum", 0)))
    return result


def pointwise_stores(parts, kernel, shape, path, slow_weight, block):
    cells = prod(shape)
    signatures = (("grad", "fast_beta", "velocity", "rail"), ("rail", "velocity"),
                  ("momentum", "fast_weight", "rail", "grad", "prepared"))
    reference = []
    for (name, source), phase in zip(parts, ("update", "copy", "lookahead"), strict=True):
        fn, = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
        reference.append(compile_pointwise((fn, ()), path, phase))
    roles = ("grad", "velocity", "momentum", "fast_beta", "fast_weight", "prepared")
    signature = {name: ("*bf16" if role in HALF else "*fp32")
                 for name, role in zip(kernel.arg_names[:6], roles, strict=True)}
    signature.update(CELLS="constexpr", BLOCK="constexpr")
    # The momentum slot is 16-byte aligned; fast beta and fast weight are
    # at byte offsets 8 and 12 in the pinned device staging record.
    attrs = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 5)}
    compiled = triton.compile(GluonASTSource(kernel, signature, dict(CELLS=cells, BLOCK=block), attrs=attrs),
                              target=HOPPER, options=dict(num_warps=4, num_stages=1, enable_fp_fusion=False))
    ptx = compiled.asm["ptx"]
    assert "bar.sync" not in ptx
    path.with_suffix(".ttgir").write_text(compiled.asm["ttgir"])
    path.with_suffix(".ptx").write_text(ptx)
    path.with_suffix(".cubin").write_bytes(compiled.asm["cubin"])

    def check_tiles(tiles, execute):
        tile, _ = tiles
        per_matrix = shape[1] * shape[2] // tile
        assert per_matrix * tile == shape[1] * shape[2]
        boundaries = {b * per_matrix + edge for b in range(shape[0]) for edge in (0, per_matrix - 1)}
        for cta in sorted(boundaries):
            assert execute(cta) == expected_cells(cells, tile, cta, slow_weight), (shape, tiles, cta)

    for tile, warps in reference[0]:
        up, copy, look = (artifacts[(tile, warps)] for artifacts in reference)
        def split(cta):
            scratch, stores = {}, {}
            for rail in (0, 1):
                slot = rail * (cells // tile) + cta
                row = replay(up, signatures[0], tile, warps, slot, half=HALF)
                scratch.update(row)
                stores.update(replay(copy, signatures[1], tile, warps, slot, row, half=HALF, intermediate_role="rail"))
            stores.update(replay(look, signatures[2], tile, warps, cta, scratch, half=HALF, intermediate_role="rail"))
            return stores
        check_tiles((tile, warps), split)
    code, _ = instructions(ptx)
    check_tiles((block, 4), lambda cta: replay(code, roles, block, 4, cta, half=HALF))


def check_bank_dispatch(source, rails, shape):
    names = ("grad", "velocity", "momentum", "fast_beta", "fast_weight")
    bank = SimpleNamespace(grad=torch.empty(shape, dtype=torch.bfloat16),
                           velocity=torch.empty((2, *shape), dtype=torch.float32),
                           compact_trace=True, split_baddbmm=shape[1] > shape[2],
                           cooperative_rounds="cooperative_rounds=bank.cooperative_rounds" in source,
                           p_slice=torch.empty(shape, dtype=torch.bfloat16),
                           lane_energy=None, beta2=None, red_dim=None, mantissa=None, eff_wd=None, eff_lr=None)
    staged = torch.empty(4, dtype=torch.float32)
    bank.momentum, bank.fast_beta, bank.fast_weight = staged[0], staged[2], staged[3]
    inputs = tuple(getattr(bank, name) for name in names)
    # The staging row's first word is 16-byte aligned; these two scalar
    # views must retain their offsets when passed to the Triton launcher.
    assert tuple(t.data_ptr() % 16 for t in inputs) == (0, 0, 0, 8, 12)
    launches = []

    class CaptureKernel:
        def __getitem__(self, grid):
            assert grid == (bank.grad.numel() // rails.BLOCK,)
            def invoke(g, v, m, b, w, x, cells, block, **config):
                assert all(actual is expected for actual, expected in zip((g, v, m, b, w), inputs, strict=True))
                assert cells == g.numel() and block == rails.BLOCK and x.shape == g.shape and x.dtype == g.dtype
                assert x.is_contiguous() and config == dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
                launches.append(x)
            return invoke

    def cascade(grad, velocity, momentum, *, split_baddbmm, fast_beta_t, fast_weight_t, compact_trace, cooperative_rounds=False):
        assert all(actual is expected for actual, expected in
                   zip((grad, velocity, momentum, fast_beta_t, fast_weight_t), inputs, strict=True))
        assert split_baddbmm == bank.split_baddbmm and compact_trace == bank.compact_trace
        assert cooperative_rounds == bank.cooperative_rounds
        return grad

    def epilogue(v, *args):
        assert v is bank.grad and args[-1].data_ptr() == bank.velocity[1].data_ptr()

    env = dict(AnvilBank=type(bank), anvil_cascade=cascade, torch=torch,
               epilogue_eligible=lambda *_: True, update_epilogue=epilogue)
    exec(function_span(source, "anvil_bank_update"), env)
    env["anvil_bank_update"](bank)
    bank.compact_trace = False
    env["anvil_bank_update"](bank)
    prepared = torch.empty_like(bank.grad)
    with patch.object(rails, "_prepare_rails", CaptureKernel()):
        assert rails.prepare_rails(*inputs, prepared) is None
    assert launches == [prepared]
    assert all(prepared.data_ptr() != value.data_ptr() for value in inputs)
    return staged, bank.grad, bank.velocity


def check(baseline, candidate, out):
    check_checkout(baseline, candidate)
    if torch.version.git_version != TORCH_REVISION or torch.version.cuda is not None or triton.__version__ != "3.6.0":
        raise RuntimeError("Use the pinned CPU PyTorch build and Triton 3.6.0")
    out.mkdir(parents=True, exist_ok=True)
    old = (baseline / ANVIL).read_text()
    _, slow_weight = anvil_rail_weight(old)
    cascade = load_cascade(candidate / ANVIL, load_polar(candidate / POLAR, "anvil_rail_polar"))
    rails = cascade.rails_module
    for bank, _, batch, rows, cols in CASES[:3]:
        shape = (batch, rows, cols)
        source = (candidate / ANVIL).read_text()
        staged, grad, velocity = check_bank_dispatch(source, rails, shape)
        split, fused = capture_cascades(cascade, out, bank, rows > cols, staged, grad, velocity)
        *_, parts = first_gram(split, shape)
        first_gram(fused, shape, fused=True)
        pointwise_stores(parts, rails._prepare_rails, shape, out / f"{bank}-rails", slow_weight, rails.BLOCK)
        print(f"{bank}: 3 preparation launches -> 1, exact FP32 rails and BF16 lookahead at boundaries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate, args.out)
