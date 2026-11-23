"""Check the installed lookup/staging dispatch and its five SM90 artifacts."""

import argparse
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from check_graph_stage import BASES, LAYOUTS, Tensor, WIDTHS, field_sizes, load_graph_stage, load_step_operations
from check_ngram_slots import load, replay_buffers
from install_candidate import DRIVER, STEP_GRAPHS
from verify_candidate import check_checkout


TYPES = (
    "*i32", "*i64", "*i32", "*i32", "*fp32", "*fp32", "*i32", "*i32",
    "*i32", "*i64", "*i32", "*i32", "*fp32", "*fp32", "*i32",
)
NAMES = ("INPUT_TILES", "END_ROWS", "MTP_ROWS", "VOCAB", "TOKEN_TILE", "SLOT_TILE", "HASHED_ROWS", "MEMBER_ROWS")


def check_dispatch(candidate, module):
    tokens, ends, mtp = LAYOUTS[0]
    events, launches = [], []
    tensors = tuple(Tensor(size, dtype, width, address, events)
                    for size, dtype, width, address in zip(field_sizes(tokens, ends, mtp) * 2,
                                                           module.DTYPES * 2, WIDTHS * 2, BASES, strict=True))
    want = Tensor(255, torch.int32, 4, 13 << 32, events)
    directory = Tensor(module.DIRECTORY_LENGTH, torch.int32, 4, want.pointer + 1024, events)
    query = module.SlotQuery(want, directory, tensors[3])
    result = replace(tensors[3], pointer=15 << 32)

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def execute(*args, **options):
                events.append(self.name)
                launches.append((self.name, grid, args, options))
            return execute

    slots_module = import_module("track_1_short.perf.kernels.ngram_slots")
    forward, _ = load_step_operations(
        candidate / STEP_GRAPHS, ("forward", "step_inputs"), Batch=object, Tensor=object,
        ForwardScheduleConfig=object, live_key=lambda *_: 0, SlotQuery=module.SlotQuery,
        try_stage_step_inputs=module.try_stage_step_inputs, try_stage_step_query=module.try_stage_step_query,
    )
    cfg = SimpleNamespace(mtp_weights=tensors[4], prefix_weight=tensors[5], sampled_loss=None)
    batch = SimpleNamespace(inputs=tensors[0], targets=tensors[1], cum_seqlens=tensors[2], ngram_ids=tensors[3],
                            targets_cpu=object())

    def run(plan, *, fresh=False, sealed=False, self_check=False):
        graph = SimpleNamespace(step_keys=[0], captured={}, _attn_scales=lambda: (), sealed=sealed,
                                unchecked={0} if self_check else set(),
                                _self_check=lambda _: events.append("self-check"))
        capture = SimpleNamespace(attn_scales=(), cfg=cfg, static=SimpleNamespace(buffers=lambda: tensors[6:]),
                                  forward=SimpleNamespace(replay=lambda: events.append("replay")))

        def capture_inputs(key, incoming, slot_tensor, config):
            assert key == 0 and incoming is batch and slot_tensor is result and config is cfg
            events.append("capture")
            graph.captured[key] = capture
            return capture

        graph._capture = capture_inputs
        if not fresh:
            graph.captured[0] = capture
        output = forward(graph, 0, batch, plan, cfg)
        assert graph.current is capture and output is result
        return output

    with (patch.object(torch.cuda, "current_device", return_value=0),
          patch.object(torch.autograd.graph, "increment_version", side_effect=events.append),
          patch.object(torch, "empty_like", return_value=result),
          patch.object(module, "stage_step", Launch("stage")),
          patch.object(module, "stage_step_with_lookup", Launch("fused")),
          patch.object(slots_module, "bucket_slots", Launch("lookup"))):
        run(query, self_check=True)
        assert events == ["fused", tensors[6:], "replay", "self-check"]
        assert launches == [("fused", (tokens // 256 + 2 * tokens // 128,),
                             (*tensors[:6], want, directory, *tensors[6:], result,
                              tokens // 256, ends, mtp, module.VOCAB_SIZE, 256, 128, False, False),
                             {"num_warps": 4, "num_stages": 1})]
        assert result.pointer != tensors[9].pointer

        events.clear()
        run(replace(query, directory=replace(directory, pointer=directory.pointer + 4)))
        assert events == ["lookup", "stage", tensors[6:], "replay"]

        events.clear()
        run(query, fresh=True)
        assert events == ["lookup", "capture", "stage", tensors[6:], "replay"]

        events.clear()
        try:
            run(query, fresh=True, sealed=True)
        except AssertionError as exc:
            assert "no graph for" in str(exc)
        else:
            raise AssertionError("A missing graph was accepted after sealing")
        assert not events

        def no_effect(*_):
            pass

        def planned(step, row_ids, *, defer_slots, hashed_rows):
            assert step == 0 and row_ids is query.rows and defer_slots and hashed_rows
            return replace(query, hashed_rows=True, member_rows=True)

        def accumulate(slots, gradient, event_step):
            assert slots is result and gradient == "sink" and not event_step
            events.append("accumulate")

        def forward_in_step(step, incoming, plan, config):
            assert step == 0 and incoming is batch and config is cfg
            return run(plan)

        row = SimpleNamespace(ngram=SimpleNamespace(accumulate_grad=accumulate, pending=[]),
                              forward_slots=planned, after_forward=no_effect, post_prep=no_effect,
                              is_update=lambda _: False, position={0: 0}, update_positions=[1])
        ngram = SimpleNamespace(
            forward=forward_in_step,
            backward=lambda: "sink",
        )
        manager = SimpleNamespace(get_forward_args=lambda _: cfg, step_optimizers=no_effect,
                                  is_adam_step=lambda _: False)
        batches = SimpleNamespace(peek=lambda _: batch, take=no_effect)
        softmax = SimpleNamespace(upload=no_effect, gather=no_effect, mark_readers_done=no_effect)
        deferred = SimpleNamespace(flush=no_effect, hand_over=lambda *a, **k: None)
        trainer = load(candidate / DRIVER, functions=("train_step",), DEFERRED_LABELS=())
        trainer["train_step"](manager, ngram, row, softmax, batches, deferred, 0, None, False)
        assert events == ["fused", tensors[6:], "replay", "accumulate"]
        assert launches[-1][2][-2:] == (True, True)
    print("Installed train_step receives the fused slots. Capture, alias rejection and sealed failure checked.", flush=True)


def check_artifacts(module, ir_dir):
    kernel = module.stage_step_with_lookup
    types = dict(zip(kernel.arg_names, TYPES + ("constexpr",) * len(NAMES), strict=True))
    alignment = {(i,): [["tt.divisibility", 16]] for i in range(len(TYPES))}
    rng = np.random.default_rng(9361)
    # Cover a full 255-entry bucket, a singleton, and an empty bucket
    # alongside irregular widths throughout the table.
    want = np.concatenate((np.arange(255, dtype=np.int32), np.array([32768], dtype=np.int32),
                           np.unique(rng.integers(98304, module.VOCAB_SIZE, 393216, dtype=np.int32))))
    directory = import_module("track_1_short.perf.kernels.ngram_slots").make_directory(want)
    assert directory is not None
    if ir_dir is not None:
        ir_dir.mkdir(parents=True, exist_ok=True)
    for tokens, ends, mtp in LAYOUTS:
        input_tiles = tokens // module.TOKEN_BLOCK
        field_counts = field_sizes(tokens, ends, mtp)
        words = [rng.integers(0, 2**(width * 8), size, dtype=np.uint64).astype(np.int64)
                 for width, size in zip(WIDTHS, field_counts, strict=True)]
        ctas = input_tiles + field_counts[3] // module.SLOT_BLOCK
        loads = []
        for hashed, member in ((False, False), (True, False), (True, True)):
            values = dict(zip(NAMES, (input_tiles, ends, mtp, module.VOCAB_SIZE, module.TOKEN_BLOCK,
                                      module.SLOT_BLOCK, hashed, member), strict=True))
            compiled = triton.compile(ASTSource(kernel, types, values, attrs=alignment),
                                      target=GPUTarget("cuda", 90, 32), options={"num_warps": 4, "num_stages": 1})
            if member:
                queries = rng.choice(want, size=field_counts[3])
                queries[:5] = (0, 254, 32768, want[-1], 127)
            else:
                queries = rng.integers(0, module.VOCAB_SIZE, field_counts[3], dtype=np.int32)
                queries[:5] = (0, 32767, 32768, 65536, module.VOCAB_SIZE - 1)
                if not hashed:
                    queries[5:7] = (-1, module.VOCAB_SIZE)
            slots = np.searchsorted(want, queries, side="left")
            words[3] = queries
            inputs = tuple(zip(WIDTHS, words, strict=True)) + ((4, want), (4, directory))
            static = words.copy()
            static[3] = slots
            outputs = tuple(zip(WIDTHS, static, strict=True)) + ((4, slots),)
            probes = replay_buffers(compiled.asm["ptx"], kernel.__name__, ctas, inputs, outputs, 6)
            assert (probes[:input_tiles * 128] == 0).all()
            if member:
                assert (probes[input_tiles * 128:] == 8).all()
            loads.append(compiled.asm["ptx"].count("ld.global"))
            if ir_dir is not None:
                label = "member-rows" if member else ("hashed-rows" if hashed else "slots")
                for stage in ("ttir", "ttgir", "ptx"):
                    (ir_dir / f"stage-with-{label}-{tokens}-{ends}-{mtp}.{stage}").write_text(compiled.asm[stage])
        print(f"{tokens:6d} tokens, {ends} ends, {mtp} weights   "
              f"{loads[0]}→{loads[1]}→{loads[2]} global-load instructions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ir-dir", type=Path)
    args = parser.parse_args()
    check_checkout(args.baseline, args.candidate)
    installed = load_graph_stage(args.candidate)
    check_dispatch(args.candidate, installed)
    check_artifacts(installed, args.ir_dir)
