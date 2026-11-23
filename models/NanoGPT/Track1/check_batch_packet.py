"""Check the installed pinned upload with CPU copies standing in for DMA."""

import argparse
import ast
from contextlib import contextmanager
from pathlib import Path
import random
from unittest.mock import patch

import torch

from install_candidate import BATCH_PACKET, PINNED_BATCHES
from verify_candidate import check_checkout, load_candidate_scale


DTYPES = (torch.int32, torch.int64, torch.int32, torch.int32)


def load_staging(path, packet=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "track_1_short.perf.batch_packet")]
    env = {"BatchPacket": packet}
    exec(compile(tree, str(path), "exec"), env)
    return env["PinnedBatchStaging"]


@contextmanager
def fake_dma(log):
    empty = torch.empty
    clone_to = torch.Tensor.to

    def unpinned(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    class Event:
        def synchronize(self):
            log.append(("ready", self))

        def record(self):
            log.append(("record", self))

    def transfer(tensor, device, *, non_blocking):
        log.append(("transfer", tensor.numel() * tensor.element_size()))
        assert non_blocking
        return clone_to(tensor, device, copy=True)

    with (patch.object(torch, "empty", unpinned), patch.object(torch.cuda, "Event", Event),
          patch.object(torch.Tensor, "to", transfer)):
        yield


def check(baseline, candidate):
    check_checkout(baseline, candidate)
    packet = load_candidate_scale(candidate / BATCH_PACKET)
    original = load_staging(baseline / PINNED_BATCHES)
    optimized = load_staging(candidate / PINNED_BATCHES, packet.BatchPacket)
    defaults = [(16384, 16384, 64, 32768), (32768, 32768, 96, 65536), (40960, 40960, 128, 81920),
                (49152, 49152, 128, 98304)]
    rng = random.Random(271448)
    ragged = [tuple(rng.randrange(n + 1) for n in (1000, 2000, 128, 400)) for _ in range(40)]
    trials = [(sizes, True) for sizes in defaults * 5 + ragged + [(0, 0, 0, 0)]]
    trials.extend([((300, 300, 128, 600), False), ((262144, 262144, 896, 524288), False)])
    log = []
    saved = []
    with fake_dma(log):
        before, after = (cls(262144, 896, "cpu") for cls in (original, optimized))
        for index, (sizes, coalesce) in enumerate(trials):
            sources = tuple(torch.randint(-(1 << 30), 1 << 30, (n,), dtype=dtype)
                            for n, dtype in zip(sizes, DTYPES, strict=True))
            want = before.upload(*sources)
            assert len(log) == 6 and all(op == "transfer" for op, _ in log[1:-1])
            log.clear()
            got = after.upload(*sources, coalesce=coalesce)
            assert len(log) == (3 if coalesce else 6)
            event = after.slots[index % len(after.slots)].uploaded
            assert log[0] == ("ready", event) and log[-1] == ("record", event)
            assert all(torch.equal(actual, expected) for actual, expected in zip(got, want, strict=True))
            if coalesce:
                regions = packet.byte_ranges(sizes)
                assert log[1] == ("transfer", regions[-1][1])
                assert all(view.data_ptr() % 16 == 0 for view in got if view.numel())
                for field, (start, stop) in enumerate(regions):
                    assert start % 16 == 0 and stop - start == sizes[field] * got[field].element_size()
                    if field:
                        assert regions[field - 1][1] <= start
                storages = {view.untyped_storage()._cdata for view in got}
                assert len(storages) == 1
            elif all(sizes):
                assert len({view.untyped_storage()._cdata for view in got}) == 4
            if index < len(after.slots):
                saved.append((sources, got))
            log.clear()
    for sources, retained in saved:
        assert all(torch.equal(actual, expected) for actual, expected in zip(retained, sources, strict=True))
    print(f"Checked {len(trials)} uploads, typed views, slot reuse, and the separate validation path.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate)
