"""Pack the training loader's four host fields into a single pinned transfer."""

import torch


DTYPES = (torch.int32, torch.int64, torch.int32, torch.int32)
WIDTHS = (4, 8, 4, 4)
ALIGNMENT = 16


def byte_ranges(counts):
    stop = 0
    ranges = []
    for count, width in zip(counts, WIDTHS, strict=True):
        start = (stop + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        stop = start + count * width
        ranges.append((start, stop))
    return tuple(ranges)


def typed_views(packet, ranges):
    return tuple(packet[start:stop].view(dtype)
                 for (start, stop), dtype in zip(ranges, DTYPES, strict=True))


class BatchPacket:
    def __init__(self, max_tokens, max_cu_seqlens_rows):
        self.capacities = (max_tokens, max_tokens, max_cu_seqlens_rows, 2 * max_tokens)
        ranges = byte_ranges(self.capacities)
        self.storage = torch.empty(ranges[-1][1], dtype=torch.uint8, pin_memory=True)
        self.separate = typed_views(self.storage, ranges)
        self.layouts = {}

    def upload(self, sources, device, coalesce):
        if not coalesce:
            out = []
            for host, buf in zip(sources, self.separate, strict=True):
                n = host.numel()
                assert n <= buf.numel() and host.dtype == buf.dtype, f"batch tensor {host.dtype}[{n}] overflows its pinned slot"
                buf[:n].copy_(host)
                out.append(buf[:n].to(device, non_blocking=True))
            return tuple(out)

        counts = tuple(host.numel() for host in sources)
        for host, n, limit, dtype in zip(sources, counts, self.capacities, DTYPES, strict=True):
            assert n <= limit and host.dtype == dtype, f"batch tensor {host.dtype}[{n}] overflows its pinned slot"

        layout = self.layouts.get(counts)
        if layout is None:
            ranges = byte_ranges(counts)
            prefix = self.storage[:ranges[-1][1]]
            layout = (prefix, ranges, typed_views(prefix, ranges))
            if len(self.layouts) < 8:
                self.layouts[counts] = layout

        prefix, ranges, host_views = layout
        for host, destination in zip(sources, host_views, strict=True):
            destination.copy_(host)
        packet = prefix.to(device, non_blocking=True)
        return typed_views(packet, ranges)
