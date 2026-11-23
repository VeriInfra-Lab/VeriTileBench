"""Replay the two SM90a RMSNorm backward layouts against the row recurrence."""

from mudd_norm_replay import BANK, FP8, GRADIENT, MIXED, SQUARE_SUM, RowOffset, inverse, mismatch, nodes, op, replay

def backward_row(encode):
    row = RowOffset(1, 0)
    width = 16 if encode else 8
    def val(bits):
        return op("f32", bits)
    def at(column):
        index = row * 768 + column
        gradient = op("widen", op("bf16", GRADIENT // BANK, index))
        mixed = op("f32-load", MIXED // BANK, index)
        return gradient, mixed
    partial = []
    for lane in range(32):
        live_words = 16 if encode else 24
        local = [at(width*lane + word%width + width*32*(word//width)) for word in range(live_words)]
        total = op("mul.f32", *local[1])
        for operands in (local[0], *local[2:]):
            total = op("fma.f32", *operands, total)
        if encode:
            for word in range(512 + 16*lane, 528 + 16*lane):
                value = op("mul.f32", *at(word)) if word < 768 else val(0)
                total = op("add.f32", value, total)
        else:
            total = op("add.f32", total, val(0))
        partial.append(total)
    for offset in (16, 8, 4, 2, 1):
        partial = [op("add.f32", partial[i], partial[i ^ offset]) for i in range(32)]
    square_sum = op("f32-load", SQUARE_SUM // BANK, row)
    rstd = inverse(square_sum)
    inverse3 = op("mul.f32", rstd, op("mul.f32", rstd, rstd))
    scales = [op("mul.f32", op("mul.f32", op("mul.f32", dot, val(0xbf000000)), inverse3),
                 val(0x3aaaaaab)) for dot in partial]
    stores = {}
    for col in range(768):
        index = row * 768 + col
        g, x = at(col)
        correction = op("mul.f32", op("add.f32", x, x), scales[col // width % 32])
        stores[GRADIENT + 2 * index] = op("narrow", op("fma.f32", rstd, g, correction))
        if encode:
            scaled = op("mul.f32", op("mul.f32", rstd, x), val(0x408f5c29))
            stores[FP8 + index] = op("e4m3", scaled)
    return stores


def check_backward(rows, encode, reference, candidate, *, retain=False):
    ports = (GRADIENT, MIXED, SQUARE_SUM, FP8) if encode else (GRADIENT, MIXED, SQUARE_SUM)
    expected = backward_row(encode)
    for ptx, save_inverse in ((reference, False), (candidate, True)):
        retain_bytes = retain and save_inverse
        stores = replay(ptx, ports[:-1] if retain_bytes else ports, cached=save_inverse)
        observed = {k: v for k, v in expected.items() if k // BANK == GRADIENT // BANK} if retain_bytes else expected
        assert stores.keys() == observed.keys(), (rows, encode, save_inverse, len(stores))
        for address, value in stores.items():
            assert value == observed[address], (rows, encode, save_inverse, address, mismatch(value, observed[address]))
    op.cache_clear()
    nodes.clear()
