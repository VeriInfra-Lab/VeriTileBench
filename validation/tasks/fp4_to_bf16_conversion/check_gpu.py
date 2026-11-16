"""Run the actual repaired Triton source against an independent format table.

This is an execution regression, separate from Lean proof replay and two-gates.
Run with Python/Torch/Triton on CUDA; pass the task source and result JSON path.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import torch
import triton


def load(path):
    spec = importlib.util.spec_from_file_location("fp4_scaled_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected(x, scales):
    magnitude = torch.tensor([0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0],
                             device="cuda", dtype=torch.int16)
    nibbles = torch.stack((x >> 4, x & 15), dim=-1).flatten().long()
    words = magnitude[nibbles & 7] | ((nibbles >= 8).to(torch.int32) * 0x8000).to(torch.int16)
    code = scales.long()
    scale_words = torch.where(code == 0, 0x40, code * 128)
    scale_words = torch.where(code == 255, 0x7fc0, scale_words).to(torch.int16)
    return words.view(torch.bfloat16) * scale_words.view(torch.bfloat16).repeat_interleave(32)[:len(words)]


def check(module, xbytes, sbytes, block):
    n = len(xbytes)
    xb = torch.zeros(n + 32, dtype=torch.uint8, device="cuda")
    sb = torch.zeros(len(sbytes) + 32, dtype=torch.uint8, device="cuda")
    x, scales = xb[7:7+n], sb[11:11+len(sbytes)]
    x.copy_(torch.tensor(xbytes, dtype=torch.uint8, device="cuda"))
    scales.copy_(torch.tensor(sbytes, dtype=torch.uint8, device="cuda"))
    before_x, before_s = xb.clone(), sb.clone()
    allocation = torch.full((2*n + 32,), 42.5, dtype=torch.bfloat16, device="cuda")
    output = allocation[16:16+2*n]
    module.triton_f4_to_scaled_bf16_kernel.fn[(triton.cdiv(n, block),)](
        x, scales, output, n, 32, 8, 1, 1, 2, 1, 23, 8, 127, 0, 0x3f000000, 127, 255,
        BLOCK_SIZE_IN=block, num_warps=4)
    want = expected(x, scales)
    torch.cuda.synchronize()
    actual_nan, expected_nan = torch.isnan(output), torch.isnan(want)
    class_bad = int((actual_nan != expected_nan).sum().item())
    bits, reference_bits = output.view(torch.int16), want.view(torch.int16)
    finite_bad = int(((bits != reference_bits) & ~expected_nan).sum().item())
    nan_payload_differences = int(((bits != reference_bits) & expected_nan & actual_nan).sum().item())
    assert torch.equal(xb, before_x) and torch.equal(sb, before_s), "input modified"
    assert bool(torch.all(allocation[:16] == 42.5)) and bool(torch.all(allocation[-16:] == 42.5)), "output guard modified"
    assert class_bad == finite_bad == nan_payload_differences == 0, (n, block, class_bad, finite_bad, nan_payload_differences)
    return {"count": n, "block": block, "output_lanes": 2*n,
            "classification_mismatches": class_bad, "non_nan_bit_mismatches": finite_bad,
            "nan_payload_differences": nan_payload_differences, "guard_and_input_checks": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    module = load(args.source.resolve())
    rows = []
    for block in (128, 256, 512, 1024, 2048):
        # Every packed byte paired with every scale byte, without random omissions.
        rows.append(check(module, list(range(256))*256,
                          [scale for scale in range(256) for _ in range(16)], block))
        for n in (1,2,15,16,17,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049):
            rows.append(check(module, [(i*73+128)%256 for i in range(n)],
                              [(i*127)%256 for i in range((2*n+31)//32)], block))
        print(f"block={block}: passed complete format table and 20 tail cases", flush=True)
    empty = module.fp4_to_bf16_conversion(torch.empty(0, dtype=torch.uint8, device="cuda"),
                                         torch.empty(0, dtype=torch.uint8, device="cuda"))
    assert empty.numel() == 0 and empty.dtype == torch.bfloat16
    report = {"accepted": True, "scope": "Actual repaired Triton execution; not a numerical admission or hardware proof",
              "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "distinct_packed_byte_scale_pairs_per_tile": 256*256,
              "torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
              "cases": rows, "case_count": len(rows), "empty_wrapper": "PASS",
              "elapsed_seconds": time.monotonic()-start}
    args.result.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
