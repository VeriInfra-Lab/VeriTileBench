"""Inspect the pinned CUDA validation kernels offline with cuobjdump."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re
import subprocess


MEAN = "_ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_7MeanOpsIffffEEjfLi4ELi4EEEEEvT1_"
SIMPLE = "_Z44ncclDevFunc_Reduce_PreMulSum_f32_RING_SIMPLEv"
LL128 = "_Z43ncclDevFunc_Reduce_PreMulSum_f32_RING_LL128v"
LL = "_Z40ncclDevFunc_Reduce_PreMulSum_f32_RING_LLv"

# The instruction encodings omit disassembler formatting and symbol names.
ENCODINGS = {
    MEAN: "0141cc36b73bb44b812126aba47bd4509f66fef0d255956e78e5c0e02f95674f",
    SIMPLE: "5cfd5b11e7faa49804bff7367ac52e6b797fc1504044c4d4e7b515e2e950e599",
    LL128: "38640060a46d84d753f9f20027745dcd4f4a445e1ed470d95f166137c39f901e",
    LL: "a9d08fc86d0a68288c3ff58b20eec8f8b387bb28f9f2ca5f4b0bc06578bfd58d",
}


def read_functions(cuobjdump, library, symbols):
    result = subprocess.run([str(cuobjdump), "--dump-sass", "--gpu-architecture", "sm_90",
                             "--function", ",".join(symbols), str(library)],
                            check=True, capture_output=True, text=True)
    chunks = re.split(r"(?m)^\s*Function : (\S+)\s*$", result.stdout)
    if sorted(chunks[1::2]) != sorted(symbols):
        raise ValueError(f"Missing or repeated SM90 functions in {library}")
    functions = {}
    for name, body in zip(chunks[1::2], chunks[2::2], strict=True):
        words = re.findall(r"/\*\s*0x([0-9a-f]{16})\s*\*/", body)
        if sha256(bytes.fromhex("".join(words))).hexdigest() != ENCODINGS[name]:
            raise ValueError(f"Unrecognized SM90 instruction encodings for {name}")
        lines = re.findall(r"(?m)^\s*/\*([0-9a-f]+)\*/\s+(?:@!?U?P\d+\s+)?(.*?)\s*;", body)
        functions[name] = {int(pc, 16): text.replace(".reuse", "") for pc, text in lines}
    return functions


def fp_ops(instructions):
    return Counter(text.split()[0] for text in instructions.values() if text.startswith(("FADD", "FMUL", "FFMA")))


def check(cuobjdump, torch, nccl):
    mean = read_functions(cuobjdump, torch, (MEAN,))[MEAN]
    assert fp_ops(mean) == Counter({"FADD": 54, "FMUL": 4})
    assert mean[0x1750] == "FADD R9, R4, R9" and mean[0x1780] == "FADD R2, R7, R2"
    assert mean[0x1900] == "FADD R16, R16, R9" and mean[0x1920] == "FADD R9, R9, R2"
    assert mean[0xb0a0].startswith("SHFL.DOWN") and mean[0xe190].startswith("SHFL.DOWN")
    assert mean[0xd9c0].startswith("ATOMG.E.ADD") and mean[0xe2a0] == "FMUL R19, R11, UR4"

    rank = read_functions(cuobjdump, nccl, (LL, LL128, SIMPLE))
    assert fp_ops(rank[LL]) == Counter({"FFMA": 4, "FMUL": 2})
    assert fp_ops(rank[SIMPLE]) == Counter({"FFMA": 106, "FMUL": 53})
    ll128 = rank[LL128]
    # First word only. Follow the initial send, intermediate send, and root
    # publication through the LL128 packing and the scalar scratch tail.
    words = {
        0x70b0: "FMUL R8, R14, R8",
        0x70f0: "IMAD.MOV.U32 R64, RZ, RZ, R8",
        0x7220: "STG.E.128.STRONG.SYS desc[UR4][R72.64], R64",
        0x33b0: "FFMA R4, R14, R4, R8",
        0x3440: "IMAD.MOV.U32 R24, RZ, RZ, R4",
        0x3550: "STG.E.128.STRONG.SYS desc[UR4][R36.64], R24",
        0x51f0: "FFMA R4, R15, R4, R8",
        0x5360: "STS.128 [R66], R4",
        0x5880: "LDS R67, [R26]",
        0x58a0: "ST.E desc[UR4][R58.64], R67",
    }
    assert all(ll128[pc] == expected for pc, expected in words.items())
    print("Validated SM90 ATen FP32 mean and NCCL scalar PreMulSum for LL, LL128 and Simple.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_cuda", type=Path, help="libtorch_cuda.so from Torch 2.10.0+cu128")
    parser.add_argument("nccl", type=Path, help="libnccl.so.2 from NVIDIA NCCL 2.27.5")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    args = parser.parse_args()
    check(args.cuobjdump, args.torch_cuda, args.nccl)
