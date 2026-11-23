"""Fuse the private ANVIL equalized gradient into its parameter update."""
import torch
import triton

TORCH_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
BANKS = {
    (3, 2816, 768): -1,  # MLP
    (2, 768, 768): -1,   # VO
    (6, 256, 768): -2,   # QK
}


def _epilogue(v, energy, beta2, red_dim, p, mantissa, wd, lr, gate):
    lane_power = v.float().square().mean(dim=red_dim, keepdim=True)
    length = v.size(red_dim)
    pre_norm = lane_power.sum(dim=(-2, -1), keepdim=True).mul_(length).sqrt_()
    energy.lerp_(lane_power.to(dtype=energy.dtype), 1 - beta2)
    gain = energy.clamp_min(1e-10).rsqrt_()
    post_power = (lane_power * length) * gain.float().square()
    post_norm = post_power.sum(dim=(-2, -1), keepdim=True).sqrt_()
    scale = gain * (pre_norm / post_norm.clamp_min_(1e-10))
    # Preserve the BF16 store from the separate equalizer inside the fused graph.
    update = torch.bitwise_or((v * scale.type_as(v)).view(torch.int16), 0).view(torch.bfloat16).float()

    raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    precise = raw.view(torch.float32)
    aligned = (gate.float() * precise) >= 0
    wd_factor = wd.to(torch.float32)
    lr_factor = lr.to(torch.float32)
    precise.copy_(precise - (precise * aligned * wd_factor * lr_factor) - (update * lr_factor))
    p.copy_((raw >> 16).to(torch.uint16))
    mantissa.copy_(raw.to(torch.uint16))


update_epilogue = torch.compile(_epilogue, dynamic=False, fullgraph=True)


def eligible(v, energy, p, mantissa, wd, lr, gate, beta2, red_dim):
    if (torch.version.git_version != TORCH_REVISION or triton.__version__ != "3.6.0"
            or v.device.type != "cuda" or torch.cuda.get_device_capability(v.device) != (9, 0)
            or beta2 != 0.9 or BANKS.get(tuple(v.shape)) != red_dim):
        return False
    matrices, rows, columns = v.shape
    lane_shape = (matrices, rows, 1) if red_dim == -1 else (matrices, 1, columns)
    inputs = ((v, torch.bfloat16, v.shape), (energy, torch.float32, lane_shape),
              (p, torch.uint16, v.shape), (mantissa, torch.uint16, v.shape),
              (wd, torch.float32, ()), (lr, torch.float32, (matrices, 1, 1)),
              (gate, torch.float32, v.shape))
    return all(t.device == v.device and t.dtype == dtype and tuple(t.shape) == shape
               and t.is_contiguous() for t, dtype, shape in inputs) and all(
                   t.data_ptr() % 16 == 0 for t in (v, energy, p, mantissa, gate))
