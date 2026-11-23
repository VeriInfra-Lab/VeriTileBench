"""Token-by-channel Triton tiles for the Track 1 n-gram embedding."""

# Each channel tile has the same 1024 value lanes as the upstream launch.
TOKEN_TILE, CHANNEL_TILE = 4, 256

# These definitions replace the corresponding kernels in the upstream module.
# There they share its triton imports and its original _sign_rows function.

@triton.jit
def _ngram_embed_fwd_kernel(cache_ptr, slots_ptr, pool_ptr, inp_ptr, out_ptr, T,
                            D: tl.constexpr, RMASK: tl.constexpr,
                            BT: tl.constexpr, BC: tl.constexpr,
                            B0: tl.constexpr, B1: tl.constexpr, T0: tl.constexpr,
                            T1: tl.constexpr, T2: tl.constexpr, FULL_ROWS: tl.constexpr = False):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    live_token = tl.full((BT,), True, tl.int1) if FULL_ROWS else token < T
    slot_mask = None if FULL_ROWS else live_token
    bigram_slot = tl.load(slots_ptr + token, slot_mask, eviction_policy="evict_last").to(tl.int64)
    trigram_slot = tl.load(slots_ptr + (T + token), slot_mask, eviction_policy="evict_last").to(tl.int64)
    bigram_sign, trigram_sign = _sign_rows(inp_ptr, token, live_token, RMASK, B0, B1, T0, T1, T2)
    first_channel = tl.arange(0, BC) if FULL_ROWS else tl.program_id(1) * BC + tl.arange(0, BC)
    for tile in tl.static_range(D // BC if FULL_ROWS else 1):
        channel = first_channel + tile * BC
        live = None if FULL_ROWS else live_token[:, None] & (channel[None, :] < D)
        a = tl.load(cache_ptr + bigram_slot[:, None] * D + channel[None, :], live).to(tl.float32)
        b = tl.load(cache_ptr + trigram_slot[:, None] * D + channel[None, :], live).to(tl.float32)
        sa = tl.load(pool_ptr + bigram_sign[:, None] * D + channel[None, :], live,
                     eviction_policy="evict_last").to(tl.float32)
        sb = tl.load(pool_ptr + trigram_sign[:, None] * D + channel[None, :], live,
                     eviction_policy="evict_last").to(tl.float32)
        offset = token[:, None] * D + channel[None, :]
        tl.store(out_ptr + offset, (a * sa + b * sb).to(tl.bfloat16), live)


@triton.jit
def _ngram_embed_bwd_kernel(g_ptr, pool_ptr, inp_ptr, out_ptr, T,
                            D: tl.constexpr, RMASK: tl.constexpr,
                            BT: tl.constexpr, BC: tl.constexpr,
                            B0: tl.constexpr, B1: tl.constexpr, T0: tl.constexpr,
                            T1: tl.constexpr, T2: tl.constexpr, FULL_ROWS: tl.constexpr = False):
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    live_token = tl.full((BT,), True, tl.int1) if FULL_ROWS else token < T
    bigram_sign, trigram_sign = _sign_rows(inp_ptr, token, live_token, RMASK, B0, B1, T0, T1, T2)
    first_channel = tl.arange(0, BC) if FULL_ROWS else tl.program_id(1) * BC + tl.arange(0, BC)
    for tile in tl.static_range(D // BC if FULL_ROWS else 1):
        channel = first_channel + tile * BC
        live = None if FULL_ROWS else live_token[:, None] & (channel[None, :] < D)
        offset = token[:, None] * D + channel[None, :]
        g = tl.load(g_ptr + offset, live).to(tl.float32)
        sa = tl.load(pool_ptr + bigram_sign[:, None] * D + channel[None, :], live,
                     eviction_policy="evict_last").to(tl.float32)
        sb = tl.load(pool_ptr + trigram_sign[:, None] * D + channel[None, :], live,
                     eviction_policy="evict_last").to(tl.float32)
        tl.store(out_ptr + offset, (g * sa).to(tl.bfloat16), live)
        tl.store(out_ptr + T * D + offset, (g * sb).to(tl.bfloat16), live)
