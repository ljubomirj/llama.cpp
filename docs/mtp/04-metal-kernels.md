# Metal Kernel Requirements for MTP

## Background

Hybrid models like Ling-2.6-flash and Qwen3.6 use recurrent attention layers
(GLA, GDN) that maintain per-position state. Standard speculative decoding
needs to checkpoint and roll back this state when draft tokens are rejected.
Without custom Metal kernels, this state lives in Metal buffers that the
generic checkpoint mechanism cannot capture or restore.

There are two separate kernel requirements:

1. **GLA compute kernel** — run GLA attention on Metal instead of CPU fallback
2. **GDN state capture/replay kernels** — snapshot and restore recurrent state
   for speculative decoding rollback

## GLA Compute Kernel

### Current State

`ggml_gated_linear_attn` in ggml has NO Metal backend. All GLA (local) layers
in Ling-2.6-flash fall back to CPU. This means 30 out of 32 layers transfer
data between GPU and CPU for every token.

### Reference: MLX GLA Kernel

`~/LJ-asi-mlx/omlx/mlx_lm/models/bailing_hybrid.py:64-125`

The `_make_recurrent_gla_kernel` is a ~60 line MSL kernel that implements
GLA attention on Metal. Key characteristics:

- **Inputs**: q, k, v (each `[B, T, H, D]`), g (decay, `[H]`), state_in (`[B, H, D, D]`)
- **Outputs**: y (`[B, T, H, D]`), state_out (`[B, H, D, D]`)
- **State**: D×D matrix per head, updated recurrently: `h_t = h_{t-1} * exp(g) + k_t * v_t`
- **Output**: `y_t = q_t @ h_t` (matmul with accumulated state)
- **Threadgroup**: splits D dimension across 32 threads, uses `simd_sum` for reduction
- **Decay**: `exp(g[h_idx])` per head, constant over time (head-level gating)

### Porting to ggml-metal

The ggml-metal op would be `GGML_OP_GATED_LINEAR_ATTN`. The kernel needs:

1. Read q, k, v from ggml tensors (may be quantized — need dequant or f16 path)
2. Read/write recurrent state from a separate state buffer (not the KV cache)
3. Support batched processing (multiple tokens in one call)
4. Thread group sizing: D/32 threads per group, H*B groups

The state buffer format in ggml would need to match what `ggml_gated_linear_attn`
expects on CPU. Check `src/ggml/ggml.c` for the CPU implementation's state layout.

## GDN State Capture/Replay Kernels

### Current State

No Metal kernels exist for GDN state capture/replay. The generic
`llama_state_seq_get_data` / `llama_state_seq_set_data` checkpoint functions
handle standard KV cache but not per-layer recurrent state buffers.

### Reference: MTPLX GDN Kernels

`~/LJ-asi-mlx/MTPLX/mtplx/gdn_capture.py` — 1832 lines, 8 Metal kernels.

#### Kernel Inventory

| Kernel | Lines | Purpose |
|--------|-------|---------|
| `_make_linear_conv1d_kernel` | 61-120 | 1D convolution for GDN preprocessing |
| `_make_linear_gated_delta_kernel` | 128-196 | Core GDN: gated delta rule with decay |
| `_make_linear_gated_delta_final_kernel` | 204-262 | Final state aggregation for GDN |
| `_make_linear_gated_delta_from_conv_kernel` | 270-378 | GDN from convolution output |
| `_make_linear_gated_delta_from_conv_stream_kernel` | 386-494 | Streaming GDN (incremental, per-token) |
| `_make_linear_gated_delta_from_conv_tape_kernel` | 502-605 | GDN with state tape recording |
| `_make_linear_gated_delta_from_conv_tape_replay_kernel` | 613-679 | **State tape replay for rollback** |
| `_make_linear_gated_delta_from_conv_inline_g_kernel` | 687-815 | GDN with inline gating |

The critical kernels for speculative decoding are:
- **Tape kernel** (`_make_linear_gated_delta_from_conv_tape_kernel`): records
  per-position state into a tape buffer during forward pass
- **Replay kernel** (`_make_linear_gated_delta_from_conv_tape_replay_kernel`):
  replays the tape from a given position to restore state after draft rejection

#### State Tape Concept

The tape approach works like this:

```
Forward pass (draft generation):
  for each token t:
    state[t] = update(state[t-1], input[t])
    tape[t] = state[t]  // record snapshot

Verification:
  if draft rejected at position r:
    state = replay(tape, from=r-1)  // restore to last known-good state
```

The tape is a `[num_layers, max_seq_len, state_dim]` buffer that stores
per-position recurrent state snapshots. For rollback, the replay kernel
reads the tape and reconstructs the state from the last accepted position.

### Porting to ggml-metal

This requires:

1. **State tape buffer**: allocate per-layer Metal buffers for state snapshots
2. **Tape write hook**: after each token decode, write recurrent state to tape
3. **Tape read/replay**: on draft rejection, restore state from tape position
4. **Integration with checkpoint**: hook into `llama_state_seq_get/set_data`
   to include tape buffers in checkpoint data

The GDN kernels in MTPLX use MLX's `metal_kernel` API which generates MSL
from inline source strings. The ggml-metal equivalent would use the standard
`ggml_metal_kernels` infrastructure with `.metal` source files.

## Implementation Priority

1. **GLA compute kernel** — standalone, enables GPU acceleration for local
   layers even without MTP. Significant perf improvement for trunk-only mode.
2. **GDN tape+replay kernels** — required for MTP speculative decoding on
   Metal. Without these, any hybrid model with recurrent layers will have
   corrupted state on draft rejection.
3. **GLA state capture** — if GLA layers also need state rollback (they do
   for speculative decoding), the tape/replay mechanism needs to cover GLA
   state too. The GLA kernel's `state_out` is the state that needs capturing.

## Detailed Architecture Reference (from bailing_hybrid.py)

Source: `~/LJ-asi-mlx/omlx/mlx_lm/models/bailing_hybrid.py` (758 lines)
This is the MLX model definition submitted upstream as mlx-lm PR #1227.

### GLA (LinearAttention) — lines 372-482

```
query_key_value  → single projection for Q, K, V  (no separate projections)
query_layernorm  → RMSNorm on Q  (when use_qk_norm=True)
key_layernorm    → RMSNorm on K  (when use_qk_norm=True)
g_proj           → learned gate projection  (hidden_size → num_heads * head_dim)
g_norm           → GroupRMSNorm on GLA output  (groups=group_norm_size)
dense            → output projection
```

The GLA forward pass:
1. QKV projection → split into Q, K, V
2. Optional QK norm
3. Partial RoPE on Q and K (partial_rotary_factor=0.5, so only half the head_dim)
4. Recurrent GLA: `h_t = h_{t-1} * exp(g) + k_t * v_t^T`, `y_t = q_t @ h_t`
5. Gating: `output = g_norm(y) * sigmoid(g_proj(x))`
6. Output projection

### GLA Decay Slopes — lines 417-432

The decay `g` is **deterministic, not learned**. Computed from head count with
a layer-dependent factor:

```python
slopes = [2^(-(2^(-(log2(n)-3))) * (i+1)) for i in range(n_heads)]
layer_factor = 1 - layer_idx / (num_hidden_layers - 1) + 1e-5
g = -slopes * layer_factor   # shape: [n_heads]
```

Earlier layers decay faster (layer 0: factor ≈ 1.0), later layers retain more
history (layer 31: factor ≈ 0.0). The negative sign means `exp(g)` is always < 1.

### GLA State Format

```
state h: [Batch, n_heads, head_dim, head_dim]   # D×D matrix per head
```

For Ling-2.6-flash (head_dim=128, n_heads=32):
- Per layer: 32 × 128 × 128 × 4 bytes = 2 MB (f32)
- 28 GLA layers: 56 MB total

This is tiny compared to the KV cache. In ggml, this maps to the recurrent
state memory (`llama_memory_recurrent`).

### GLA Metal Kernel — lines 64-128

The kernel processes T timesteps in a single call:
- Grid: `(32, D, B*H)` — 32 threads for D/32 elements, D threads for v_dim, B*H groups
- Threadgroup: `(32, 4, 1)`
- Each thread holds `D/32` elements of the state row
- Per timestep: update state, compute output via `simd_sum`
- State is loaded once at start, stored once at end

```
for t in 0..T:
    for each dk_slice:
        state[dk, dv] = state[dk, dv] * decay + k[t, dk] * v[t, dv]
        out += state[dk, dv] * q[t, dk]
    y[t, dv] = simd_sum(out)   # reduce across 32 threads
```

This is directly portable to ggml-metal as `GGML_OP_GATED_LINEAR_ATTN`.
~60 lines of MSL, no complex dependencies.

### MLA (MultiLatentAttention) — lines 256-369

Key detail: **decode vs prefill optimization** (lines 355-366):

```
if L == 1:  # decode (single token)
    q_nope = embed_q(q_nope)      # absorb Q into KV
    k = v = kv_latent              # use compressed KV directly
else:        # prefill (multiple tokens)
    k = embed_q(kv_latent, transpose=False)  # expand KV
    v = unembed_out(kv_latent)               # expand KV
```

For decode, Q is absorbed into the KV latent space (avoids expanding KV).
For prefill, KV is expanded to full dimension for flash attention.

This matches what our bailing-hybrid.cpp does with the `wk_b` absorption.

### Cache Setup — lines 751-758

```python
def make_cache(self):
    for layer in self.layers:
        if layer.is_global:    # MLA layers (7, 15, 23, 31)
            caches.append(KVCache())         # standard KV cache
        else:                  # GLA layers (all others)
            caches.append(ArraysCache(size=1))  # single state array [B, H, D, D]
```

Global layers use a growing KV cache. Local layers use a fixed-size state
array — no cache growth at all.

### Layer Determination — lines 582-584

```python
is_global = (layer_idx + 1) % group_size == 0 or layer_idx >= (n_layers // group_size) * group_size
```

For Ling-2.6 (group_size=8, n_layers=32): global at indices 7, 15, 23, 31.
This means layers 0-6, 8-14, 16-22, 24-30 are GLA (local), and 7, 15, 23, 31
are MLA (global). 28 GLA + 4 MLA = 32 total.

### MoE Expert Selection — lines 485-518

```python
# Group-based selection with sigmoid scoring
n_group=8, topk_group=4, num_experts_per_tok=8
# 256 experts → 8 groups of 32
# Select top 4 groups (by sum of top-2 expert scores in each group)
# Then select top 2 experts from each selected group → 8 active
score_function = "sigmoid"  # not softmax
expert_bias correction applied
```

### Weight Sanitization — lines 672-729

- **Drops MTP layers**: `layer_idx >= n_layers` → deleted. The MLX model
  deliberately strips MTP. Our llama.cpp `bailing_hybrid_mtp` loads only
  these dropped tensors — same approach.
- **Stacks MoE expert weights**: `experts.{e}.{m}.*` → `switch_mlp.{m}.*`
- **Splits kv_b_proj**: into `embed_q` (absorbed K) and `unembed_out` (V)
- **Gate remap**: `mlp.gate.weight` → `mlp.gate.gate_proj.weight`

## Estimated Effort

| Task | Lines of MSL | Complexity | Depends On |
|------|-------------|------------|------------|
| GLA compute kernel | ~100-150 | Medium | Understanding ggml-metal op registration |
| GDN tape write kernel | ~80-120 | Medium | GDN state layout in ggml |
| GDN tape replay kernel | ~60-100 | Medium | Tape write kernel |
| Checkpoint integration | ~200-300 C++ | High | Both tape kernels, ggml checkpoint internals |
| Testing/debugging | — | High | All of the above |
