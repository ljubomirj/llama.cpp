# MTPLX — MTP Speculative Decoding Reference (MLX)

Source: `~/LJ-asi-mlx/MTPLX/` — native MTP runtime for Apple MLX
README: `~/LJ-asi-mlx/MTPLX/README.md`

## What MTPLX Is

MTPLX is a speculative decoding runtime that uses the target model's own MTP
heads as drafters, achieving ~2.24x speedup on Qwen3.6-27B with **verified
bit-exact output** (`max_diff = 0.0` vs AR). Built specifically for Apple
Silicon / MLX.

Key differentiator from our llama.cpp MTP: **mathematically exact acceptance**
using Leviathan-Chen rejection sampling with `(p - q)+` residual correction —
not greedy argmax. This means it produces identical output distribution to
standard autoregressive at any temperature.

**No bailing_hybrid support.** MTPLX is built for Qwen3.5/3.6's GDN
(Gated Delta Network) hybrid architecture. Ling-2.6-flash uses GLA (Gated
Linear Attention), which is a different recurrent mechanism. But the
architectural patterns (state capture, tape replay, draft/verify loop) are
directly applicable.

## Draft/Verify Loop

The core is `generate_mtpk` (`generation.py:2117`, ~4200 lines total).

Each cycle:
1. Sample primary token from target logits
2. Draft K tokens through MTP heads (depth d=1..K)
3. Verify all K+1 tokens in a **single batched target forward pass**
4. Accept/reject per position using Leviathan-Chen: `min(1, p(x)/q(x))`
5. If ALL accepted: bonus token from verify logits at position K+1
6. On rejection: rollback GDN state to last accepted position

### Verify Strategies

| Strategy | How | Speed |
|----------|-----|-------|
| `batched` | Simple batched forward + rollback on reject | Baseline |
| `capture_commit` | Forward with per-position GDN state capture, selective commit | ~640x faster commit |
| `graphbank` | Pre-compiled `mx.compile` graphs per (suffix_length, depth, profile) | Fast dispatch |
| `graphbank_capture_commit` | Both combined | Best |

### GDN State Capture/Commit

This is MTPLX's core innovation for hybrid models:

1. **Before verify**: `snapshot_untrimmable_cache(cache)` saves cache state
2. **During verify**: `gdn_forward_with_capture()` runs GDN for all K+1 tokens
   but records recurrent state at each intermediate position (the "tape")
3. **On reject at position j**: `commit_captured_prefix()` extracts captured
   state at position j and installs it into the cache. Attention layers are
   trimmed back to accepted length. No re-forward needed.

## Metal Kernel Inventory

All in `mtplx/gdn_capture.py` (1832 lines, 8 kernels).

| Kernel | Line | Purpose |
|--------|------|---------|
| `linear_conv1d` | 61 | Custom Conv1d for GDN, per-position state recording |
| `linear_gated_delta` | 128 | Core GDN: delta update with per-position state capture |
| `linear_gated_delta_final` | 204 | GDN returning only final state (diagnostic) |
| `linear_gated_delta_from_conv` | 270 | **Fused**: conv_out → split q/k/v → norm → delta update |
| `linear_gated_delta_from_conv_stream` | 386 | Streaming: only captures from `CaptureStart` onward |
| `linear_gated_delta_from_conv_tape` | 502 | **Tape kernel**: records `(v_t - kv_mem) * beta_t` per step |
| `linear_gated_delta_from_conv_tape_replay` | 613 | **Replay**: replays steps 0..j-1 from tape to reconstruct state |
| `linear_gated_delta_from_conv_inline_g` | 687 | Fused gate computation (sigmoid/softplus/exp inline) |

### The Tape/Replay Mechanism

The most sophisticated approach to recurrent state rollback:

**During verify** (tape kernel): for each timestep t, records
`delta[t] = (v_t - kv_mem) * beta_t` — the incremental state update.

**On reject at position j** (replay kernel): takes the original state, conv
outputs, gate values, and the tape. Replays steps 0..j-1:
`state *= g; state += k * delta[t]` — reconstructs exact state at position j.

This avoids storing all T intermediate full states. The tape stores only the
compact delta vectors, and replay reconstructs from those.

**Backend selection** (`resolve_gdn_capture_backend`, line 898): auto-selects
between `stock` (sequential single-token), `linear_gdn` (custom kernel),
`linear_gdn_from_conv` (fused), `linear_gdn_from_conv_tape` (tape+replay),
etc. based on model config and hardware.

## MTP Patching System

MTPLX uses runtime monkey-patching: loads model via stock `mlx-lm`, then
replaces `model.__class__` with an enhanced subclass. Each architecture has
its own patch:

| Patch | Models | Lines |
|-------|--------|-------|
| `mtp_patch.py` | Qwen3.5/3.6/Qwen3-Next | 677 |
| `deepseek_mtp_patch.py` | DeepSeek V3/V3.2, GLM MoE DSA | 413 |
| `glm_mtp_patch.py` | GLM-4 MoE / MoE Lite | 400 |
| `mimo_mtp_patch.py` | MiMo | 300 |
| `nemotron_h_mtp_patch.py` | Nemotron-H | 381 |

### Common MTP Patch Pattern

All patches share the same structure (from DeepSeek, which is closest to
Bailing/Ling-2.6):

```
MTP head components:
  enorm          — RMSNorm on token embedding
  hnorm          — RMSNorm on hidden state
  eh_proj        — project [e_norm; h_norm] back to n_embd
  mtp_block      — full decoder layer (attention + FFN)
  shared_head_norm — separate output norm (or reuse trunk's)
  shared_head_head — separate LM head (or reuse trunk's)

Forward:
  tok_embd = embed(token)
  h_norm = RMSNorm(hidden_state)
  e_norm = RMSNorm(tok_embd)
  concat = [e_norm, h_norm]
  projected = eh_proj @ concat
  output = decoder_layer(projected)
  logits = lm_head(RMSNorm(output))
```

This is **exactly** the structure of our `bailing_hybrid_mtp` in
`bailing-hybrid.cpp:651-890`.

## Novel Techniques Beyond Basic MTP

1. **Leviathan-Chen acceptance** (`sampling.py:143-183`): True rejection
   sampling with `(p - q)+` residual. Uses fp32 for p/q ratio to avoid BF16
   underflow. Our llama.cpp uses greedy argmax — less mathematically correct.

2. **GDN tape/replay**: The delta-tape + replay kernel approach for O(1)
   rollback of recurrent state. The compact representation avoids storing
   full state at every position.

3. **GraphBank** (`graphbank.py:581`): Pre-compiled `mx.compile` graphs keyed
   by `(suffix_length, depth, profile)`. Each verify shape gets one compiled
   graph. Achieves 0.073ms per commit vs 47ms for full verify.

4. **Adaptive depth** (`adaptive.py`): EWMA-tracked acceptance rates, draft
   confidence (top2_margin, entropy), and cost estimates to dynamically
   adjust MTP depth D per cycle. Increases depth after N consecutive
   full-accepts, decreases on early rejection.

5. **Draft-only LM head** (`draft_lm_head.py`): Requantizes target's lm_head
   to 3-4 bit for draft proposals only. ~29% draft time savings without
   affecting target accuracy.

6. **Online hidden corrector**: EWMA-tracked deltas between draft and target
   hidden states, applied to correct systematic drift in future drafts.

7. **Session Bank** (`session_bank.py`): Warm-prefix state reuse across turns,
   preserving exact KV cache state to avoid re-prefilling shared context.

## What We Can Apply to Ling-2.6-flash / llama.cpp

### Directly Applicable

| Technique | What to Port | Effort |
|-----------|-------------|--------|
| Tape/replay for GLA | GLA state is `[B, H, D, D]` (56MB total). Record delta `k_t * v_t` per step, replay from tape on reject. | Medium |
| Capture/commit verify | Replace checkpoint restore with per-position state capture during verify. | Medium |
| Adaptive depth | Track acceptance EWMA, adjust `spec-draft-n-max` dynamically. | Low |
| Draft-only LM head | Requantize lm_head for MTP draft proposals. | Low |

### Needs Adaptation

| Technique | Issue |
|-----------|-------|
| GDN kernels → GLA | GDN uses conv1d + gated delta rule. GLA uses simple recurrence `h = h * decay + k * v^T`. Need GLA-specific tape/replay. |
| MLX metal_kernel → ggml-metal | Different APIs. MLX generates MSL from inline strings; ggml uses `.metal` files + op registration. |
| GraphBank | ggml doesn't have `mx.compile`. Would need pre-allocated compute graphs per verify shape. |
| Leviathan-Chen acceptance | Requires draft logits (probabilities), not just argmax. Need to pass draft distribution to verify step. |

### Not Applicable

| Technique | Why |
|-----------|-----|
| GDN conv1d kernels | Ling-2.6 uses GLA, not GDN. No conv1d in GLA. |
| Thermal management | Hardware-specific, not algorithmic. |
| MLX-specific patches | Monkey-patching `__class__` doesn't apply to llama.cpp's C++ model loading. |

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `mtplx/generation.py` | ~4200 | Core AR, MTP1, MTP-K generation loops |
| `mtplx/gdn_capture.py` | 1832 | Metal kernels for GDN state capture, commit, tape replay |
| `mtplx/sampling.py` | 235 | Leviathan-Chen acceptance, residual distribution |
| `mtplx/mtp_patch.py` | 677 | Qwen3.5/3.6 MTP injection (closest to our model) |
| `mtplx/deepseek_mtp_patch.py` | 413 | DeepSeek V3 MTP (MLA + MoE, most similar architecture) |
| `mtplx/adaptive.py` | 212 | Adaptive depth policies |
| `mtplx/graphbank.py` | 581 | Pre-compiled verify graph bank |
| `mtplx/cache_state.py` | 2157 | Cache snapshot, rollback, detach |
| `mtplx/draft_lm_head.py` | 203 | Draft-only requantized LM head |
| `mtplx/backends/registry.py` | 1090 | Architecture catalog and compatibility |
