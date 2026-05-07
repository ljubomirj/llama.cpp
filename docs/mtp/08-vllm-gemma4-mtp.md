# vLLM Gemma-4 MTP Reference

Source: https://github.com/vllm-project/vllm/pull/41745
PR: "[Spec Decode] Add Gemma4 MTP speculative decoding support" by lucianommartins
Status: OPEN (as of 2026-05-05)

## Architecture

Gemma-4 MTP uses a **separate assistant model** (sidecar) that shares the
target's KV cache. Key design choices:

- **Q-only attention**: the assistant has Q and O projections but **NO K or V
  projections**. It reads K, V directly from the target model's KV cache.
- **Position anchoring**: all draft steps reuse the last target position
  (constant_draft_positions). The assistant was trained this way.
- **Centroids masking** (E2B/E4B only): reduces lm_head computation from ~262K
  to ~4K candidate tokens via learned centroid-based vocabulary selection.
- **CUDA graph acceleration**: pre-captured graphs for centroids at batch sizes
  [1,2,4,8,16,32,64].

## Performance (H100)

| Model | gamma | Output TPS | Speedup |
|-------|-------|-----------|---------|
| E2B | baseline | 267 | 100% |
| E2B | 2 | 348 | 130% |
| E4B | baseline | 171 | 100% |
| E4B | 4 | 305 | 178% |
| 26B | baseline | 169 | 100% |
| 26B | 4 | 270 | 160% |
| 31B | baseline | 43 | 100% |
| 31B | 8 | 136 | **319%** |

Gemma-4 31B with gamma=8 achieves **3.19x speedup** — the highest reported
MTP speedup we've seen.

On GB10 (DGX Spark) with NVFP4 26B + BF16 assistant: 67-69% acceptance,
2.34x sequential speedup.

## What's Unique vs Our Approach

1. **Separate model loading**: The assistant is loaded as a completely separate
   vLLM model instance (`gemma4_mtp` architecture), not as extra tensors in
   the same GGUF. This is similar to how draft models work in llama.cpp, but
   the assistant is tiny (just Q projections + MLP layers).

2. **No separate KV cache**: The biggest architectural difference. The
   assistant attends over the target's KV cache directly. This eliminates the
   entire MTP KV management problem we've been debugging (position gaps, state
   sync, rollback).

3. **Centroids masking**: An optimization specific to large-vocabulary models.
   Learns ~2048 centroids that map to clusters of tokens, then only computes
   logits for the ~4K tokens in the top-K clusters. Reduces lm_head from
   O(V*D) to O(K*D) where K << V.

4. **Constant draft positions**: All draft steps use the same position as the
   last target token. This is Gemma-4 specific training, not applicable to
   Ling-2.6.

## Ling-2.6 MTP Depth

Ling-2.6-flash has `nextn_predict_layers = 1`. This means **one MTP head**.

The MTP head predicts **1 extra token** (t+1) per cycle. The trunk generates
token t via standard autoregressive sampling, then the MTP head predicts t+1
as a draft. So:

```
STP (trunk alone):  generates t, then t+1, then t+2, ...
MTP (trunk + 1 head): trunk generates t, MTP drafts t+1,
                       verify batch [t, t+1] → accepted or rejected
                       if accepted: 2 tokens generated per cycle
                       if rejected: 1 token generated per cycle
```

This is **depth-1 MTP** (also called "MTP1" in MTPLX terminology). It can
predict at most 1 draft token per cycle. Compare with:

| Model | MTP heads (depth) | Draft tokens per cycle |
|-------|------------------|----------------------|
| Ling-2.6-flash | 1 | 1 (t+1) |
| Qwen3.6-27B | 1 | 1 (t+1) |
| Gemma-4 31B | ~8 layers | up to 7 (t+1 through t+7) |
| DeepSeek V3 | 2 | up to 2 (t+1, t+2) |

With depth-1, the theoretical maximum speedup is 2x (if every draft is
accepted). In practice, Qwen3.6-27B on Metal gets ~47% speedup (17→25 t/s).
For Ling-2.6-flash with GLA layers on CPU, the actual speedup would be lower
due to the CPU bottleneck for GLA layers.

## Relevance to Our Work

**Low relevance.** Gemma-4 MTP is architecturally different:
- Separate model (not embedded tensors)
- Shared KV cache (no MTP KV management)
- Q-only attention (no K/V projections)
- Position anchoring (specific training)

The only transferable insight: **vLLM achieves 319% speedup with depth-8** on
the 31B model. This validates that deeper MTP (more draft heads) gives much
better speedups than our depth-1 approach. If Ling-2.6 had more MTP heads
(2-4), the potential speedup would be significantly higher.
