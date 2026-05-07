# mlx-vlm PR #1112 — Gemma 4 MTP Speculative Decoding

Source: https://github.com/Blaizzy/mlx-vlm/pull/1112
PR: "Add Gemma 4 MTP speculative-decoding drafter" by Blaizzy
Status: MERGED (May 5, 2026)

## Architecture

Gemma 4 MTP uses a **separate assistant model** (sidecar) that shares the
target's KV cache. The assistant is a tiny decoder-only model with Q-only
attention.

### Key Design Choices

1. **Shared KV cache**: The assistant has Q and O projections but NO K or V
   projections. It reads K, V directly from the target model's KV cache. This
   eliminates the entire MTP KV management problem — no position gaps, no
   separate KV sync, no KV rollback.

2. **Position anchoring**: All draft steps reuse the last target position
   (constant_draft_positions). The assistant was trained this way — each draft
   step produces one token anchored at `pos = last_target_pos`.

3. **Centroids masking** (E2B/E4B only): Reduces lm_head computation from
   ~262K to ~4K candidate tokens via learned centroid-based vocabulary
   selection. `MaskedEmbedder` with `use_ordered_embeddings=True`. Not needed
   for 26B-A4B/31B (tied dense head).

4. **Batched verification**: All drafts verified in one target forward pass.

## Rollback: Speculative Cache Wrapper Pattern

The assistant uses a clean write-buffer + commit approach:

```
BeginSpeculation():
  wraps all KV caches in speculative wrappers
  writes are buffered, not committed to live cache

Speculation.Commit(n):
  commits only first n accepted tokens' KV to live caches
  discards rest
```

**Relevance to Ling-2.6 GLA rollback**: This pattern applies to GLA state
rollback too. Instead of checkpoint/restore of the entire recurrent state
buffer, wrap writes in a speculation buffer and commit only accepted prefix.
The `n_confirmed` + snapshot approach (from mlx-lm PR #990, doc 09) is
essentially the same pattern applied to SSM/GLA state.

## Draft/Verify Flow

```
1. Get last token embedding from target's embedding table
2. Get target's last hidden state (pre-norm)
3. Concatenate [embedding, hidden_state]
4. Pass through assistant's Draft() → logits + projected hidden
5. Sample draft token
6. Position anchored at last target token (not advancing)
7. Batch verify: [drafts...] through target in one forward pass
8. Acceptance: match check (greedy) or Leviathan-Chen (stochastic)
9. Rollback: target-side `rollback_speculative_cache` hooks per layer
```

## Performance (Apple Silicon, greedy, byte-identical)

| Model | B | Best bs | tot tok/s | Speedup |
|-------|---|---------|-----------|---------|
| 26B-A4B | 4 | 3 | 85.5 | **3.94×** |
| 26B-A4B | 8 | 3 | 165.1 | 1.55× |
| 31B | 4 | 3 | 17.1 | 2.29× |
| 31B | 8 | 2 | 21.4 | 1.41× |
| E4B | 4 | 4 | 62.1 | 1.56× |

Note: B=1 is a net loss for 31B (0.64-0.87× depending on block size). MTP
benefits from batching.

## Target-Side Hooks

New hooks added to the gemma4 model:
- `shared_kv_sink`: assistant reads target KV directly
- `hidden_sink`: captures pre-norm hidden states
- `capture_layer_ids`: which layers feed the assistant
- `rollback_speculative_cache`: per-layer rollback on draft rejection

For Ling-2.6, `rollback_speculative_cache` would restore GLA state from
snapshot + trim MLA KV cache.

## What Applies to Ling-2.6

| Technique | Applicable? | Notes |
|-----------|-------------|-------|
| Shared KV cache | No | Ling MTP has own attention with K/V projections |
| Position anchoring | No | Ling MTP was trained with advancing positions |
| Centroids masking | Future | Could reduce lm_head cost for draft proposals |
| Batched verification | Yes | Already doing this |
| Speculative cache wrapper | **Yes** | Cleaner than checkpoint restore for GLA rollback |
| Batching for speedup | **Yes** | B>1 makes MTP much more effective |
| Leviathan-Chen acceptance | **Yes** | Already in MTPLX reference; matches Gemma sampled mode |

## Key Difference from Our Architecture

Gemma 4 assistant is architecturally simpler than Ling MTP:
- No separate attention (Q-only, reads target KV)
- No recurrent layers (pure attention)
- No KV management (shares target's)

Ling MTP has its own MLA attention with K/V projections + own KV cache.
This makes the implementation more complex but is what the model was trained for.

## Reference Files in PR

```
mlx_vlm/speculative/drafters/gemma4_assistant/
├── __init__.py          # DraftModel: bind, make_cache, reset, draft_block, __call__, sanitize
├── mask.py              # MaskedEmbedder for centroid-routed sparse LM head
└── parity_check.py      # End-to-end parity test vs no-drafter baseline
mlx_vlm/models/gemma4/
└── (model)              # Target model with shared KV, hidden_sink, rollback hooks
mlx_vlm/generate.py      # _mtp_rounds, _mtp_rounds_batch
scripts/mtp_batch_sweep.py  # B ∈ {4,8,16}, block ∈ {2,3,4} performance sweep
```
