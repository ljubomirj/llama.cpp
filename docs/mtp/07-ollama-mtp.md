# Ollama MTP Implementation (v0.23.1-rc0)

Source: `~/Downloads/ollama-0.23.1-rc0/`
PR: https://github.com/ollama/ollama/pull/15980 ("mlx: Gemma4 MTP speculative decoding")
Scope: **Gemma-4 only, MLX backend only** (Apple Silicon)

## Architecture

Ollama has three execution backends, only one supports MTP:

| Backend | Runner | MTP? |
|---------|--------|------|
| llama.cpp (C++) | `runner/llamarunner/` | No |
| Ollama native (Go) | `runner/ollamarunner/` | No |
| MLX (Apple Silicon) | `x/mlxrunner/` | **Yes — all MTP here** |

The MTP code is entirely in the MLX runner. It does NOT use llama.cpp's
speculative decoding infrastructure.

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `x/mlxrunner/mtp.go` | 959 | Entire draft/verify loop |
| `x/mlxrunner/pipeline.go` | — | Entry point, decides MTP vs normal |
| `x/models/gemma4/assistant.go` | — | Gemma-4 draft model (reads target's KV) |
| `x/mlxrunner/cache/cache.go` | — | Speculative KV cache wrappers |
| `x/models/gemma4/gemma4.go` | — | Target model with KV sharing |

## How It Works

### Draft Phase (`generateMTPDrafts`, mtp.go:463)

```
1. Get last token embedding from target's embedding table
2. Get target's last hidden state
3. Concatenate [embedding, hidden_state]
4. Pass through assistant's Draft() method → logits + projected hidden
5. Sample draft token from logits
6. CRITICAL: position anchored at last target token (not advancing)
   "Gemma4 assistant MTP is trained as single-position drafting"
```

### Verify Phase — Two Modes

**Greedy** (`acceptMTPDrafts`, mtp.go:533):
- Batch all draft tokens into single target forward pass
- Compare argmax of target logits at each position vs draft tokens
- Simple exact match

**Sampled** (`acceptSampleMTPDrafts`, mtp.go:617):
- Rejection sampling: `accept_prob = min(p_target / p_draft, 1)`
- Bernoulli trial per position
- Residual distribution `(p - q)+` for correction token on rejection
- Similar to MTPLX's Leviathan-Chen approach

### Cache Rollback (cache/cache.go)

```
BeginSpeculation():
  wraps all KV caches in speculative wrappers
  writes are buffered, not committed to live cache

Speculation.Commit(n):
  commits only first n accepted tokens' KV to live caches
  discards rest
```

This is a clean implementation of the "buffer writes, commit only accepted
prefix" pattern. Exactly what we need for our KV rollback problem.

## Gemma-4 Shared KV — The Key Innovation

The assistant (draft) model **reads the target model's KV cache directly**.
It has NO separate KV cache.

```
assistant.sharedHistories():
  extracts target's sliding-window KV + full-attention KV
  → assistant attention layers use nn.WithKVHistory(target_kv)
```

The assistant only has Q and O projections — **no K or V projections**.
It reuses the target's precomputed K, V. This eliminates the entire MTP KV
cache management problem.

This works because Gemma-4's MTP heads were specifically trained this way.
Ling-2.6's MTP heads have their own attention with separate K/V projections,
so they need their own KV cache.

## Differences from Our Approach (mtp-clean PR)

| Aspect | Ollama (MLX) | Our approach (llama.cpp) |
|--------|-------------|------------------------|
| Model | Gemma-4 (non-hybrid) | Ling/Bailing (MLA+GLA hybrid) |
| Draft model | Separate loaded model | MTP heads in same GGUF |
| KV cache | Draft reads target's KV directly | MTP has own KV cache |
| Position | Anchored (single-position drafting) | Advances per draft step |
| Verify | Batched (all drafts in one forward) | Serial (one forward per draft) |
| Sampling | Greedy + stochastic rejection | Greedy argmax only |
| Cache rollback | Speculative wrappers (buffer + commit) | Checkpoint restore |
| Recurrent state | No recurrent layers in Gemma-4 | GLA layers need state management |

## What's Relevant to Us

### Applicable

1. **Speculative cache wrapper pattern** — buffer writes, commit only accepted
   prefix. Clean implementation at `cache/cache.go`. This is the right pattern
   for KV rollback without expensive checkpoint/restore.

2. **Batched verification** — all drafts verified in one target forward pass.
   Much faster than serial verification. Our server already does this for the
   mtp-clean approach, but it's worth confirming the batch construction.

3. **Stochastic acceptance** — rejection sampling with probability ratios.
   More correct than greedy argmax. MTPLX does the same thing.

### Not Applicable

1. **Shared KV cache** — Gemma-4 specific. Ling's MTP heads have their own
   attention and need their own KV cache.

2. **Position anchoring** — Gemma-4 specific training. Ling's MTP advances.

3. **No recurrent state handling** — Gemma-4 has no GLA/GDN layers. The
   recurrent state management problem doesn't exist for them.

### Interesting Observation

Gemma-4's shared KV approach is architecturally cleaner — the draft model
doesn't duplicate any attention state. If future MTP models adopt this pattern,
it eliminates the entire MTP KV management problem. But for current models
(Ling, DeepSeek, Qwen) that have separate MTP attention, we still need
separate KV management.
