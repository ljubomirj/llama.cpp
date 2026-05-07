# mlx-lm MTP PR #990 — Qwen3.5 Hybrid SSM+Attention

Source: https://github.com/ml-explore/mlx-lm/pull/990
PR: "feat: native MTP speculative decoding for qwen3_5_moe (Qwen3.5-3.6)" by AirRunner
Status: OPEN (as of 2026-05-05)

**This is the most directly relevant reference for our work.** Qwen3.5 is a
hybrid model (GatedDeltaNet SSM + standard attention), same class of problem
as Ling-2.6 (GLA + MLA). The SSM state snapshot/rollback mechanism is exactly
what we need for GLA.

## Performance (M4 Pro)

| Model | Baseline | MTP | Speedup | Acceptance |
|-------|----------|-----|---------|------------|
| Qwen3.5-27B 4-bit (dense) | 15.3 t/s | 24.0 t/s | 1.57x | 46% |
| Qwen3.5-35B-A3B 4-bit (MoE) | 85.3 t/s | 87.9 t/s | 1.04x | 46% |
| Qwen3.5-122B-A10B 5-bit (MoE) | 43.0 t/s | 46.7 t/s | 1.09x | — |

MoE speedup is marginal because baseline is already fast (3B active params).
MTP benefit scales with baseline decode time. For our Ling-2.6 (18 t/s, 7.4B
active), we should see meaningful speedup if MTP works.

## SSM State Snapshot/Rollback — The Key Innovation

### The Problem

GatedDeltaNet (and GLA) layers have stateful recurrent state. You can't
"unsee" a token. When a draft is rejected, you must restore the state to
*before* the draft was processed.

### The Solution: `n_confirmed` + Split Processing

When verifying a draft, the backbone processes `[confirmed_token, draft_token]`
with `n_confirmed=1`. This tells each SSM layer to split processing:

```
GatedDeltaNet.__call__(inputs, n_confirmed=1):
  1. Process confirmed_token → new (conv_state, ssm_state)
  2. SNAPSHOT: cache.rollback_state = (conv_state, ssm_state)
  3. Process draft_token from that state → final (conv_state, ssm_state)
```

On **rejection**: restore `(conv_state, ssm_state)` from snapshot.
On **acceptance**: discard snapshot, state is already correct.

### Implementation (cache.py)

```python
class ArraysCache(_BaseCache):
    rollback_state: Optional[tuple] = None
    # Holds: (conv_state_snapshot, ssm_state_snapshot)
```

### Rollback Function (generate.py)

```python
def _rollback_draft():
    for c in model_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            conv_snap, ssm_snap = c.rollback_state
            c[0] = conv_snap    # restore conv_state
            c[1] = ssm_snap     # restore ssm_state
            c.rollback_state = None
        elif c.is_trimmable():
            c.trim(1)            # attention layers: trim KV cache
```

Polymorphic: SSM layers restore from snapshot, attention layers just trim KV.

### Applicability to Ling-2.6 GLA

GLA state is even simpler than GatedDeltaNet:
- **GDN state**: `(conv_state, ssm_state)` — two separate arrays
- **GLA state**: single `[B, H, D, D]` array per layer

For GLA, the snapshot is just one memcpy per layer (2MB × 28 layers = 56MB).
The split-processing pattern is identical:
1. Process confirmed token through GLA → new state
2. Snapshot the GLA state
3. Process draft token → final state
4. On rejection: restore from snapshot

In ggml-metal, this would be a `memcpy` from the Metal GLA state buffer to a
snapshot buffer. ~56MB copy is fast on unified memory (Apple Silicon).

## MTP Head Architecture

```
Input:  h_t (pre-norm hidden) + embed(t+1)
        |                            |
  pre_fc_norm_hidden      pre_fc_norm_embedding
        |                            |
        +-------- concat -----------+
                     |
                fc (2H → H)    ← KEPT IN FULL PRECISION
                     |
        MTPDecoderLayer × N      ← Standard attention ONLY (no SSM)
                     |
                 RMSNorm
                     |
                lm_head           ← SHARED with backbone
```

### Key Details

1. **Attention-only MTP head**: No GatedDeltaNet/SSM. Standard causal attention.
   This avoids managing a second set of SSM states in the MTP head.
   Ling-2.6 does the same — MTP head uses MLA (attention), not GLA.

2. **Fusion projection in full precision**: `mtp.fc` (2H → H) is not quantized.
   Small layer, minimal cost.

3. **`mtp_num_hidden_layers: 1`**: Only 1 MTP head. Same as Ling-2.6.

4. **Pre-norm hidden state**: The backbone returns hidden state *before* the
   final RMSNorm, not after. In llama.cpp, this is `t_h_pre_norm`.

## Draft/Verify Flow

```
Each cycle:
  if no pending draft:
    1. Backbone forward(y) → logits + pre_norm_hidden
    2. Sample main_tok from logits
    3. MTP: mtp_forward(pre_norm_hidden, main_tok) → draft_tok
    4. Yield main_tok
    5. y = [main_tok]

  else (draft pending):
    1. Concatenate: y_with_draft = [y, draft_tok]
    2. Backbone forward(y_with_draft, n_confirmed=1)
       → SSM layers snapshot state after confirmed token
    3. Acceptance check:
       - Greedy: verify_pred == draft_tok?
       - Stochastic: min(1, p_target/p_draft) with residual correction
    4. If ACCEPT:
       - Clear rollback snapshots
       - Yield draft_tok + bonus_tok
       - MTP: mtp_forward(hidden_at_draft, bonus_tok) → next draft
    5. If REJECT:
       - Rollback SSM states, trim KV caches
       - Sample verify_tok (or from residual distribution)
       - MTP: mtp_forward(hidden_at_confirmed, verify_tok) → next draft
```

### Probabilistic Acceptance

For non-greedy sampling, implements Leviathan-Chen rejection sampling:
```python
log_accept = verify_lp[draft_tok] - draft_lp[draft_tok]
accept = log_accept >= 0 or random() < exp(log_accept)
# On rejection: sample from max(p_target - p_draft, 0) / Z
```

Guarantees output distribution exactly equals target distribution.

## Key Patterns for llama.cpp Adaptation

### Pattern 1: Split-Process-with-Snapshot (GLA)

Modify GLA graph building to support `n_confirmed`:
1. Process confirmed token through GLA → new state
2. Copy GLA state to snapshot buffer (memcpy, ~2MB/layer)
3. Process draft token → final state
4. On rejection: restore from snapshot

### Pattern 2: Per-Cache-Type Rollback

```
On draft rejection:
  For each layer:
    if GLA layer:     restore from snapshot
    if MLA/KV layer:  trim(1) to remove draft entry
```

### Pattern 3: MTP Head Uses Attention Only

Both Qwen3.5 and Ling-2.6 MTP heads use standard attention, not recurrent
layers. This is deliberate — avoids double state management in the draft model.

### Pattern 4: Pre-Norm Hidden State

Backbone returns hidden state before final RMSNorm. In llama.cpp this is
`t_h_pre_norm`, which our hook already intercepts.

## Key Files

| File | Purpose |
|------|---------|
| `mlx_lm/models/qwen3_5.py` | Model with MTP head, GDN n_confirmed handling |
| `mlx_lm/generate.py` | MTP generation loop (mtp_generate_step) |
| `mlx_lm/models/cache.py` | ArraysCache with rollback_state slot |
