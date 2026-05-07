# MTP (Multi-Token Prediction) Architecture

## What MTP Does

MTP is a speculative decoding technique where the model itself generates draft
tokens, eliminating the need for a separate smaller draft model. The model's
trunk produces a hidden state; an MTP head (a lightweight transformer layer)
takes that hidden state and predicts the next token. The trunk then verifies
the draft in a single batched forward pass.

For a model with one MTP head (`num_nextn_predict_layers=1`), each generation
step works like this:

```
1. Trunk generates token at position t
2. MTP head takes h_t (trunk's hidden state) and predicts token at t+1
3. Trunk decodes batch [t, t+1] — verifies draft and gets bonus token
4. If draft accepted → 2 tokens generated in one step
   If rejected    → trunk's verification token is the real output
5. Repeat
```

## Model-Level Architecture (BailingMoeV2 / Ling-2.6-flash)

Ling-2.6-flash is a 104B MoE model (7.4B active params) with hybrid attention:

- **Global layers** (every 8th: layers 7, 15, 23, 31): MLA (Multi-Latent Attention)
- **Local layers** (all others): GLA (Gated Linear Attention)
- **MoE**: 256 experts, 8 active per token, group expert selection
- **MTP**: One MTP head (layer 32, index = `n_layer - nextn_predict_layers = 32`)
- **Head dim**: 128, RoPE dim: 64

### MTP Head Structure

The MTP head is a single transformer layer that shares the trunk's embedding
and LM head. Its computation graph (from `bailing-hybrid.cpp:651-890`):

```
Input: token_t (from trunk's sampled token)
       h_t     (from trunk's pre-normalization hidden state)

1. tok_embd = Embed(token_t)           -- token embedding lookup
2. h_norm   = RMSNorm(h_t)             -- normalize hidden state
3. e_norm   = RMSNorm(tok_embd)        -- normalize embedding
4. concat   = [e_norm, h_norm]          -- concatenate along dim 0 (2*n_embd)
5. cur      = eh_proj @ concat          -- project back to n_embd

6. cur      = RMSNorm(cur)              -- attention norm
7. Q,K,V    = MLA projections           -- DeepSeek2-style MLA attention
8. attn_out = MLA_attention(Q, K, V)    -- with KV cache
9. cur      = attn_out + residual       -- residual connection (skip from step 5)

10. cur     = RMSNorm(cur)              -- FFN norm
11. cur     = MoE_FF(cur)              -- MoE (same as trunk layer 31)
12. cur     = cur + residual            -- residual connection (skip from step 9)

13. cur     = RMSNorm(cur)              -- shared head norm (or output_norm)
14. logits  = LM_head @ cur            -- shared LM head (or model.output)
```

### MTP-Specific Tensors (loaded from GGUF)

The MTP head loads tensors from layer index `n_layer - nextn_predict_layers`:

```
nextn.embed_tokens   -- optional separate token embedding (usually NULL, uses trunk's)
nextn.eh_proj        -- projection from [e_norm; h_norm] back to n_embd (shape: n_embd x 2*n_embd)
nextn.enorm          -- RMSNorm weights for token embedding
nextn.hnorm          -- RMSNorm weights for hidden state
nextn.shared_head_norm -- optional separate head norm (or reuses model.output_norm)
nextn.shared_head_head -- optional separate LM head (or reuses model.output)
```

Plus all standard layer tensors (attn_norm, wq_a, wq_b, wkv_a_mqa, wk_b,
wv_b, attn_kv_a_norm, wo, ffn_norm, ffn_gate_inp, ffn_*_exps, ffn_*_shexp).

For Ling-2.6-flash: 22 MTP tensors out of 540 total in the GGUF.

## llama.cpp Implementation

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| MTP model definition | `src/models/bailing-hybrid.cpp:651-890` | `llama_model_bailing_hybrid_mtp::graph` |
| Streaming hook | `src/llama-context.cpp:3313-3391` | `handle_mtp_for_ubatch` |
| Draft function | `common/speculative.cpp:604-767` | `common_speculative_state_mtp` |
| Batch allocator | `src/llama-batch.cpp` | Consecutive position check |
| Server integration | `tools/server/server-context.cpp:3005-3158` | Verification + accept/reject |
| MTP registration | `src/llama-context.cpp:3282-3311` | `llama_set_mtp` |

### Two-Context Architecture

MTP uses **two separate llama_context** objects:

```
ctx_tgt (trunk context):
  - Main model (bailing_hybrid)
  - Its own KV cache, samplers, memory
  - Runs the full model graph
  - Hook intercepts t_h_pre_norm tensor after each ubatch

ctx_mtp (MTP head context):
  - Lightweight model (bailing_hybrid_mtp) — single layer
  - Its own KV cache (MLA compressed KV)
  - Runs only the MTP head graph
  - Takes hidden states from trunk as input embeddings
```

### Data Flow

```
                   TRUNK (ctx_tgt)
                   ┌──────────────────────────────────────┐
  prompt tokens ──►│  embed → [layers 0..31] → t_h_pre_norm │──► sample → token_t
                   │         ▲                              │
                   │         │ hook intercepts this tensor   │
                   └─────────┼──────────────────────────────┘
                             │
                             │  hidden state (n_embd floats per row)
                             ▼
                   MTP HEAD (ctx_mtp)
                   ┌──────────────────────────────────────┐
  token_t ────────►│  embed(token_t) + h_t → eh_proj →    │
                   │  MLA_attn → MoE_FF → LM_head → logits│──► sample → draft_{t+1}
                   │         ▲                              │
                   │         │ MTP's own KV cache           │
                   └──────────────────────────────────────┘
```

### The Streaming Hook

`handle_mtp_for_ubatch` is called after every trunk ubatch decode. It reads
`res->t_h_pre_norm` (the trunk's hidden states before the final norm) and
feeds them to the MTP head's context.

**During prefill:**
- Each ubatch of N tokens produces N rows of hidden states
- The hook feeds rows 0..N-2 to the MTP at positions 1..N-1
  (shifted by 1: hidden state from position k predicts token at position k+1)
- The last row (position N-1) is saved as `pending_h` for the next ubatch
- The next ubatch picks up the pending: feeds `pending_h` at its first position,
  then rows 0..N-2 at subsequent positions

**During decode (verification batches):**
- The trunk decodes a batch of 2 tokens: [sampled, draft]
- The hook checks if `pos_start <= pos_max_mtp` (MTP already has these positions)
- If so, it returns early but saves the last hidden state as pending
- The draft function handles MTP state for decode tokens

### The Draft Function

`common_speculative_state_mtp::draft()` in `common/speculative.cpp:668-741`:

```python
# Pseudocode for the draft loop:
pos = mtp_kv_pos_max + 1
cond_tok = id_last  # trunk's last sampled token

for k in 0..n_max-1:
    if k == 0:
        h = trunk.t_h_pre_norm[row = last_n_accepted]  # from trunk's decode output
    else:
        h = mtp.t_mtp_out[last_row]  # from MTP's own previous output (AR loop)

    mtp_batch = {token: cond_tok, pos: pos, embd: h}
    mtp_logits = llama_decode(ctx_mtp, mtp_batch)
    draft_tok = sample(mtp_logits)
    draft_tokens.append(draft_tok)
    cond_tok = draft_tok
    pos += 1
```

### Verification and Accept/Reject

The server's speculative flow (`server-context.cpp:3005-3158`):

1. Build batch: `[sampled_token, draft_token]`
2. Trunk decodes both → `common_sampler_sample_and_accept_n`
3. Compare trunk's prediction at each position with the draft
4. Accept matching prefix, reject first mismatch
5. If partial acceptance + FULL checkpoint mode: restore trunk checkpoint
6. Call `common_speculative_accept(n_accepted)` to update MTP KV

### MTP KV Cache Management

The MTP head has its own KV cache (MLA compressed). Three operations manage it:

1. **Hook adds entries** during prefill (streaming, one ubatch at a time)
2. **Draft function adds entries** during decode (one position per draft step)
3. **Accept removes entries** when drafts are rejected:
   - `n_to_drop = max(0, last_n_drafted - n_accepted - 1)`
   - Removes KV entries from `pos_max - n_to_drop + 1` to end

## Command-Line Usage

```bash
./llama-server \
  -m model.gguf \
  -ngl 99 -fa on \
  --ctx-size 4096 \
  --spec-type mtp \
  --spec-draft-n-max 1 \
  --port 8080
```

Key flags:
- `--spec-type mtp` — enables MTP speculative decoding
- `--spec-draft-n-max N` — max drafts per step (typically 1 for single MTP head)
- `--spec-draft-ngl N` — GPU layers for MTP head (0 = CPU only)

## Other MTP Implementations in llama.cpp

| Model | File | Notes |
|-------|------|-------|
| Qwen3.5 | `src/models/qwen35_mtp.cpp` | Standard MHA attention |
| Qwen3.5-MoE | `src/models/qwen35moe_mtp.cpp` | MoE variant |
| Bailing Hybrid | `src/models/bailing-hybrid.cpp` | MLA attention + MoE (our model) |

All share the same `common_speculative_state_mtp` draft function and hook mechanism.
