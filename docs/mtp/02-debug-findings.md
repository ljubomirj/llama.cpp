# MTP Debug Findings

## Bug #1: Hook Position Gap (FIXED)

### Symptoms

`llama_decode(ctx_mtp)` returns rc=-1 starting at position 27+ during
decode. The MTP head never gets valid hidden states after the first
verification batch.

### Root Cause

The streaming hook (`handle_mtp_for_ubatch`) returns early when
`pos_start <= pos_max_mtp`, meaning the MTP's KV already covers the
batch positions. But it was NOT saving the trunk's hidden state as
`pending_h` before returning.

This caused a stale `pending_pos` from prefill to persist through
verification batches. The next hook call would see misaligned
`pending_pos` and couldn't feed the MTP consecutively.

### Trace (before fix)

```
hook: pos_start=0,  n_tokens=21, pos_max_mtp=-1, pending_pos=-1   ← prefill batch 1
hook: pos_start=21, n_tokens=4,  pos_max_mtp=20, pending_pos=20   ← prefill batch 2
hook: pos_start=25, n_tokens=2,  pos_max_mtp=25, pending_pos=24   ← returns early, pending NOT updated
hook: pos_start=27, n_tokens=2,  pos_max_mtp=26, pending_pos=24   ← pending stale! gap at pos 27
→ rc=-1: positions not consecutive (27 ≠ 26+1)
```

### Fix

When returning early, also read the last hidden state row from the trunk
and save it as `pending_h` with `pending_pos = pos_start + n_rows - 1`.

```cpp
if (pos_start <= pos_max_mtp) {
    synchronize();
    const size_t row_bytes = (size_t) n_embd * sizeof(float);
    ggml_backend_tensor_get(t, mtp.pending_h.data(),
        (size_t) (n_rows - 1) * row_bytes, row_bytes);
    mtp.pending_pos = pos_start + n_rows - 1;
    return;
}
```

### Trace (after fix)

```
hook: pos_start=0,  n_tokens=21, pos_max_mtp=-1, pending_pos=-1
hook: pos_start=21, n_tokens=4,  pos_max_mtp=20, pending_pos=20
hook: pos_start=25, n_tokens=2,  pos_max_mtp=25, pending_pos=24   ← early return, saves pending at 26
hook: pos_start=27, n_tokens=2,  pos_max_mtp=26, pending_pos=26   ← pending aligns! consecutive ✓
hook: pos_start=28, n_tokens=2,  pos_max_mtp=28, pending_pos=28   ← all subsequent calls OK
...
```

**Status**: Fixed. No more rc=-1 decode failures.

---

## Bug #2: MTP Draft Position Mismatch vs Verification (UNFIXED)

### Symptoms

Even with Bug #1 fixed, draft acceptance rate stays at ~14% (1/7).
First draft is correct (`capital`), but subsequent drafts are wrong
(`import` instead of `of`, then `.LoggerFactory` garbage loop).

### Root Cause

A **position offset** between the MTP's draft position and the server's
verification position. After each acceptance cycle, the MTP falls one
position behind the trunk.

#### Detailed Trace

Prompt: "What is the capital of France?" (25 tokens, positions 0-24)

**Cycle 1** (works):
```
Trunk prefill → samples "The" at position 25
Draft fn: pos_max_mtp=24, starts at pos=25, predicts " capital" for pos 26
Server: batch=[sampled@25, draft@26], verification checks trunk@25 → predicts for 26
        MTP predicted for 26, trunk verifies for 26 → MATCH ✓
accept(1): MTP KV has 1-25, pos_next=27
```

**Cycle 2** (fails):
```
Draft fn: pos_max_mtp=25, starts at pos=26, predicts for pos 27
Server: batch=[sampled@27, draft@28], verification checks trunk@27 → predicts for 28
        MTP predicted for 27, trunk verifies for 28 → MISMATCH ✗
```

The offset: after accepting 1 draft, the server advances `pos_next` by
2 (accepted draft + bonus sample), but the MTP only advances by 1
(because the hook returned early and didn't fill the verification batch
positions into the MTP KV).

#### Why it happens

1. Hook returns early for verification batch → MTP KV doesn't get positions 25-26
2. Draft function starts at `pos_max_mtp + 1 = 26` → predicts for 27
3. Server builds batch at `pos_next = 27` → trunk verifies for 28
4. Off by 1: MTP predicts for 27, trunk expects match for 28

#### The `src_row` and `id_last` are also wrong

For the second draft call:
- `src_row = last_n_accepted = 1` → reads row 1 of verification batch (position 26's hidden state)
  - Should read position 25's hidden state (the previous position) = row 0
- `id_last = slot.sampled = trunk_prediction_for_27` → token at position 27
  - Should be the accepted token at position 26 (`capital` = 7706)

The MTP head receives the wrong hidden state AND the wrong token embedding.

### Potential Fixes (not yet implemented)

**Option A**: Remove stale MTP KV entries on hook early return, then re-feed
from trunk. This keeps MTP KV in sync with trunk positions.

```cpp
if (pos_start <= pos_max_mtp) {
    // Remove draft function's stale entries, re-feed from trunk
    llama_memory_seq_rm(mtp.memory, 0, pos_start, -1);
    // Now pos_max_mtp < pos_start, continue to normal processing
}
```

**Option B**: Change draft function to start from trunk's position, not
MTP's pos_max. Fill the gap with multiple MTP decode calls.

**Option C**: Have the hook always process verification batches (don't
return early) by removing and re-adding MTP KV entries.

**Option D**: Make the draft function track `pos_next` from the server
instead of `pos_max_mtp`.

---

## Bug #3: Metal GDN State Corruption on Draft Rejection

### Symptoms (Qwen3.5 on Metal)

- MTP head predictions are CORRECT (e.g., `capital`, `France`)
- But trunk's verification step on Metal produces wrong logits
- Low acceptance rate (6%) vs CPU (100%)
- After rejection, checkpoint restore corrupts GDN/SSM state

### Root Cause

The GDN (Gated Delta Network) / SSM layers used in hybrid models maintain
per-position recurrent state. When speculative decoding rejects drafts and
rolls back via checkpoint restore, this state must also be rolled back.
On Metal, there are **no custom kernels** for GDN state capture/replay.

The CPU fallback path handles this correctly because the recurrent state
lives in host memory and can be checkpointed/restored via the standard
`llama_state_seq_*` functions. On Metal, the state lives in Metal buffers
that the generic checkpoint mechanism doesn't know about.

### Ling-2.6-flash Impact

Ling-2.6-flash uses GLA (Gated Linear Attention) for local layers, which
has the same class of problem. GLA has NO Metal backend at all in ggml
(`ggml_gated_linear_attn` falls back to CPU). Even if MTP head logits were
correct, the trunk's GLA layers would corrupt state on draft rejection.

### Required Fix

Custom Metal kernels for GLA/GDN state:
1. **Capture kernel**: snapshot per-position recurrent state after each decode
2. **Replay kernel**: restore state from snapshot on draft rejection
3. **Tape kernel**: maintain incremental state tape for rollback

Reference implementation: MTPLX (`~/LJ-asi-mlx/MTPLX/mtplx/gdn_capture.py`)
has ~800 lines of custom MSL kernels for this exact purpose.

---

## Bug #4: GLA Local Layers — No Metal Backend

### Symptoms

`ggml_gated_linear_attn` has no Metal implementation. All GLA (local) layers
run on CPU while MLA (global) and MoE run on Metal GPU.

### Impact

- Significant performance penalty: local layers (30 out of 32) hit CPU fallback
- Data transfer between GPU and CPU for every layer transition
- MTP head (layer 32, MLA-based) runs on GPU but must receive data from
  the last GLA layer via CPU

### Potential Fix

Port the GLA Metal kernel from `bailing_hybrid.py`
(`_make_recurrent_gla_kernel`) to ggml-metal. This is separate from the
GDN state capture/replay kernels needed for speculative decoding.
