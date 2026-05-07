# PR #22673 Comments — Key Takeaways

Source: https://github.com/ggml-org/llama.cpp/pull/22673
PR: "llama + spec: MTP Support" by am17an (OPEN)

## Critical Architectural Feedback

### ggerganov (project lead) — Review on `src/llama-context.h:248`

> This logic has to be extracted into the server/speculative contexts. One of
> the main reasons is that just the token positions are not enough to be able to
> do this correctly. For example, for multi-modal use cases, this implementation
> does not take into account that multiple tokens have the same position and
> there is no way to resolve this inside `llama_context`.
>
> Also, restoring prompts in a slot from the prompt cache does not work because
> we forget the computed embeddings for the prompt.
>
> Hence, we have to actually extract the embeddings from the context (in a
> similar way as we extract the logits) and manage them along side the prompts
> (store them, prefix cache them, etc.). Quite a lot of work is needed, but I
> think that's the proper way to implement support for these methods.
>
> For MTP, we can already use the existing embeddings API because MTP only
> needs the output embeddings and we already have the API for that
> `llama_get_embeddings_*` (edit: on second look, we actually need the
> embeddings before the output norm, so need API for that as well). But for
> Eagle3 this is not enough. For that I am preparing new embeddings API
> (#22728) that will be used to extract the internal layer embeddings.

**Implication for us**: GG wants the hook/embedding extraction moved out of
`llama_context` into the server/speculative layer. The new embeddings API
(#22728) will provide `llama_get_embeddings_*` for pre-norm hidden states.
This would replace our current `t_h_pre_norm` hook approach. The current hook
in `llama_context` is a temporary solution that won't survive upstream review.

### ngxson — on `src/llama-memory-recurrent.h:74`

> not 100% sure but maybe the naming with `_seq` is a bit confusing. I imagine
> that we want to keep a buffer ring style of recurrent-state(s), similar to
> SWA in KV cache, right? if that's the case, probably better call it
> `n_rs_window`

Related to PR #22400 (recurrent state memory). The naming convention for
recurrent state (SSM/GDN) is still being discussed.

### ngxson — on `tools/server/server-context.cpp:841`

> if you look at https://github.com/ggml-org/llama.cpp/pull/18886, the better
> way is to move `llama_graph_type` to the public API, then load the context
> with the appropriate graph type

am17an agreed: "Yes that seems like the correct way to do this if we want to
support MTP in a generic way"

## Metal/GDN Issues and Fixes

### am17an on Metal gibberish (multiple comments)

When asked about gibberish on M5 Pro / Metal:
> you can ask an LLM to implement the `keep_intermediates=true` GDN path for
> metal (make sure it passes `test-backend-ops` tests), it should probably work
> then. Same for any non-CUDA backend users.

This confirms the GDN state issue we identified — Metal needs the
`keep_intermediates=true` path (which is the non-fused, per-step GDN that
stores intermediate states for checkpoint/rollback).

Later: "Metal should work now as well" — am17an fixed Metal by implementing
the GDN intermediate state path for Metal.

### Metal Results (after fix)

PkmX (M1 Ultra):
> Qwen3.6 27B Q8_0 went from 17 t/s to 25 t/s with `--spec-type mtp --spec-draft-n-max 3`

This is a ~47% speedup on Metal. The GDN fix works for Qwen3.5/3.6 (which
use GDN/SSM layers), but **Ling-2.6-flash uses GLA, not GDN**, so this fix
doesn't apply to our model.

## Performance Reports (CUDA/Vulkan)

| Hardware | Model | No MTP | With MTP | n_max | Accept Rate |
|----------|-------|--------|----------|-------|-------------|
| RTX 5090 | Qwen3.6-27B Q4_K_M | — | good | 3 | — |
| RTX 3090+3060 | Qwen3.6-27B Q6_K | 33 t/s | 39 t/s | 3 | 0.53 |
| RTX A6000 | Qwen3.6-27B Q8_0 | 21 t/s | 50 t/s | 4 | — |
| R9 9700 (Vulkan) | Qwen3.6-27B Q8_0 | 32 t/s | 55-60 t/s | 2 | — |
| 3x RTX3060 (tensor) | Qwen3.6-27B Q4_K_M | 18.5 t/s | 24 t/s | 3 | — |
| M1 Ultra (Metal) | Qwen3.6-27B Q8_0 | 17 t/s | 25 t/s | 3 | — |
| 3060 Laptop + CPU | Qwen3.6-35BA3B | 22.9 t/s | 29.4 t/s | 2 | 0.815 |
| R9 6900HX (ROCm iGPU) | Qwen3.5-4B Q8_0 | 6.6 t/s | 11.6 t/s | 3 | 0.659 |
| 5070 Ti + 3080 | Qwen3.6-27B Q4_K_M | 39 t/s | 58 t/s | 3 | — |

**Key insight**: Acceptance rates are 50-88% on CUDA/Vulkan for Qwen3.6
(non-hybrid models). Our 14% on Ling-2.6-flash is far below this, confirming
that the hybrid GLA/MLA architecture has additional issues.

## Gemma 4 MTP — Shared KV Cache

coder543 noted:
> From the Gemma 4 blog: "The draft models seamlessly utilize the target
> model's activations and share its KV cache, meaning they don't have to waste
> time recalculating context the larger model has already computed."

Gemma 4 MTP shares the KV cache with the trunk (no separate MTP KV). This is
different from Qwen3.6/Ling-2.6 where the MTP has its own KV. This will need
different handling in llama.cpp.

## Known Issues Reported

1. **Vulkan was slow**: fell back to CPU silently when VRAM was tight
2. **Token generation freezes**: some users reported generation stopping
   mid-stream (fixed in later commits)
3. **Gibberish on Metal (pre-fix)**: GDN intermediate states not available
4. **Tensor parallelism crashes**: `-sm tensor` doesn't work with MTP
5. **Prefill performance**: cturan reported degraded prefill with MTP enabled
6. **NVFP4 quant GGUF**: missing MTP layers in some community GGUFs
   causes assertion failure

## Relevance to Our Work

1. **GG's feedback** means our hook approach will need reworking to use the
   new embeddings API (#22728) instead of hooking into `llama_context`

2. **Metal GDN fix** works for Qwen models (GDN layers) but not for
   Ling-2.6-flash (GLA layers). We still need GLA Metal kernel + state
   capture/replay.

3. **The position mismatch bug (#2)** in our docs is likely specific to our
   bailing_hybrid MTP implementation — Qwen3.6 users report 50-88% acceptance,
   so the generic draft function works for them.

4. **Gemma 4 shared KV** approach is architecturally different and may inform
   future MTP implementations.
