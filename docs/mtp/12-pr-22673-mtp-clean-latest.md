# PR #22673 mtp-clean latest lessons (2026-05-07)

Sources checked:

- Upstream draft PR: <https://github.com/ggml-org/llama.cpp/pull/22673>
- Local worktree: `~/llama.cpp/worktrees/mtp-clean`
- Latest fetched PR head: `refs/remotes/am17an/mtp-clean` at `5d5f1b46e fix: use rs for only MTP`

The local `mtp-clean` worktree was dirty when inspected (`AGENTS.md` deleted and debug changes in
`common/speculative.cpp`), so the conclusions below are based on the fetched PR head rather than
the dirty local branch.

## Short version

The PR has moved a long way since the older notes in `05-pr-22673-comments.md`: it now carries Qwen
3.5 dense and MoE MTP support, partial tensor loading, GDN backend fixes, server lifetime fixes, and
more practical performance reports.

It is still mostly the hook-based design: `common_speculative_state_mtp` creates a second
`llama_context` and calls `llama_set_mtp(ctx_tgt, ctx_mtp)`, while `llama_context` streams target
hidden states into the MTP context during decode. That is the working PR design, not GG's preferred
long-term factoring where speculative decoding owns the MTP state, consumes internal/pre-norm
embeddings explicitly, and prompt cache can persist those embeddings.

For the Ling r2 port, do not copy the hook back in. The useful parts are the model/loader details,
rollback-slot fixes, backend lessons, and failure reports.

## Code lessons

1. MTP as a sibling model is a valid loader shape.

   The PR adds separate sibling architectures such as `LLM_ARCH_QWEN35_MTP` and
   `LLM_ARCH_QWEN35MOE_MTP`. The loader accepts partial tensor loading via
   `llama_model_loader::done_getting_tensors(bool partial)` and logs that unused tensors belong to a
   sibling model in the same GGUF. This matches the direction already used for Ling/Bailing Hybrid:
   load only the appended MTP head, not a second full trunk.

2. The MTP auxiliary model should use a plain KV cache.

   Qwen MTP sets `hparams.kv_only_nextn = true`, disables recurrent layers for the MTP sibling, and
   loads only the appended prediction layers. This reinforces the r2 fix for Ling: the MTP head must
   not be classified as a hybrid/recurrent trunk, even when the parent architecture is hybrid.

3. Hidden-state rows must be complete.

   The PR still gathers output rows near the end of the trunk graph before exposing
   `res->t_h_pre_norm` for MTP. That can leave only logits-requested rows available. It works for
   narrow decode cases, but it is a poor fit for prompt prefill, cache restore, and multimodal
   batches. The r2 approach of preserving a full pre-norm hidden stream and gathering logits
   separately is closer to the design GG has been asking for.

4. The MTP graph consumes token IDs and hidden embeddings together.

   Qwen MTP uses both normal token input and an explicit hidden-embedding input named like
   `mtp_h_input`. The head fuses the current token embedding, the trunk hidden state, attention, FFN,
   and final next-token head. Ling should keep the same contract: the speculative layer sends the
   draft head the sampled token plus the trunk pre-norm hidden for that row.

5. Rollback/checkpoint slots belong to the target context only.

   The latest PR commit changes `n_rs_seq` handling so recurrent sequence-removal state is allocated
   only when the speculative type is MTP, and regular draft/MTP auxiliary contexts get `n_rs_seq = 0`.
   For Ling this matters because rollback support is needed to verify accepted/rejected speculative
   tokens against the target hybrid state, but the MTP head itself should not allocate extra hybrid
   rollback memory.

6. Lifetime order matters while hooks still exist.

   The PR has a server fix for a double free: destroy/reset speculative state before destroying the
   target model/context. That is only essential for the hook design, but the broader lesson remains:
   any auxiliary MTP context that references target context state, backends, buffers, or cache data
   must be released before the owning target context is torn down.

## Backend and runtime lessons

1. Duplicating backend instances is a real bug class.

   A ROCm user reported that creating a separate MTP context also created a second HIP backend
   instance, leaving GPUs busy even after generation. Their local workaround reused the target
   context backends for the auxiliary MTP context and required freeing speculative/MTP contexts before
   the target context.

   This lines up with our Metal finding: a second MTP context can map/allocate too much backend memory
   from the same GGUF. The r2 default of keeping the MTP head on CPU unless explicitly offloaded is a
   reasonable guard, but the real fix is backend/buffer sharing or genuinely sparse partial offload.

2. Tensor split is not solved.

   PR users report that MTP does not respect tensor split reliably and may place the whole MTP sibling
   on the last GPU. This can OOM even when the trunk fits. Any Ling MTP GPU path should be tested with
   layer split and tensor split separately before considering multi-GPU support complete.

3. Memory estimation must include MTP.

   Users hit Metal OOM because fit estimates did not include the extra MTP context/buffers. The server
   should either account for MTP memory in planning or refuse unsafe automatic offload. Hidden prompt
   cache storage will add another memory term once implemented.

4. Multimodal prompts are an open problem.

   Several PR comments report crashes or bad behavior with image prompts. The maintainer response was
   that this likely needs the architecture refactor and may not be solved in PR #22673. It is also not
   obvious that every MTP head was trained to draft over vision-token positions. For Ling r2, MTP
   should be disabled or explicitly guarded for multimodal until there is a clear hidden-state and
   position mapping story.

5. Draft depth is empirical.

   Performance reports show the best `--spec-draft-n-max` is often 2-4 for Qwen, and deeper is not
   automatically better because acceptance falls. Ling currently has one extra head for `t+2`, so the
   initial r2 target should stay at depth 1 and optimize correctness/acceptance before adding generic
   depth machinery.

## Open questions for the Ling r2 port

- What exact upstream internal/pre-norm embedding API will replace the temporary raw graph result
  plumbing? PR #22728 is related, but the final shape is still unsettled.
- How should prompt cache store and restore the pre-norm hidden rows needed by MTP?
- Can the MTP sibling share the trunk backend and mapped GGUF buffers without resurrecting the
  `llama_context` hook?
- Should MTP be rejected for multimodal requests until model-specific training and position semantics
  are known?
- How should server memory planning include MTP compute buffers, KV, recurrent rollback slots, and
  future hidden-state cache storage?
- Should separate MTP GGUF files be supported in addition to same-GGUF sibling heads, especially for
  models where the MTP head is optional or distributed separately?

## Concrete r2 guidance

- Keep the GG-style direction: MTP orchestration should live in speculative/common/server code, not as
  a hidden callback inside `llama_context`.
- Keep Ling `BAILING_HYBRID_MTP` on the plain KV path with `kv_only_nextn`; do not let parent hybrid
  memory classification leak into the MTP sibling.
- Preserve full pre-norm hidden rows from the target graph before logits-row gathering.
- Keep `--parallel 1` as the supported mode until prompt cache, hidden rows, and sequence bookkeeping
  are all explicit.
- Treat GPU offload for the MTP sibling as experimental until duplicate Metal/HIP backend allocation
  and tensor split behavior are fixed.
- Add clear runtime guards for MTP plus multimodal and for unsafe offload choices rather than allowing
  late crashes.
