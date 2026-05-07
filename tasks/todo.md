# MTP port todo

- [x] Read shared README.LJ tail and docs/mtp status notes.
- [x] Compare r2 MTP graph/memory handling with LJ-Ling-2.6-flash-mtp.
- [x] Identify current graph-reserve failure: BAILING_HYBRID_MTP is incorrectly classified as hybrid in r2, so the MTP graph receives a hybrid memory context and crashes when cast to llama_kv_cache_context.
- [x] Patch arch classification and remove temporary graph fprintf diagnostics.
- [x] Rebuild and run focused MTP context-load reproduction.
- [x] Move MTP hidden-state streaming out of llama_context and into common speculative/server flow.
- [x] Preserve full Bailing Hybrid pre-norm hidden states for MTP while keeping logits gathered to requested output rows.
- [x] Default MTP draft head to CPU to avoid mapping a second full Metal copy unless --spec-draft-ngl is explicitly forced.
- [x] Add MTP decode failure guard to avoid cascading sequence-position errors after backend failure.
- [x] Verify server MTP smoke tests and record remaining open issue.
- [x] Review latest PR #22673 mtp-clean head and write docs/mtp lessons note.

## Open

- [ ] Avoid duplicate huge Metal mapping for forced GPU-offloaded MTP head, or make partial MTP tensor loading/offload truly sparse.
- [ ] Replace temporary raw tensor getters with the upstream internal/pre-norm embeddings API when available.
- [ ] Integrate MTP hidden-state stream with prompt-cache restore once the server cache can persist computed pre-norm embeddings.

## Research Backlog

- [ ] Pin down the exact Ling MTP math from the original model implementation or converter metadata.
- [ ] Trace `-mtp` vs `-r2` on the same prompt and log where the outputs first diverge.
- [ ] Re-read the latest `mtp-clean` PR discussion for architecture intent, rollback semantics, and loader behavior.
- [ ] Inspect recurrent rollback / `seq_rm` behavior for hybrid trunk verification and MTP draft acceptance.
- [ ] Map prompt-cache restore requirements for pre-norm hidden rows and MTP verification state.
- [ ] Research backend and memory sharing options for trunk + MTP sibling contexts, especially Metal and HIP.
- [ ] Compare MLX, vLLM, Ollama, and MTPLX MTP implementations for model math and cache handling ideas.
- [ ] Verify whether MTP should be disabled or guarded for multimodal prompts until positions and hidden-state flow are explicit.
- [ ] Confirm how tensor split and layer split should place the MTP sibling without forcing it onto the last GPU.
- [ ] Define how server memory estimation should account for MTP buffers, rollback slots, and future hidden-state cache storage.
