# MTP Documentation for Ling-2.6-flash in llama.cpp

Internal reference for the MTP (Multi-Token Prediction) integration effort
for the Ling-2.6-flash model in llama.cpp.

## Documents

| Doc | Description |
|-----|-------------|
| [01-architecture.md](01-architecture.md) | MTP model architecture, llama.cpp implementation, data flow, key source files |
| [02-debug-findings.md](02-debug-findings.md) | Bugs found and fixed/unfixed: hook position gap, draft mismatch, Metal state, GLA |
| [03-status-and-reference.md](03-status-and-reference.md) | Current state, what works/blocks, build commands, file locations, session history |
| [04-metal-kernels.md](04-metal-kernels.md) | Metal kernel requirements, MTPLX reference, **detailed bailing_hybrid.py architecture** |
| [05-pr-22673-comments.md](05-pr-22673-comments.md) | Key comments from upstream PR #22673: GG's architecture feedback, Metal fix, perf reports |
| [06-mtplx-reference.md](06-mtplx-reference.md) | MTPLX project analysis: GDN tape/replay, Leviathan-Chen acceptance, adaptive depth, what applies to us |
| [07-ollama-mtp.md](07-ollama-mtp.md) | Ollama v0.23.1-rc0 MTP: Gemma-4 shared KV, speculative cache wrappers, batched verify |
| [08-vllm-gemma4-mtp.md](08-vllm-gemma4-mtp.md) | vLLM PR #41745: Gemma-4 Q-only attention, centroids masking, 319% speedup on 31B, **Ling MTP depth = 1** |
| [09-mlx-lm-mtp.md](09-mlx-lm-mtp.md) | mlx-lm PR #990: Qwen3.5 native MTP, SSM snapshot/rollback, n_confirmed pattern |
| [10-mlx-vlm-gemma4-mtp.md](10-mlx-vlm-gemma4-mtp.md) | mlx-vlm PR #1112: Gemma 4 MTP drafter, shared KV, centroids masking, spec cache wrapper |
| [11-cudnn-frontend-kernels.md](11-cudnn-frontend-kernels.md) | NVIDIA cuDNN Frontend: fused kernel patterns (Grouped GEMM+GLU, SDPA, NSA) for Metal ref |
| [12-pr-22673-mtp-clean-latest.md](12-pr-22673-mtp-clean-latest.md) | Latest PR #22673 / mtp-clean lessons: hook design status, partial MTP loader, rollback slots, backend/multimodal risks |

## Quick Start

```bash
# Build
cd ~/llama.cpp/worktrees/LJ-Ling-2.6-flash-mtp
cmake -B build -DLLAMA_METAL=ON -DLLAMA_ACCELERATE=ON && cmake --build build -j$(sysctl -n hw.ncpu) --target llama-server

# Run with MTP (currently broken - see 02-debug-findings.md Bug #2)
./build/bin/llama-server \
  -m /Volumes/NVME_4TB_SSD_GRAUGEAR/Users_ljubomir/llama.cpp/bailing-hybrid/Ling-2.6-flash-IQ4_NL-fixed.gguf \
  -ngl 99 -t 8 -fa on --ctx-size 4096 \
  --spec-type mtp --spec-draft-n-max 1 --port 8080

# Run trunk only (works)
./build/bin/llama-server \
  -m /Volumes/NVME_4TB_SSD_GRAUGEAR/Users_ljubomir/llama.cpp/bailing-hybrid/Ling-2.6-flash-IQ4_NL-fixed.gguf \
  -ngl 99 -t 8 -fa on --ctx-size 4096 --port 8080
```

## TL;DR of Bugs

**2026-05-07 PR #22673 update:** latest `mtp-clean` still uses the hook-based
`llama_set_mtp()` design, but adds useful fixes: partial sibling-model loading,
target-only recurrent rollback slots for MTP, double-free lifetime ordering, and
Qwen dense/MoE MTP heads. New PR reports confirm the main integration risks for
Ling r2: duplicate backend/GPU memory allocation, tensor split placement, missing
MTP memory estimation, and multimodal prompt crashes. See
[12-pr-22673-mtp-clean-latest.md](12-pr-22673-mtp-clean-latest.md).

1. **Hook position gap** (FIXED) — streaming hook didn't save pending state on early return
2. **Draft position mismatch** (FIXED) — MTP fell behind trunk by 1 position. Fixed by GLA slot fix (`slot = n_tokens - t`) + n_max clamping. Acceptance now ~40%.
3. **Output quality divergence** (NEW, OPEN) — MTP produces "I'm Bailing, a_type" loop instead of meaningful output. keep_intermediates hardcoded false for debugging (just reverted). Investigation ongoing.
4. **Metal ki kernel snapshot indexing** (FIXED) — was saving wrong token's intermediate state. Now matches CPU semantics (skip t=0, reverse snap_idx).
5. **GLA Metal backend** (UNFIXED) — all GLA layers fall back to CPU
6. **Metal GDN state capture/replay** (UNFIXED) — no tape/replay kernels for GLA rollback
