# MTP Status and Reference

## Current State (2026-05-05)

### What Works

| Component | Status | Notes |
|-----------|--------|-------|
| Trunk (no MTP) | **Working** | ~18 tok/s gen, 212-361 tok/s prefill |
| MTP model loading | **Working** | 22 tensors loaded, graph builds OK |
| MTP hook prefill | **Working** | Hidden states streamed to MTP during prefill |
| Hook position gap fix | **Working** | No more rc=-1 after verification batches |
| First MTP draft | **Correct** | `capital` predicted correctly for "capital of France" |
| Hook early-return pending | **Working** | pending_h/pending_pos saved on early return |

### What Doesn't Work

| Issue | Severity | Notes |
|-------|----------|-------|
| MTP draft position mismatch | **Critical** | MTP 1 position behind trunk after acceptance. Low acceptance rate (14%) |
| Metal GDN state rollback | **Critical** | No Metal kernels for GDN/GLA state capture/replay on draft rejection |
| GLA Metal backend | **Major** | All GLA layers fall back to CPU. 30/32 layers on CPU |
| MTP head quality degradation | **Major** | After first correct draft, predictions rapidly degrade to garbage |

### Blocked

- Full CPU test of MTP: 57GB model needs ~115GB with repack, only 96GB RAM
- Custom Metal GLA/GDN kernels: major engineering effort (~800 lines MSL)

### Dependency Chain

```
MTP working end-to-end
├── Bug #2: Position mismatch (software fix needed)
│   └── Fix hook/draft position synchronization
├── Bug #3: Metal GDN state (kernel work needed)
│   ├── Port GLA compute kernel from bailing_hybrid.py
│   ├── Write GDN state capture kernel
│   └── Write GDN state replay/rollback kernel
└── Bug #4: GLA Metal backend (kernel work needed)
    └── Port _make_recurrent_gla_kernel to ggml-metal
```

Bug #2 can be fixed independently (software only). Bugs #3 and #4 are
interrelated Metal kernel work.

## Worktree Layout

```
~/llama.cpp/
├── worktrees/
│   ├── LJ-Ling-2.6-flash/         ← WORKING trunk (branch from upstream/master)
│   ├── LJ-Ling-2.6-flash-old/     ← old worktree (ignore)
│   ├── LJ-Ling-2.6-flash-mtp/     ← MTP integration (this worktree)
│   ├── mtp-clean/                 ← upstream MTP PR branch (also broken on Metal)
│   └── upstream-master/           ← upstream llama.cpp master
```

## Model Files

```
/Volumes/NVME_4TB_SSD_GRAUGEAR/Users_ljubomir/llama.cpp/bailing-hybrid/
├── Ling-2.6-flash-IQ4_NL-fixed.gguf   # 57GB — working quantized GGUF
├── Ling-2.6-flash-F16.gguf             # 200GB — original bfloat16 GGUF
└── *.log                               # conversion/quantization logs
```

## Build & Run

### Build (MTP worktree)
```bash
cd ~/llama.cpp/worktrees/LJ-Ling-2.6-flash-mtp
mkdir -p build && cd build
cmake .. -DLLAMA_METAL=ON -DLLAMA_ACCELERATE=ON
make -j$(sysctl -n hw.ncpu) llama-server
```

### Run with MTP
```bash
./build/bin/llama-server \
  -m /Volumes/NVME_4TB_SSD_GRAUGEAR/Users_ljubomir/llama.cpp/bailing-hybrid/Ling-2.6-flash-IQ4_NL-fixed.gguf \
  -ngl 99 -t 8 -fa on \
  --ctx-size 4096 --batch-size 256 --ubatch-size 64 \
  --parallel 1 \
  --spec-type mtp --spec-draft-n-max 1 \
  --port 8080
```

### Run trunk only (baseline)
```bash
./build/bin/llama-server \
  -m /Volumes/NVME_4TB_SSD_GRAUGEAR/Users_ljubomir/llama.cpp/bailing-hybrid/Ling-2.6-flash-IQ4_NL-fixed.gguf \
  -ngl 99 -t 8 -fa on \
  --ctx-size 4096 \
  --port 8080
```

## MLX Reference Benchmarks

For comparing llama.cpp performance against MLX:

```
pp1024/tg128:  TTFT 3513ms, TPOT 27ms, 291 tok/s prefill, 37 tok/s gen
pp8192/tg128:  TTFT 32563ms, TPOT 27ms, 252 tok/s prefill, 37 tok/s gen
pp65536/tg128: TTFT 296868ms, TPOT 33ms, 221 tok/s prefill, 31 tok/s gen
```

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/models/bailing-hybrid.cpp:534-890` | 357 | Trunk + MTP model definitions and graph builders |
| `src/llama-context.cpp:3270-3391` | 122 | Hook mechanism (`handle_mtp_for_ubatch`, `set_mtp`) |
| `common/speculative.cpp:604-767` | 164 | MTP draft function (`common_speculative_state_mtp`) |
| `src/llama-batch.cpp` | — | Batch allocator with consecutive position check |
| `tools/server/server-context.cpp:3005-3158` | 154 | Server speculative verification loop |

## External References

| Resource | Path/URL | Purpose |
|----------|----------|---------|
| MTPLX GDN capture | `~/LJ-asi-mlx/MTPLX/mtplx/gdn_capture.py` | ~800 lines custom MSL kernels for GDN state management |
| MTPLX generation | `~/LJ-asi-mlx/MTPLX/mtplx/generation.py` | Draft/verify loop with GDN state management |
| MLX GLA kernel | `~/LJ-asi-mlx/omlx/mlx_lm/models/bailing_hybrid.py` | `_make_recurrent_gla_kernel` Metal kernel for GLA |
| MLX model patch | `~/LJ-asi-mlx/mlx-lm/mlx_lm/models/bailing_hybrid.py` | Auto-generated model definition (submitted upstream) |
| Upstream MTP PR | `worktrees/mtp-clean/` (am17an/llama.cpp) | Reference MTP implementation (broken on Metal for hybrid models) |

## Session History

- **2026-04**: Trunk working in LJ-Ling-2.6-flash worktree. MTP model registered.
- **2026-05-01**: Confirmed mtp-clean PR broken on Metal for Qwen3.6 models too.
- **2026-05-02**: Identified Metal GDN state corruption as root cause for Qwen3.6.
- **2026-05-03**: Analyzed MTPLX custom Metal kernels for GDN capture/replay.
- **2026-05-05**: Fixed hook position gap bug (Bug #1). Identified draft position
  mismatch (Bug #2). MTP still not working end-to-end.
