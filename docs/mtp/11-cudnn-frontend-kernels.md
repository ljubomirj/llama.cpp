# NVIDIA cuDNN Frontend — Kernel Patterns Reference

Source: https://github.com/NVIDIA/cudnn-frontend (Python Frontend + Frontend OSS APIs)
Docs: https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/fe-oss-apis.html

**Note: CUDA-specific. These patterns inform Metal kernel design for future
ggml-metal optimizations, not directly portable.**

## Relevant Fused Kernel Patterns

### 1. Grouped GEMM Fusions (MoE-Specific)

```
Grouped GEMM + SwiGLU (SM100)
Grouped GEMM + dSwiGLU
Grouped GEMM + GLU + Hadamard
Grouped GEMM + Quant (Dense, Unified)
Discrete Grouped GEMM + SwiGLU  (per-expert pointer arrays)
```

**Relevance to Ling-2.6**: Ling has MoE with 256 experts, 8 active/token.
Fused Grouped GEMM + SwiGLU would eliminate separate gate/up matrix
multiplications and the SwiGLU activation, reducing memory traffic.

**Metal port feasibility**: High for SwiGLU fusion (simple fused kernel).
Lower for discrete grouped (uses CUDA tensor core instructions).

### 2. SDPA Forward/Backward (Attention)

```
SDPA Forward  (SM100, D=256)
SDPA Backward (SM100, D=256)
```

Highly optimized attention with fused softmax + matmul. Not directly
applicable since ggml-metal already has flash attention kernels.

### 3. Fused RMSNorm + SiLU

```
RMSNorm + SiLU (SM100)
```

**Relevance**: Ling-2.6 uses RMSNorm extensively (attention norm, FFN norm,
MTP norms). Fusing RMSNorm + SiLU would reduce kernel launch overhead for
FFN activation.

### 4. RHT (Row-wise Hadamard Transform)

```
RMSNorm + RHT + Amax (SM100)
```

Optional quantization pre-processing. Not relevant for Ling-2.6 inference.

### 5. Native Sparse Attention (NSA) — In Progress

```
NSA (Native Sparse Attention)
```

Sparse attention pattern for very long contexts. Niche for Ling-2.6 but
could be relevant for >128K context inference.

## Frontend Architecture

The cuDNN Python frontend provides three layers:

### Graph API
Low-level primitives for building, compiling, and executing cuDNN operation
graphs. Operations: Attention, Causal Conv1d, Convolutions, Matmul,
Normalizations, Grouped Matmul (MoE), Pointwise, Reshape, etc.

### Graph Wrapper (`Graph`)
Convenience layer: manages workspace, tensor mapping, execution.
Relevant to ggml's compute graph builder pattern.

### Frontend-Only APIs ("FE-OSS")
Turnkey fused kernels exposed as standalone Python APIs:
- `cudnn.gemm_swiglu` — GEMM + SwiGLU (FP16, SM100)
- `cudnn.rmsnorm_rht_amax` — RMSNorm + RHT + Amax
- `cudnn.grouped_gemm` — Grouped GEMM for MoE
- `cudnn.sdpa` — SDPA Forward/Backward
- `cudnn.native_sparse_attention` — NSA (in progress)

## Kernel Fusion Strategy for Metal

The cuDNN approach of fusing GEMM + elementwise ops is the right pattern
for Metal kernel optimization:

### Priority 1: GEMM + SwiGLU (MoE FFN)
```
Current: gate_proj → SiLU → up_proj → multiply → down_proj
Fused:   [gate_proj + up_proj] → SiLU → multiply → ... (still need down_proj separately)
```

Metal: Single kernel that reads gate/up weights, computes both matmuls,
applies SiLU, multiplies elementwise, writes intermediate result.

### Priority 2: RMSNorm + Linear
```
Current: RMSNorm kernel → linear kernel (two launches)
Fused:   RMSNorm + matmul in one kernel
```

Common in Ling-2.6: attention norm followed by Q/K/V projections.

### Priority 3: GLA + Gate + RMSNorm
```
Current: GLA output → gate (sigmoid) → GroupRMSNorm → dense proj
Fused:   Single kernel for GLA output + gate + norm
```

This would be the biggest win for Ling-2.6 since GLA is on CPU and the
gate/norm/dense are separate ops. A fused Metal kernel for the entire
GLA layer (GLA attention → gate → norm → output projection) would
eliminate multiple GPU↔CPU transfers.

## Discrete Grouped MoE Pattern

For MoE with many experts and small batches, the "discrete grouped" approach
uses per-expert pointer arrays instead of a packed `B` tensor:

```
b_ptrs:  CUDA pointers array (num_experts,) → each points to one expert's weights
```

This avoids data duplication in the packed tensor and is more memory-efficient.
For Ling-2.6's 256 experts, this could reduce MoE kernel memory requirements
on Metal if the API supported per-expert pointers.

## What's NOT Applicable

- **SM100-specific instructions**: Blackwell GPU features, no Metal equivalent
- **CUDA graph capture**: Linux/DMA-BUF specific, no Metal equivalent
- **Tensor core instructions**: NVIDIA-specific matrix multiply-accumulate
- **FP8/NVFP4 quant**: Hardware-specific formats, Metal uses INT4/INT8

## Key Takeaways

1. **Fuse GEMM + elementwise ops** — standard optimization pattern, applies to Metal
2. **Discrete grouped MoE** — pointer-based expert dispatch, future Metal possibility
3. **RMSNorm + Linear fusion** — biggest bang for buck on Metal (reduces kernel launches)
4. **Kernel fusion reduces memory traffic** — the main bottleneck on Apple Silicon
