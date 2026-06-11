# DiffusionGemma 26B-A4B Metal Benchmark Report

**Date:** June 11, 2026  
**Hardware:** MacBook Pro M2 Max, 96 GB unified memory  
**Model:** `diffusiongemma-26B-A4B-it-Q8_0.gguf` (26,878,831,328 bytes)  
**Runtime:** `danielhanchen/llama.cpp:diffusion-visual-updates` at `15ad8f4201d05fee7be94e42ac73fc934ff20235`

## Result

The Q8_0 model builds and runs successfully with Metal. The optimized single-GPU path uses prompt KV caching and device-resident self-conditioning. A long-form 2048-token run completed in 127.78 seconds at 16.0 full-budget output tokens/s and about 30.9 GB maximum RSS.

The model produced coherent output in both normal and live visual denoising modes. The 2048-token test generated a technically sound tutorial with a structured reasoning trace, then continued into the requested answer until the token budget ended.

## Checkout And Build

The independent fork belongs under `contrib/diffusion-llama.cpp`, matching the existing standalone forks in `contrib/`. It is not an upstream branch or a suitable main-repository worktree: Daniel Hanchen's branch and upstream PR ref currently resolve to the same commit, but only the fork has the named `diffusion-visual-updates` branch.

```bash
git clone --branch diffusion-visual-updates --single-branch \
  https://github.com/danielhanchen/llama.cpp.git \
  contrib/diffusion-llama.cpp

cmake -S contrib/diffusion-llama.cpp \
  -B contrib/diffusion-llama.cpp/build-macbook2-metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON

cmake --build contrib/diffusion-llama.cpp/build-macbook2-metal \
  --config Release -j12 \
  --target llama-diffusion-cli llama-diffusion-gemma-eval \
           llama-diffusion-gemma-server llama-gguf
```

The built runtime sees `MTL0: Apple M2 Max (88000 MiB)` and Accelerate BLAS.

## Native Benchmark

The full-budget sweep used a fixed long-form prompt, seed 123, thinking enabled, 48-step Entropy-Bound sampling, Flash Attention, all layers on Metal, prompt KV caching, and GPU-resident self-conditioning.

| Requested/generated | Blocks | Denoising steps | Generation time | Full-budget tok/s | Step time | Max RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 17 | 9.90 s | 25.9 | 582 ms | 30.82 GB |
| 512 | 2 | 41 | 28.95 s | 17.7 | 706 ms | 30.82 GB |
| 1024 | 4 | 84 | 64.01 s | 16.0 | 762 ms | 30.82 GB |
| 2048 | 8 | 158 | 127.78 s | 16.0 | 809 ms | 30.88 GB |

Throughput falls after the first canvas because every new block commits another 256 tokens to the prefix and requires a fresh prompt prefill. The 1024- and 2048-token runs stabilize at 16.0 tok/s. Memory use is nearly flat because the model weights dominate the footprint and the tested context is small.

Adaptive stopping is strongly prompt-dependent. A simple capital-of-France request converged in 8 steps and reported 56.3 canvas tok/s. That figure is not 256 visible answer tokens per second: the runner denoises a complete canvas and then trims it at the end token.

## Optimization Ablations

One 256-token long-form canvas was used for controlled ablations.

| Configuration | Steps | Time | Canvas tok/s | Change |
|---|---:|---:|---:|---:|
| KV cache on, GPU sampling on | 17 | 9.90 s | 25.9 | baseline |
| KV cache off, GPU sampling on | 16 | 12.62 s | 20.3 | -21.6% |
| KV cache on, GPU sampling off | 17 | 10.96 s | 23.4 | -9.7% |

Both optimized defaults should remain enabled on the M2 Max. Prompt KV caching is the larger gain even for this short prompt; its value should increase as the committed prefix grows.

## Comparison With Prior M2 Max Runs

These are different generation algorithms, so the comparison is user-perceived completion throughput, not equivalent kernel work. DiffusionGemma emits completed 256-token blocks; autoregressive models stream one accepted token at a time.

| Model/test | Context or output depth | Reported generation rate |
|---|---:|---:|
| DiffusionGemma Q8_0 | 256-token full canvas | 25.9 tok/s |
| DiffusionGemma Q8_0 | 1024-2048 full output | 16.0 tok/s |
| Gemma 4 26B-A4B Q8_K_XL | 10K-token historical server runs | 21.16 tok/s median |
| Qwen3.6-35B-A3B | 1K context | 40.5 tok/s |
| Qwen3.6-35B-A3B | 32K context | 19.4 tok/s |
| Qwen3.6-35B-A3B | 64K context | 12.5 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 1K context | 51.0 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 64K context | 26.5 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 128K context | 23.5 tok/s |

At short context, this proof-of-concept DiffusionGemma runtime is slower than the established autoregressive MoE servers. Its sustained 16.0 tok/s is roughly 60% below Qwen's 1K rate and 69% below Nemotron's 1K rate. It is also below Nemotron at 128K, despite the DiffusionGemma test using only a few thousand tokens of total context. The current Metal implementation therefore does not demonstrate the architecture's advertised accelerator speed advantage.

The same-family Gemma 4 figure comes from 81 completed 10,000-token generations in a historical long-running M2 Max server log, not a controlled benchy run. Its median was 21.16 tok/s, so DiffusionGemma's sustained 16.0 tok/s was about 24% lower; prompts, cache state, and server conditions varied in the Gemma 4 workload.

The branch's `in-step parallel` rate of 317-440 tok/s is not comparable to autoregressive TG. It counts 256 canvas positions refined in parallel on every denoising step, while effective completed-token throughput divides that work by the number of adaptive steps.

## Why `llama-benchy` Was Not Run

`llama-benchy` requires an OpenAI-compatible `/v1/models` and streaming `/v1/chat/completions` server. PR #24423 does not provide one:

- `llama-diffusion-cli` is the supported generator.
- The normal `llama-server` has no DiffusionGemma generation loop.
- `llama-diffusion-gemma-server` is a stdin/stdout binary-logits worker for a Python driver, not an HTTP server.

Running the standard PP/TG matrix would therefore require a new diffusion-aware HTTP serving layer. Treating canvas denoising as ordinary token streaming would produce misleading TTFT and TG measurements. The native benchmark above is the closest valid measurement available from this draft branch.

## Artifacts

- Launcher: `llama_server_diffusiongemma-26b-a4b_macbook2.sh`
- Raw logs: `benches/diffusiongemma/`
- Upstream draft PR: https://github.com/ggml-org/llama.cpp/pull/24423
- Fork branch: https://github.com/danielhanchen/llama.cpp/tree/diffusion-visual-updates
- Model: https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF
