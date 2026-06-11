# DiffusionGemma 26B-A4B Q8_0 on Apple M2 Max: Technical Report

## Scope

This report records the complete June 11, 2026 setup and test session for:

- Host: macbook2, 2023 MacBook Pro, Apple M2 Max, 96 GB unified memory
- OS: macOS 15.7.7, arm64
- Model: models/diffusiongemma-26B-A4B-it-Q8_0.gguf
- Model size: 26,878,831,328 bytes, approximately 25.0 GiB
- Runtime branch: danielhanchen/llama.cpp:diffusion-visual-updates
- Runtime commit: 15ad8f4201d05fee7be94e42ac73fc934ff20235
- Upstream review: ggml-org/llama.cpp PR #24423

The goals were to choose a location for the special llama.cpp implementation, build it with Metal, create a reusable launcher, test actual generation and visual denoising, run the closest valid performance test, and compare the results with prior M2 Max measurements.

## Executive Result

The model loads and generates successfully on Metal. Both normal output and live visual denoising work. The stable long-form result was 16.0 generated canvas tokens/s at 1024 and 2048 requested/generated tokens, using prompt KV caching and device-resident self-conditioning. Maximum resident memory was about 30.8 GB.

The special branch does not provide an OpenAI-compatible DiffusionGemma HTTP server. Therefore the standard llama-benchy endpoint test could not be run honestly. The valid benchmark is the timing emitted by llama-diffusion-cli over complete 256-token diffusion canvases.

## Repository Placement Decision

Two locations were considered:

1. worktrees/, alongside local branches managed by the main ~/llama.cpp/.git repository.
2. contrib/, alongside independent llama.cpp forks and external tools.

The existing layout was inspected with:

~~~bash
git worktree list --porcelain
find contrib -maxdepth 2 -type d -name .git -prune -print
find contrib -maxdepth 1 -mindepth 1 -type d -print | sort
~~~

The relevant existing independent repositories included:

~~~text
contrib/llama.cpp-MTP/
contrib/atomic-llama-cpp-turboquant/
contrib/bati.cpp/
contrib/kernel-anvil/
contrib/llama-benchy/
~~~

The special implementation is a fork-owned named branch rather than a branch maintained in the main local repository. The project instructions also treat the main .git/ directory as read-only. An independent checkout under contrib/ therefore preserves the fork origin and avoids modifying main-repository worktree metadata.

Chosen location:

~~~text
~/llama.cpp/contrib/diffusion-llama.cpp
~~~

## Branch And Commit Verification

The fork branch and upstream pull-request ref were compared before cloning:

~~~bash
git ls-remote https://github.com/danielhanchen/llama.cpp.git \
  refs/heads/diffusion-visual-updates

git ls-remote https://github.com/ggml-org/llama.cpp.git \
  refs/pull/24423/head
~~~

Both resolved to:

~~~text
15ad8f4201d05fee7be94e42ac73fc934ff20235
~~~

This established that Daniel Hanchen's named fork branch and the current upstream PR head were identical at test time. The named branch is not an ordinary branch in the upstream repository; upstream exposes it only through the pull-request ref until it is merged.

## Checkout

The fork was cloned as a single-branch independent repository:

~~~bash
git clone \
  --branch diffusion-visual-updates \
  --single-branch \
  https://github.com/danielhanchen/llama.cpp.git \
  contrib/diffusion-llama.cpp
~~~

Verification:

~~~bash
git -C contrib/diffusion-llama.cpp status --short --branch
git -C contrib/diffusion-llama.cpp rev-parse HEAD
git -C contrib/diffusion-llama.cpp remote -v
~~~

Result:

~~~text
## diffusion-visual-updates...origin/diffusion-visual-updates
15ad8f4201d05fee7be94e42ac73fc934ff20235
origin https://github.com/danielhanchen/llama.cpp.git
~~~

## Runtime Interface Inspection

The branch was searched before building:

~~~bash
rg -n 'diffusion|time per step|tokens per second|server' \
  contrib/diffusion-llama.cpp/examples/diffusion \
  contrib/diffusion-llama.cpp/examples/CMakeLists.txt \
  contrib/diffusion-llama.cpp/CMakeLists.txt
~~~

The relevant targets were:

- llama-diffusion-cli: actual DiffusionGemma generation loop
- llama-diffusion-gemma-eval: exactness/logits comparison harness
- llama-diffusion-gemma-server: persistent stdin/stdout logits worker
- llama-gguf: GGUF inspection utility

The target named llama-diffusion-gemma-server was inspected in source. It is not an HTTP server. It reads request-file paths from stdin, performs forward passes, writes raw canvas logits to a response file, and prints a small line protocol to stdout. It does not expose /v1/models or /v1/chat/completions.

The ordinary llama-server source contains the standard OpenAI-compatible routes, but no DiffusionGemma denoising loop. Loading architecture support into the library is not sufficient: generation requires repeated full-canvas denoising, self-conditioning, adaptive stopping, block commitment, and canvas trimming.

## Metal Build

Host checks:

~~~bash
sysctl -n hw.logicalcpu
uname -m
sw_vers
~~~

Result:

~~~text
12 logical CPUs
arm64
macOS 15.7.7
~~~

The existing machine GPU wired-memory limit was already suitable:

~~~bash
sysctl -n iogpu.wired_limit_mb
~~~

Result:

~~~text
88000
~~~

The dedicated Release build was configured with explicit Metal, embedded Metal shaders, and Accelerate BLAS:

~~~bash
cd ~/llama.cpp/contrib/diffusion-llama.cpp

cmake -S . -B build-macbook2-metal \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON
~~~

Configuration identified:

~~~text
AppleClang 17.0.0
ARM64 CPU backend with dotprod and i8mm
Accelerate framework BLAS
Metal backend
ggml commit 15ad8f420
~~~

The targets were built with:

~~~bash
cmake --build build-macbook2-metal \
  --config Release \
  --target \
    llama-diffusion-cli \
    llama-diffusion-gemma-eval \
    llama-diffusion-gemma-server \
    llama-gguf \
  -j12
~~~

All four targets built successfully. Binary verification included:

~~~bash
file build-macbook2-metal/bin/llama-diffusion-cli \
     build-macbook2-metal/bin/libggml-metal.dylib

otool -L build-macbook2-metal/bin/llama-diffusion-cli

build-macbook2-metal/bin/llama-diffusion-cli --list-devices
~~~

Results:

~~~text
llama-diffusion-cli: Mach-O 64-bit executable arm64
libggml-metal.dylib: Mach-O 64-bit dynamically linked shared library arm64
MTL0: Apple M2 Max (88000 MiB, 87999 MiB free)
BLAS: Accelerate
~~~

The executable linked against libggml-metal, libggml-blas, libggml-cpu, libggml, and libllama from the dedicated build.

## GGUF Inspection

The file was checked with:

~~~bash
stat -f 'size=%z bytes modified=%Sm' \
  -t '%Y-%m-%d %H:%M:%S %z' \
  models/diffusiongemma-26B-A4B-it-Q8_0.gguf
~~~

Result:

~~~text
size=26878831328 bytes modified=2026-06-10 22:22:18 +0100
~~~

The branch's GGUF reader was also run:

~~~bash
contrib/diffusion-llama.cpp/build-macbook2-metal/bin/llama-gguf \
  models/diffusiongemma-26B-A4B-it-Q8_0.gguf r n
~~~

The command confirmed GGUF v3, 44 metadata keys, 692 tensors, general.architecture=diffusion-gemma, 30 blocks, and diffusion.canvas_length=256. The tool is extremely verbose and reads tensor data; it is useful for exact inspection but unnecessary for routine startup checks.

## Launcher

The following executable launcher was created:

~~~text
llama_server_diffusiongemma-26b-a4b_macbook2.sh
~~~

The name follows the local llama_server_*.sh discovery convention, but its header explicitly states that it launches llama-diffusion-cli, not an HTTP server.

Default behavior:

- Model: local Q8_0 GGUF
- Metal offload: --gpu-layers all
- CPU threads: 8 generation and 8 batch threads
- Flash Attention: on
- Memory mapping and locking: on
- Conversation mode: on
- Requested output ceiling: 2048 tokens
- Thinking enabled with system prompt <|think|>
- Entropy-Bound sampling: automatic, with reference settings
- Prompt KV cache: automatic, on for this single-GPU machine
- GPU-resident self-conditioning: automatic, on for this single-GPU machine
- Live diffusion canvas: on by default

The launcher is parameterized through environment variables:

~~~text
DIFFUSION_BIN
MODEL_FILE
N_PREDICT
SYSTEM_PROMPT
DIFFUSION_VISUAL
PROMPT
~~~

Extra command-line arguments are passed directly to llama-diffusion-cli, allowing test overrides such as --no-conversation, --seed, or optimization ablations.

Validation:

~~~bash
chmod 700 llama_server_diffusiongemma-26b-a4b_macbook2.sh
bash -n llama_server_diffusiongemma-26b-a4b_macbook2.sh
git diff --check -- llama_server_diffusiongemma-26b-a4b_macbook2.sh
~~~

## Smoke Test 1: Concise Sky Explanation

The first non-interactive smoke test used a 256-token canvas:

~~~bash
mkdir -p benches/diffusiongemma

/usr/bin/time -l env \
  PROMPT='Explain why the sky appears blue in two concise paragraphs.' \
  N_PREDICT=256 \
  DIFFUSION_VISUAL=0 \
  ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
    --no-conversation \
    --seed 42 \
    --log-colors off \
  > benches/diffusiongemma/smoke-256.log 2>&1
~~~

The model loaded and selected:

~~~text
canvas_length=256
max_steps=48
temperature schedule 0.8 -> 0.4
entropy_bound=0.1
confidence=0.005
kv_cache=on
gpu_sampling=on
~~~

Result:

~~~text
16 denoising steps
9.09888 s generation time
568.68 ms per step
28.1 canvas tok/s
450 in-step parallel canvas positions/s
17.73 s total process wall time including load
30,805,311,488 bytes maximum RSS
~~~

The response correctly explained Rayleigh scattering, shorter-wavelength scattering, and why the sky appears blue rather than violet. It ended mid-sentence because thinking plus answer text filled the fixed 256-token canvas. This established that 256 tokens was enough for a smoke test but not a robust quality test with thinking enabled.

## Initial 512/1024/2048 Sweep And Its Interpretation Error

The same concise sky prompt was initially tested with larger N_PREDICT values:

~~~bash
for n in 512 1024 2048; do
  /usr/bin/time -l env \
    PROMPT='Explain why the sky appears blue in two concise paragraphs.' \
    N_PREDICT="$n" \
    DIFFUSION_VISUAL=0 \
    ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
      --no-conversation \
      --seed 42 \
      --log-colors off \
    > "benches/diffusiongemma/native-${n}.log" 2>&1
done
~~~

Observed results:

| Requested ceiling | Actual canvases | Counted canvas tokens | Time | Reported throughput |
|---:|---:|---:|---:|---:|
| 512 | 2 | 512 | 16.57 s | 30.9 tok/s |
| 1024 | 2 | 512 | 17.14 s | 29.9 tok/s |
| 2048 | 2 | 512 | 17.59 s | 29.1 tok/s |

This exposed a critical semantic detail: -n is a maximum block budget, not a promise to generate that many tokens. The runner trims a canvas at the first end-of-generation token or repetition loop and stops creating additional blocks. The concise answer naturally completed after two canvases, so the 1024 and 2048 requests did not exercise four and eight blocks.

These runs were retained as evidence of natural adaptive completion, but they were rejected as the full-budget scaling benchmark.

## Controlled Long-Form Sweep

To force all requested blocks to execute, the prompt was changed to demand at least 2500 words and explicitly prevent an early conclusion.

Exact command:

~~~bash
for n in 256 512 1024 2048; do
  /usr/bin/time -l env \
    PROMPT='Write a continuous, detailed technical tutorial of at least 2500 words explaining how virtual memory, page tables, translation lookaside buffers, memory mapping, page faults, and unified memory interact on modern operating systems. Do not conclude early.' \
    N_PREDICT="$n" \
    DIFFUSION_VISUAL=0 \
    ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
      --no-conversation \
      --seed 123 \
      --log-colors off \
    > "benches/diffusiongemma/long-${n}.log" 2>&1
done
~~~

This prompt produced a structured thought section and a coherent tutorial beginning with virtual-memory abstraction, physical versus virtual addressing, MMUs, multi-level page tables, PTE fields, TLBs, and page-fault handling. The 2048-token run consumed all eight canvases and ended at the requested token ceiling rather than at a natural end token.

Results:

| Requested/generated | Canvases | Total denoising steps | Steps/canvas | Generation time | Effective tok/s | In-step parallel | Process wall time | Max RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 17 | 17.0 | 9.90 s | 25.9 | 440 tok/s | 18.72 s | 30.82 GB |
| 512 | 2 | 41 | 20.5 | 28.95 s | 17.7 | 363 tok/s | 37.65 s | 30.82 GB |
| 1024 | 4 | 84 | 21.0 | 64.01 s | 16.0 | 336 tok/s | 72.81 s | 30.82 GB |
| 2048 | 8 | 158 | 19.8 | 127.78 s | 16.0 | 317 tok/s | 136.79 s | 30.88 GB |

Interpretation:

- First-canvas throughput is not sustained throughput.
- Each later block commits another 256 tokens into the prefix.
- The branch prefills that longer prefix before denoising the next canvas.
- Per-step time rose from 582 ms at one block to 809 ms over the eight-block run.
- The 1024- and 2048-token runs converged on 16.0 effective tok/s.
- Memory remained nearly flat because the 25 GiB model dominates the footprint and the tested 4096-token context is small relative to 96 GB unified memory.

## Prompt KV Cache Ablation

The long-form 256-token case was rerun with prefix KV caching disabled:

~~~bash
/usr/bin/time -l env \
  PROMPT='Write a continuous, detailed technical tutorial of at least 2500 words explaining how virtual memory, page tables, translation lookaside buffers, memory mapping, page faults, and unified memory interact on modern operating systems. Do not conclude early.' \
  N_PREDICT=256 \
  DIFFUSION_VISUAL=0 \
  ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
    --no-conversation \
    --seed 123 \
    --log-colors off \
    --diffusion-kv-cache off \
  > benches/diffusiongemma/ablation-kv-off-256.log 2>&1
~~~

Result:

~~~text
kv_cache=off, gpu_sampling=on
16 steps
12.62442 s
789.03 ms per step
20.3 tok/s
30.82 GB maximum RSS
~~~

Against the 25.9 tok/s optimized baseline, disabling prompt KV caching reduced throughput by 21.6%. The step count changed from 17 to 16 because generation is not bitwise invariant across all backend execution paths, but the wall-time loss was unambiguous.

## GPU-Resident Self-Conditioning Ablation

The same case was rerun with GPU-resident self-conditioning disabled:

~~~bash
/usr/bin/time -l env \
  PROMPT='Write a continuous, detailed technical tutorial of at least 2500 words explaining how virtual memory, page tables, translation lookaside buffers, memory mapping, page faults, and unified memory interact on modern operating systems. Do not conclude early.' \
  N_PREDICT=256 \
  DIFFUSION_VISUAL=0 \
  ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
    --no-conversation \
    --seed 123 \
    --log-colors off \
    --diffusion-gpu-sampling off \
  > benches/diffusiongemma/ablation-gpu-sampling-off-256.log 2>&1
~~~

Result:

~~~text
kv_cache=on, gpu_sampling=off
17 steps
10.96130 s
644.78 ms per step
23.4 tok/s
30.80 GB maximum RSS
~~~

Against the optimized baseline, disabling device-resident self-conditioning reduced throughput by 9.7%. The branch comments explain the cost: without the device path, approximately 268 MB of previous-step canvas logits must be supplied through the host path on every step.

## Visual Denoising TTY Test

The default launcher behavior, including live in-place canvas updates, was tested in a real pseudo-terminal:

~~~bash
env \
  PROMPT='In one sentence, state the capital of France.' \
  N_PREDICT=256 \
  ./llama_server_diffusiongemma-26b-a4b_macbook2.sh \
    --no-conversation \
    --seed 7 \
    --log-colors off
~~~

The terminal displayed the evolving thought and answer canvas without producing a new line for each denoising step. The final answer correctly stated that Paris is the capital of France.

Result:

~~~text
8 adaptive denoising steps
4.54595 s generation time
568.24 ms per step
56.3 canvas tok/s
451 in-step parallel canvas positions/s
~~~

The 56.3 tok/s number must not be interpreted as 256 visible answer tokens per second. The runtime always denoises a 256-position canvas and then trims at the end token. The visible answer was short.

## Why Standard llama-benchy Was Not Applicable

The local llama-benchy implementation was inspected. It requires:

- an OpenAI-compatible base URL
- GET /v1/models
- streaming POST /v1/chat/completions
- usage accounting suitable for prompt-processing and generated-token timing

No target in PR #24423 satisfies that interface for DiffusionGemma generation. The normal server cannot substitute for the dedicated denoising loop, and the binary logits worker is not an HTTP service.

A compatibility wrapper would need to implement:

1. Chat-template application.
2. Prompt tokenization.
3. Repeated entropy-bound canvas denoising.
4. Self-conditioning state.
5. Prefix KV prefill and canvas decode phases.
6. Block commitment and prefix growth.
7. End-token and repetition-loop trimming.
8. OpenAI response objects and usage accounting.
9. A defined streaming policy for a canvas whose tokens change until convergence.

The last point is fundamental. Streaming provisional tokens would repeatedly revise prior output and violate normal append-only SSE expectations. Streaming only converged canvases would make time-to-first-token effectively time-to-first-block. Ordinary autoregressive TTFT and inter-token-latency metrics do not transfer directly.

The native runner's own timing was therefore used instead of manufacturing invalid PP/TG measurements.

## Comparisons

### Same Hardware: Qwen And Nemotron

Prior standardized M2 Max server measurements provide context:

| Model | Depth | TG |
|---|---:|---:|
| Qwen3.6-35B-A3B | 1K | 40.5 tok/s |
| Qwen3.6-35B-A3B | 32K | 19.4 tok/s |
| Qwen3.6-35B-A3B | 64K | 12.5 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 1K | 51.0 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 64K | 26.5 tok/s |
| Nemotron-Cascade-2-30B-A3B Q8_0 | 128K | 23.5 tok/s |
| DiffusionGemma Q8_0 | 1024-2048 output | 16.0 tok/s |

DiffusionGemma's current Metal path is slower than short-context Qwen and Nemotron, and slower than Nemotron even at 128K context. It only exceeds the recorded Qwen 64K TG rate. This does not demonstrate the architecture's advertised H100-class speed advantage on Apple Metal.

### Same Family: Gemma 4 26B-A4B

The historical M2 Max log log_llama-server-gemma4-ppid_54758-20260410_085304.log contains 81 completed 10,000-token autoregressive generation samples for Gemma 4 26B-A4B Q8_K_XL under a long-running server workload. Their reported TG distribution was:

~~~text
median 21.16 tok/s
mean   25.53 tok/s
range  12.74-93.38 tok/s
~~~

This is not a controlled benchy comparison: requests, prompts, cache state, speculative behavior, and server conditions varied. The median is still a useful same-family reference. DiffusionGemma's 16.0 tok/s sustained full-budget rate was about 24% below that historical Gemma 4 median.

### Architectural Metric Warning

The reported DiffusionGemma in-step parallel rate of 317-451 positions/s is internal work throughput. It counts all 256 positions processed during every denoising step. It is not generated text throughput and must never be placed in the same table as autoregressive TG without a separate label.

## Raw Artifacts

~~~text
benches/diffusiongemma/smoke-256.log
benches/diffusiongemma/native-512.log
benches/diffusiongemma/native-1024.log
benches/diffusiongemma/native-2048.log
benches/diffusiongemma/long-256.log
benches/diffusiongemma/long-512.log
benches/diffusiongemma/long-1024.log
benches/diffusiongemma/long-2048.log
benches/diffusiongemma/ablation-kv-off-256.log
benches/diffusiongemma/ablation-gpu-sampling-off-256.log
~~~

Other artifacts:

~~~text
contrib/diffusion-llama.cpp/
contrib/diffusion-llama.cpp/build-macbook2-metal/
llama_server_diffusiongemma-26b-a4b_macbook2.sh
DIFFUSIONGEMMA-26B-A4B_METAL_BENCHMARK_REPORT.md
~~~

## Final Assessment

Confidence is high that the Q8_0 model and PR #24423 runtime are functioning correctly on this M2 Max. The output is coherent, the visual denoising path works, all intended blocks execute under a sufficiently long prompt, and the branch-specific optimizations show measurable benefits.

Confidence is also high that the current implementation is not yet a competitive Metal serving path. The sustained long-form rate is 16.0 tok/s, there is no OpenAI-compatible diffusion server, output arrives naturally by converged canvas rather than by append-only token, and the current speed is below established Gemma 4 and Nemotron results on the same machine.

