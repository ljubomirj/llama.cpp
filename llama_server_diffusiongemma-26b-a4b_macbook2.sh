#!/usr/bin/env bash
#
# Interactive DiffusionGemma runner for macbook2. Despite the llama_server_ name,
# PR #24423 currently provides a dedicated CLI, not an HTTP llama-server backend.
#
# ljubomir@macbook2:~/llama.cpp$ ./llama_server_diffusiongemma-26b-a4b_macbook2.sh
# Non-interactive example:
# PROMPT='Why is the sky blue?' N_PREDICT=256 DIFFUSION_VISUAL=0 \
#   ./llama_server_diffusiongemma-26b-a4b_macbook2.sh --no-conversation

set -euo pipefail

DIFFUSION_BIN="${DIFFUSION_BIN:-$HOME/llama.cpp/contrib/diffusion-llama.cpp/build-macbook2-metal/bin/llama-diffusion-cli}"
MODEL_FILE="${MODEL_FILE:-$HOME/llama.cpp/models/diffusiongemma-26B-A4B-it-Q8_0.gguf}"
N_PREDICT="${N_PREDICT:-2048}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-<|think|>}"
DIFFUSION_VISUAL="${DIFFUSION_VISUAL:-1}"

if [[ ! -x "$DIFFUSION_BIN" ]]; then
    printf 'Diffusion runner is missing or not executable: %s\n' "$DIFFUSION_BIN" >&2
    exit 1
fi

if [[ ! -r "$MODEL_FILE" ]]; then
    printf 'Model is missing or not readable: %s\n' "$MODEL_FILE" >&2
    exit 1
fi

DIFFUSION_ARGS=(
    --model "$MODEL_FILE"
    --gpu-layers all
    --threads 8
    --threads-batch 8
    --flash-attn on
    --mmap
    --mlock
    --conversation
    --n-predict "$N_PREDICT"
    --system-prompt "$SYSTEM_PROMPT"
    --diffusion-eb auto
    --diffusion-eb-max-steps 48
    --diffusion-eb-t-max 0.8
    --diffusion-eb-t-min 0.4
    --diffusion-eb-entropy-bound 0.1
    --diffusion-eb-confidence 0.005
    --diffusion-kv-cache auto
    --diffusion-gpu-sampling auto
)

if [[ "$DIFFUSION_VISUAL" == "1" ]]; then
    DIFFUSION_ARGS+=(--diffusion-visual)
fi

if [[ -n "${PROMPT:-}" ]]; then
    DIFFUSION_ARGS+=(--prompt "$PROMPT")
fi

printf 'Starting DiffusionGemma with %s\n' "$DIFFUSION_BIN"
printf 'Model: %s\n' "$MODEL_FILE"
set -x
exec "$DIFFUSION_BIN" "${DIFFUSION_ARGS[@]}" "$@"
