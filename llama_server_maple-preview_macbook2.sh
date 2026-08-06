#!/usr/bin/env bash
#
# ljubomir@macbook2:~/llama.cpp$ ./llama_server_maple-preview_macbook2.sh >& log-maple-preview-ppid_$$.log &
#
# Maple Preview (20B, TQ2_0 ternary 2.06 bpw, 5.5 GB) on macbook2 (M2 Max, 96 GB).
# Fork: deepgrove-ai/llama.cpp in worktrees/deepgrove-main (branch deepgrove-main
# @ 8ce8ca6c6 maplesupport + local TQ2_0 Metal kernels, LJ-maple-tq2_0-metal).
#
# TQ2_0 runs on GPU via the local Metal port (pp ~1660 t/s, tg ~175 t/s).
# The stock fork (no Metal kernels) is CPU-only for TQ2.
#
# Code repo llama.cpp fork
# https://github.com/deepgrove-ai/llama.cpp
#
# Model files
# https://huggingface.co/deepgrove/maple-preview-GGUF
# Our model file
# https://huggingface.co/deepgrove/maple-preview-GGUF/resolve/main/maple-preview-TQ2_0-head-Q4_K.gguf
#
# The llama.cpp fork and GGUF annuncement
# https://x.com/deepgrove_ai/status/2085190212427411715
#
# Original non-GGUF HF
# https://huggingface.co/deepgrove/maple-preview

set -euo pipefail

SERVER_BIN="$HOME/llama.cpp/worktrees/deepgrove-main/build-macbook2-metal/bin/llama-server"
BIND_HOST="0.0.0.0"
BIND_PORT="8081"
LOG_DIR="$HOME/llama.cpp"

wait_for_port() {
    local port="$1" name="$2" logfile="$3"
    local retries=150
    echo "Waiting for $name on port $port log file $logfile ..."
    while ! curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
        retries=$((retries - 1))
        if [ $retries -le 0 ]; then
            echo " TIMEOUT"
            return 1
        fi
        if (( retries % 10 == 0 )); then echo -n ":"; else echo -n "."; fi
        sleep 2
    done
    echo " OK"
}

kill_servers() {
    echo "Shutting down all servers..."
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
PIDS=""
trap kill_servers EXIT INT TERM

# ── Tunables (documented, most left at defaults) ────────────────────────────
# CPU-only fallback (stock fork has no TQ2 Metal kernels):
#    --gpu-layers 0                      # ~360 pp / ~77 tg
# GPU (local Metal port):
#    --gpu-layers all                    # ~1660 pp / ~175 tg (default below)
# Model context is 131072 (declared); KV is tiny (ternary, shared-head arch).
#    --ctx-size 131072
# KV cache types: q8_0/q8_0 is a safe default; f16/f16 otherwise.
#    --cache-type-k q8_0 --cache-type-v q8_0
# Maple has no separate reasoning-format config needed (chat template in GGUF).

# === Model 1: Maple Preview (port 8083) ===
MODEL_FILE="${MODEL_FILE:-$HOME/llama.cpp/models/maple-preview-TQ2_0-head-Q4_K.gguf}"
MODEL_ALIAS="maple-preview"
MODEL_NAME="Maple Preview"
echo "=== Starting $MODEL_NAME ($MODEL_FILE) on ${BIND_HOST}:${BIND_PORT} ==="

SERVER_ARGS=(
    --model "$MODEL_FILE" --alias "$MODEL_ALIAS"
    --host "$BIND_HOST" --port "$BIND_PORT"
    --log-verbosity 4
    --gpu-layers all
    --ctx-size 131072
    --temp 0.7 --top-k 40 --top-p 0.90 --min-p 0.05
    --repeat-penalty 1.1 --presence-penalty 1.0
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
    --threads 8 --parallel 1
    --mmap --mlock
    --n-predict 4096
    --jinja
)
LOG_FILE="${LOG_DIR}/log-${MODEL_ALIAS}-ppid_$$.log"
set -x
"$SERVER_BIN" "${SERVER_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
set +x
PIDS="$PIDS $PID"

wait_for_port "$BIND_PORT" "$MODEL_NAME" "$LOG_FILE"

echo ""
echo "=== All 1 models loaded ==="
echo "  ${BIND_PORT}: $PID ${MODEL_ALIAS} ($MODEL_FILE)"
echo ""
echo "Logs: $LOG_DIR/log-*-ppid_$$.log"
echo "Press Ctrl+C to shut down all servers."
echo ""

# Wait for any child to exit
wait

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                 MAPLE PREVIEW (TQ2_0 TERNARY) BENCHMARK RESULTS             ║
# ║                    20.21B @ 2.06 bpw, 5.49 GiB, M2 Max                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# ┌───────────┬──────────┬────────────┬────────────┐
# │   Test    │   GPU    │    CPU     │  speedup   │
# ├───────────┼──────────┼────────────┼────────────┤
# │  pp512    │ 1643 t/s │  360 t/s   │   4.6×     │
# │  pp2048   │ 1659 t/s │  364 t/s   │   4.6×     │
# │  tg128    │  173 t/s │   77 t/s   │   2.2×     │
# │ 8K-depth  │ 1261/156 │     —      │     —      │
# └───────────┴──────────┴────────────┴────────────┘

