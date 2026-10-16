#!/usr/bin/env bash
# Starts llama-server with settings tuned for a 4 GB GTX 1050 Ti.
#
#   --jinja is REQUIRED. Without it llama-server silently ignores the "tools"
#   field and the model answers from (nonexistent) memory instead of calling tools.
#
# Install llama.cpp with CUDA support first:
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61   # 61 = Pascal
#   cmake --build build --config Release -j
#   export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server

set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODEL="${MODEL:-models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PORT="${PORT:-8081}"

# 36 = all layers of Qwen3-4B on GPU. If you see CUDA OOM, step down to 28, then 20.
NGL="${NGL:-36}"
# 4096 context. Each extra 1k costs ~30 MB of VRAM for this model.
CTX="${CTX:-4096}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL" >&2
  echo "run: bash scripts/download-models.sh --llm" >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias qwen3-4b-instruct \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --jinja \
  --flash-attn off \
  --no-warmup \
  --threads 4 \
  --batch-size 256 \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
