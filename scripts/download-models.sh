#!/usr/bin/env bash
# Downloads the models that are not fetched automatically at runtime.
# Whisper (faster-whisper) and the embedding model self-download on first use.
#
# Usage:  bash scripts/download-models.sh [--llm]
#         --llm  also download the ~2.4 GB Qwen3-4B GGUF

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/piper models/whisper models/fastembed

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "  exists: $out"
    return
  fi
  echo "  downloading $(basename "$out")"
  curl -fL --progress-bar "$url" -o "$out.part"
  mv "$out.part" "$out"
}

echo "[1/3] silero VAD (ONNX, 2 MB)"
fetch \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
  "models/silero_vad.onnx"

echo "[2/3] Piper voice en_US-amy-medium (63 MB)"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
fetch "$PIPER_BASE/en_US-amy-medium.onnx"      "models/piper/en_US-amy-medium.onnx"
fetch "$PIPER_BASE/en_US-amy-medium.onnx.json" "models/piper/en_US-amy-medium.onnx.json"

if [ "${1:-}" = "--llm" ]; then
  echo "[3/3] Qwen3-4B-Instruct Q4_K_M (2.4 GB)"
  # Fits a 4 GB card alongside Whisper. Drop to Qwen3-1.7B if you hit CUDA OOM.
  fetch \
    "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
    "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
else
  echo "[3/3] skipping LLM weights (pass --llm to download)"
fi

echo
echo "done. models/:"
du -h models/* 2>/dev/null | sort -h || true
