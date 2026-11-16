#!/usr/bin/env bash
# Download reranker GGUFs for the pm1 model bake-off into ~/models/rerankers.
set -euo pipefail
DIR="${PM1_MODELS:-$HOME/models/rerankers}"
mkdir -p "$DIR"
get() {  # repo file
  if [ -s "$DIR/$2" ]; then echo "have $2"; return; fi
  curl -sSL --retry 3 -o "$DIR/$2.part" "https://huggingface.co/$1/resolve/main/$2" && mv "$DIR/$2.part" "$DIR/$2"
  echo "got $2 ($(du -h "$DIR/$2" | cut -f1))"
}
get gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-Q4_K_M.gguf
get gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-Q8_0.gguf
get gpustack/bge-reranker-v2-m3-GGUF bge-reranker-v2-m3-FP16.gguf
get ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
echo done
