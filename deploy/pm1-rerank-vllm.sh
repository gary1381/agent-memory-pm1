#!/bin/bash
# Qwen3-Reranker-0.6B (fp16) on vLLM for pm1 agents: 127.0.0.1:18166, OpenAI-style /score (pm1 sends the full
# instruction prompt itself, see pm1.rerank.vllm_qwen_scorer). ~5x faster than the llama.cpp Q8 server on 18162 with
# the same rankings as the original model (eval/rerank_engines_20261018). The llama.cpp server stays on 18162 for
# experiments that pre-registered it (BEAM-1M).
#   bash deploy/pm1-rerank-vllm.sh [gpu_memory_fraction, default 0.17] [max_num_batched_tokens, default 16384]
FRAC=${1:-0.17}   # 0.30 + 32K batch tokens measured no faster (1.12 s vs 1.03 s)
podman rm -f pm1-rerank-qwen-vllm >/dev/null 2>&1
podman run -d --name pm1-rerank-qwen-vllm --label pm1=reranker --restart unless-stopped \
  --security-opt label=disable --device nvidia.com/gpu=all --ipc=host \
  -v "${PM1_RERANK_MODEL_DIR:-$HOME/models/hf/Qwen3-Reranker-0.6B}":/model:ro -p 127.0.0.1:18166:8000 \
  docker.io/vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9 \
  --model /model --served-model-name qwen3-reranker --runner pooling --convert classify --dtype float16 \
  --enforce-eager \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --max-model-len 2048 --max-num-batched-tokens "${2:-16384}" --gpu-memory-utilization "$FRAC" --port 8000
