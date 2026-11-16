#!/usr/bin/env bash
# Start / stop a llama.cpp rerank server on the GPU in its own container.
#   scripts/reranker.sh start <name> <file.gguf> <port>   e.g. start bge-q8 bge-reranker-v2-m3-Q8_0.gguf 18140
#   scripts/reranker.sh stop  <name>
# Binds 127.0.0.1 only. Containers are labelled pm1=reranker.
set -euo pipefail
IMAGE="${PM1_LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
DIR="${PM1_MODELS:-$HOME/models/rerankers}"
cmd="$1"; name="pm1-rerank-$2"
if [ "$cmd" = stop ]; then podman rm -f "$name" >/dev/null 2>&1 || true; echo "stopped $name"; exit 0; fi
file="$3"; port="$4"; device="${5:-gpu}"   # gpu | cpu (quality is identical; cpu avoids VRAM pressure)
if [ "$device" = cpu ]; then gpu_args=(); ngl=0; else
  gpu_args=(--device nvidia.com/gpu=all); ngl=99; fi
podman rm -f "$name" >/dev/null 2>&1 || true
podman run -d --name "$name" --label pm1=reranker --restart unless-stopped --security-opt label=disable \
  "${gpu_args[@]}" \
  -v "$DIR:/models:ro" -p "127.0.0.1:$port:8080" \
  --entrypoint /app/llama-server "$IMAGE" \
  -m "/models/$file" --host 0.0.0.0 --port 8080 --rerank -ngl "$ngl" -fa on -ctk "${PM1_KV_TYPE:-q8_0}" -ctv "${PM1_KV_TYPE:-q8_0}" \
  -c "${PM1_RERANK_CTX:-4096}" --parallel "${PM1_RERANK_PAR:-4}" -b 2048 -ub 1024 -t "${PM1_RERANK_THREADS:-6}" --alias "$2" >/dev/null
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then echo "ready $name on $port"; exit 0; fi
  sleep 1
done
echo "not ready: $(podman logs --tail 20 "$name" 2>&1)" >&2; exit 1
