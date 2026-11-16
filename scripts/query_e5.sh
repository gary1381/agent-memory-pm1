#!/usr/bin/env bash
# Local CPU query encoder: the same multilingual-e5-small Q8_0 GGUF that a GPU host may serve for documents.
# Documents can be embedded in bulk on a GPU; short queries are embedded here so the online path stays local. Same weights + same runtime => same vector space.
#   scripts/query_e5.sh start [cpu|gpu] | stop   (127.0.0.1:${PM1_QE5_PORT:-18170}; gpu needs ~0.3 GB VRAM)
set -euo pipefail
IMAGE="${PM1_LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
DIR="${PM1_E5_DIR:-$HOME/models/embedding-e5-gguf}"
FILE="multilingual-e5-small-q8_0-512.gguf"
SHA="76d049169b67f063c16a47958a46b6fc36f917eb416bce03ab1a174f48c390be"   # must match the document-side GGUF
PORT="${PM1_QE5_PORT:-18170}"; NAME=pm1-query-e5
if [ "${1:-}" = stop ]; then podman rm -f "$NAME" >/dev/null 2>&1 || true; echo "stopped $NAME"; exit 0; fi
printf '%s  %s\n' "$SHA" "$DIR/$FILE" | sha256sum --check --status || { echo "model sha mismatch" >&2; exit 2; }
DEVICE="${2:-cpu}"
if [ "$DEVICE" = gpu ]; then
  gpu_args=(--device nvidia.com/gpu=all); ngl=99
else gpu_args=(); ngl=0; fi
podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --name "$NAME" --label pm1=query-e5 --restart unless-stopped --security-opt label=disable "${gpu_args[@]}" \
  -v "$DIR:/models:ro" -p "127.0.0.1:$PORT:8080" --entrypoint /app/llama-server "$IMAGE" \
  -m "/models/$FILE" --embedding --pooling mean --host 0.0.0.0 --port 8080 -ngl "$ngl" \
  -c 1024 --parallel 2 -b 1024 -ub 1024 -t "${PM1_QE5_THREADS:-2}" --alias e5-small-q8 >/dev/null
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "ready $NAME on $PORT"; exit 0; }
  sleep 1
done
echo "not ready: $(podman logs --tail 20 "$NAME" 2>&1)" >&2; exit 1
