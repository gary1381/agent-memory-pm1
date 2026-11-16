#!/usr/bin/env bash
# Download the public benchmark data the eval scripts read. Nothing here is redistributed by this repository:
# each file comes from its publisher, under its own license (see docs/THIRD_PARTY.md). Run only what you need:
#   scripts/prepare_data.sh lme | beam | beam10m | locomo | all
# Files go to ${PM1_DATA:-~/.cache/pm1/data}; the eval scripts look there by default.
set -euo pipefail
DATA="${PM1_DATA:-$HOME/.cache/pm1/data}"
HF="https://huggingface.co"

get() {  # url dest
  if [ -s "$2" ]; then echo "have $2"; return; fi
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  echo "got  $2 ($(du -h "$2" | cut -f1))"
}

lme() {     # LongMemEval-S (cleaned), MIT -- github.com/xiaowu0162/LongMemEval
  get "$HF/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
      "$DATA/longmemeval_s_cleaned.json"
}
beam() {    # BEAM 100K / 500K / 1M, CC BY-SA 4.0 -- huggingface.co/datasets/Mohammadta/BEAM
  for s in 100K 500K 1M; do
    d="$DATA/beam$(echo "$s" | tr 'A-Z' 'a-z')"
    get "$HF/datasets/Mohammadta/BEAM/resolve/main/data/$s-00000-of-00001.parquet" "$d/$s-00000-of-00001.parquet"
  done
}
beam10m() { # BEAM 10M, CC BY-SA 4.0 -- huggingface.co/datasets/Mohammadta/BEAM-10M
  for i in 0 1; do
    get "$HF/datasets/Mohammadta/BEAM-10M/resolve/main/data/10M-0000$i-of-00002.parquet" "$DATA/beam10m/10M-0000$i-of-00002.parquet"
  done
}
locomo() {  # LoCoMo, CC BY-NC 4.0 (non-commercial) -- github.com/snap-research/locomo
  get "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" "$DATA/locomo10.json"
}

case "${1:-}" in
  lme) lme ;; beam) beam ;; beam10m) beam10m ;; locomo) locomo ;;
  all) lme; beam; beam10m; locomo ;;
  *) echo "usage: $0 lme|beam|beam10m|locomo|all" >&2; exit 2 ;;
esac
