# Third-party models, datasets and code

This repository contains **no model weights and no benchmark data**. Everything below is downloaded separately and
stays under its publisher's license. The licenses listed were read from each publisher's model card, dataset card or
LICENSE file on 2026-11-16. Check them again before you redistribute anything.

## Vendored code

| Files | Source | License |
|---|---|---|
| `eval/beam_official/compute_metrics.py`, `eval/beam_official/prompts.py` | BEAM benchmark, github.com/mohammadtavakoli78/BEAM | MIT, Copyright (c) 2025 Mohammad Tavakoli (`eval/beam_official/LICENSE`) |

`eval/lme_qa.py` builds LongMemEval's answer-check judge prompts from the wording in the LongMemEval repository (MIT).
`eval/qwen3_reranker_memory.jinja` and `src/pm1/rerank.py` use the Qwen3-Reranker prompt format, with pm1's own task
instruction.

## Models

| Model | Used for | License | How you get it |
|---|---|---|---|
| `intfloat/multilingual-e5-small` (384-d) | dense retrieval (documents and queries) | MIT | `PM1_DENSE=local`: `fastembed` downloads the int8 ONNX file (`onnx/model_qint8_avx512_vnni.onnx`) from Hugging Face on first use. Optionally, run a Q8_0 GGUF of the same model with llama.cpp (`scripts/query_e5.sh`, `PM1_DENSE=split` or `http`). |
| `Qwen/Qwen3-Reranker-0.6B` | cross-encoder rerank of the top candidates | Apache-2.0 | Optional. Download the HF weights yourself and serve them with vLLM (`deploy/pm1-rerank-vllm.sh`, `PM1_RERANK_QWEN_VLLM=<url>`), or get the GGUF `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` (`scripts/fetch_rerankers.sh`) and serve it with llama.cpp `--rerank` (`scripts/reranker.sh`, `PM1_RERANK_QWEN=<url>`). |
| `quickmt/quickmt-en-zh`, `quickmt/quickmt-zh-en` | optional query translation between Chinese and English (`pm1.xlate`) | CC-BY-4.0 | Optional. Download them yourself and set `PM1_XLATE_MODEL_ZH` / `PM1_XLATE_MODEL_EN`. |
| `BAAI/bge-reranker-v2-m3` (GGUF from `gpustack/bge-reranker-v2-m3-GGUF`) | optional CPU reranker; eval bake-offs | Apache-2.0 (license not re-checked for this release) | `scripts/fetch_rerankers.sh` |
| `sentence-transformers/all-MiniLM-L6-v2` | BEAM's official event-ordering alignment (eval only) | Apache-2.0 (not re-checked) | `sentence-transformers` downloads it when `eval/beam_qa.py judge` runs |

Without any of these, pm1 still runs. It falls back to BM25 only, with no network access.

## Python dependencies (installed by pip, not vendored)

Core: `numpy` (BSD-3-Clause), `jieba` (MIT). Extras: `fastembed` (Apache-2.0); `pypdf` (BSD), `python-docx` (MIT),
`openpyxl` (MIT); `ctranslate2` (MIT), `sentencepiece` (Apache-2.0). Eval scripts also use `pyarrow`, `nltk`,
`rouge-score`, `sentence-transformers`, `scipy`, `json-repair` and `onnxruntime`, all under permissive licenses.

## Benchmark datasets

`scripts/prepare_data.sh` downloads them into `~/.cache/pm1/data`. The prompt and answer files that were derived from
them during the experiments (about 436 MB) are not included.

| Dataset | Source | License | Used for |
|---|---|---|---|
| LongMemEval-S (cleaned) | HF `xiaowu0162/longmemeval-cleaned`; github.com/xiaowu0162/LongMemEval | MIT (code repository) | retrieval: recall-all@5 (`eval/lme.py`, `eval/lme_strict.py`), QA (`eval/lme_qa.py`) |
| BEAM 100K / 500K / 1M | HF `Mohammadta/BEAM` | CC BY-SA 4.0 | long-conversation QA (`eval/beam_*.py`, `eval/mv_arms.py`) |
| BEAM 10M | HF `Mohammadta/BEAM-10M` | CC BY-SA 4.0 | same |
| LoCoMo | github.com/snap-research/locomo (`data/locomo10.json`) | CC BY-NC 4.0 (**non-commercial**) | LoCoMo results in [EVALUATION.md](EVALUATION.md) |
| SWE-agent trajectories | HF `nebius/SWE-agent-trajectories` (shard 0) | CC-BY-4.0 | coding-trajectory retrieval (`eval/traj_pm1.py`) |
| PerLTQA | arXiv 2402.16288 | CC BY-NC 4.0 | Chinese retrieval check (`eval/perltqa_eval.py`) |

The LoCoMo and coding-trajectory comparisons against other memory systems (Mem0, Graphiti, Hindsight, OpenViking,
graphify, MemoryOS, agentmemory, claude-mem) ran in a separate harness. That harness held copies of those systems
and their logs, so it is **not published**. Their numbers are reported in [EVALUATION.md](EVALUATION.md) and [RESULTS.md](RESULTS.md).
