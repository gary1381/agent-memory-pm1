# Vendored BEAM evaluation code

`compute_metrics.py` and `prompts.py` are copied from the official BEAM benchmark repository
(github.com/mohammadtavakoli78/BEAM, MIT License, Copyright (c) 2025 Mohammad Tavakoli; see `LICENSE` in this folder).
Local change: the module-level LLM client objects were removed; `eval/beam_qa.py` injects an OpenAI-compatible client.
These two files remain under the MIT License, not the repository's Apache-2.0 license.
